// The TAA resolve's implementation: the barriers, the fullscreen draw into the HDR target, and the copy that
// becomes the next frame's history. Moved out of `vulkan.runtime` unchanged in behaviour - the same barrier
// batches in the same order, the same attachment, the same copy - so the capture gate can decide the move on
// `deferred_taa_fxaa`, the scenario that runs with TAA on. What did NOT move, and why, is in the header: the
// G-buffer depth's transition stays with the host, because the flag that says whether it needs one belongs to
// the G-buffer pass.

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

module vulkan.pass.taa;

import vulkan.render_resource;
import vulkan.constant_init;
import vulkan.pipelines;
import utility;

namespace vulkan::pass {

    taa_pass::~taa_pass() {
        this->release_owned();
    }

    void taa_pass::release_owned() noexcept {
        // The order they were made: the set layout first, the pipeline layout FROM it, the pipeline from that.
        this->pipeline_.reset();
        if (this->pipeline_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->device_, this->pipeline_layout_, nullptr);
            this->pipeline_layout_ = VK_NULL_HANDLE;
        }
        if (this->set_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(this->device_, this->set_layout_, nullptr);
            this->set_layout_ = VK_NULL_HANDLE;
        }
    }

    render_resource::pass_io const& taa_pass::io() const noexcept {
        return render_resource::taa_io;
    }

    vulkan::pass::behaviour const& taa_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view taa_pass::feature() const noexcept {
        return "taa";
    }

    bool taa_pass::pipeline_ready() const noexcept {
        return this->pipeline_.has_value();
    }

    VkPipeline taa_pass::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout taa_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    bool taa_pass::wrote_history() const noexcept {
        return this->wrote_history_;
    }

    void taa_pass::reset_history() noexcept {
        // The off -> on edge: the renderer calls this when TAA is switched on, because blending against
        // frames that were never resolved shows the alias instead of hiding it.
        this->history_valid_.assign(this->history_valid_.size(), false);
    }

    void taa_pass::on_swapchain_recreated(pass_host const&) {
        // THE PASS'S OWN GENERATION RESET, and the framework's `recreate_stage` is what guarantees it happens:
        // the family is retired (its pool may still be named by a recorded command buffer, so it is retired
        // rather than destroyed), the histories belong to images that no longer exist, and the cached
        // generation fingerprint is exactly the thing that identifies the OLD generation.
        this->family_.retire_all();
        this->history_valid_.assign(this->history_valid_.size(), false);
        this->generation_views_valid_ = false;
        this->wrote_history_ = false;
    }

    void taa_pass::create(pass_context const& context) {
        if (context.device == VK_NULL_HANDLE) {
            return;
        }
        if (this->device_ != VK_NULL_HANDLE && this->device_ != context.device) {
            this->release_owned();
        }
        this->device_ = context.device;
        this->samplers_ = context.samplers;
        if (this->set_layout_ != VK_NULL_HANDLE) {
            return; // already built for this device
        }
        std::span<unsigned char const> const vertex_spirv = context.shader != nullptr ? context.shader(context.owner, vertex_shader_name) : std::span<unsigned char const>{};
        std::span<unsigned char const> const fragment_spirv = context.shader != nullptr ? context.shader(context.owner, fragment_shader_name) : std::span<unsigned char const>{};
        if (vertex_spirv.empty() || fragment_spirv.empty()) {
            utility::log("taa disabled: the owner has no {} or {}", vertex_shader_name, fragment_shader_name);
            return;
        }
        std::expected<VkDescriptorSetLayout, std::string> const layout = bindings::make_set_layout(context.device, render_resource::taa_io, render_resource::taa_io.own_set);
        if (!layout.has_value()) {
            utility::log("taa disabled: {}", layout.error());
            return;
        }
        this->set_layout_ = *layout;
        // The pipeline layout is built from the DECLARATION's own set and the DECLARATION's push range, so
        // the range the driver is told about is the one the host composes into.
        auto built = pipelines::build_taa(context.device, this->set_layout_, render_resource::taa_io.push->size, vertex_spirv, fragment_spirv);
        if (!built) {
            utility::log("taa disabled: {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->resolve);
        utility::log("SUCCESS: TAA resolve pipeline created (history reprojection over the deferred path)");
    }

    void taa_pass::record(resolved_io const& io) {
        this->wrote_history_ = false;
        if (io.own.size() < own_binding_count || io.targets.empty() || io.push.size() < sizeof(push_constants) || this->set_layout_ == VK_NULL_HANDLE) {
            return; // the runner resolves all of this or skips the pass
        }
        uint32_t const index = io.frame.image_index;
        if (this->history_valid_.size() != io.frame.image_count) {
            this->history_valid_.assign(io.frame.image_count, false);
        }
        if (index >= this->history_valid_.size()) {
            return;
        }
        bool const history_valid = this->history_valid_[index];

        // THE GENERATION FINGERPRINT: the family must rebind when the swapchain was rebuilt, and the views of
        // the image being resolved RIGHT NOW are the wrong thing to compare - they differ every frame, which
        // would make the family rewrite every set (and rewriting a set a pending frame names is a validation
        // error). One image's views identify the generation, so the first frame after a recreation caches
        // them, and `on_swapchain_recreated` is what drops them.
        if (!this->generation_views_valid_) {
            for (uint32_t b = 0; b < own_binding_count; ++b) {
                this->generation_views_[b] = io.own[b].view;
            }
            this->generation_views_valid_ = true;
        }
        // One list of four views is the whole fingerprint: the four inputs are one KIND of thing as far as the
        // family is concerned (it compares what the sets point at, and every set points at one image's four).
        auto const write_sets = [&io, this](uint32_t const /*image_index*/, std::span<VkDescriptorSet const> const sets) {
            // The four views are all this lambda decides: everything else - the binding numbers, the
            // descriptor type, the count, the SHADER_READ layout each declares and the sampler the
            // declaration chose - is generated from the pass's own declaration by `bindings::write_set`.
            std::array<VkImageView, own_binding_count> const views = {io.own[0].view, io.own[1].view, io.own[2].view, io.own[3].view};
            auto const written = bindings::write_set(this->device_, render_resource::taa_io, render_resource::taa_io.own_set, sets[0], views, {}, this->samplers_);
            if (!written) {
                utility::log("taa: {}", written.error());
            }
        };
        uint32_t const descriptors_per_set = render_resource::descriptor_counts_for(render_resource::taa_io, render_resource::taa_io.own_set).total();
        if (!this->family_.ensure(this->device_, this->set_layout_, io.frame.image_count, 1u, descriptors_per_set, std::span<VkImageView const>(this->generation_views_), write_sets)) {
            utility::log("taa: descriptor sets unavailable - TAA skipped");
            return;
        }
        VkDescriptorSet const set = this->family_.set(index, 0);
        if (set == VK_NULL_HANDLE) {
            utility::log("taa: no descriptor set - the frame is shown unresolved");
            return;
        }

        // Layouts, all before vkCmdBeginRendering: this frame's colour, the motion vectors and (on its first
        // use for this image) the history become inputs, and the HDR target - still untouched this frame -
        // becomes the resolve's attachment. The history is left in SHADER_READ_ONLY by the previous frame's
        // copy and is only ever read as a texture, so it needs no barrier once it is valid.
        std::array<VkImageMemoryBarrier2, 4> barriers = {};
        barriers[0] = vulkan::hdr_sampling_transition; // scene_color: COLOR_ATTACHMENT -> SHADER_READ
        barriers[0].image = io.own[0].image;
        barriers[1] = vulkan::hdr_sampling_transition; // velocity: the same transition, COLOR aspect
        barriers[1].image = io.own[2].image;
        uint32_t barrier_count = 2;
        if (!history_valid) {
            barriers[barrier_count] = vulkan::undefined_to_sampling_transition;
            barriers[barrier_count].image = io.own[1].image;
            ++barrier_count;
        }
        VkDependencyInfo const dependency = make_image_dependency_info(barrier_count, barriers.data());
        vkCmdPipelineBarrier2(io.cmd, &dependency);

        // NOTE: the G-buffer depth the disocclusion guard samples is transitioned by the HOST, just before
        // this stage (see runtime::record_taa_pass): it is a shared per-image transition whose flag belongs
        // to the G-buffer pass, and a pass can only declare its own bindings. The order the host has to
        // preserve is "after this batch, before the draw" only in the sense that the barrier must precede the
        // draw - the barrier commands are independent of this batch, so the host places them first.

        std::array<VkImageMemoryBarrier2, 1> output_barrier = {vulkan::color_attachment_transition};
        output_barrier[0].image = io.targets[0].image;
        VkDependencyInfo const output_dependency = make_image_dependency_info(1, output_barrier.data());
        vkCmdPipelineBarrier2(io.cmd, &output_dependency);

        // The runner has bound the pipeline and set the viewport and scissor from io.extent (this pass
        // declared resync_viewport); what a fullscreen pass still owns is its instance - the load op is its
        // knowledge - and the state the viewport fields do not cover.
        VkClearValue clear = {};
        VkRenderingAttachmentInfo const color_attachment = make_color_attachment_info(io.targets[0].view, clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
        VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, io.extent}, true, &color_attachment, nullptr);
        vkCmdBeginRendering(io.cmd, &rendering_info);
        vkCmdSetCullMode(io.cmd, VK_CULL_MODE_NONE);
        VkDescriptorSet const draw_set = set;
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, io.pipeline_layout, 0, 1, &draw_set, 0, nullptr);
        push_constants push = {};
        std::memcpy(&push, io.push.data(), sizeof(push));
        push.history_valid = history_valid ? 1.0f : 0.0f; // the pass's own lane, not the owner's value
        vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(io.cmd, 3, 1, 0, 0);
        vkCmdEndRendering(io.cmd);

        // ---- the resolved frame becomes the next frame's history ----
        // A copy rather than a ping-pong: the resolve necessarily writes the image the post chain reads, so
        // the history has to be a separate image, and copying into it keeps every descriptor set in the frame
        // stable (no per-frame rewrites). The barriers move the HDR target out to TRANSFER_SRC and back - the
        // post chain still finds it in COLOR_ATTACHMENT_OPTIMAL, exactly where it expects it.
        std::array<VkImageMemoryBarrier2, 2> copy_barriers = {};
        copy_barriers[0] = vulkan::color_attachment_to_transfer_transition; // HDR -> TRANSFER_SRC
        copy_barriers[0].image = io.targets[0].image;
        copy_barriers[1] = vulkan::sampling_to_transfer_dst_transition; // history: SHADER_READ -> TRANSFER_DST
        copy_barriers[1].image = io.own[1].image;
        VkDependencyInfo const copy_dependency = make_image_dependency_info(static_cast<uint32_t>(copy_barriers.size()), copy_barriers.data());
        vkCmdPipelineBarrier2(io.cmd, &copy_dependency);

        VkImageCopy const region = {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffset = {0, 0, 0},
            .extent = {io.extent.width, io.extent.height, 1},
        };
        vkCmdCopyImage(io.cmd, io.targets[0].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, io.own[1].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // hand both images on: the HDR target back to the post chain, the history copy to the next frame's
        // resolve (which will find it in TRANSFER_DST and transition it from there)
        std::array<VkImageMemoryBarrier2, 2> hand_back = {};
        hand_back[0] = vulkan::transfer_to_color_attachment_transition; // HDR -> COLOR_ATTACHMENT
        hand_back[0].image = io.targets[0].image;
        hand_back[1] = vulkan::transfer_dst_to_sampling_transition; // history -> SHADER_READ
        hand_back[1].image = io.own[1].image;
        VkDependencyInfo const hand_back_dependency = make_image_dependency_info(static_cast<uint32_t>(hand_back.size()), hand_back.data());
        vkCmdPipelineBarrier2(io.cmd, &hand_back_dependency);

        this->history_valid_[index] = true;
        this->wrote_history_ = true;
    }

} // namespace vulkan::pass
