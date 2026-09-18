// The temporal resolve's implementation: the barriers around its accumulation, the copy that becomes the next
// frame's history, the hand-off to the lighting stage, and the two things it owns outside a frame - the set
// layout generated from its declaration plus the family that holds one set per swapchain image, and its
// compute pipeline. Its barrier reasoning is the shape any running mean has: the accumulation is read as history
// at one end of the frame and written at the other, so it needs a transition on each side, and the copy that
// becomes the next frame's history is a second one. The depth and velocity it also reads are its OWN bindings,
// so those two transitions belong to the pass as well.

module;

#include <algorithm>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

module vulkan.pass.megalights_temporal;

import vulkan.render_resource;
import vulkan.constant_init;
import vulkan.pipelines; // build_megalights_temporal: the compute pipeline this pass owns
import utility;

namespace vulkan::pass {

    megalights_temporal_pass::~megalights_temporal_pass() {
        this->release_owned();
    }

    void megalights_temporal_pass::release_owned() noexcept {
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

    render_resource::pass_io const& megalights_temporal_pass::io() const noexcept {
        return render_resource::megalights_temporal_io;
    }

    vulkan::pass::behaviour const& megalights_temporal_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view megalights_temporal_pass::feature() const noexcept {
        // The CHAIN's feature: this pass is required, so "the chain is on" is the right gate - and the
        // renderer's predicate for it includes this pass having built its pipeline.
        return "megalights";
    }

    bool megalights_temporal_pass::ready() const noexcept {
        return this->pipeline_.has_value() && this->set_layout_ != VK_NULL_HANDLE;
    }

    VkPipeline megalights_temporal_pass::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout megalights_temporal_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    VkDescriptorSetLayout megalights_temporal_pass::set_layout() const noexcept {
        return this->set_layout_;
    }

    bool megalights_temporal_pass::resolved() const noexcept {
        return this->resolved_;
    }

    void megalights_temporal_pass::set_accumulation(float const depth_tolerance, float const max_frames) noexcept {
        // The clamps live with the values: a tolerance of 0 would reject every history (the accumulation could
        // never grow past one frame), and a cap below 1 would divide the running mean by zero.
        this->depth_tolerance_ = std::max(depth_tolerance, 0.0f);
        this->max_frames_ = std::clamp(max_frames, 1.0f, max_frames_limit);
    }

    void megalights_temporal_pass::set_spatial(float const sigma) noexcept {
        this->spatial_sigma_ = std::clamp(sigma, 0.0f, 4.0f);
    }

    void megalights_temporal_pass::set_frame(megalights_temporal_frame const& frame) noexcept {
        this->frame_ = frame;
        // A NEW FRAME BEGINS, the same per-frame answer the GI resolve gives: `resolved()` is exactly "this
        // frame's dispatch happened", which the renderer reads to decide whether the lighting stage may add the
        // accumulation at all.
        this->resolved_ = false;
    }

    void megalights_temporal_pass::prepare_frame(frame_facts const& facts) noexcept {
        megalights_temporal_frame frame = this->frame_; // the callback the owner installed survives this call
        frame.history_valid = facts.megalights_history_valid;
        this->set_frame(frame);
    }

    void megalights_temporal_pass::on_swapchain_recreated(pass_host const&) {
        this->resolved_ = false;
        // The family is OURS to retire: its sets name that generation's images, so they are stale the moment
        // the swapchain is rebuilt.
        this->family_.retire_all();
    }

    void megalights_temporal_pass::create(pass_context const& context) {
        if (context.device == VK_NULL_HANDLE) {
            return;
        }
        if (this->device_ != VK_NULL_HANDLE && this->device_ != context.device) {
            this->release_owned();
        }
        this->device_ = context.device;
        this->samplers_ = context.samplers;
        if (this->ready()) {
            return; // already built for this device
        }
        std::span<unsigned char const> const spirv = context.shader != nullptr ? context.shader(context.owner, shader_name) : std::span<unsigned char const>{};
        if (spirv.empty()) {
            utility::log("stochastic punctual lighting's temporal resolve disabled (the chain will stay off): the owner has no {}", shader_name);
            return;
        }
        // THE SET LAYOUT IS GENERATED FROM THE DECLARATION (the same call the GI denoiser makes): the three
        // bindings and the family's pool count cannot drift apart because both read this declaration.
        std::expected<VkDescriptorSetLayout, std::string> const layout = bindings::make_set_layout(context.device, render_resource::megalights_temporal_io, render_resource::megalights_temporal_io.own_set);
        if (!layout.has_value()) {
            utility::log("stochastic punctual lighting's temporal resolve disabled (the chain will stay off): {}", layout.error());
            return;
        }
        this->set_layout_ = *layout;
        // ONE set layout, this pass's own (see the declaration): its depth and velocity are own bindings too, so
        // there is no shared set to hand over - which is the shape `build_resolve_pipeline` takes.
        auto built = pipelines::build_resolve_pipeline(context.device, this->set_layout_, render_resource::megalights_temporal_io.push->size, spirv);
        if (!built) {
            utility::log("stochastic punctual lighting's temporal resolve disabled (the chain will stay off): {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->resolve);
        utility::log("SUCCESS: stochastic punctual lighting's temporal resolve created (running mean with a per-pixel frame count)");
    }

    void megalights_temporal_pass::record(resolved_io const& io) {
        this->resolved_ = false;
        if (!this->ready() || io.barrier_images.size() < render_resource::megalights_temporal_barriers.size() || io.frame.image_count == 0 || io.pipelines.empty() ||
            io.pipelines[0] == VK_NULL_HANDLE || io.pipeline_layout == VK_NULL_HANDLE || io.extent.width == 0 || io.extent.height == 0) {
            return; // the runner resolves all of this or skips the pass (the declaration's own gates are the table's)
        }
        VkImage const resolve_image = io.barrier_images[barrier_resolve].image;
        VkImage const history_image = io.barrier_images[barrier_history].image;

        // ---- the set this dispatch binds: OURS, written from the per-image views ----
        constexpr std::size_t own_binding_count = render_resource::megalights_temporal_io.bindings.size();
        auto const views_for = [&io](uint32_t const image, std::array<VkImageView, own_binding_count>& out) {
            for (std::size_t b = 0; b < out.size(); ++b) {
                if (io.own_per_image[b].size() <= image) {
                    return false;
                }
                out[b] = io.own_per_image[b][image];
            }
            return true;
        };
        auto const write_sets = [&views_for, this](uint32_t const image_index, std::span<VkDescriptorSet const> const sets) {
            std::array<VkImageView, own_binding_count> views = {};
            if (!views_for(image_index, views)) {
                utility::log("megalights_temporal: no per-image views for image {} - this frame has no stochastic lighting", image_index);
                return;
            }
            auto const written = bindings::write_set(this->device_, render_resource::megalights_temporal_io, render_resource::megalights_temporal_io.own_set, sets[0], views, {}, this->samplers_);
            if (!written) {
                utility::log("megalights_temporal: {}", written.error());
            }
        };
        std::array<VkImageView, own_binding_count> signature = {};
        if (!views_for(0u, signature)) {
            return; // the host filled nothing: not a frame this pass can resolve
        }
        uint32_t const descriptors_per_set = render_resource::descriptor_counts_for(render_resource::megalights_temporal_io, render_resource::megalights_temporal_io.own_set).total();
        if (!this->family_.ensure(this->device_, this->set_layout_, io.frame.image_count, 1u, descriptors_per_set, signature, write_sets)) {
            utility::log("megalights_temporal: descriptor sets unavailable - this frame has no stochastic lighting");
            return;
        }
        VkDescriptorSet const set = this->family_.set(io.frame.image_index, 0);
        if (set == VK_NULL_HANDLE) {
            utility::log("megalights_temporal: no descriptor set for image {} - this frame has no stochastic lighting", io.frame.image_index);
            return;
        }

        // Layouts, all before the dispatch. The accumulation is READ across frames (the lighting stage samples
        // it after this pass) and the history only by the resolve, which is a copy's destination first.
        std::array<VkImageMemoryBarrier2, 2> barriers = {};
        uint32_t count = 0;
        barriers[count] = vulkan::undefined_to_general_transition; // the accumulation is fully overwritten
        barriers[count].image = resolve_image;
        ++count;
        if (!this->frame_.history_valid) {
            // FIRST USE for this image: the history's contents are whatever the allocation held, so the
            // descriptor has to be legal without their being readable - UNDEFINED -> SHADER_READ.
            barriers[count] = vulkan::undefined_to_sampling_transition;
            barriers[count].image = history_image;
            ++count;
        }
        VkDependencyInfo const dependency = make_image_dependency_info(count, barriers.data());
        vkCmdPipelineBarrier2(io.cmd, &dependency);

        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, io.pipeline_layout, 0, 1, &set, 0, nullptr);

        push_constants push = {};
        push.params = glm::vec4(io.constants.proj[2][2], io.constants.proj[3][2], this->max_frames_, this->depth_tolerance_);
        push.extents = glm::vec4(static_cast<float>(io.extent.width), static_cast<float>(io.extent.height), this->spatial_sigma_, 0.0f);
        static_assert(sizeof(push) <= pass::max_push_bytes, "the resolve's push block must fit the guaranteed minimum");
        [[maybe_unused]] bool const pushed = io.push_block(io.cmd, pass::push_bytes(push));
        vkCmdDispatch(io.cmd, (io.extent.width + group_size - 1u) / group_size, (io.extent.height + group_size - 1u) / group_size, 1);

        // ---- the accumulation becomes the next frame's history ----
        // A copy rather than a ping-pong, exactly like the GI resolve: the accumulation is what the lighting
        // stage samples, so the history has to be a second image and copying into it keeps every descriptor set
        // in the frame stable.
        std::array<VkImageMemoryBarrier2, 2> copy_barriers = {};
        copy_barriers[0] = vulkan::general_to_transfer_src_transition; // resolve: GENERAL -> TRANSFER_SRC
        copy_barriers[0].image = resolve_image;
        copy_barriers[1] = vulkan::sampling_to_transfer_dst_transition;
        copy_barriers[1].image = history_image;
        VkDependencyInfo const copy_dependency = make_image_dependency_info(static_cast<uint32_t>(copy_barriers.size()), copy_barriers.data());
        vkCmdPipelineBarrier2(io.cmd, &copy_dependency);

        VkImageCopy const region = {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffset = {0, 0, 0},
            .extent = {io.extent.width, io.extent.height, 1},
        };
        vkCmdCopyImage(io.cmd, resolve_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, history_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // Hand both on: the accumulation to the lighting stage that adds it (SHADER_READ, which its binding 17
        // declares) and the history copy to the next frame's resolve.
        std::array<VkImageMemoryBarrier2, 2> hand_back = {};
        hand_back[0] = vulkan::transfer_src_to_sampling_transition; // resolve -> SHADER_READ
        hand_back[0].image = resolve_image;
        hand_back[1] = vulkan::transfer_dst_to_sampling_transition; // history -> SHADER_READ
        hand_back[1].image = history_image;
        VkDependencyInfo const hand_back_dependency = make_image_dependency_info(static_cast<uint32_t>(hand_back.size()), hand_back.data());
        vkCmdPipelineBarrier2(io.cmd, &hand_back_dependency);

        this->resolved_ = true;
    }

} // namespace vulkan::pass
