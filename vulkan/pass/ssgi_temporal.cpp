// The diffuse temporal resolve's implementation: the barriers, the dispatch, the push lanes that describe this
// pass's own state, the copy that becomes the next frame's history, and the hand-backs. The recording was moved
// out of `runtime::record_ssgi_resolve_pass`'s mode-0 path UNCHANGED in behaviour - the same barrier order, the
// same bind, the same 48-byte push, the same copy and the same two last barriers - so the capture gate decides
// the move on the four GI scenarios. Its set layout, pipeline layout and pipeline are the pass's as well now:
// built here from its own declaration and its own shader (the runtime's `make_ssgi_temporal_pipeline` is gone).

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

module vulkan.pass.ssgi_temporal;

import vulkan.render_resource;
import vulkan.constant_init;
import vulkan.pipelines; // build_ssgi_temporal: the compute pipeline this pass owns
import utility;

namespace vulkan::pass {

    ssgi_temporal_pass::~ssgi_temporal_pass() {
        this->release_owned();
    }

    void ssgi_temporal_pass::release_owned() noexcept {
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

    render_resource::pass_io const& ssgi_temporal_pass::io() const noexcept {
        return render_resource::ssgi_temporal_io;
    }

    vulkan::pass::behaviour const& ssgi_temporal_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view ssgi_temporal_pass::feature() const noexcept {
        // The CHAIN's feature: the resolve is required, so "the chain is on" is the right gate - and the
        // renderer's predicate for it already includes this pass's pipeline.
        return "ssgi";
    }

    bool ssgi_temporal_pass::pipeline_ready() const noexcept {
        return this->pipeline_.has_value() && this->set_layout_ != VK_NULL_HANDLE;
    }

    VkPipeline ssgi_temporal_pass::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout ssgi_temporal_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    VkDescriptorSetLayout ssgi_temporal_pass::set_layout() const noexcept {
        return this->set_layout_;
    }

    void ssgi_temporal_pass::create(pass_context const& context) {
        if (context.device == VK_NULL_HANDLE) {
            return;
        }
        if (this->device_ != VK_NULL_HANDLE && this->device_ != context.device) {
            this->release_owned();
        }
        this->device_ = context.device;
        this->samplers_ = context.samplers;
        if (this->pipeline_ready()) {
            return; // already built for this device
        }
        std::span<unsigned char const> const spirv = context.shader != nullptr ? context.shader(context.owner, shader_name) : std::span<unsigned char const>{};
        if (spirv.empty()) {
            utility::log("GI temporal denoiser disabled (screen-space GI will stay off): the owner has no {}", shader_name);
            return;
        }
        // THE SET LAYOUT IS GENERATED FROM THE DECLARATION - the whole reason `render_resource::ssgi_temporal_io`
        // exists: the seven bindings and the family's pool count were two hand-written lists once, and the
        // validation layer named the mismatch between them. `descriptor_counts_for` (the pool) and this call (the
        // layout) now read the same declaration, and they are read on opposite sides of the ownership line: the
        // layout here, the pool in the renderer's family, which takes this layout through `set_layout()`.
        std::expected<VkDescriptorSetLayout, std::string> const layout = bindings::make_set_layout(context.device, render_resource::ssgi_temporal_io, render_resource::ssgi_temporal_io.own_set);
        if (!layout.has_value()) {
            utility::log("GI temporal denoiser disabled (screen-space GI will stay off): {}", layout.error());
            return;
        }
        this->set_layout_ = *layout;
        auto built = pipelines::build_ssgi_temporal(context.device, this->set_layout_, render_resource::ssgi_temporal_io.push->size, spirv);
        if (!built) {
            utility::log("GI temporal denoiser disabled (screen-space GI will stay off): {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->resolve);
        utility::log("SUCCESS: GI temporal denoiser created (history accumulation)");
    }

    void ssgi_temporal_pass::on_swapchain_recreated(pass_host const&) {
        this->resolved_ = false; // the renderer clears the history flags it owns
        // ... and the family is OURS to retire: its sets name that generation's images, so they are stale the
        // moment the swapchain is rebuilt, and the runner calls this for every pass in a stage - which is what
        // makes the reset unforgettable rather than something the host has to remember about someone else's
        // object.
        this->family_.retire_all();
    }

    bool ssgi_temporal_pass::resolved() const noexcept {
        return this->resolved_;
    }

    void ssgi_temporal_pass::set_frame(ssgi_temporal_frame const& frame) noexcept {
        this->frame_ = frame;
    }

    void ssgi_temporal_pass::record(resolved_io const& io) {
        this->resolved_ = false;
        // NOT `io.own`: this pass does not read the CURRENT frame's binding handles (it writes its sets from the
        // PER-IMAGE views below), so requiring them would be a guard on something it never uses - and it was: the
        // first version of this guard made the pass silently record nothing, which showed up one step later as
        // the reflection's dispatch sampling a motion-vector image still in its attachment layout.
        if (io.barrier_images.size() < render_resource::ssgi_temporal_barriers.size() || io.frame.image_count == 0 ||
            io.pipelines.empty() || io.pipelines[0] == VK_NULL_HANDLE || io.pipeline_layout == VK_NULL_HANDLE ||
            io.push.size() < sizeof(push_constants) || io.extent.width == 0 || io.extent.height == 0) {
            return; // the runner resolves all of this or skips the pass (see runtime::resolve_ssgi_temporal)
        }
        VkImage const resolve_image = io.barrier_images[barrier_resolve].image;
        VkImage const history_image = io.barrier_images[barrier_history].image;

        // ---- the set this dispatch binds: OURS, and written from the per-image views ----
        // Every set the family holds belongs to one swapchain image and must name THAT image's seven views; that
        // is the whole reason `resolved_io::own_per_image` exists, and the two failed extractions this pass's
        // history records are what it cost to learn. NOTE THE INDEX ORDER: `own_per_image[binding][image]`, one
        // span per BINDING whose length is the generation's image count. `bindings::write_set` decides everything
        // else - the binding numbers, the types, the layouts and the sampler each binding's hint chose - from the
        // pass's own declaration, so what is written cannot drift from what `set_layout_` was generated from.
        constexpr std::size_t own_binding_count = render_resource::ssgi_temporal_io.bindings.size();
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
                utility::log("ssgi_temporal: no per-image views for image {} - this frame has no GI", image_index);
                return;
            }
            // ONE SUBSTITUTION, and it is the same one the renderer made before this family was the pass's:
            // binding 6 is the REFLECTION's reprojection, which only mode 1 reads and only the lobe's frame
            // maintains. The diffuse dispatch still has to name a VALID view there (a shader that samples it in a
            // branch leaves the access in the SPIR-V, so validation checks the descriptor whether or not the
            // branch is taken), and naming the lobe's image would make this resolve require a layout that only
            // the lobe establishes - which is UNDEFINED on exactly the frames the lobe is OFF. The depth target is
            // readable on every frame that resolves anything, and mode 0 ignores the value. So the declaration
            // keeps naming the resource the REFLECTION's set binds, and this pass, which alone knows it is mode 0,
            // puts the depth there.
            views[binding_spec_reproject] = views[binding_gbuffer_depth];
            auto const written = bindings::write_set(this->device_, render_resource::ssgi_temporal_io, render_resource::ssgi_temporal_io.own_set, sets[0], views, {}, this->samplers_);
            if (!written) {
                utility::log("ssgi_temporal: {}", written.error());
            }
        };
        // The fingerprint is image 0's views, which are stable for as long as the target generation lives
        // (unlike the frame's own handles, which change image every frame): a recreation rebinds every set, and
        // `on_swapchain_recreated` is what makes that rewrite safe rather than a set a pending frame still names.
        std::array<VkImageView, own_binding_count> signature = {};
        if (!views_for(0u, signature)) {
            return; // the host filled nothing: not a frame this pass can resolve
        }
        uint32_t const descriptors_per_set = render_resource::descriptor_counts_for(render_resource::ssgi_temporal_io, render_resource::ssgi_temporal_io.own_set).total();
        if (!this->family_.ensure(this->device_, this->set_layout_, io.frame.image_count, 1u, descriptors_per_set, signature, write_sets)) {
            utility::log("ssgi_temporal: descriptor sets unavailable - this frame has no GI (its weight stays 0)");
            return;
        }
        VkDescriptorSet const set = this->family_.set(io.frame.image_index, 0);
        if (set == VK_NULL_HANDLE) {
            utility::log("ssgi_temporal: no descriptor set for image {} - this frame has no GI", io.frame.image_index);
            return;
        }

        // Layouts, all before the dispatch (a compute pass may barrier anywhere, but keeping them together is
        // what makes the set of states one image passes through readable):
        //   resolve -> GENERAL (storage write). The old layout is SHADER_READ once the image has been resolved
        //   before, and UNDEFINED on its first frame: the resolve is READ across frames (the tracer samples the
        //   previous frame's copy at a hit, the multi-bounce feedback), so this write has to keep its contents -
        //   claiming UNDEFINED every frame would discard exactly what the feedback reads, which is why the
        //   tracer's first-use barrier leaves it in SHADER_READ.
        //   history -> SHADER_READ, and only on its FIRST use for this image: the previous frame's copy left it
        //   readable (see the hand-back below), so a later frame needs no barrier at all - claiming TRANSFER_DST
        //   as the old layout would be a layout the image is not in. Exactly the TAA resolve's arrangement.
        //   The raw trace needs no barrier either: the tracer handed it to SHADER_READ with a barrier that names
        //   COMPUTE as well as FRAGMENT, which is the read this dispatch does.
        std::array<VkImageMemoryBarrier2, 2> barriers = {};
        uint32_t count = 0;
        barriers[count] = this->frame_.history_valid ? vulkan::sampling_to_general_transition : vulkan::undefined_to_general_transition;
        barriers[count].image = resolve_image;
        ++count;
        if (!this->frame_.history_valid) {
            barriers[count] = vulkan::undefined_to_sampling_transition;
            barriers[count].image = history_image;
            ++count;
        }
        VkDependencyInfo const dependency = make_image_dependency_info(count, barriers.data());
        vkCmdPipelineBarrier2(io.cmd, &dependency);

        // The depth guard samples the G-buffer depth and the reprojection reads the motion-vector target: two
        // shared per-image transitions whose flags belong to the pass that WROTE those images, so the renderer
        // does them (see the header). Called here - after this pass's own barriers, before the dispatch - which
        // is where the moved code called the two accessors.
        if (this->frame_.ensure_inputs != nullptr) {
            this->frame_.ensure_inputs(this->frame_.owner, io.cmd, io.frame.image_index);
        }

        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, io.pipeline_layout, 0, 1, &set, 0, nullptr);
        // The two lanes that describe THIS pass's state are written here rather than by the renderer: whether the
        // history may be trusted, and which signal this dispatch resolves (always the diffuse bounce).
        push_constants push = {};
        std::memcpy(&push, io.push.data(), sizeof(push));
        push.history_valid = this->frame_.history_valid ? 1.0f : 0.0f;
        push.mode = 0.0f;
        vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(io.cmd, (io.extent.width + group_size - 1u) / group_size, (io.extent.height + group_size - 1u) / group_size, 1);

        // ---- the accumulation becomes the next frame's history ----
        // A copy rather than a ping-pong, exactly like the TAA resolve: the resolve writes the image the
        // composite reads, so the history has to be separate, and copying into it keeps every descriptor set in
        // the frame stable. The accumulation is a storage image (GENERAL), so it goes out through TRANSFER_SRC
        // and comes back as a sample - the post chain still finds it in SHADER_READ, where it expects it.
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

        // Hand both on: the accumulation to the composite, the history copy to the next frame's resolve (which
        // finds it in TRANSFER_DST and transitions it from there). The raw trace has been readable since the
        // tracer left it that way and nothing here touches it.
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
