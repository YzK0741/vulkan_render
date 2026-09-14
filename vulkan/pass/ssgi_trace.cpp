// The SSGI tracer's implementation: the first-use barrier batch, the dispatch and the hand-off. Moved out of
// `runtime::record_ssgi_pass` UNCHANGED in behaviour - the same barrier list in the same order, the same bind
// order (two sets, then the pipeline), the same half-resolution dispatch with the same workgroup size, the same
// hand-off that the glossy lobe's presence postpones - so the capture gate can decide the move on the five GI
// scenarios, of which `default_gi` (traced, hit shading) is the one that exercises this pass hardest.

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

module vulkan.pass.ssgi_trace;

import vulkan.render_resource;
import vulkan.constant_init;
import vulkan.pipelines; // build_ssgi: the compute pipeline this pass owns
import utility;

namespace vulkan::pass {

    ssgi_trace_pass::~ssgi_trace_pass() {
        this->release_owned();
    }

    void ssgi_trace_pass::release_owned() noexcept {
        this->pipeline_.reset();
        if (this->pipeline_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->device_, this->pipeline_layout_, nullptr);
            this->pipeline_layout_ = VK_NULL_HANDLE;
        }
    }

    render_resource::pass_io const& ssgi_trace_pass::io() const noexcept {
        return render_resource::ssgi_trace_io;
    }

    vulkan::pass::behaviour const& ssgi_trace_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view ssgi_trace_pass::feature() const noexcept {
        // The renderer's registry answers it: the chain needs the temporal and spatial stages too, so "the
        // tracer's pipeline exists" is a necessary condition and not a sufficient one - which is exactly what
        // the host's predicate adds.
        return "ssgi";
    }

    bool ssgi_trace_pass::pipeline_ready() const noexcept {
        return this->pipeline_.has_value();
    }

    VkPipeline ssgi_trace_pass::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout ssgi_trace_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    void ssgi_trace_pass::set_frame(ssgi_trace_frame const& frame) noexcept {
        this->frame_ = frame;
    }

    void ssgi_trace_pass::on_swapchain_recreated(pass_host const&) {
        // The probe grid's images belong to a generation; so does the fact that this pass has seen them. The
        // runner calls this for every pass in a stage, which is what makes the reset unforgettable.
        this->probe_grid_seen_ = false;
    }

    void ssgi_trace_pass::create(pass_context const& context) {
        if (context.device == VK_NULL_HANDLE) {
            return;
        }
        if (this->device_ != VK_NULL_HANDLE && this->device_ != context.device) {
            this->release_owned();
        }
        this->device_ = context.device;
        if (this->pipeline_.has_value()) {
            return; // already built for this device
        }
        std::span<unsigned char const> const spirv = context.shader != nullptr ? context.shader(context.owner, shader_name) : std::span<unsigned char const>{};
        if (spirv.empty()) {
            utility::log("screen-space GI disabled: the owner has no {}", shader_name);
            return;
        }
        // THE TWO SET LAYOUTS COME FROM THE CONTEXT, not from this pass: it binds the shared scene set and the
        // shared G-buffer set, and their layouts are their owners' (see pass_context::shared_set_layout). That
        // is also why this pass has no set layout of its own to build.
        VkDescriptorSetLayout const scene_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 0) : VK_NULL_HANDLE;
        VkDescriptorSetLayout const gbuffer_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 1) : VK_NULL_HANDLE;
        if (scene_layout == VK_NULL_HANDLE || gbuffer_layout == VK_NULL_HANDLE) {
            utility::log("screen-space GI disabled: the owner has no layout for the shared sets this pass binds");
            return;
        }
        auto built = pipelines::build_ssgi(context.device, scene_layout, gbuffer_layout, render_resource::ssgi_trace_io.push->size, spirv);
        if (!built) {
            utility::log("screen-space GI disabled: {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->trace);
        utility::log("SUCCESS: ssgi compute pipeline created (screen-space global illumination)");
    }

    void ssgi_trace_pass::record(resolved_io const& io) {
        if (!this->pipeline_.has_value() || io.barrier_images.size() < render_resource::ssgi_trace_barriers.size() || io.push.size() < sizeof(push_constants) ||
            io.extent.width == 0 || io.extent.height == 0) {
            return; // the runner resolves all of this or skips the pass (see runtime::resolve_ssgi_trace)
        }

        // ---- the first-use transitions ----
        // The GI image is the one this pass writes: it becomes GENERAL here and the hand-off below takes it to
        // SHADER_READ for the composite that samples it. The other entries are FIRST USE ONLY, and why each one
        // needs the layout it gets is the kind of fact that belongs next to the barrier rather than in a
        // summary:
        //  * `gi_resolve` is READ first (the bounce feedback samples it), even though this frame's denoiser will
        //    later WRITE it - a first-use transition is for the layout the frame's first reader needs, and
        //    claiming GENERAL here is the validation error this line was written for;
        //  * `gi_spec_resolve` is the same case one step further out: the G-buffer set names it at binding 15,
        //    so it must be in the layout THAT set declares before this - the first pass to bind the set - runs;
        //  * the probe grid's four coefficient volumes are sampled here and its four scratch volumes are only
        //    ever written, so the two halves start in different layouts; the per-cell geometry is written by the
        //    injection. Nine images, and they are needed even when the cache is OFF: the descriptor is always
        //    written with a real view (a null one is illegal), so the images behind it must be legal on every
        //    frame this pass dispatches.
        std::array<VkImageMemoryBarrier2, 12> barriers = {};
        barriers[0] = vulkan::undefined_to_general_transition; // gi_trace: written as a storage image
        barriers[0].image = io.barrier_images[barrier_gi_trace].image;
        uint32_t count = 1;
        if (!this->frame_.history_valid) {
            // ... the reflection's accumulation (the G-buffer set names it at binding 15, so it has to be in
            // that set's layout before this - the first pass to bind the set - runs).
            barriers[count] = vulkan::undefined_to_sampling_transition;
            barriers[count].image = io.barrier_images[barrier_gi_spec_resolve].image;
            ++count;
        }
        if (!this->frame_.history_valid) {
            barriers[count] = vulkan::undefined_to_sampling_transition;
            barriers[count].image = io.barrier_images[barrier_gi_resolve].image;
            ++count;
        }
        if (this->frame_.probe_grid_first_use && !this->probe_grid_seen_) {
            for (uint32_t c = 0; c < 4; ++c) {
                barriers[count] = vulkan::undefined_to_sampling_transition;
                barriers[count].image = io.barrier_images[barrier_probe_grid_read + c].image;
                ++count;
            }
            for (uint32_t c = 0; c < 4; ++c) {
                barriers[count] = vulkan::undefined_to_general_transition;
                barriers[count].image = io.barrier_images[barrier_probe_grid_write + c].image;
                ++count;
            }
            barriers[count] = vulkan::undefined_to_general_transition; // the per-cell geometry: written by the injection
            barriers[count].image = io.barrier_images[barrier_probe_surface].image;
            ++count;
            this->probe_grid_seen_ = true;
        }
        VkDependencyInfo const start = make_image_dependency_info(count, barriers.data());
        vkCmdPipelineBarrier2(io.cmd, &start);

        // Two sets, then the pipeline - the order the moved code used. A compute dispatch has no viewport, so
        // the runner's mechanical half is only the pipeline bind (behaviour::resync_viewport is false).
        std::array<VkDescriptorSet, 2> const sets = {io.shared.scene, io.shared.gbuffer};
        if (io.pipeline_layout == VK_NULL_HANDLE || sets[0] == VK_NULL_HANDLE || sets[1] == VK_NULL_HANDLE) {
            return; // nothing to dispatch into: the sets are the host's, and an unbound one is not a frame
        }
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, io.pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);

        push_constants push = {};
        std::memcpy(&push, io.push.data(), sizeof(push));
        vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(io.cmd, (io.extent.width + group_size - 1u) / group_size, (io.extent.height + group_size - 1u) / group_size, 1);

        // A compute SHADER_WRITE is not visible to a later read without this, and the reader is the denoiser's
        // resolve - UNLESS the glossy lobe runs next, in which case the lobe both reads and writes this image
        // and owes the hand-off itself (see vulkan.pass.ssgi_spec). The predicate is the renderer's, because
        // only it knows which of the two passes runs; getting it wrong is either a validation error or a
        // dependency the resolve does not need yet.
        if (this->frame_.specular_next) {
            return;
        }
        VkImageMemoryBarrier2 to_sampling = vulkan::general_to_sampling_transition;
        to_sampling.image = io.barrier_images[barrier_gi_trace].image;
        VkDependencyInfo const hand_off = make_image_dependency_info(1, &to_sampling);
        vkCmdPipelineBarrier2(io.cmd, &hand_off);
    }

} // namespace vulkan::pass
