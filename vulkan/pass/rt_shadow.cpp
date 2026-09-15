// The ray-traced shadow pass's implementation: the two barriers around the visibility image, the two shared sets
// and the full-resolution dispatch, plus the pipeline it owns. The recording was moved out of
// `runtime::record_rt_shadow_pass` UNCHANGED in behaviour - the same UNDEFINED -> GENERAL transition before the
// dispatch, the same bind order (scene set, then G-buffer set), the same 80-byte push, the same workgroup size
// and the same GENERAL -> SHADER_READ hand-back.
//
// HOW IT WAS VERIFIED, and this is the interesting part: **the capture gate cannot decide this pass.** No
// scenario in `scripts/windows/check_render.ps1` sets `rt_shadows`, and the config default is `false`, so all
// twelve reference frames were captured with the pass NOT running - which is why the 12 x 2 gate is green for
// this change and why that green is NOT the evidence. The evidence is an A/B against the parent commit's binary
// with a scenario that pins `rt_shadows = true`: both builds produce the SAME hash
// (AEEB757EA347CC4178006F7FD5C1795AFD7274556D72C9028ABB3970B3DA72BB, twice each, validation clean). See
// docs/pass_chain_plan.md for the exact command and what it means for the gate's coverage.

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

module vulkan.pass.rt_shadow;

import vulkan.render_resource;
import vulkan.constant_init;
import vulkan.pipelines; // build_rt_shadow: the compute pipeline this pass owns
import utility;

namespace vulkan::pass {

    rt_shadow_pass::~rt_shadow_pass() {
        this->release_owned();
    }

    void rt_shadow_pass::release_owned() noexcept {
        this->pipeline_.reset();
        if (this->pipeline_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->device_, this->pipeline_layout_, nullptr);
            this->pipeline_layout_ = VK_NULL_HANDLE;
        }
    }

    render_resource::pass_io const& rt_shadow_pass::io() const noexcept {
        return render_resource::rt_shadow_io;
    }

    vulkan::pass::behaviour const& rt_shadow_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view rt_shadow_pass::feature() const noexcept {
        // The renderer's registry answers whether the ray-traced path is on at all (the knob AND a device with
        // ray queries); this pass's own answer is whether it built its pipeline, which the registry also asks.
        return "rt_shadow";
    }

    bool rt_shadow_pass::pipeline_ready() const noexcept {
        return this->pipeline_.has_value();
    }

    VkPipeline rt_shadow_pass::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout rt_shadow_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    void rt_shadow_pass::create(pass_context const& context) {
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
            utility::log("ray-traced shadows unavailable: the owner has no {}", shader_name);
            return;
        }
        // The two set layouts come from the CONTEXT: this pass binds the shared scene set and the shared G-buffer
        // set and owns no layout of its own (see pass_context::shared_set_layout).
        VkDescriptorSetLayout const scene_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 0) : VK_NULL_HANDLE;
        VkDescriptorSetLayout const gbuffer_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 1) : VK_NULL_HANDLE;
        if (scene_layout == VK_NULL_HANDLE || gbuffer_layout == VK_NULL_HANDLE) {
            utility::log("ray-traced shadows unavailable: the owner has no layout for the shared sets this pass binds");
            return;
        }
        auto built = pipelines::build_rt_shadow(context.device, scene_layout, gbuffer_layout, static_cast<uint32_t>(sizeof(push_constants)), spirv);
        if (!built) {
            utility::log("ray-traced shadows unavailable: {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->trace);
        // The line the runtime used to log when it built this pipeline: a pass that says what it built is what
        // makes a missing one visible in the startup log rather than in a frame that looks merely shadowless.
        utility::log("SUCCESS: ray-traced sun shadow pipeline created (one ray per pixel, terminated on first hit)");
    }

    void rt_shadow_pass::on_swapchain_recreated(pass_host const&) {
        // Nothing to reset: this pass owns no per-generation state. Its visibility image is per FRAME SLOT and
        // belongs to the acceleration structures' own lifetime (the renderer rebuilds both together), and its
        // pipeline does not depend on the surface's format or size - so the generation change is not its event.
        // The one-shot log line stays set, because "this pass traces rays" does not become untrue on a resize.
    }

    void rt_shadow_pass::record(resolved_io const& io) {
        if (!this->pipeline_.has_value() || io.barrier_images.size() < render_resource::rt_shadow_barriers.size() ||
            io.pipelines.empty() || io.pipelines[0] == VK_NULL_HANDLE || io.pipeline_layout == VK_NULL_HANDLE ||
            io.shared.scene == VK_NULL_HANDLE || io.shared.gbuffer == VK_NULL_HANDLE || io.push.size() < sizeof(push_constants) ||
            io.extent.width == 0 || io.extent.height == 0) {
            return; // the runner resolves all of this or skips the pass (see runtime::resolve_rt_shadow)
        }
        VkImage const visibility = io.barrier_images[barrier_visibility].image;
        if (visibility == VK_NULL_HANDLE) {
            return;
        }

        // The image is written as a storage image (GENERAL) and read by the lighting stage as a sampler
        // (SHADER_READ). Both transitions happen here, around the dispatch, because this is the only place that
        // knows the image is being rewritten - the lighting stage's descriptor declares SHADER_READ whether or
        // not this pass ran (see the off path in the frame loop).
        VkImageMemoryBarrier2 to_general = vulkan::undefined_to_general_transition;
        to_general.image = visibility;
        VkDependencyInfo const general_dependency = make_image_dependency_info(1, &to_general);
        vkCmdPipelineBarrier2(io.cmd, &general_dependency);

        std::array<VkDescriptorSet, 2> const sets = {io.shared.scene, io.shared.gbuffer};
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, io.pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);

        push_constants push = {};
        std::memcpy(&push, io.push.data(), sizeof(push));
        vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(io.cmd, (io.extent.width + group_size - 1u) / group_size, (io.extent.height + group_size - 1u) / group_size, 1);

        VkImageMemoryBarrier2 to_sampling = vulkan::general_to_sampling_transition;
        to_sampling.image = visibility;
        VkDependencyInfo const sampling_dependency = make_image_dependency_info(1, &to_sampling);
        vkCmdPipelineBarrier2(io.cmd, &sampling_dependency);

        if (!this->logged_) {
            this->logged_ = true;
            utility::log("ray-traced shadows: tracing {}x{} rays per frame (one per pixel, terminated on the first hit)", io.extent.width, io.extent.height);
        }
    }

} // namespace vulkan::pass
