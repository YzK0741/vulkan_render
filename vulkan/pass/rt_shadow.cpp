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
// (AEEB757EA347CC4178006F7FD5C1795AFD7274556D72C9028ABB3970B3DA72BB, twice each, validation clean).

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>
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
        auto const fetch = [&context](std::string_view const name) {
            return context.shader != nullptr ? context.shader(context.owner, name) : std::span<unsigned char const>{};
        };
        std::span<unsigned char const> const raygen = fetch(raygen_name);
        std::span<unsigned char const> const closest_hit = fetch(closest_hit_name);
        std::span<unsigned char const> const miss = fetch(miss_name);
        std::span<unsigned char const> const any_hit = fetch(any_hit_name);
        if (raygen.empty() || closest_hit.empty() || miss.empty() || any_hit.empty()) {
            utility::log("ray-traced shadows unavailable: the owner has not registered all of {}, {}, {} and {}", raygen_name, closest_hit_name, miss_name, any_hit_name);
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
        auto built = pipelines::build_rt_shadow_ray_tracing(context.device, scene_layout, gbuffer_layout, static_cast<uint32_t>(sizeof(push_constants)), raygen, closest_hit, miss, any_hit);
        if (!built) {
            utility::log("ray-traced shadows unavailable: {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(*built->pipeline);

        // ---- THE SHADER BINDING TABLE, the half of a ray-tracing pipeline that belongs to its CALLER: the
        //      group handles are per-pipeline data, and the STRIDE is a property of the device rather than a
        //      constant. Here a handle is 32 bytes while a region's address must be 64-byte aligned, so using the
        //      handle size as the stride is exactly the first-attempt VUID this pass would otherwise hit. ----
        auto const get_group_handles = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(vkGetDeviceProcAddr(context.device, "vkGetRayTracingShaderGroupHandlesKHR"));
        this->trace_rays_ = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetDeviceProcAddr(context.device, "vkCmdTraceRaysKHR"));
        if (get_group_handles == nullptr || this->trace_rays_ == nullptr) {
            utility::log("ray-traced shadows unavailable: the device did not publish the traceRays entry points");
            this->release_owned();
            return;
        }
        uint32_t const handle_size = context.ray_tracing_properties.shaderGroupHandleSize;
        uint32_t const handle_alignment = context.ray_tracing_properties.shaderGroupHandleAlignment;
        uint32_t const base_alignment = context.ray_tracing_properties.shaderGroupBaseAlignment;
        if (handle_size == 0 || handle_alignment == 0 || base_alignment == 0 || context.create_upload_buffer == nullptr) {
            utility::log("ray-traced shadows unavailable: this device published no shader binding table numbers to build one against");
            this->release_owned();
            return;
        }
        uint32_t const region_size = ((handle_size + base_alignment - 1u) / base_alignment) * base_alignment;
        uint32_t const group_count = built->group_count;
        std::vector<unsigned char> handles(static_cast<size_t>(group_count) * handle_size);
        if (get_group_handles(context.device, this->pipeline_->get_pipeline(), 0, group_count, handles.size(), handles.data()) != VK_SUCCESS) {
            utility::log("ray-traced shadows unavailable: the shader group handles could not be read back");
            this->release_owned();
            return;
        }
        // One REGION per group, each starting on a base-aligned offset and holding exactly one record: the order
        // is the builder's (raygen, miss, hit), so `handles[group]` lands in region `group`.
        std::vector<unsigned char> table(static_cast<size_t>(group_count) * region_size, 0);
        for (uint32_t group = 0; group < group_count; ++group) {
            std::memcpy(table.data() + static_cast<size_t>(group) * region_size, handles.data() + static_cast<size_t>(group) * handle_size, handle_size);
        }
        VkDeviceAddress address = 0;
        VkBuffer const table_buffer = context.create_upload_buffer(context.owner, table.data(), static_cast<uint64_t>(table.size()), VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR, &address);
        if (table_buffer == VK_NULL_HANDLE || address == 0) {
            utility::log("ray-traced shadows unavailable: the shader binding table buffer could not be created");
            this->release_owned();
            return;
        }
        auto const region = [region_size](VkDeviceAddress const at) { return VkStridedDeviceAddressRegionKHR{.deviceAddress = at, .stride = region_size, .size = region_size}; };
        this->raygen_region_ = region(address);
        this->miss_region_ = region(address + region_size);
        this->hit_region_ = region(address + 2u * region_size);
        // The line the runtime used to log when it built this pipeline: a pass that says what it built is what
        // makes a missing one visible in the startup log rather than in a frame that looks merely shadowless.
        utility::log("SUCCESS: ray-traced sun shadow pipeline created (raygen + miss + closest hit + any hit, one ray per pixel, terminated on first hit)");
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
            io.shared.scene == VK_NULL_HANDLE || io.shared.gbuffer == VK_NULL_HANDLE ||
            this->hit_region_.deviceAddress == 0 ||
            io.extent.width == 0 || io.extent.height == 0) {
            return; // the runner resolves all of this or skips the pass (see frame_pass::resolve)
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
        // THE PRODUCER IS A RAY-TRACING STAGE, NOT A DISPATCH. The shared constants name COMPUTE_SHADER because
        // every other writer of this image was one; a barrier whose masks do not cover the stage that actually
        // ran leaves the writes unsynchronized, and the lighting stage then samples an image the trace has not
        // necessarily finished writing (measured: half the model lost its sun, deterministically, while a
        // constant write - which no ordering can make wrong - came out right).
        to_general.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        VkDependencyInfo const general_dependency = make_image_dependency_info(1, &to_general);
        vkCmdPipelineBarrier2(io.cmd, &general_dependency);

        std::array<VkDescriptorSet, 2> const sets = {io.shared.scene, io.shared.gbuffer};
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, io.pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);

        // THE PUSH BLOCK IS THE PASS'S OWN NOW (S3): the renderer used to compose it and hand it over as raw
        // bytes, which was the last thing it knew about this pass's frame. What it carries is this frame's
        // inverse view-projection - a frame CONSTANT, so it arrives through `resolved_io::constants`, the channel
        // that exists for exactly this - and the three ray-offset terms, which are this shader's own constants and
        // therefore the struct's defaults rather than values anybody has to pass in.
        push_constants push = {};
        push.inv_view_proj = io.constants.inv_view_proj;
        vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR, 0, sizeof(push), &push);
        // THE LAUNCH DIMS ARE THE EXTENT: one invocation per pixel of the visibility image, which is what the
        // compute form got from its dispatch and its bounds check.
        this->trace_rays_(io.cmd, &this->raygen_region_, &this->miss_region_, &this->hit_region_, &this->callable_region_, io.extent.width, io.extent.height, 1);

        VkImageMemoryBarrier2 to_sampling = vulkan::general_to_sampling_transition;
        to_sampling.image = visibility;
        to_sampling.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR; // the trace is the writer (see above)
        VkDependencyInfo const sampling_dependency = make_image_dependency_info(1, &to_sampling);
        vkCmdPipelineBarrier2(io.cmd, &sampling_dependency);

        if (!this->logged_) {
            this->logged_ = true;
            utility::log("ray-traced shadows: tracing {}x{} rays per frame through the ray-tracing pipeline (one per pixel, terminated on the first hit)", io.extent.width, io.extent.height);
        }
    }

} // namespace vulkan::pass
