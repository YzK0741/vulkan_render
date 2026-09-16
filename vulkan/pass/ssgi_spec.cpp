// The glossy lobe's implementation: the ordering barrier, the two first-use transitions, the dispatch and the
// hand-off. Moved out of `runtime::record_ssgi_spec_pass` UNCHANGED in behaviour - the same barrier list in the
// same order, the same two sets bound before the pipeline, the same half-resolution dispatch, the same three
// images handed to the denoiser - so the capture gate decides the move on `metal_rough_glossy` and
// `glossy_motion` (the two scenarios built AROUND this feature) plus `default_gi`.

module;

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

module vulkan.pass.ssgi_spec;

import vulkan.render_resource;
import vulkan.constant_init;
import vulkan.pipelines; // build_ssgi_spec: the compute pipeline this pass owns
import utility;

namespace vulkan::pass {

    ssgi_spec_pass::~ssgi_spec_pass() {
        this->release_owned();
    }

    void ssgi_spec_pass::release_owned() noexcept {
        this->pipeline_.reset();
        if (this->pipeline_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->device_, this->pipeline_layout_, nullptr);
            this->pipeline_layout_ = VK_NULL_HANDLE;
        }
    }

    render_resource::pass_io const& ssgi_spec_pass::io() const noexcept {
        return render_resource::ssgi_spec_io;
    }

    vulkan::pass::behaviour const& ssgi_spec_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view ssgi_spec_pass::feature() const noexcept {
        // The renderer's registry answers it (the knob AND the two conditions that make a hit shadeable), which
        // is also the predicate the tracer reads to decide who owes the denoiser the hand-off barrier - one
        // place, so the two cannot disagree.
        return "ssgi_specular";
    }

    bool ssgi_spec_pass::pipeline_ready() const noexcept {
        return this->pipeline_.has_value();
    }

    VkPipeline ssgi_spec_pass::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout ssgi_spec_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    void ssgi_spec_pass::reset_first_use() noexcept {
        this->seen_.assign(this->seen_.size(), false);
    }

    void ssgi_spec_pass::set_reach(float const radius, uint32_t const rays) noexcept {
        // The clamps are the renderer's old ones, moved here with the values: the reach is a fraction of the scene
        // radius (0.001 rather than 0 so a rounded-away value still traces), and one to eight rays per pixel is
        // what the shader's loop and the feature's own definition allow.
        this->radius_ = std::clamp(radius, 0.001f, 8.0f);
        this->rays_ = std::clamp(rays, 1u, 8u);
    }

    void ssgi_spec_pass::on_swapchain_recreated(pass_host const&) {
        // The lobe's two outputs belong to the target generation: new images are in UNDEFINED, so the first-use
        // transition is owed again. The runner calls this for every pass in a stage, which is what makes the
        // reset unforgettable.
        this->seen_.assign(this->seen_.size(), false);
    }

    void ssgi_spec_pass::create(pass_context const& context) {
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
            utility::log("the glossy lobe is disabled: the owner has no {}", shader_name);
            return;
        }
        VkDescriptorSetLayout const scene_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 0) : VK_NULL_HANDLE;
        VkDescriptorSetLayout const gbuffer_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 1) : VK_NULL_HANDLE;
        if (scene_layout == VK_NULL_HANDLE || gbuffer_layout == VK_NULL_HANDLE) {
            utility::log("the glossy lobe is disabled: the owner has no layout for the shared sets this pass binds");
            return;
        }
        auto built = pipelines::build_ssgi_spec(context.device, scene_layout, gbuffer_layout, render_resource::ssgi_spec_io.push->size, spirv);
        if (!built) {
            utility::log("the glossy lobe is disabled: {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->trace);
        utility::log("SUCCESS: the glossy lobe's compute pipeline created (traced specular GI)");
    }

    void ssgi_spec_pass::record(resolved_io const& io) {
        if (!this->pipeline_.has_value() || io.barrier_images.size() < render_resource::ssgi_spec_barriers.size() || io.extent.width == 0 || io.extent.height == 0 ||
            io.pipeline_layout == VK_NULL_HANDLE || io.shared.scene == VK_NULL_HANDLE || io.shared.gbuffer == VK_NULL_HANDLE) {
            return; // the runner resolves all of this or skips the pass (the declaration's own gates are the table's)
        }
        // THE INSTANCE TABLE IS THE SWITCH: with none, a hit cannot be shaded, and the pass does nothing rather
        // than replacing the environment's answer with a worse one. The renderer's feature registry already
        // refuses such a frame (`ssgi_specular`); this is the pass being unable to proceed without it even if it
        // is handed one - the same defensive pair the tracer's frame carries.
        if (io.constants.gi_instance_table == 0u) {
            return;
        }

        uint32_t const index = io.frame.image_index;
        if (this->seen_.size() != io.frame.image_count) {
            this->seen_.assign(io.frame.image_count, false);
        }
        bool const first_use = index < this->seen_.size() && !this->seen_[index];

        // 1. THE ORDERING BARRIER: `gi_trace` is about to be READ as well as written, by a dispatch that is not
        //    the one that wrote it. Consecutive dispatches in one command buffer have no memory dependency
        //    between them, so without this the lobe can read a texel the tracer has not finished writing. Same
        //    layout on both sides (GENERAL), which is why it is the compute-storage barrier rather than a
        //    transition - and why the tracer left the image in GENERAL instead of handing it over itself.
        // 2. THE LOBE'S OWN TWO OUTPUTS: storage images whose first write needs GENERAL, from UNDEFINED once per
        //    target generation and from the readable layout every frame after (the hand-back below leaves them
        //    in SHADER_READ). Claiming UNDEFINED every frame would discard them for nothing; claiming a layout
        //    they are not in is the thing that is actually illegal.
        std::array<VkImageMemoryBarrier2, 3> barriers = {};
        barriers[0] = vulkan::compute_storage_transition;
        barriers[0].image = io.barrier_images[barrier_gi_trace].image;
        uint32_t count = 1;
        VkImageMemoryBarrier2 const to_general = first_use ? vulkan::undefined_to_general_transition : vulkan::sampling_to_general_transition;
        barriers[count] = to_general;
        barriers[count].image = io.barrier_images[barrier_gi_spec_trace].image;
        ++count;
        barriers[count] = to_general;
        barriers[count].image = io.barrier_images[barrier_gi_spec_reproject].image;
        ++count;
        if (index < this->seen_.size()) {
            this->seen_[index] = true;
        }
        VkDependencyInfo const order = make_image_dependency_info(count, barriers.data());
        vkCmdPipelineBarrier2(io.cmd, &order);

        // Two sets, then the pipeline: the tracer's sets, unchanged (set 0 the shared scene set - the camera,
        // the environment, the BRDF LUT, the material table, the top level structure - and set 1 the G-buffer
        // set, whose binding 6 is the raw trace).
        std::array<VkDescriptorSet, 2> const sets = {io.shared.scene, io.shared.gbuffer};
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, io.pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);

        // THE PUSH BLOCK, from its owners: the camera, the scene radius (the reach is a fraction of it), the
        // instance table's address and the ray sequence are the FRAME's (`io.constants`); the reach and the ray
        // count are this PASS's. The address goes into two float lanes because a push constant is raw bytes and
        // half an address survives that trip exactly - the same trick the tracer's `proj_terms` uses.
        float const scene_radius = io.constants.scene_radius;
        float const table_low = std::bit_cast<float>(static_cast<uint32_t>(io.constants.gi_instance_table & 0xFFFFFFFFu));
        float const table_high = std::bit_cast<float>(static_cast<uint32_t>(io.constants.gi_instance_table >> 32u));
        push_constants push = {};
        push.inv_view_proj = io.constants.inv_view_proj;
        // The ray length is the lobe's OWN reach, and z is the self-intersection bias as an explicit WORLD length
        // rather than a fraction of x, so that raising the reach does not also lift every ray's origin further off
        // its surface. w is the ray sequence the shader seeds its sampling with.
        push.params = glm::vec4(this->radius_ * scene_radius, static_cast<float>(this->rays_), scene_radius * 0.0002f,
                                static_cast<float>(io.constants.gi_frame_index));
        push.table = glm::vec4(0.0f, 0.0f, table_low, table_high);
        static_assert(sizeof(push) <= pass::max_push_bytes, "the lobe's push block must fit the guaranteed minimum");
        vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(io.cmd, (io.extent.width + group_size - 1u) / group_size, (io.extent.height + group_size - 1u) / group_size, 1);

        // ... and hand the COMPLETED raw trace to the denoiser: this is the transition the tracer would have
        // written when this pass does not run, moved here because the image has a second writer now, and the
        // barrier has to come after the LAST one or the resolve could sample a half-written trace. The lobe's
        // own two outputs come along: the reflection's resolve reads BOTH as samplers.
        VkImageMemoryBarrier2 const to_sampling = vulkan::general_to_sampling_transition;
        std::array<VkImageMemoryBarrier2, 3> hand_off = {to_sampling, to_sampling, to_sampling};
        hand_off[0].image = io.barrier_images[barrier_gi_trace].image;
        hand_off[1].image = io.barrier_images[barrier_gi_spec_trace].image;
        hand_off[2].image = io.barrier_images[barrier_gi_spec_reproject].image;
        VkDependencyInfo const sampling = make_image_dependency_info(static_cast<uint32_t>(hand_off.size()), hand_off.data());
        vkCmdPipelineBarrier2(io.cmd, &sampling);
    }

} // namespace vulkan::pass
