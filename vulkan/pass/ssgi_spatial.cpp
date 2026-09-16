// The spatial filter's implementation: the two barriers around its storage output, the two shared sets, its push
// block and the dispatch, plus the two things it owns outside a frame - its pipeline layout and its compute pipeline,
// built from `ssgi_spatial.comp` and the two shared set layouts its owner hands over at create time (the runtime used
// to build this pipeline and pass it in; that entry point is gone), and its filter WIDTH (one pass reads it, so it is
// this pass's parameter - see set_sigma). The recording itself was moved out of `runtime::record_ssgi_spatial_pass`
// UNCHANGED in behaviour - the same barrier pair, the same bind order (scene set, then G-buffer set), the same 48-byte
// push and the same half-resolution dispatch - so the capture gate decides the move on the five GI scenarios.

module;

#include <algorithm>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vulkan/vulkan.h>

module vulkan.pass.ssgi_spatial;

import vulkan.render_resource;
import vulkan.constant_init;
import vulkan.pipelines; // build_ssgi_spatial: the compute pipeline this pass owns
import utility;

namespace vulkan::pass {

    ssgi_spatial_pass::~ssgi_spatial_pass() {
        this->release_owned();
    }

    void ssgi_spatial_pass::release_owned() noexcept {
        this->pipeline_.reset();
        if (this->pipeline_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->device_, this->pipeline_layout_, nullptr);
            this->pipeline_layout_ = VK_NULL_HANDLE;
        }
    }

    render_resource::pass_io const& ssgi_spatial_pass::io() const noexcept {
        return render_resource::ssgi_spatial_io;
    }

    vulkan::pass::behaviour const& ssgi_spatial_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view ssgi_spatial_pass::feature() const noexcept {
        // ITS OWN feature name, and not the chain's, because this pass runs on a SECOND condition the chain
        // cannot express (see chain.cppm's header): the frame's accumulation has to have been resolved THIS
        // frame, or the filter would smooth a stale image. The renderer's registry answers that
        // (`feature_active("ssgi_spatial")` = the chain is on AND the temporal pass resolved this frame), and the
        // pass's own `resolved()` is what the composite's weight is read from.
        return "ssgi_spatial";
    }

    bool ssgi_spatial_pass::pipeline_ready() const noexcept {
        return this->pipeline_.has_value();
    }

    VkPipeline ssgi_spatial_pass::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout ssgi_spatial_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    void ssgi_spatial_pass::create(pass_context const& context) {
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
            utility::log("GI spatial filter disabled (screen-space GI will stay off): the owner has no {}", shader_name);
            return;
        }
        // The two set layouts come from the CONTEXT, not from this pass: it binds the shared scene set and the
        // shared G-buffer set and owns no layout of its own (see pass_context::shared_set_layout) - which is
        // also why `build_ssgi_spatial` creates a pipeline layout and no set layout.
        VkDescriptorSetLayout const scene_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 0) : VK_NULL_HANDLE;
        VkDescriptorSetLayout const gbuffer_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 1) : VK_NULL_HANDLE;
        if (scene_layout == VK_NULL_HANDLE || gbuffer_layout == VK_NULL_HANDLE) {
            utility::log("GI spatial filter disabled (screen-space GI will stay off): the owner has no layout for the shared sets this pass binds");
            return;
        }
        auto built = pipelines::build_ssgi_spatial(context.device, scene_layout, gbuffer_layout, static_cast<uint32_t>(sizeof(push_constants)), spirv);
        if (!built) {
            utility::log("GI spatial filter disabled (screen-space GI will stay off): {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->trace);
        utility::log("SUCCESS: GI spatial filter created (joint-bilateral, depth + normal edge stops)");
    }

    void ssgi_spatial_pass::on_swapchain_recreated(pass_host const&) {
        this->resolved_ = false;
    }

    bool ssgi_spatial_pass::resolved() const noexcept {
        return this->resolved_;
    }

    void ssgi_spatial_pass::set_sigma(float const sigma) noexcept {
        // The clamp lives HERE, with the value: 0 is the shader's pass-through (a legal setting, and one the
        // documentation names), and 8 is where the filter is wider than any GI texel neighbourhood it can read.
        this->sigma_ = std::clamp(sigma, 0.0f, 8.0f);
    }

    float ssgi_spatial_pass::sigma() const noexcept {
        return this->sigma_;
    }

    void ssgi_spatial_pass::record(resolved_io const& io) {
        this->resolved_ = false;
        if (io.barrier_images.size() < render_resource::ssgi_spatial_barriers.size() || io.pipelines.empty() || io.pipelines[0] == VK_NULL_HANDLE ||
            io.pipeline_layout == VK_NULL_HANDLE || io.shared.scene == VK_NULL_HANDLE || io.shared.gbuffer == VK_NULL_HANDLE || io.extent.width == 0 || io.extent.height == 0) {
            return; // the runner resolves all of this or skips the pass (see the pass-level review in the header)
        }
        VkImage const output = io.barrier_images[barrier_output].image;

        // The output is a storage image whose contents are fully overwritten this frame: UNDEFINED -> GENERAL
        // here, and GENERAL -> SHADER_READ at the end for the composite. The INPUT needs no barrier - the
        // temporal resolve handed it to SHADER_READ through a transition that names COMPUTE as well as FRAGMENT,
        // which is the read this dispatch does.
        VkImageMemoryBarrier2 to_general = vulkan::undefined_to_general_transition;
        to_general.image = output;
        VkDependencyInfo const general_dependency = make_image_dependency_info(1, &to_general);
        vkCmdPipelineBarrier2(io.cmd, &general_dependency);

        // Two sets, then the pipeline: the shared scene set and the shared G-buffer set, in that order.
        std::array<VkDescriptorSet, 2> const sets = {io.shared.scene, io.shared.gbuffer};
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, io.pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);

        // THE PUSH BLOCK, and each lane comes from where its owner is: the two projection terms and the
        // depth/normal criteria are the FRAME's (the same `proj` terms and the same `render_settings` pair the
        // composite's upsample reads - they have to agree, which is why they are not here), and the extents are
        // the frame's and this pass's own. `subtract_ambient` is NOT written: it is a RETIRED lane that stays 0
        // (see the struct), because the removal it used to request now happens in `shaders/deferred.frag`,
        // where the ambient it removed is added. The one flag left is this frame's answer about the
        // REFLECTION's accumulation - not about the diffuse one, which needs no answer from this pass.
        render_settings const& settings = io.constants.settings;
        push_constants push = {};
        push.depth_scale = io.constants.proj[2][2];
        push.depth_offset = io.constants.proj[3][2];
        push.sigma_spatial = this->sigma_;
        push.sigma_depth = settings.gi_depth_sigma;
        push.normal_power = settings.gi_normal_power;
        // The reflection's own accumulation is summed in by the filter at binding 15. THIS lane is about
        // how much of it to include, and it keys on whether the reflection was actually RESOLVED this frame rather
        // than on whether the lobe is enabled: if its descriptor set could not be had, the accumulation holds an
        // older frame and must not be summed in. The denoise pass runs before this one, so the flag is this frame's.
        push.spec_weight = io.constants.gi_spec_resolved ? 1.0f : 0.0f;
        push.gi_size = glm::vec4(static_cast<float>(io.extent.width), static_cast<float>(io.extent.height),
                                 static_cast<float>(io.frame.extent.width), static_cast<float>(io.frame.extent.height));
        vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(io.cmd, (io.extent.width + group_size - 1u) / group_size, (io.extent.height + group_size - 1u) / group_size, 1);

        // Hand the filtered image to the composite. This is also where the frame's GI becomes usable: the
        // composite's weight is read from `gi_resolved`, and only this pass writes the image that weight applies
        // to - which is why the renderer sets it from this pass's own answer.
        VkImageMemoryBarrier2 to_sampling = vulkan::general_to_sampling_transition;
        to_sampling.image = output;
        VkDependencyInfo const sampling_dependency = make_image_dependency_info(1, &to_sampling);
        vkCmdPipelineBarrier2(io.cmd, &sampling_dependency);

        this->resolved_ = true;
    }

} // namespace vulkan::pass
