// The stochastic punctual lighting pass's implementation: the first-use transition around its output image, the
// half-resolution dispatch, the hand-off that makes the image readable by the lighting stage that adds it, and the
// two things it owns outside a frame - its pipeline layout and its compute pipeline, built from
// `megalights_trace.comp` and the two shared set layouts its owner hands over at create time.
//
// THE SHAPE IS `ssgi_trace.cpp`'s, and the two differ only in what the estimator's parameters are: the tracer
// pushes a ray budget, a reach and a world-space origin bias, this pass pushes a sample count, a minimum sample
// weight, a tmin and the two bias terms the shader's `lerp` takes. Read that file's comments for the barrier
// reasoning; what is repeated here is only what a reader of this pass needs.

module;

#include <algorithm>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

module vulkan.pass.megalights_trace;

import vulkan.render_resource;
import vulkan.constant_init;
import vulkan.pipelines; // build_megalights_trace: the compute pipeline this pass owns
import utility;

namespace vulkan::pass {

    megalights_trace_pass::~megalights_trace_pass() {
        this->release_owned();
    }

    void megalights_trace_pass::release_owned() noexcept {
        this->pipeline_.reset();
        if (this->pipeline_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->device_, this->pipeline_layout_, nullptr);
            this->pipeline_layout_ = VK_NULL_HANDLE;
        }
    }

    render_resource::pass_io const& megalights_trace_pass::io() const noexcept {
        return render_resource::megalights_trace_io;
    }

    vulkan::pass::behaviour const& megalights_trace_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view megalights_trace_pass::feature() const noexcept {
        // The pass's own feature name: "the knob and this pass having built its pipeline". The renderer's
        // `megalights_active()` composes the same answer for the lighting stage, which is what keeps the two
        // from disagreeing about whether the punctual lights were already handled this frame.
        return "megalights";
    }

    bool megalights_trace_pass::pipeline_ready() const noexcept {
        return this->pipeline_.has_value();
    }

    VkPipeline megalights_trace_pass::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout megalights_trace_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    void megalights_trace_pass::set_light_angle(float const radians) noexcept {
        // The angle is the emitter's half-size: past a tenth of a radian the "small emitter" the BRDF's
        // representative-point approximation assumes stops being small, so the clamp is where that approximation is
        // still honest rather than where the math breaks.
        this->light_angle_ = std::clamp(radians, 0.0f, 0.1f);
    }

    void megalights_trace_pass::set_estimator(uint32_t const samples, float const min_weight, float const bias_floor, float const bias_grazing) noexcept {
        // The clamps are this pass's, with the values: the sample count is bounded by the shader's own
        // compile-time array (see the header), and the three floats are bounded by what they MEAN - a negative
        // minimum weight inverts the smooth cut, and a negative bias would start the ray inside the surface it
        // is leaving.
        this->samples_ = std::clamp(samples, 1u, max_samples);
        this->min_weight_ = std::max(min_weight, 0.0f);
        this->bias_floor_ = std::max(bias_floor, 0.0f);
        this->bias_grazing_ = std::max(bias_grazing, this->bias_floor_);
    }

    void megalights_trace_pass::on_swapchain_recreated(pass_host const&) {
        // Nothing per-generation is kept here yet: the chain is one pass until the temporal resolve lands, and
        // that one is where a history and its reset will live (see docs/megalights.md's staging).
        this->frame_index_ = 0;
    }

    void megalights_trace_pass::create(pass_context const& context) {
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
            utility::log("stochastic punctual lighting disabled: the owner has no {}", shader_name);
            return;
        }
        VkDescriptorSetLayout const scene_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 0) : VK_NULL_HANDLE;
        VkDescriptorSetLayout const gbuffer_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 1) : VK_NULL_HANDLE;
        if (scene_layout == VK_NULL_HANDLE || gbuffer_layout == VK_NULL_HANDLE) {
            utility::log("stochastic punctual lighting disabled: the owner has no layout for the shared sets this pass binds");
            return;
        }
        auto built = pipelines::build_megalights_trace(context.device, scene_layout, gbuffer_layout, render_resource::megalights_trace_io.push->size, spirv);
        if (!built) {
            utility::log("stochastic punctual lighting disabled: {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->trace);
        utility::log("SUCCESS: stochastic punctual lighting pipeline created (sampled lights with ray-traced visibility)");
    }

    void megalights_trace_pass::record(resolved_io const& io) {
        if (!this->pipeline_.has_value() || io.barrier_images.size() < render_resource::megalights_trace_barriers.size() || io.extent.width == 0 || io.extent.height == 0 ||
            io.pipeline_layout == VK_NULL_HANDLE || io.shared.scene == VK_NULL_HANDLE || io.shared.gbuffer == VK_NULL_HANDLE) {
            return; // the runner resolves all of this or skips the pass (the declaration's own gates are the table's)
        }
        VkImage const output = io.barrier_images[barrier_output].image;

        // The image is this pass's to place, and it needs TWO transitions per frame:
        //  * UNDEFINED -> GENERAL here, because the pass writes it as a STORAGE image and its descriptor in the
        //    shared G-buffer set declares GENERAL (the same statement the tracer makes about `gi_trace`);
        //  * GENERAL -> SHADER_READ at the end, because the deferred lighting stage samples the SAME image
        //    through its binding 17 and that descriptor declares SHADER_READ - one image, two descriptors in one
        //    set, and each is only accessed while the image is in the layout it names.
        VkImageMemoryBarrier2 to_general = vulkan::undefined_to_general_transition;
        to_general.image = output;
        VkDependencyInfo const first_use = make_image_dependency_info(1, &to_general);
        vkCmdPipelineBarrier2(io.cmd, &first_use);

        // Two sets, then the pipeline: the shared scene set and the shared G-buffer set, the order the tracer uses.
        std::array<VkDescriptorSet, 2> const sets = {io.shared.scene, io.shared.gbuffer};
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, io.pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);

        push_constants push = {};
        push.inv_view_proj = io.constants.inv_view_proj;
        push.params = glm::vec4(static_cast<float>(this->samples_), this->min_weight_, this->tmin_, static_cast<float>(this->frame_index_));
        push.bias = glm::vec4(this->bias_floor_, this->bias_grazing_, this->light_angle_, 0.0f);
        static_assert(sizeof(push) <= pass::max_push_bytes, "the estimator's push block must fit the guaranteed minimum");
        vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(io.cmd, (io.extent.width + group_size - 1u) / group_size, (io.extent.height + group_size - 1u) / group_size, 1);

        // ... and the hand-off: a compute SHADER_WRITE is not visible to the lighting stage's texture fetch
        // without this, and the layout it leaves the image in is the one the lighting stage's descriptor
        // declares.
        VkImageMemoryBarrier2 to_sampling = vulkan::general_to_sampling_transition;
        to_sampling.image = output;
        VkDependencyInfo const hand_off = make_image_dependency_info(1, &to_sampling);
        vkCmdPipelineBarrier2(io.cmd, &hand_off);

        ++this->frame_index_; // the next frame's ray sequence must differ (see the header)
    }

} // namespace vulkan::pass
