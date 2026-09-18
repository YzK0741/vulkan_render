// The deferred lighting stage's implementation: the scene-colour dependency barrier, the frame's two per-image
// input transitions, the LOAD instance over the frame's scene target and the fullscreen draw. Moved out of
// `runtime::record_lighting_pass` UNCHANGED in behaviour - the same barrier, the same two ensures, the same LOAD
// attachment, the same bind order (scene set then G-buffer set), the same 88-byte push and the same 3-vertex draw -
// so the capture gate decides the move on all twelve scenarios, every one of which runs this stage.

module;

#include <algorithm> // std::clamp / std::max in set_ssao's clamps
#include <array>
#include <cstdint>
#include <cstring>
#include <glm/glm.hpp> // the push block's ssao lane and the frame's inverse view-projection
#include <span>
#include <string>
#include <vulkan/vulkan.h>

module vulkan.pass.deferred;

import vulkan.constant_init;
import vulkan.pipelines; // build_deferred: the pipeline this pass owns
import utility;

namespace vulkan::pass {

    deferred_pass::~deferred_pass() {
        this->release_owned();
    }

    void deferred_pass::release_owned() noexcept {
        this->pipeline_.reset();
        if (this->pipeline_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->device_, this->pipeline_layout_, nullptr);
            this->pipeline_layout_ = VK_NULL_HANDLE;
        }
    }

    render_resource::pass_io const& deferred_pass::io() const noexcept {
        return render_resource::deferred_io;
    }

    vulkan::pass::behaviour const& deferred_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view deferred_pass::feature() const noexcept {
        return "deferred";
    }

    bool deferred_pass::pipeline_ready() const noexcept {
        return this->pipeline_.has_value();
    }

    VkPipeline deferred_pass::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout deferred_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    void deferred_pass::set_frame(deferred_frame const& frame) noexcept {
        this->frame_ = frame;
    }

    void deferred_pass::prepare_frame(frame_facts const& facts) noexcept {
        // ONE flag, and it is the composed predicate (the knob AND ray queries AND this frame's structures), not
        // the similarly named feature fact - see frame_facts' own note. The punctual lane is the OTHER kind of fact:
        // whether the stochastic pass recorded this frame, which is what makes the two paths exclusive.
        this->set_frame(deferred_frame{.punctual_replaced = facts.megalights_resolved});
    }

    void deferred_pass::create(pass_context const& context) {
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
        std::span<unsigned char const> const vertex_spirv = context.shader != nullptr ? context.shader(context.owner, vertex_shader_name) : std::span<unsigned char const>{};
        std::span<unsigned char const> const fragment_spirv = context.shader != nullptr ? context.shader(context.owner, fragment_shader_name) : std::span<unsigned char const>{};
        if (vertex_spirv.empty() || fragment_spirv.empty()) {
            utility::log("deferred lighting disabled: the owner has no {} or {}", vertex_shader_name, fragment_shader_name);
            return;
        }
        // The two set layouts come from the CONTEXT (the scene set's is the core's, the G-buffer set's is the
        // renderer's) and the blend state is the stage's own: the attachment is LOADed and the lighting ADDS to the
        // emissive the G-buffer pass wrote.
        VkDescriptorSetLayout const scene_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 0) : VK_NULL_HANDLE;
        VkDescriptorSetLayout const gbuffer_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 1) : VK_NULL_HANDLE;
        if (scene_layout == VK_NULL_HANDLE || gbuffer_layout == VK_NULL_HANDLE) {
            utility::log("deferred lighting disabled: the owner has no layout for the shared sets this pass binds");
            return;
        }
        std::array<VkPipelineColorBlendAttachmentState, 1> const blend = {make_color_blend_attachment_additive()};
        auto built = pipelines::build_deferred(context.device, scene_layout, gbuffer_layout, static_cast<uint32_t>(sizeof(push_constants)),
                                               std::span<VkPipelineColorBlendAttachmentState const>(blend), vertex_spirv, fragment_spirv);
        if (!built) {
            utility::log("deferred lighting disabled: {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->lighting);
        utility::log("SUCCESS: deferred lighting pipeline created (shades the stored surface, additive over the emissive)");
    }

    void deferred_pass::on_swapchain_recreated(pass_host const&) {
        // Nothing to reset: no per-generation handle here. The pipeline does not depend on the surface's size and
        // the target arrives per frame (the host picks it - see the header).
    }

    void deferred_pass::set_ssao(bool const enabled, float const radius, float const intensity, uint32_t const samples) noexcept {
        // The clamps came with the parameters, because they are the same fact: an intensity above 1 darkens past
        // black, a negative radius is meaningless, and the sample count is bounded by the SHADER's own array (16).
        this->ssao_enabled_ = enabled;
        this->ssao_radius_ = std::max(radius, 0.0f);
        this->ssao_intensity_ = std::clamp(intensity, 0.0f, 1.0f);
        this->ssao_samples_ = std::clamp(samples, 0u, 16u);
    }

    bool deferred_pass::ssao_enabled() const noexcept {
        return this->ssao_enabled_;
    }

    void deferred_pass::set_unlit(bool const unlit) noexcept {
        this->unlit_ = unlit;
    }

    bool deferred_pass::unlit() const noexcept {
        return this->unlit_;
    }

    void deferred_pass::record(resolved_io const& io) {
        if (!this->pipeline_.has_value() || io.targets.empty() || io.pipelines.empty() || io.pipelines[0] == VK_NULL_HANDLE ||
            io.pipeline_layout == VK_NULL_HANDLE || io.shared.scene == VK_NULL_HANDLE || io.extent.width == 0 || io.extent.height == 0) {
            return; // the runner resolves all of this or skips the pass (see frame_pass::resolve)
        }
        VkImage const target = io.targets[0].image;
        VkImageView const target_view = io.targets[0].view;
        if (target == VK_NULL_HANDLE || target_view == VK_NULL_HANDLE) {
            return;
        }
        // The scene-colour dependency, which cannot be folded into the sampling transitions below: it does not
        // change layout, and its consumer is this instance's LOAD - a color-attachment access, not the
        // FRAGMENT_SHADER read those transitions publish. Dynamic rendering inserts no dependency of its own between
        // two instances, so without it the load is not ordered after the G-buffer pass's store.
        VkImageMemoryBarrier2 dependency_barrier = vulkan::color_attachment_dependency;
        dependency_barrier.image = target;
        VkDependencyInfo const dependency = make_image_dependency_info(1, &dependency_barrier);
        vkCmdPipelineBarrier2(io.cmd, &dependency);
        // The three stored targets and the G-buffer depth became samples in this STAGE's preamble, in the
        // renderer - the frame's ordering rule about images the G-buffer pass wrote (see deferred_frame's note).
        VkRenderingAttachmentInfo const color_attachment = make_load_color_attachment_info(target_view);
        VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, io.extent}, true, &color_attachment, nullptr);
        vkCmdBeginRendering(io.cmd, &rendering_info);
        vkCmdSetCullMode(io.cmd, VK_CULL_MODE_NONE); // the synthetic triangle has no facing to cull
        std::array<VkDescriptorSet, 2> const sets = {io.shared.scene, io.shared.gbuffer};
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, io.pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);
        // THE PUSH BLOCK IS THE PASS'S OWN (S3): its shape was always the pass's (`push_constants`, `static_assert`ed
        // against the declaration), and its VALUES are now the pass's too - the SSAO parameters and the flat-render
        // flag it owns, the frame's inverse view-projection from `resolved_io::constants`, and the frame's answer to
        // what the lighting chain is doing. The renderer used to compose all of it and hand it over as raw bytes.
        push_constants push = {};
        push.inv_view_proj = io.constants.inv_view_proj;
        push.ssao = glm::vec4(this->ssao_radius_, this->ssao_enabled_ ? this->ssao_intensity_ : 0.0f, static_cast<float>(this->ssao_samples_), this->ssao_bias_);
        push.unlit = this->unlit_ ? 1.0f : 0.0f;
        push.punctual_replaced = this->frame_.punctual_replaced ? 1.0f : 0.0f;
        [[maybe_unused]] bool const pushed = io.push_block(io.cmd, pass::push_bytes(push));
        vkCmdDraw(io.cmd, 3, 1, 0, 0);
        vkCmdEndRendering(io.cmd);
    }

} // namespace vulkan::pass