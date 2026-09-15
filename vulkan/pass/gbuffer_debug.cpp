// The G-buffer debug view's implementation: the four images the pass moves to a sampled layout, the frame's two
// bookkeeping callbacks, the CLEAR instance over the HDR target, the one-set bind, the 16-byte push and the
// fullscreen draw. Moved out of `runtime::record_gbuffer_debug_pass` UNCHANGED in behaviour except for the two
// things the pass does not own - the HDR target's own transition and the missing-set fallback, both of which are
// the frame loop's (see the pass's header) - so the knob-on A/B against the parent commit is what decides the move.

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

module vulkan.pass.gbuffer_debug;

import vulkan.constant_init;
import vulkan.pipelines; // build_gbuffer_debug: the G-buffer set layout, the pipeline layout and the view pipeline
import utility;

namespace vulkan::pass {

    gbuffer_debug_pass::~gbuffer_debug_pass() {
        this->release_owned();
    }

    void gbuffer_debug_pass::release_owned() noexcept {
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

    render_resource::pass_io const& gbuffer_debug_pass::io() const noexcept {
        return render_resource::gbuffer_debug_io;
    }

    vulkan::pass::behaviour const& gbuffer_debug_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view gbuffer_debug_pass::feature() const noexcept {
        // THE NAME THE RENDERER ALREADY ANSWERS: `f.gbuffer_debug` is "the knob is on, the G-buffer pipeline and this
        // pass's pipeline exist", which is also what the overlay asks when it offers the view and what the composite
        // asks when it zeroes the bloom weight for it - one answer, not three spellings of the same question.
        return "gbuffer-debug";
    }

    void gbuffer_debug_pass::create(pass_context const& context) {
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
            utility::log("gbuffer debug view disabled: the owner has no {} or {}", vertex_shader_name, fragment_shader_name);
            return;
        }
        // THE G-BUFFER SET LAYOUT IS BUILT HERE, by the builder that needs it, and the deferred lighting stage binds
        // it as its set 1 - which is why the host answers `shared_set_layout(1)` with THIS pass's layout.
        auto built = pipelines::build_gbuffer_debug(context.device, static_cast<uint32_t>(sizeof(push_constants)), vertex_spirv, fragment_spirv);
        if (!built) {
            utility::log("gbuffer debug view disabled: {}", built.error());
            this->release_owned();
            return;
        }
        this->set_layout_ = built->set_layout;
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->debug);
        utility::log("SUCCESS: gbuffer debug pipeline created (the stored surface, one channel at a time)");
    }

    void gbuffer_debug_pass::on_swapchain_recreated(pass_host const&) {
        // Nothing to reset: the pipeline depends on the surface's FORMAT and on nothing whose size changes, and the
        // set this pass binds belongs to the G-buffer family, whose owner retires it.
    }

    bool gbuffer_debug_pass::pipeline_ready() const noexcept {
        return this->pipeline_.has_value() && this->set_layout_ != VK_NULL_HANDLE;
    }

    VkPipeline gbuffer_debug_pass::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout gbuffer_debug_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    VkDescriptorSetLayout gbuffer_debug_pass::set_layout() const noexcept {
        return this->set_layout_;
    }

    void gbuffer_debug_pass::set_frame(gbuffer_debug_frame const& frame) noexcept {
        this->frame_ = frame;
    }

    void gbuffer_debug_pass::record(resolved_io const& io) {
        if (!this->pipeline_ready() || io.targets.empty() || io.pipelines.empty() || io.pipelines[0] == VK_NULL_HANDLE || io.pipeline_layout == VK_NULL_HANDLE ||
            io.shared.gbuffer == VK_NULL_HANDLE || io.push.size() < sizeof(push_constants) || io.extent.width == 0 || io.extent.height == 0) {
            return; // the runner resolves all of this or skips the pass (see runtime::resolve_gbuffer_debug)
        }
        VkImageView const target_view = io.targets[0].view;
        if (target_view == VK_NULL_HANDLE) {
            return;
        }
        // THE FOUR IMAGES THIS PASS READS BECOME SAMPLES, in ONE dependency info and in the declaration's order -
        // the three stored targets, then the motion-vector target. A pipeline barrier may not be recorded inside a
        // rendering instance, which is why this is here and not after vkCmdBeginRendering below.
        std::array<VkImageMemoryBarrier2, render_resource::gbuffer_debug_barriers.size()> barriers = {};
        for (std::size_t b = 0; b < barriers.size(); ++b) {
            barriers[b] = vulkan::hdr_sampling_transition; // COLOR_ATTACHMENT -> SHADER_READ
            barriers[b].image = io.barrier_images[b].image;
        }
        VkDependencyInfo const dependency = make_image_dependency_info(static_cast<uint32_t>(barriers.size()), barriers.data());
        vkCmdPipelineBarrier2(io.cmd, &dependency);
        // ... and the two pieces of per-image bookkeeping the frame carries: the depth's hand-back (its old layout
        // depends on whether the G-buffer instance rendered this frame) and clearing the flag that says the
        // motion-vector target has been handed to a sampler this frame.
        if (this->frame_.ensure_inputs != nullptr) {
            this->frame_.ensure_inputs(this->frame_.owner, io.cmd, io.frame.image_index);
        }
        VkClearValue clear = {};
        VkRenderingAttachmentInfo const attachment = make_color_attachment_info(target_view, clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
        VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, io.extent}, true, &attachment, nullptr);
        vkCmdBeginRendering(io.cmd, &rendering_info);
        vkCmdSetCullMode(io.cmd, VK_CULL_MODE_NONE);   // the synthetic triangle has no facing to cull
        VkDescriptorSet const set = io.shared.gbuffer; // the G-buffer family's set 0, written by the host
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, io.pipeline_layout, 0, 1, &set, 0, nullptr);
        push_constants push = {};
        std::memcpy(&push, io.push.data(), sizeof(push));
        vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(io.cmd, 3, 1, 0, 0);
        vkCmdEndRendering(io.cmd);
    }

} // namespace vulkan::pass
