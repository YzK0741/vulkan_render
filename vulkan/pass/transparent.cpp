// The transparent pass's implementation: the two hand-off barriers, the LOAD instance, one secondary and the
// depth hand-back. Moved unchanged from `runtime::record_transparent_pass` - same barrier constants, same
// attachment helpers, same order - so the gate decides it on `transparent_blend`, the scenario with an
// alphaMode BLEND material, and on every other scenario that has none (where the pass does not run at all).

module;

#include <array>
#include <cstdint>
#include <span>
#include <vulkan/vulkan.h>

module vulkan.pass.transparent;

import vulkan.render_resource;
import vulkan.constant_init;
import utility;

namespace vulkan::pass {

    render_resource::pass_io const& transparent_pass::io() const noexcept {
        return render_resource::transparent_io;
    }

    vulkan::pass::behaviour const& transparent_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view transparent_pass::feature() const noexcept {
        // THE RENDERER'S GATE, as a feature name (see the scene pass): a frame whose culling left nothing blended
        // is a frame this pass is INACTIVE on - which is what the runner checks before it resolves anything, so
        // the pass neither records nor resolves and costs nothing on those frames.
        return "transparent";
    }

    void transparent_pass::create(pass_context const&) {
        // Nothing to build: no own set (everything is in the shared scene set) and no pipeline (the leaves name
        // theirs). See the scene pass.
    }

    void transparent_pass::on_swapchain_recreated(pass_host const&) {
        // Nothing is per-image here; the secondary is the frame loop's and the leaves own their buffers.
    }

    void transparent_pass::set_frame(transparent_frame const& frame) noexcept {
        this->frame_ = frame;
    }

    void transparent_pass::record(resolved_io const& io) {
        if (this->frame_.make_environment == nullptr || this->frame_.secondary == VK_NULL_HANDLE || this->frame_.leaves.empty() || io.targets.size() < 2) {
            return; // the runner resolves all of this or skips the pass (see runtime::resolve_transparent_pass)
        }
        VkImageView const target_view = io.targets[0].view; // the scene colour target (declaration order)
        VkImage const target_image = io.targets[0].image;
        VkImageView const depth_view = io.targets[1].view; // the surface depth
        VkImage const depth_image = io.targets[1].image;

        // The lighting stage sampled the surface depth, so it is in SHADER_READ_ONLY: hand it back to the
        // attachment layout for the depth test. The scene target is already in COLOR_ATTACHMENT (the lighting
        // instance ended as an attachment write), but dynamic rendering inserts no dependency between two
        // instances, so that store still has to be published before this instance LOADs the same image.
        std::array<VkImageMemoryBarrier2, 2> barriers = {};
        barriers[0] = vulkan::sampling_to_depth_attachment_transition;
        barriers[0].image = depth_image;
        barriers[1] = vulkan::color_attachment_dependency;
        barriers[1].image = target_image;
        VkDependencyInfo const dependency = make_image_dependency_info(static_cast<uint32_t>(barriers.size()), barriers.data());
        vkCmdPipelineBarrier2(io.cmd, &dependency);

        // The leaves go into the frame slot's transparent secondary, with inheritance matching the instance
        // below: ONE colour attachment at 1x. A secondary does not inherit state from its primary, so it binds
        // the shared scene set for itself - the same bind the scene pass's segments make.
        std::array<VkFormat, 1> const color_formats = {this->frame_.color_format};
        VkCommandBufferInheritanceRenderingInfo const inheritance =
            make_inheritance_rendering_info(color_formats.data(), 1, this->frame_.depth_format, VK_SAMPLE_COUNT_1_BIT);
        VkCommandBufferInheritanceInfo const secondary_inherit = make_inheritance_info(&inheritance);
        VkCommandBufferBeginInfo const secondary_begin = make_command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT, &secondary_inherit);
        bool recorded = false;
        if (vkBeginCommandBuffer(this->frame_.secondary, &secondary_begin) == VK_SUCCESS) {
            render_environment env = this->frame_.make_environment(this->frame_.owner, this->frame_.secondary, /*gbuffer=*/false);
            // No set to bind (see scene_pass::record_segment): every slot these leaves read comes from the heaps,
            // which the runtime binds on this same secondary before executing it.
            for (primitive const* const leaf : this->frame_.leaves) {
                leaf->draw(env);
            }
            vkEndCommandBuffer(this->frame_.secondary);
            recorded = true;
        } else {
            utility::log("transparent pass: secondary begin failed - transparent leaves skipped this frame");
        }

        // loadOp LOAD on both attachments: the scene target holds the shaded frame and the depth holds the
        // opaque surface, and neither may be cleared.
        VkRenderingAttachmentInfo const color_attachment = make_load_color_attachment_info(target_view);
        VkRenderingAttachmentInfo const depth_attachment = make_load_depth_attachment_info(depth_view);
        VkRenderingInfo const rendering_info =
            make_rendering_info(VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT, {{0, 0}, this->frame_.extent}, &color_attachment, 1, &depth_attachment);
        vkCmdBeginRendering(io.cmd, &rendering_info);
        if (recorded) {
            vkCmdExecuteCommands(io.cmd, 1, &this->frame_.secondary);
        }
        vkCmdEndRendering(io.cmd);

        // Hand the depth back to the layout everything downstream samples it in: this pass took it out of
        // SHADER_READ to depth-test against it, and TWO later stages read the same image (the resolve's
        // disocclusion guard and the composite's edge test). Nothing else would move it - the flag-driven
        // transition was already consumed by the lighting stage - so a frame with blended geometry would leave
        // the image as an attachment and every read after it would be a layout error.
        VkImageMemoryBarrier2 to_sampling = vulkan::shadow_map_sampling_transition; // attachment -> SHADER_READ
        to_sampling.image = depth_image;
        VkDependencyInfo const sampling_dependency = make_image_dependency_info(1, &to_sampling);
        vkCmdPipelineBarrier2(io.cmd, &sampling_dependency);
    }

} // namespace vulkan::pass
