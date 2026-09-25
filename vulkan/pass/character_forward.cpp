// The character-forward pass's implementation: the two hand-off barriers, the LOAD instance, and the
// session state that makes `ZTest Equal` + `ZWrite Off` true rather than merely intended.
//
// MODELLED ON transparent.cpp AND DELIBERATELY SIMPLER: that pass records into a per-slot SECONDARY
// command buffer because the renderer fans its leaves out across the task pool, and this one draws
// directly into the primary. A character has hundreds of leaves, not the scene's thousands, and the
// secondary's inheritance plumbing (the heaps, the rendering info, the begin/end pair) buys nothing at
// that size. If the leaf count ever argues for it, this is the file that changes.

module;

#include <array>
#include <cstdint>
#include <span>
#include <vulkan/vulkan.h>

module vulkan.pass.character_forward;

import vulkan.render_resource;
import vulkan.constant_init;
import utility;

namespace vulkan::pass {

    render_resource::pass_io const& character_forward_pass::io() const noexcept {
        return render_resource::character_forward_io;
    }

    vulkan::pass::behaviour const& character_forward_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view character_forward_pass::feature() const noexcept {
        // THE RENDERER'S GATE, as a feature name (see the scene pass): a scene with no toon character is a
        // scene this pass is INACTIVE on - which is what the runner checks before it resolves anything, so
        // the pass neither records nor resolves and costs nothing on those frames. That is also what keeps
        // every capture-gate scenario byte-identical while the feature is off.
        return "character_forward";
    }

    void character_forward_pass::create(pass_context const&) {
        // Nothing to build: no own set (every slot these leaves read comes from the frame's heaps) and no
        // pipeline (the renderer registers the named one, and the leaves bind it through `default_name`).
    }

    void character_forward_pass::on_swapchain_recreated(pass_host const&) {
        // Nothing is per-image here: the targets belong to the frame loop and the leaves own their buffers.
    }

    void character_forward_pass::set_frame(character_forward_frame const& frame) noexcept {
        this->frame_ = frame;
    }

    void character_forward_pass::record(resolved_io const& io) {
        if (this->frame_.make_environment == nullptr || this->frame_.pipeline_name.empty() || this->frame_.leaves.empty() || io.targets.size() < 2) {
            return; // the runner resolves all of this or skips the pass (see make_character_forward_frame)
        }
        VkImageView const target_view = io.targets[0].view; // the scene colour target (declaration order)
        VkImage const target_image = io.targets[0].image;
        VkImageView const depth_view = io.targets[1].view; // the surface depth
        VkImage const depth_image = io.targets[1].image;

        // TWO HAND-OFFS, and both are needed for the same reason the transparent pass needs them: the
        // lighting stage (or the transparent pass after it) left the depth in SHADER_READ_ONLY, so it has
        // to go back to the attachment layout before this instance can test against it; and the scene
        // colour's last store has to be published before this instance LOADs the same image, because
        // dynamic rendering inserts no dependency between two instances.
        std::array<VkImageMemoryBarrier2, 2> barriers = {};
        barriers[0] = vulkan::sampling_to_depth_attachment_transition;
        barriers[0].image = depth_image;
        barriers[1] = vulkan::color_attachment_dependency;
        barriers[1].image = target_image;
        VkDependencyInfo const dependency = make_image_dependency_info(static_cast<uint32_t>(barriers.size()), barriers.data());
        vkCmdPipelineBarrier2(io.cmd, &dependency);

        // LOAD on both attachments: the lit frame and the opaque surface are what this pass draws OVER.
        VkRenderingAttachmentInfo const color_attachment = make_load_color_attachment_info(target_view);
        VkRenderingAttachmentInfo const depth_attachment = make_load_depth_attachment_info(depth_view);
        VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, this->frame_.extent}, &color_attachment, 1, &depth_attachment);
        vkCmdBeginRendering(io.cmd, &rendering_info);

        render_environment env = this->frame_.make_environment(this->frame_.owner, io.cmd, /*gbuffer=*/false);
        // THE TWO THINGS THIS PASS STATES ABOUT ITSELF, neither of which the renderer could know:
        //
        //   WHICH PIPELINE. Every primitive's draw() calls `bind_default()`, so a session's default name is
        //   what its leaves bind - there is no redirect flag and no per-leaf pipeline name to set. The
        //   renderer registered the character pipeline under `pipeline_name`; from here on the leaves are
        //   drawn by it.
        //
        //   DEPTH WRITE OFF, THEN LOCKED. The set comes first (the state starts "unknown", so it is really
        //   emitted), and the lock comes second because every primitive's draw() sets its OWN depth write a
        //   few instructions later - without the lock this pass's `ZWrite Off` would be a statement of
        //   intent that the very next leaf undoes. See render_environment::depth_write_locked.
        env.default_name = this->frame_.pipeline_name;
        env.set_depth_write(false);
        env.depth_write_locked = true;

        for (primitive const* const leaf : this->frame_.leaves) {
            leaf->draw(env);
        }

        vkCmdEndRendering(io.cmd);

        // Hand the depth back to the layout everything downstream samples it in. TWO later stages read this
        // image (the resolve's disocclusion guard and the composite's edge test), and this pass is what took
        // it out of SHADER_READ - so leaving it as an attachment would make every read after it a layout
        // error. The colour target needs no hand-back: the composite transitions it for itself.
        VkImageMemoryBarrier2 to_sampling = vulkan::shadow_map_sampling_transition; // attachment -> SHADER_READ
        to_sampling.image = depth_image;
        VkDependencyInfo const sampling_dependency = make_image_dependency_info(1, &to_sampling);
        vkCmdPipelineBarrier2(io.cmd, &sampling_dependency);
    }

} // namespace vulkan::pass
