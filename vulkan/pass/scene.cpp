// The scene pass's implementation: the rendering instance, the parallel segments, and the draw loop that used
// to live in `runtime::record_opaque_scene` / `record_main_segment` / `sub_render_task`. Moved unchanged in
// behaviour - same attachment order, same clear values, same segment count rule, same per-segment bind and
// draw order, same secondary inheritance - so the capture gate decides the move on the twelve scenarios, which
// between them cover the deferred surface write, the forward shading path and the lighting stage that reads the
// depth this pass writes.

module;

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>
#include <vulkan/vulkan.h>

module vulkan.pass.scene;

import vulkan.render_resource;
import vulkan.constant_init;
import utility;

namespace vulkan::pass {

    render_resource::pass_io const& scene_pass::io() const noexcept {
        return render_resource::scene_io;
    }

    vulkan::pass::behaviour const& scene_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view scene_pass::feature() const noexcept {
        // THE RENDERER'S GATE, as a feature name: this pass runs when the surface pipelines exist. It used to be
        // the first line of `runtime::resolve_scene_pass` (which returned false), and the OTHER half of that gate -
        // whether this frame's target generation exists - needs no answer here: the pass's declared targets are
        // resolved from the frame's resource table, so a frame without them does not resolve the pass at all.
        return "scene";
    }

    void scene_pass::create(pass_context const&) {
        // NOTHING TO BUILD, and that is the pass: it owns no set layout (its bindings all live in the shared
        // scene set) and no pipeline (a leaf names the pipeline it wants, and the renderer's registry owns it).
        // A pass whose create is empty is not a pass that is missing something - it is a pass that draws other
        // people's content, which is exactly what a scene pass is.
    }

    void scene_pass::on_swapchain_recreated(pass_host const&) {
        // Nothing of this pass is per-image: the targets are the frame's, the secondaries are the frame loop's,
        // and the leaves hold their own buffers.
    }

    void scene_pass::set_frame(scene_frame const& frame) noexcept {
        this->frame_ = frame;
    }

    bool scene_pass::begin_segment(VkCommandBuffer const command_buffer) const {
        // The inheritance is built FRESH here, so the pNext chains point at this invocation's stack: a task is
        // moved into the pool and may run later, on another thread.
        VkCommandBufferInheritanceRenderingInfo const inheritance =
            make_inheritance_rendering_info(this->frame_.color_formats.data(), static_cast<uint32_t>(this->frame_.color_formats.size()), this->frame_.depth_format, this->frame_.samples);
        VkCommandBufferInheritanceInfo const inherit = make_inheritance_info(&inheritance);
        VkCommandBufferBeginInfo const begin = make_command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT, &inherit);
        return vkBeginCommandBuffer(command_buffer, &begin) == VK_SUCCESS;
    }

    void scene_pass::record_segment(VkCommandBuffer const command_buffer, VkDescriptorSet const scene_set, std::span<primitive const* const> const leaves) const {
        render_environment env = this->frame_.make_environment(this->frame_.owner, command_buffer, this->frame_.gbuffer);
        // The shared scene set is bound once per segment, not per draw: every scene pipeline shares the layout,
        // so the set stays valid across pipeline binds and only the per-draw state varies. A SECONDARY does not
        // inherit state from its primary, so each segment binds it for itself.
        if (scene_set != VK_NULL_HANDLE && env.layout != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, env.layout, 0, 1, &scene_set, 0, nullptr);
        }
        for (primitive const* const leaf : leaves) {
            leaf->draw(env); // polymorphic: normal / instanced / static / custom, each through its own pipeline
        }
    }

    void scene_pass::record(resolved_io const& io) {
        if (this->frame_.make_environment == nullptr || this->frame_.segments.empty() || this->frame_.leaves.empty() || io.targets.empty()) {
            return; // the runner resolves all of this or skips the pass (see runtime::resolve_scene_pass)
        }

        // THE INSTANCE, over the targets the declaration names: every colour target in declaration order, and
        // the one DEPTH target in the depth slot. The declarations and the resolved handles are index-aligned,
        // which is what lets the pass tell the two apart without the framework carrying a kind in resolved_io.
        std::array<VkRenderingAttachmentInfo, max_render_targets> color_attachments = {};
        uint32_t color_count = 0;
        VkRenderingAttachmentInfo depth_attachment = {};
        VkRenderingAttachmentInfo const* depth = nullptr;
        VkClearValue const clear = {}; // every target clears to zero: no geometry, no motion, no emissive
        render_resource::pass_io const& declared = this->io();
        for (std::size_t t = 0; t < io.targets.size() && t < declared.targets.size() && t < max_render_targets; ++t) {
            if (declared.targets[t].kind == render_resource::target_kind::depth) {
                // the surface depth must SURVIVE the instance: the lighting stage, the transparent pass, the
                // resolve and the debug view all read it later in the same submission
                depth_attachment = make_depth_attachment_info(io.targets[t].view, VK_ATTACHMENT_STORE_OP_STORE);
                depth = &depth_attachment;
            } else {
                color_attachments[color_count++] = make_color_attachment_info(io.targets[t].view, clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
            }
        }
        VkRenderingInfo const rendering_info =
            make_rendering_info(VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT, {{0, 0}, this->frame_.extent}, color_attachments.data(), color_count, depth);
        vkCmdBeginRendering(io.cmd, &rendering_info);

        // THE SEGMENTS: the renderer's own rule, kept exactly - one segment when there is only one, or when
        // there are too few leaves for the fan-out to pay for itself.
        std::size_t const leaf_count = this->frame_.leaves.size();
        std::size_t const segment_count = std::min<std::size_t>(this->frame_.segments.size(), std::max<std::size_t>(1, leaf_count));
        if (segment_count == 1 || leaf_count < min_leaves_for_parallel) {
            VkCommandBuffer const single = this->frame_.segments[0].buffer;
            if (this->begin_segment(single)) {
                this->record_segment(single, io.shared.scene, this->frame_.leaves);
                vkEndCommandBuffer(single);
                vkCmdExecuteCommands(io.cmd, 1, &single);
            } else {
                utility::log("scene pass: main secondary begin failed - scene skipped this frame");
            }
        } else {
            // One task per contiguous span of leaves, each recording its own secondary on its own pool (a pool
            // is not thread safe, and a worker never shares one). The tasks read shared state and write only
            // their own command buffer; the batch is waited on before the primary executes the segments IN
            // ORDER, so the recorded command stream is the sequential one.
            std::vector<std::function<void()>> tasks;
            tasks.reserve(segment_count);
            std::vector<std::atomic<bool>> segment_recorded(segment_count);
            for (std::size_t s = 0; s < segment_count; ++s) {
                std::size_t const seg_first = leaf_count * s / segment_count;
                std::size_t const seg_last = leaf_count * (s + 1) / segment_count;
                VkCommandBuffer const segment = this->frame_.segments[s].buffer;
                std::span<primitive const* const> const segment_leaves(this->frame_.leaves.data() + seg_first, seg_last - seg_first);
                std::atomic<bool>* const recorded = &segment_recorded[s];
                VkDescriptorSet const scene_set = io.shared.scene;
                tasks.emplace_back([this, segment, segment_leaves, recorded, scene_set] {
                    if (!this->begin_segment(segment)) {
                        utility::log("scene pass: main segment secondary begin failed - segment skipped this frame");
                        recorded->store(false, std::memory_order_relaxed);
                        return;
                    }
                    this->record_segment(segment, scene_set, segment_leaves);
                    vkEndCommandBuffer(segment);
                    recorded->store(true, std::memory_order_relaxed);
                });
            }
            this->frame_.run_tasks(this->frame_.owner, tasks);
            for (std::size_t s = 0; s < segment_count; ++s) {
                if (!segment_recorded[s].load(std::memory_order_relaxed)) {
                    continue; // never execute a secondary whose begin failed
                }
                VkCommandBuffer const segment = this->frame_.segments[s].buffer;
                vkCmdExecuteCommands(io.cmd, 1, &segment);
            }
        }

        // THE INSTANCE ENDS HERE, which is the whole point of the pass owning it: the lighting stage, the
        // transparent pass, the resolve and the lighting stage all run after this, each opening its own.
        vkCmdEndRendering(io.cmd);
    }

} // namespace vulkan::pass
