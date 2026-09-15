// The diffuse temporal resolve's implementation: the barriers, the dispatch, the push lanes that describe this
// pass's own state, the copy that becomes the next frame's history, and the hand-backs. Moved out of
// `runtime::record_ssgi_resolve_pass`'s mode-0 path UNCHANGED in behaviour - the same barrier order, the same
// bind, the same 48-byte push, the same copy and the same two last barriers - so the capture gate decides the
// move on the four GI scenarios.

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <vulkan/vulkan.h>

module vulkan.pass.ssgi_temporal;

import vulkan.render_resource;
import vulkan.constant_init;
import utility;

namespace vulkan::pass {

    render_resource::pass_io const& ssgi_temporal_pass::io() const noexcept {
        return render_resource::ssgi_temporal_io;
    }

    vulkan::pass::behaviour const& ssgi_temporal_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view ssgi_temporal_pass::feature() const noexcept {
        // The CHAIN's feature: the resolve is required, so "the chain is on" is the right gate - and the
        // renderer's predicate for it already includes this pass's pipeline.
        return "ssgi";
    }

    void ssgi_temporal_pass::create(pass_context const&) {
        // Nothing to build YET: the set layout, the pipeline and the per-image family are still the renderer's
        // (see the header). What the pass reads arrives through `resolved_io`.
    }

    void ssgi_temporal_pass::on_swapchain_recreated(pass_host const&) {
        this->resolved_ = false; // the renderer retires the family and clears the history flags it owns
    }

    bool ssgi_temporal_pass::resolved() const noexcept {
        return this->resolved_;
    }

    void ssgi_temporal_pass::set_frame(ssgi_temporal_frame const& frame) noexcept {
        this->frame_ = frame;
    }

    void ssgi_temporal_pass::record(resolved_io const& io) {
        this->resolved_ = false;
        // NOT `io.own`: this pass does not read its own binding handles (its set is ensured by the renderer and
        // handed over whole), so requiring them would be a guard on something it never uses - and it was: the
        // first version of this guard made the pass silently record nothing, which showed up one step later as
        // the reflection's dispatch sampling a motion-vector image still in its attachment layout.
        if (io.barrier_images.size() < render_resource::ssgi_temporal_barriers.size() ||
            io.own_set == VK_NULL_HANDLE || io.pipelines.empty() || io.pipelines[0] == VK_NULL_HANDLE || io.pipeline_layout == VK_NULL_HANDLE ||
            io.push.size() < sizeof(push_constants) || io.extent.width == 0 || io.extent.height == 0) {
            return; // the runner resolves all of this or skips the pass (see runtime::resolve_ssgi_temporal)
        }
        VkImage const resolve_image = io.barrier_images[barrier_resolve].image;
        VkImage const history_image = io.barrier_images[barrier_history].image;

        // Layouts, all before the dispatch (a compute pass may barrier anywhere, but keeping them together is
        // what makes the set of states one image passes through readable):
        //   resolve -> GENERAL (storage write). The old layout is SHADER_READ once the image has been resolved
        //   before, and UNDEFINED on its first frame: the resolve is READ across frames (the tracer samples the
        //   previous frame's copy at a hit, the multi-bounce feedback), so this write has to keep its contents -
        //   claiming UNDEFINED every frame would discard exactly what the feedback reads, which is why the
        //   tracer's first-use barrier leaves it in SHADER_READ.
        //   history -> SHADER_READ, and only on its FIRST use for this image: the previous frame's copy left it
        //   readable (see the hand-back below), so a later frame needs no barrier at all - claiming TRANSFER_DST
        //   as the old layout would be a layout the image is not in. Exactly the TAA resolve's arrangement.
        //   The raw trace needs no barrier either: the tracer handed it to SHADER_READ with a barrier that names
        //   COMPUTE as well as FRAGMENT, which is the read this dispatch does.
        std::array<VkImageMemoryBarrier2, 2> barriers = {};
        uint32_t count = 0;
        barriers[count] = this->frame_.history_valid ? vulkan::sampling_to_general_transition : vulkan::undefined_to_general_transition;
        barriers[count].image = resolve_image;
        ++count;
        if (!this->frame_.history_valid) {
            barriers[count] = vulkan::undefined_to_sampling_transition;
            barriers[count].image = history_image;
            ++count;
        }
        VkDependencyInfo const dependency = make_image_dependency_info(count, barriers.data());
        vkCmdPipelineBarrier2(io.cmd, &dependency);

        // The depth guard samples the G-buffer depth and the reprojection reads the motion-vector target: two
        // shared per-image transitions whose flags belong to the pass that WROTE those images, so the renderer
        // does them (see the header). Called here - after this pass's own barriers, before the dispatch - which
        // is where the moved code called the two accessors.
        if (this->frame_.ensure_inputs != nullptr) {
            this->frame_.ensure_inputs(this->frame_.owner, io.cmd, io.frame.image_index);
        }

        VkDescriptorSet const set = io.own_set;
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, io.pipeline_layout, 0, 1, &set, 0, nullptr);
        // The two lanes that describe THIS pass's state are written here rather than by the renderer: whether the
        // history may be trusted, and which signal this dispatch resolves (always the diffuse bounce).
        push_constants push = {};
        std::memcpy(&push, io.push.data(), sizeof(push));
        push.history_valid = this->frame_.history_valid ? 1.0f : 0.0f;
        push.mode = 0.0f;
        vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(io.cmd, (io.extent.width + group_size - 1u) / group_size, (io.extent.height + group_size - 1u) / group_size, 1);

        // ---- the accumulation becomes the next frame's history ----
        // A copy rather than a ping-pong, exactly like the TAA resolve: the resolve writes the image the
        // composite reads, so the history has to be separate, and copying into it keeps every descriptor set in
        // the frame stable. The accumulation is a storage image (GENERAL), so it goes out through TRANSFER_SRC
        // and comes back as a sample - the post chain still finds it in SHADER_READ, where it expects it.
        std::array<VkImageMemoryBarrier2, 2> copy_barriers = {};
        copy_barriers[0] = vulkan::general_to_transfer_src_transition; // resolve: GENERAL -> TRANSFER_SRC
        copy_barriers[0].image = resolve_image;
        copy_barriers[1] = vulkan::sampling_to_transfer_dst_transition;
        copy_barriers[1].image = history_image;
        VkDependencyInfo const copy_dependency = make_image_dependency_info(static_cast<uint32_t>(copy_barriers.size()), copy_barriers.data());
        vkCmdPipelineBarrier2(io.cmd, &copy_dependency);

        VkImageCopy const region = {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffset = {0, 0, 0},
            .extent = {io.extent.width, io.extent.height, 1},
        };
        vkCmdCopyImage(io.cmd, resolve_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, history_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // Hand both on: the accumulation to the composite, the history copy to the next frame's resolve (which
        // finds it in TRANSFER_DST and transitions it from there). The raw trace has been readable since the
        // tracer left it that way and nothing here touches it.
        std::array<VkImageMemoryBarrier2, 2> hand_back = {};
        hand_back[0] = vulkan::transfer_src_to_sampling_transition; // resolve -> SHADER_READ
        hand_back[0].image = resolve_image;
        hand_back[1] = vulkan::transfer_dst_to_sampling_transition; // history -> SHADER_READ
        hand_back[1].image = history_image;
        VkDependencyInfo const hand_back_dependency = make_image_dependency_info(static_cast<uint32_t>(hand_back.size()), hand_back.data());
        vkCmdPipelineBarrier2(io.cmd, &hand_back_dependency);

        this->resolved_ = true;
    }

} // namespace vulkan::pass
