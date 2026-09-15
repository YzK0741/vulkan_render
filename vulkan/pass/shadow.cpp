// The shadow pass's implementation: the depth-only pipeline (built against the SCENE pipeline layout and the
// context's depth format) and the per-cascade recording - the layer's transition to a depth attachment, the
// depth-only instance at the map's edge, the pre-recorded secondary's execution and the instance's end. Moved out of
// `runtime::make_shadow_pipeline` + the shadow block of `begin_recording` UNCHANGED in behaviour, except for the
// things the pass does not own (the hand-back barrier over every allocated layer, and the per-cascade content, which
// the renderer records through the frame's callback) - so the capture gate, which runs with shadows on in all twelve
// scenarios, decides the move.

module;

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

module vulkan.pass.shadow;

import vulkan.constant_init;
import vulkan.pipelines; // build_shadow: the depth-only pipeline this pass owns
import utility;

namespace vulkan::pass {

    shadow_pass::~shadow_pass() {
        this->release_owned();
    }

    void shadow_pass::release_owned() noexcept {
        // The pipeline LAYOUT is not destroyed here: it is the SCENE's (the owner handed it over), which is why this
        // pass creates its pipeline against it rather than around it.
        this->pipeline_.reset();
        this->pipeline_layout_ = VK_NULL_HANDLE;
    }

    render_resource::pass_io const& shadow_pass::io() const noexcept {
        return render_resource::shadow_io;
    }

    vulkan::pass::behaviour const& shadow_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view shadow_pass::feature() const noexcept {
        // THE NAME THE RENDERER ALREADY ANSWERS: `f.shadow` is the two knobs, this pipeline's existence and "the flat
        // render mode reads no shadow map" - one answer shared by the frame loop's stage gate, the overlay and the
        // startup log, so the three cannot disagree about whether this pass runs.
        return "shadow";
    }

    void shadow_pass::create(pass_context const& context) {
        if (context.device == VK_NULL_HANDLE || context.depth_format == VK_FORMAT_UNDEFINED) {
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
            utility::log("shadow disabled: the owner has no {} or {}", vertex_shader_name, fragment_shader_name);
            return;
        }
        // THE SCENE'S PIPELINE LAYOUT, which this pass does not own: its draw is a subset of the scene's (the same
        // scene set, the same per-leaf push block) and its per-cascade push lands at that block's end - so a layout
        // of its own would be a second object whose push ranges are this pass's guess.
        VkPipelineLayout const scene_layout = context.shared_pipeline_layout != nullptr ? context.shared_pipeline_layout(context.owner) : VK_NULL_HANDLE;
        if (scene_layout == VK_NULL_HANDLE) {
            utility::log("shadow disabled: the owner has no scene pipeline layout for this pass's draw");
            return;
        }
        auto built = pipelines::build_shadow(context.device, scene_layout, context.depth_format, create_bias_constant, create_bias_slope, create_bias_clamp, vertex_spirv, fragment_spirv);
        if (!built) {
            utility::log("shadow disabled: {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_layout_ = scene_layout;
        this->pipeline_ = std::move(*built);
        utility::log("SUCCESS: shadow pipeline created (directional depth-only cascade pass)");
    }

    void shadow_pass::on_swapchain_recreated(pass_host const&) {
        // Nothing to reset: the pipeline depends on the DEPTH format and on nothing whose size changes, and the map
        // it renders into is sized by the frame's `map_size` rather than by the surface.
    }

    bool shadow_pass::pipeline_ready() const noexcept {
        return this->pipeline_.has_value() && this->pipeline_layout_ != VK_NULL_HANDLE;
    }

    VkPipeline shadow_pass::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout shadow_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    void shadow_pass::set_frame(shadow_frame const& frame) noexcept {
        this->frame_ = frame;
    }

    void shadow_pass::record(resolved_io const& io) {
        if (!this->pipeline_ready() || io.targets.empty() || io.pipeline_layout == VK_NULL_HANDLE) {
            return;
        }
        if (this->frame_.record_cascade == nullptr || this->frame_.run_tasks == nullptr || this->frame_.map_size == 0u) {
            return;
        }
        // ONE LAYER PER CASCADE, and never more than the map has: the DECLARATION names the map and its first layer
        // and the FRAME hands over the layers this frame has (see render_resource::shadow_io), so the two counts meet
        // here - and `cascades` (the secondaries) is the frame's own count.
        uint32_t const layers = static_cast<uint32_t>(std::min<std::size_t>(this->frame_.cascades.size(), io.targets.size()));
        if (layers == 0u) {
            return;
        }
        VkPipeline const pipeline = this->pipeline();
        VkPipelineLayout const layout = this->pipeline_layout_;
        // ---- THE CONTENT: one task per cascade, each into its OWN secondary ----
        // A VkCommandPool is not thread safe, which is why every cascade has its own {pool, buffer} pair (the same
        // rule the main pass's workers follow). Only the CONTENT moves off the primary thread: the barriers, the
        // instances and the executions below stay here, in the layer order the attachments require, so a parallel
        // frame's recorded commands are identical to a sequential one's.
        std::vector<std::function<void()>> tasks;
        tasks.reserve(layers);
        std::vector<bool> recorded(layers, false);
        for (uint32_t cascade = 0; cascade < layers; ++cascade) {
            tasks.emplace_back([this, cascade, pipeline, layout, &recorded] {
                VkCommandBuffer const secondary = this->frame_.cascades[cascade];
                if (secondary == VK_NULL_HANDLE) {
                    return;
                }
                recorded[cascade] = this->frame_.record_cascade(this->frame_.owner, secondary, cascade, pipeline, layout);
            });
        }
        this->frame_.run_tasks(this->frame_.owner, tasks);

        // ---- ONE INSTANCE PER CASCADE, in the primary ----
        VkExtent2D const map_extent = {this->frame_.map_size, this->frame_.map_size};
        for (uint32_t cascade = 0; cascade < layers; ++cascade) {
            VkImageView const layer_view = io.targets[cascade].view;
            VkImage const layer_image = io.targets[cascade].image;
            if (layer_view == VK_NULL_HANDLE || layer_image == VK_NULL_HANDLE) {
                continue;
            }
            // THIS layer to a renderable depth attachment: one barrier per layer, because the transition constant's
            // subresource range is single-layer and each layer is its own attachment here. Its loadOp CLEAR discards
            // the previous frame's contents, so UNDEFINED as the old layout is valid.
            VkImageMemoryBarrier2 layer_barrier = vulkan::depth_attachment_transition;
            layer_barrier.image = layer_image;
            layer_barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, cascade, 1};
            VkDependencyInfo const layer_dependency = make_image_dependency_info(1, &layer_barrier);
            vkCmdPipelineBarrier2(io.cmd, &layer_dependency);
            // Depth-only rendering into this cascade (no colour attachment), with loadOp CLEAR (the far plane) and
            // storeOp STORE - the map has to survive for the shading stages that sample it.
            VkRenderingAttachmentInfo const depth_attachment = make_depth_attachment_info(layer_view, VK_ATTACHMENT_STORE_OP_STORE);
            VkRenderingInfo const rendering_info = make_rendering_info(VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT, {{0, 0}, map_extent}, false, nullptr, &depth_attachment);
            vkCmdBeginRendering(io.cmd, &rendering_info);
            // Never execute a secondary whose begin failed - executing an unrecorded command buffer is a VUID and can
            // wedge the frame slot, which is what the recorded flags are for (a null buffer or a log line from the
            // callback leaves its flag false).
            if (recorded[cascade]) {
                VkCommandBuffer const secondary = this->frame_.cascades[cascade];
                vkCmdExecuteCommands(io.cmd, 1, &secondary);
            }
            vkCmdEndRendering(io.cmd);
        }
    }

} // namespace vulkan::pass
