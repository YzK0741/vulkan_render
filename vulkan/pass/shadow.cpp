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
        this->pipeline_.reset();
        this->mesh_pipeline_.reset();
        this->meshlet_pipeline_.reset();
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
        // The pipeline is heap-native: nothing about the draw's descriptors or push travels through a layout.
        auto built = pipelines::build_shadow(context.device, context.depth_format, create_bias_constant, create_bias_slope, create_bias_clamp, vertex_spirv, fragment_spirv);
        if (!built) {
            utility::log("shadow disabled: {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_ = std::move(*built);
        utility::log("SUCCESS: shadow pipeline created (directional depth-only cascade pass)");
        // ---- ... and the MESH form of the same pass, when the device can run one (docs/mesh_shaders.md step 1).
        // The two differ in ONE stage type: `shadow.mesh.spv`'s mesh entry fetches the vertices the input
        // assembler would have bound and emits the same triangles, through the same fragment stage - so the two
        // pipelines are interchangeable per frame, and `pipeline()` answers with the mesh one whenever it is
        // there. A device without mesh shaders (`context.mesh_shaders` false) must not even create the module,
        // and a shader file that is missing is the same answer: keep the vertex form and say so.
        if (!context.mesh_shaders) {
            utility::log("shadow: the device has no mesh shaders, so the pass keeps its vertex stage");
            return;
        }
        std::span<unsigned char const> const mesh_spirv = context.shader != nullptr ? context.shader(context.owner, mesh_shader_name) : std::span<unsigned char const>{};
        if (mesh_spirv.empty()) {
            utility::log("shadow: no {} beside {}, so the pass keeps its vertex stage", mesh_shader_name, vertex_shader_name);
            return;
        }
        auto mesh_built = pipelines::build_shadow(context.device, context.depth_format, create_bias_constant, create_bias_slope, create_bias_clamp, mesh_spirv, fragment_spirv, VK_SHADER_STAGE_MESH_BIT_EXT);
        if (!mesh_built) {
            utility::log("shadow: the mesh pipeline was refused ({}), so the pass keeps its vertex stage", mesh_built.error());
            return;
        }
        this->mesh_pipeline_ = std::move(*mesh_built);
        // ---- ... and the MESHLET form (docs/mesh_shaders.md step 3): only the MESH module differs (same fragment
        // stage), and a missing shader or a refusal is a log line - the two pipelines above are complete answers.
        std::span<unsigned char const> const meshlet_spirv = context.shader != nullptr ? context.shader(context.owner, meshlet_shader_name) : std::span<unsigned char const>{};
        if (!meshlet_spirv.empty()) {
            auto meshlet_built = pipelines::build_shadow(context.device, context.depth_format, create_bias_constant, create_bias_slope, create_bias_clamp, meshlet_spirv, fragment_spirv, VK_SHADER_STAGE_MESH_BIT_EXT);
            if (meshlet_built) {
                this->meshlet_pipeline_ = std::move(*meshlet_built);
                utility::log("SUCCESS: shadow MESHLET pipeline created (one workgroup per meshlet, window read from the table)");
            } else {
                utility::log("shadow: the meshlet pipeline was refused ({}), so the pass keeps the mesh form", meshlet_built.error());
            }
        }
        utility::log("SUCCESS: shadow MESH pipeline created (the same depth-only pass, fed by mesh dispatches)");
    }

    void shadow_pass::on_swapchain_recreated(pass_host const&) {
        // Nothing to reset: the pipeline depends on the DEPTH format and on nothing whose size changes, and the map
        // it renders into is sized by the frame's `map_size` rather than by the surface.
    }

    bool shadow_pass::pipeline_ready() const noexcept {
        return this->pipeline_.has_value();
    }

    VkPipeline shadow_pass::pipeline() const noexcept {
        // THE MESH FORM WHEN THERE IS ONE, because it is the same pass: it emits the same triangles through the
        // same fragment stage, so which pipeline draws is not the frame's business. A device that cannot run one
        // (or a refused pipeline) leaves the vertex form, which is why the fallback is a fallback rather than a
        // configuration (see create).
        if (this->meshlet_pipeline_.has_value()) {
            return this->meshlet_pipeline_->get_pipeline();
        }
        if (this->mesh_pipeline_.has_value()) {
            return this->mesh_pipeline_->get_pipeline();
        }
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    void shadow_pass::set_frame(shadow_frame const& frame) noexcept {
        this->frame_ = frame;
    }

    void shadow_pass::record(resolved_io const& io) {
        if (!this->pipeline_ready() || io.targets.empty()) {
            return;
        }
        if (this->frame_.record_cascade == nullptr || this->frame_.run_tasks == nullptr || this->frame_.map_size == 0u) {
            return;
        }
        // ONE LAYER PER CASCADE, and never more than the map has: the DECLARATION claims a RUN of cascade layers
        // (render_resource::shadow_targets) and the FRAME caps it at the layers the image actually has - so the
        // two counts meet here. The secondaries are the frame's own count, and they are what decides how many
        // layers this frame renders: a cascade with no secondary to record into is not rendered at all.
        uint32_t const layers = static_cast<uint32_t>(std::min<std::size_t>(this->frame_.cascades.size(), io.targets.size()));
        if (layers == 0u) {
            return;
        }
        VkPipeline const pipeline = this->pipeline();
        // ... and HOW it must be fed travels with it: a mesh pipeline has no input assembler, so its casters are
        // dispatched rather than drawn (see the frame's record_cascade).
        bool const meshlets = this->meshlet_pipeline_.has_value();
        bool const mesh_stage = meshlets || this->mesh_pipeline_.has_value();
        // ---- THE CONTENT: one task per cascade, each into its OWN secondary ----
        // A VkCommandPool is not thread safe, which is why every cascade has its own {pool, buffer} pair (the same
        // rule the main pass's workers follow). Only the CONTENT moves off the primary thread: the barriers, the
        // instances and the executions below stay here, in the layer order the attachments require, so a parallel
        // frame's recorded commands are identical to a sequential one's.
        std::vector<std::function<void()>> tasks;
        tasks.reserve(layers);
        std::vector<bool> recorded(layers, false);
        for (uint32_t cascade = 0; cascade < layers; ++cascade) {
            tasks.emplace_back([this, cascade, pipeline, mesh_stage, meshlets, &recorded] {
                VkCommandBuffer const secondary = this->frame_.cascades[cascade];
                if (secondary == VK_NULL_HANDLE) {
                    return;
                }
                recorded[cascade] = this->frame_.record_cascade(this->frame_.owner, secondary, cascade, pipeline, mesh_stage, meshlets);
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
