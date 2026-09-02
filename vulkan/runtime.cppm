module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>

export module vulkan.runtime;
export import std;
export import vulkan.core;
export import vulkan.core.filter;
export import vulkan.model;
import utility;

/**
 * @file runtime.cppm
 * @defgroup vulkan_runtime Vulkan Runtime Facade
 * @brief runtime facade: a thin wrapper exposing all functionality of vulkan::core
 * @note
 *      - use operator-> to access the filtered core view (core_filter, e.g. runtime->get_device())
 *      - the inner core's lifetime is tied to the runtime
 */
namespace vulkan {
    /**
     * @ingroup vulkan_runtime
     * @brief orbit camera state, updated by the mouse callbacks registered in the runtime constructor
     */
    export struct orbit_camera {
        double last_x = 0.0;
        double last_y = 0.0;
        bool dragging = false;
        float yaw = 0.0f;
        // level view: the skybox horizon (the direction parallel to the ground plane) then sits
        // exactly at the screen center, where the model is framed against it
        float pitch = 0.0f;
        float distance = 2.2f;
        // point the camera looks at and orbits around (default origin; main may sink it
        // together with the scene so the camera follows the model)
        glm::vec3 target = glm::vec3(0.0f);
    };

    /**
     * @ingroup vulkan_runtime
     * @brief result of one runtime::render_frame() call; the caller reacts to it
     */
    export enum class frame_result {
        render_success, // a frame was recorded, submitted and presented
        skipped,        // not renderable this iteration (window minimized / swapchain recreated); caller yields and retries
        closed,         // the window was closed (ESC or the native close button); caller exits the loop
        failed,         // a fatal Vulkan error occurred; caller exits the loop
    };

    /**
     * @ingroup vulkan_runtime
     * @brief vulkan runtime facade class
     * @note
     *      - use operator-> to access the filtered core view (core_filter, e.g. runtime->get_device())
     *      - default construction performs the whole core initialization (window/instance/device/swap chain etc.)
     *        and registers the orbit camera mouse callbacks on the window
     */
    export class runtime {
        core vulkan_core;

        // set while the window is iconified; the restore transition recreates the swapchain
        bool was_minimized = false;

        // ---- shared scene resources (single flat descriptor set, see core::init_scene_layouts) ----
        // camera UBO: one buffer per frame slot, updated once per frame, shared by every model
        std::vector<uint64_t> camera_buffer_handles = {};
        std::vector<void*> camera_mapped = {};
        // texture registry: flat entries of the set 0 binding 1 array (raw handles; the white
        // fallback view may repeat); the owning views / vma handles live in the vectors below
        std::vector<VkImageView> texture_array_views = {};
        std::vector<vk_image_view> owned_texture_views = {};
        std::vector<uint64_t> owned_texture_handles = {};
        uint32_t white_texture_index = 0;
        // scene-wide IBL (bindings 2-4): prefiltered env / irradiance / BRDF LUT, uploaded once
        std::vector<vk_image_view> ibl_views = {};
        std::vector<uint64_t> ibl_handles = {};
        vk_sampler texture_sampler = {};
        vk_sampler env_sampler = {};
        // GPU material table (set 0 binding 5): one material_record per entry (texture indices +
        // factors + flags); models only push their material_index. Host-visible, written at
        // registration, read-only for the GPU.
        uint64_t material_buffer_handle = 0;
        void* material_mapped = nullptr;
        uint32_t material_count = 0;
        // the single scene descriptor set: all pipelines share the layout, so one set covers them all
        vk_descriptor_set scene_set = {};
        bool scene_set_created = false;
        bool ibl_ready = false;
        // background pass (fullscreen triangle, no depth test): drawn first every frame
        std::optional<vk_pipeline> skybox_pipeline = std::nullopt;

        std::mutex access_mutex;
        // string keys (not string_view): the runtime owns the pipeline/model names, so lookups
        // stay valid regardless of the caller's storage lifetime. std::less<> enables heterogeneous
        // lookup, so the string_view-based API (get_pipeline / make_model / ...) still works
        // without constructing a std::string per call.
        std::map<std::string, vk_pipeline, std::less<>> pipelines;
        // models grouped by the pipeline they bind against: bind the pipeline once, draw them all
        std::map<std::string, std::vector<model>, std::less<>> models;
        // one command buffer per frame slot, used and reused by render_frame()
        std::vector<vk_command_buffer> command_buffers;
        // filtered view over vulkan_core, exposed via operator-> (external code never sees the raw core)
        core_filter filtered_core;

        /**
         * @ingroup vulkan_runtime
         * @brief begin the frame's rendering on the given command buffer: clears the color
         *        attachment with a dark background and the depth attachment
         * @param command_buffer the command buffer being recorded
         * @param image_index the acquired swapchain image index (selects the attachment views)
         * @note uses vkCmdBeginRendering (dynamic rendering) when the device supports it,
         *       otherwise falls back to the classic render pass + framebuffer path
         */
        void begin_rendering(VkCommandBuffer command_buffer, uint32_t image_index) const;

        // ---- scene resource management (see the members above) ----
        void init_scene_resources();                               // camera UBO buffers + white fallback texture + texture sampler + material table
        void ensure_scene_set();                                   // lazily create the scene set and write camera + IBL + material bindings
        void write_ibl_bindings() const;                           // (re)write bindings 2-4 with the current IBL views / placeholders
        uint32_t register_material(model_create_info const& info); // upload textures into the array, append a material_record, return its index

    public:
        // A non-const runtime exposes a mutable filter (e.g. runtime->get_vma()); a const runtime
        // gets a read-only filter, so mutating operations are impossible through const access.
        core_filter* operator->() noexcept {
            return &this->filtered_core;
        }
        core_filter const* operator->() const noexcept {
            return &this->filtered_core;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief orbit camera state; left-drag rotates, wheel zooms
         */
        orbit_camera camera;

        runtime();

        /**
         * @ingroup vulkan_runtime
         * @brief destroy cached pipelines before the inner core (and thus the VkDevice) is destroyed
         * @note explicit destructor: member destruction order is reverse declaration order, which
         *      currently already destroys pipelines before vulkan_core; making it explicit keeps
         *      that guarantee even if members are reordered later
         */
        ~runtime();

        /**
         * @ingroup vulkan_runtime
         * @brief drive one application frame: poll window events, respond to ESC / native close,
         *        skip rendering while the window is minimized, recreate the swapchain on restore
         *        and on resize (VK_ERROR_OUT_OF_DATE_KHR / VK_SUBOPTIMAL_KHR), then record,
         *        submit and present one frame when renderable
         * @return frame_result: render_success when a frame was presented; skipped when not renderable
         *         (minimized or swapchain recreated — caller yields and calls again); closed on
         *         window close; failed on a fatal Vulkan error (caller exits the loop)
         * @note all window-event handling, swapchain recreation and frame management live here,
         *       so the caller's loop needs no Vulkan or GLFW knowledge
         */
        frame_result render_frame();

        std::expected<void, std::string> make_pipeline(
            std::string_view pipeline_name,
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief create the skybox background pipeline: a fullscreen triangle (drawn with
         *        vkCmdDraw(3), no vertex/index buffers) that samples the environment cubemap
         * @param vertex_shader_code raw SPIR-V binary of the skybox vertex shader
         * @param fragment_shader_code raw SPIR-V binary of the skybox fragment shader
         * @return success, or an error message on failure
         * @note drawn first in every frame with depth test/write disabled, so models render over it;
         *       uses the shared scene set (camera UBO binding 0, env cubemap binding 2)
         */
        std::expected<void, std::string> make_skybox_pipeline(
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief get a cached pipeline by its name
         * @param pipeline_name the name passed to make_pipeline()
         * @return pointer to the cached pipeline, or nullptr if no pipeline with that name exists
         */
        [[nodiscard]] vk_pipeline const* get_pipeline(std::string_view pipeline_name) const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief upload the scene-wide IBL resources (prefiltered env / irradiance / BRDF LUT)
         *        into the shared scene set; call it before creating models that use IBL
         * @param info precomputed split-sum IBL bytes (see vulkan::generate_* helpers)
         * @note the images are uploaded once and shared by every model (they used to be
         *       duplicated per model)
         */
        void set_ibl(ibl_input const& info);

        /**
         * @ingroup vulkan_runtime
         * @brief create and cache a model (geometry + texture array registration)
         * @param pipeline_name the pipeline the model draws with (must already exist)
         * @param info geometry and material textures
         * @return pointer to the appended model, or nullptr if the pipeline does not exist
         * @note the model is owned by the runtime and released in its destructor (before the VkDevice);
         *       its textures are appended to the shared scene texture array, its descriptor state
         *       is the single scene set bound once per frame by render_frame()
         * @warning appending another model to the same pipeline may reallocate its vector and
         *          invalidate previously returned pointers; use get_models() for index-based access
         */
        model* make_model(std::string_view pipeline_name, model_create_info const& info);

        /**
         * @ingroup vulkan_runtime
         * @brief get all models cached under a pipeline
         * @param pipeline_name the pipeline name passed to make_model()
         * @return pointer to the model vector, or nullptr if no model uses that pipeline
         */
        [[nodiscard]] std::vector<model> const* get_models(std::string_view pipeline_name) const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief batch-import a scene by traversing an iterator of drawables directly.
         *        The iterator must model vulkan::scene_drawable_iterator: ++ advances to the
         *        next drawable, and the geometry/material is read through get_vertex() /
         *        get_index() / get_transform() / get_albedo() ... get_factors(). The runtime
         *        drives the whole traversal: it uploads buffers, registers materials and
         *        creates one model per drawable on the "pbr" pipeline. No glTF (or any scene
         *        format) knowledge lives in the runtime.
         * @param first,last iterator pair over the scene's drawables
         * @param offset translation applied before each model's matrix (e.g. -scene_center + sink)
         * @return counts of imported primitives and materials
         */
        template <class I, class S>
            requires scene_drawable_iterator<I> && std::same_as<S, I>
        scene_import_result import_scene(I first, S last, glm::vec3 const& offset) {
            scene_import_result result = {};
            uint32_t const materials_before = this->material_count;
            for (; first != last; ++first) {
                vulkan::vertex_data_view const vertex = first.get_vertex();
                vulkan::index_data_view const index = first.get_index();
                model_create_info info = {};
                info.vertex_data = vertex.data;
                info.vertex_stride = vertex.stride;
                info.vertex_count = vertex.count;
                info.index_data = index.data;
                info.index_type = index.type;
                info.index_count = index.count;
                info.albedo = first.get_albedo();
                info.metallic_roughness = first.get_metallic_roughness();
                info.normal = first.get_normal();
                info.occlusion = first.get_occlusion();
                info.emissive = first.get_emissive();
                info.factors = first.get_factors();
                info.model_matrix = glm::translate(glm::mat4(1.0f), offset) * first.get_transform();
                if (this->make_model("pbr", info) == nullptr) {
                    utility::panic(std::source_location::current(), "failed to import drawable (pipeline 'pbr' missing)");
                }
                ++result.primitive_count;
            }
            result.material_count = this->material_count - materials_before;
            return result;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief destroy and remove all models of a pipeline, no-op if the pipeline has none
         * @param pipeline_name the pipeline name passed to make_model()
         */
        void clear_models(std::string_view pipeline_name);
    };
} // namespace vulkan