module;

#include <vulkan/vulkan.h>

export module vulkan.runtime;
export import std;
export import vulkan.core;
export import vulkan.core.filter;
export import vulkan.model;

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
        float pitch = 0.35f; // slight downward tilt
        float distance = 2.2f;
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
         * @brief begin the swapchain render pass on the given command buffer
         * @param command_buffer the command buffer being recorded
         * @param image_index the acquired swapchain image index (selects the framebuffer)
         * @note clears the color attachment with a dark background and the depth attachment
         */
        void begin_render_pass(VkCommandBuffer command_buffer, uint32_t image_index) const;

    public:
        const core_filter* operator->() const noexcept {
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
            std::span<const unsigned char> vertex_shader_code,
            std::span<const unsigned char> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief get a cached pipeline by its name
         * @param pipeline_name the name passed to make_pipeline()
         * @return pointer to the cached pipeline, or nullptr if no pipeline with that name exists
         */
        [[nodiscard]] const vk_pipeline* get_pipeline(std::string_view pipeline_name) const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief create and cache a model (geometry + material + per-frame UBOs)
         * @param pipeline_name the pipeline whose descriptor set layouts the model binds against (must already exist)
         * @param info geometry / material / IBL data
         * @return pointer to the appended model, or nullptr if the pipeline does not exist
         * @note the model is owned by the runtime and released in its destructor (before the VkDevice)
         * @warning appending another model to the same pipeline may reallocate its vector and
         *          invalidate previously returned pointers; use get_models() for index-based access
         */
        model* make_model(std::string_view pipeline_name, const model_create_info& info);

        /**
         * @ingroup vulkan_runtime
         * @brief get all models cached under a pipeline
         * @param pipeline_name the pipeline name passed to make_model()
         * @return pointer to the model vector, or nullptr if no model uses that pipeline
         */
        [[nodiscard]] const std::vector<model>* get_models(std::string_view pipeline_name) const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief destroy and remove all models of a pipeline, no-op if the pipeline has none
         * @param pipeline_name the pipeline name passed to make_model()
         */
        void clear_models(std::string_view pipeline_name);
    };
} // namespace vulkan