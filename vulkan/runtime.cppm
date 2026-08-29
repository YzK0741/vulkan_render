module;

#include <vulkan/vulkan.h>

export module vulkan.runtime;
export import std;
export import vulkan.core;

/**
 * @file runtime.cppm
 * @defgroup vulkan_runtime Vulkan Runtime Facade
 * @brief runtime facade: a thin wrapper exposing all functionality of vulkan::core
 * @note
 *      - access the inner core via operator->, same usage as a raw core
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
     * @brief vulkan runtime facade class
     * @note
     *      - use operator-> to access the inner core (e.g. runtime->vma.create_buffer(...))
     *      - default construction performs the whole core initialization (window/instance/device/swap chain etc.)
     *        and registers the orbit camera mouse callbacks on the window
     */
    export class runtime {
        core vulkan_core;

        std::mutex access_mutex;
        std::map<std::string_view, vk_pipeline> pipelines;

    public:
        core* operator->() {
            return &this->vulkan_core;
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
         * @brief begin the swapchain render pass on the given command buffer
         * @param command_buffer the command buffer being recorded
         * @param image_index the acquired swapchain image index (selects the framebuffer)
         * @note clears the color attachment with a dark background and the depth attachment
         */
        void begin_render_pass(VkCommandBuffer command_buffer, uint32_t image_index);

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
    };
} // namespace vulkan