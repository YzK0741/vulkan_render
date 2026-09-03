module;

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

export module vulkan.core;
import utility;
export import std;
export import vulkan.core.handles;
export import vulkan.core.vma;

/**
 * @file core.cppm
 */

namespace vulkan {
    /**
     * @defgroup vulkan_core Vulkan Core Objects Manager
     * @ingroup vulkan_core
     * @brief manages core vulkan objects and windows instance init and destroy.
     * @note
     *      - RAII
     *      - includes VMA decorator, which is defined in ./vma/vma.cppm
     *
     * @warning
     *      - do not call the init or create function, just use the members or other functions
     *      - no thread-safe
     */

    /**
     * @ingroup vulkan_core
     * @brief agreed flat scene descriptor set layout, shared by all pipelines (see shaders/pbr.frag):
     *        set 0 binding 0 = CameraUBO (uniform buffer, update-after-bind),
     *              binding 1 = sampler2D textures[] (runtime array, partially bound + update-after-bind + non-uniform index),
     *              binding 2/3/4 = prefiltered env / irradiance / BRDF LUT (combined image samplers),
     *              binding 5 = Material materials[] (storage buffer: per-material texture indices + factors),
     *              binding 6 = mat4 instance transforms[] (storage buffer, per-instance world matrices),
     *              binding 7 = LightUBO (uniform buffer: directional light view-proj + direction),
     *              binding 8 = shadow map (sampler2DShadow, depth comparison + hardware PCF)
     * @note hardcoded instead of parsed from SPIR-V: the indexed layout is flat, so pipelines
     *       skip descriptor / push constant parsing and share one layout object
     */
    export constexpr uint32_t scene_texture_capacity = 128;
    // material_push_constants: uint material_index + mat4 model = 80 bytes, see vulkan/runtime/scene_tree/scene_tree.cppm
    export constexpr uint32_t scene_push_constant_size = 80;
    export struct core : utility::enable_stack_destruct {
        VkInstance instance = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        uint32_t graphics_family_index = 0;
        uint32_t present_family_index = 0;
        VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
        void init_instance() noexcept;

        GLFWwindow* window = nullptr;
        void init_window(int width, int height, std::string_view window_name = "") noexcept;

        // ---- facade operations (keep raw Vulkan / GLFW calls out of the caller) ----
        void wait_idle() const noexcept;                              // vkDeviceWaitIdle
        void set_window_title(std::string_view title) const noexcept; // glfwSetWindowTitle

        VkSurfaceKHR surface = VK_NULL_HANDLE;
        void init_surface() noexcept;

        VkQueue graphics_queue = VK_NULL_HANDLE;
        VkQueue present_queue = VK_NULL_HANDLE;
        uint32_t graphics_queue_family = VK_QUEUE_FAMILY_IGNORED;
        void init_device_and_queue() noexcept;

        VkSwapchainKHR swap_chain = {};
        std::vector<VkImage> swap_chain_images = {};
        VkFormat swap_chain_image_format = {};
        VkExtent2D swap_chain_extent = {};

        void init_swap_chain() noexcept;

        std::vector<VkImageView> swap_chain_image_views = {};

        void init_image_views() noexcept;

        // MSAA related
        VkSampleCountFlagBits msaa_samples = VK_SAMPLE_COUNT_1_BIT; // no MSAA by default
        std::vector<VkImage> color_images = {};                     // MSAA color buffer images
        std::vector<VkDeviceMemory> color_image_memories = {};
        std::vector<VkImageView> color_image_views = {}; // MSAA image views
        VkFormat color_format = VK_FORMAT_UNDEFINED;
        void create_msaa_image(
            uint32_t const& width,
            uint32_t const& height,
            VkFormat const& format,
            VkSampleCountFlagBits const& num_samples,
            VkImageTiling const& tiling,
            VkImageUsageFlags const& usage,
            VkMemoryPropertyFlags const& properties,
            VkImage& image,
            VkDeviceMemory& image_memory) const noexcept;

        VkFormat depth_format = {};
        std::vector<VkImage> depth_images = {};
        std::vector<VkDeviceMemory> depth_image_memories = {};
        std::vector<VkImageView> depth_image_views = {};
        void create_depth_image(VkImage& image, VkDeviceMemory& image_memory, VkImageView& image_view) const noexcept;
        void create_depth_resources() noexcept;

        void create_color_resources();

        VkCommandPool command_pool = {};
        std::vector<VkCommandBuffer> command_buffers = {};
        void create_command_pool() noexcept;

        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        void create_descriptor_pool() noexcept;

        // shared scene layouts (see the scene_texture_capacity / scene_push_constant_size docs above);
        // all pipelines are created against scene_pipeline_layout, so one descriptor set works for all
        VkDescriptorSetLayout scene_descriptor_set_layout = VK_NULL_HANDLE;
        VkPipelineLayout scene_pipeline_layout = VK_NULL_HANDLE;
        void init_scene_layouts() noexcept;

        VkRenderPass renderpass = {};
        void init_renderpass() noexcept;

        // true when the device supports dynamic rendering (Vulkan 1.3 core): frames then use
        // vkCmdBeginRendering and no render pass / framebuffer objects exist; devices without
        // it fall back to the classic render pass path below
        bool use_dynamic_rendering = false;

        std::vector<VkFramebuffer> swap_chain_framebuffers = {};
        void create_frame_buffers() noexcept;

        vma_allocator vma = {};

        std::vector<VkSemaphore> image_available_semaphores;

        std::vector<VkSemaphore> render_finished_semaphores;

        std::vector<VkFence> in_flight_fences;

        std::vector<VkFence> images_in_flight;

        size_t current_frame = 0;

        void to_next_frame() noexcept;

        static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

        void create_sync_objects();

        core();
        ~core();

        vk_command_buffer make_command_buffer() const;
        vk_descriptor_set make_descriptor_set(VkDescriptorSetLayout layout) const;

        std::optional<vk_shader_module> make_shader_module(std::span<unsigned char> shader) const noexcept;

        /**
         * @ingroup vulkan_core
         * @brief create an image view covering the whole image (all mip levels and layers)
         * @param image the image to view
         * @param format the view format
         * @param type view type (VK_IMAGE_VIEW_TYPE_2D / VK_IMAGE_VIEW_TYPE_CUBE ...)
         * @return raii vk_image_view owning the created view
         */
        vk_image_view make_image_view(VkImage image, VkFormat format, VkImageViewType type) const;

        /**
         * @ingroup vulkan_core
         * @brief create a 2D depth image view (DEPTH aspect) over the whole image
         * @param image the image to view
         * @param format the view format (a depth format)
         * @return raii vk_image_view owning the created view
         * @note the regular make_image_view uses the COLOR aspect; depth images (e.g. the shadow
         *       map) need the DEPTH aspect to be sampled as depth
         */
        vk_image_view make_depth_image_view(VkImage image, VkFormat format) const;

        /**
         * @ingroup vulkan_core
         * @brief create a linear/min-linear sampler with the given wrap mode
         * @param address_mode wrap mode applied to all three axes
         * @param max_lod maximum mip level the sampler may access
         * @return raii vk_sampler owning the created sampler
         */
        vk_sampler make_sampler(VkSamplerAddressMode address_mode, float max_lod) const;

        /**
         * @ingroup vulkan_core
         * @brief create the shadow map sampling sampler (NEAREST + clamp-to-edge)
         * @return raii vk_sampler owning the created sampler
         * @note pbr.frag does manual percentage-closer filtering: it fetches the stored depth
         *       with this NEAREST sampler at a few neighbor texels and averages the comparisons,
         *       so no depth-comparison/linear-filter format feature is required
         */
        vk_sampler make_shadow_sampler() const;

        /**
         * @ingroup vulkan_core
         * @brief create a depth-only graphics pipeline (no color attachment, single sample),
         *        used by the shadow pass to render depth into the shadow map
         * @param vertex_shader_code raw SPIR-V binary of the vertex shader
         * @param fragment_shader_code raw SPIR-V binary of the fragment shader
         * @param depth_format depth attachment format (dynamic rendering only)
         * @return vk_pipeline on success, error message on failure
         * @note requires dynamic rendering (Vulkan 1.3); on the classic render-pass fallback
         *       path it returns an error and the caller should disable shadow mapping
         */
        std::expected<vk_pipeline, std::string_view> make_depth_pipeline(
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code,
            VkFormat depth_format) const;

        VkResult get_image_index(uint32_t& image_index) const;
        void wait_usable_image(uint32_t image_index);
        void reset_fence(uint32_t frame_index) const;
        void recreate_swap_chain();

        /**
         * @ingroup vulkan_core
         * @brief submit the recorded command buffer for the current frame
         * @param command_buffer the command buffer to submit
         * @param image_index the acquired swapchain image index (selects the render finished semaphore)
         * @return the result of vkQueueSubmit
         */
        VkResult submit(VkCommandBuffer command_buffer, uint32_t image_index) const;

        /**
         * @ingroup vulkan_core
         * @brief present the rendered swapchain image
         * @param image_index the swapchain image index to present
         * @return the result of vkQueuePresentKHR
         */
        VkResult present(uint32_t image_index) const;

        std::expected<vk_pipeline, std::string_view> make_pipeline(
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code,
            bool depth_test_enabled = true) const;
    };
} // namespace vulkan