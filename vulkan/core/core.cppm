//
// Created by 小叶 on 2026/7/27.
//
module;

#include <optional>
#include <string_view>
#include <vector>
#include <span>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

export module vulkan.core;
import utility;
export import vulkan.core.handles;
export import vulkan.vma;



namespace vulkan {

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

        VkSurfaceKHR surface = VK_NULL_HANDLE;
        void init_surface() noexcept;

        VkQueue graphics_queue = VK_NULL_HANDLE;
        VkQueue present_queue = VK_NULL_HANDLE;
        uint32_t graphics_queue_family = VK_QUEUE_FAMILY_IGNORED;
        void init_device_and_queue() noexcept;

        VkSwapchainKHR swap_chain = {};
        std::vector<VkImage> swap_chain_images{};
        VkFormat swap_chain_image_format = {};
        VkExtent2D swap_chain_extent = {};

        void init_swap_chain() noexcept;

        std::vector<VkImageView> swap_chain_image_views{};

        void init_image_views() noexcept;


        // MSAA相关
        VkSampleCountFlagBits msaa_samples = VK_SAMPLE_COUNT_1_BIT;  // 默认为无MSAA
        std::vector<VkImage> color_images = {};       // MSAA颜色缓冲图像
        std::vector<VkDeviceMemory> color_image_memories = {};
        std::vector<VkImageView> color_image_views = {};  // MSAA图像视图
        VkFormat color_format = VK_FORMAT_UNDEFINED;
        void create_msaa_image(
            const uint32_t& width,
            const uint32_t &height,
            const VkFormat &format,
            const VkSampleCountFlagBits &num_samples,
            const VkImageTiling &tiling,
            const VkImageUsageFlags &usage,
            const VkMemoryPropertyFlags &properties,
            VkImage& image,
            VkDeviceMemory& imageMemory
            ) const noexcept;

        VkFormat depth_format = {};
        std::vector<VkImage> depth_images = {};
        std::vector<VkDeviceMemory> depth_image_memories = {};
        std::vector<VkImageView> depth_image_views = {};
        void create_depth_image(VkImage& image, VkDeviceMemory& imageMemory, VkImageView& image_view) const noexcept;
        void create_depth_resources() noexcept;

        void create_color_resources();

        VkCommandPool command_pool = {};
        std::vector<VkCommandBuffer> command_buffers = {};
        void create_command_pool() noexcept;

        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        void create_descriptor_pool() noexcept;

        VkRenderPass renderpass = {};
        void init_renderpass() noexcept;

        std::vector<VkFramebuffer> swap_chain_framebuffers{};
        void create_frame_buffers() noexcept;

        vma_allocator vma = {};

        // 图像可用信号量（当交换链图像准备好渲染时触发）
        std::vector<VkSemaphore> image_available_semaphores;

        // 渲染完成信号量（当渲染完成可以呈现时触发）
        std::vector<VkSemaphore> render_finished_semaphores;

        // 每帧的栅栏（确保同一帧的命令缓冲区不会同时执行）
        std::vector<VkFence> in_flight_fences;

        // 跟踪哪些帧正在使用中
        std::vector<VkFence> images_in_flight;

        // 当前帧索引
        size_t current_frame = 0;

        // 最大并发帧数（通常是交换链图像数量）
        static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

        void create_sync_objects();

        core();
        ~core();

        vk_command_buffer make_command_buffer();
        vk_descriptor_set make_descriptor_set(VkDescriptorSetLayout layout);

        std::optional<vk_shader_module> make_shader_module(std::span<unsigned char> shader) noexcept;

        VkResult get_image_index(uint32_t &image_index) const;
        void wait_usable_image(uint32_t image_index);
        void reset_fence(uint32_t frame_index) const;
        void recreate_swap_chain();
    };
}