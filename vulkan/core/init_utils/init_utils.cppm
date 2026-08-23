//
// Created by 小叶 on 2026/7/29.
//

module;

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

export module vulkan.core.init_uitls;
export import std;

export struct logical_device {
    VkDevice device = VK_NULL_HANDLE;
    uint32_t graphics_family_index = 0;
    uint32_t present_family_index = 0;
    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue present_queue = VK_NULL_HANDLE;
};

export struct swap_chain_support_details {
    VkSurfaceCapabilitiesKHR capabilities = {};
    std::vector<VkSurfaceFormatKHR> formats = {};
    std::vector<VkPresentModeKHR> present_modes = {};
};

export struct queue_family_indices {
    std::optional<uint32_t> graphics_family = {};
    std::optional<uint32_t> present_family = {};
    std::optional<uint32_t> compute_family = {};
    std::optional<uint32_t> transfer_family = {};

    [[nodiscard]] bool is_complete() const noexcept {
        return graphics_family.has_value() && present_family.has_value();
    }
};

export struct device_creation_info {
    queue_family_indices queue_families {};
    std::vector<const char*> extensions = {};
    std::vector<const char*> validation_layers = {};
    VkPhysicalDeviceFeatures device_features = {};
    const void* pNext = nullptr;  // 用于Vulkan 1.1+的特性链
};

export [[maybe_unused]] VKAPI_ATTR VkBool32 VKAPI_CALL
    debug_callback(
    const VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
    [[maybe_unused]] void* user_data
    ) noexcept;

export bool check_validation_layer_support(const std::vector<const char*>& validation_layers) noexcept;

export bool check_device_extension_support(
        VkPhysicalDevice const& physical_device,
        std::vector<const char*> const& required_extensions
        ) noexcept;

export logical_device create_logical_device(
        VkPhysicalDevice physical_device,
        device_creation_info const& create_info
        ) noexcept;

export queue_family_indices find_queue_families(VkPhysicalDevice const& device, VkSurfaceKHR const& surface) noexcept;

export VkPhysicalDevice pick_suitable_device(VkInstance const& instance, VkSurfaceKHR surface) noexcept;

export swap_chain_support_details query_swap_chain_support(VkPhysicalDevice const& device, VkSurfaceKHR const& surface) noexcept;

export VkPresentModeKHR choose_swap_present_mode(std::vector<VkPresentModeKHR> const& available_present_modes) noexcept;

export VkSurfaceFormatKHR choose_swap_surface_format(std::vector<VkSurfaceFormatKHR> const& available_formats) noexcept;

export VkExtent2D choose_swap_extent(VkSurfaceCapabilitiesKHR const& capabilities, GLFWwindow* window) noexcept;

export uint32_t find_memory_type(const uint32_t& type_filter, const VkMemoryPropertyFlags& properties, const VkPhysicalDevice& physical_device) noexcept;

export VkFormat find_depth_format(const VkPhysicalDevice &physical_device) noexcept;

export VkSampleCountFlagBits get_max_usable_sample_count(const VkPhysicalDevice& physical_device) noexcept;

export VkImageView create_image_view(const VkImage& image, const VkFormat& format, const VkImageAspectFlags& aspectFlags, const VkDevice& device) noexcept;