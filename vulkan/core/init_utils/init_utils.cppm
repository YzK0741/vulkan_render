module;

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

export module vulkan.core.init_utils;
export import std;

/**
 * @file init_utils.cppm
 * @defgroup vulkan_init_utils Vulkan Init Utils
 * @brief pure utility functions and data types for device/queue/swap chain selection and creation
 * @note
 *      - most functions are noexcept, failures either return a sentinel or panic
 *      - used by vulkan.core during initialization
 */
/**
 * @ingroup vulkan_init_utils
 * @brief a created logical device together with its queue handles
 */
export struct logical_device {
    VkDevice device = VK_NULL_HANDLE;
    uint32_t graphics_family_index = 0;
    uint32_t present_family_index = 0;
    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue present_queue = VK_NULL_HANDLE;
};

/**
 * @ingroup vulkan_init_utils
 * @brief capabilities, formats and present modes supported by a swap chain
 */
export struct swap_chain_support_details {
    VkSurfaceCapabilitiesKHR capabilities = {};
    std::vector<VkSurfaceFormatKHR> formats = {};
    std::vector<VkPresentModeKHR> present_modes = {};
};

/**
 * @ingroup vulkan_init_utils
 * @brief queue family indices found on a physical device
 * @note is_complete() is true when both graphics and present families are found
 */
export struct queue_family_indices {
    std::optional<uint32_t> graphics_family = {};
    std::optional<uint32_t> present_family = {};
    std::optional<uint32_t> compute_family = {};
    std::optional<uint32_t> transfer_family = {};

    [[nodiscard]] bool is_complete() const noexcept {
        return graphics_family.has_value() && present_family.has_value();
    }
};

/**
 * @ingroup vulkan_init_utils
 * @brief collected physical device capabilities: core 1.x features and properties
 * @note
 *      - query() builds the pNext chains from the requested api_version and issues
 *        vkGetPhysicalDeviceFeatures2 + vkGetPhysicalDeviceProperties2 in one pass
 *      - after query(), features stay enabled where the driver supports them (except the
 *        deliberately disabled ones), so the struct is directly usable for vkCreateDevice
 *      - device_pnext() returns the feature chain head to pass via device_creation_info::pNext
 */
export struct device_capabilities {
    // ---- 特性链（查询与设备创建共用） ----
    VkPhysicalDeviceFeatures2 features2 = {};
    VkPhysicalDeviceVulkan11Features features_1_1 = {};
    VkPhysicalDeviceVulkan12Features features_1_2 = {};
    VkPhysicalDeviceVulkan13Features features_1_3 = {};
    VkPhysicalDeviceVulkan14Features features_1_4 = {};

    // ---- 属性链（只查询，供渲染器决策/诊断） ----
    VkPhysicalDeviceProperties2 properties2 = {};
    VkPhysicalDeviceDriverProperties driver_properties = {};
    VkPhysicalDeviceSubgroupProperties subgroup_properties = {};
    VkPhysicalDeviceDescriptorIndexingProperties descriptor_indexing_properties = {};
    VkPhysicalDeviceMaintenance4Properties maintenance4_properties = {};

    /**
     * @brief query all features and properties of the physical device
     * @param physical_device the device to query
     * @param api_version the API version to target, decides which 1.x structs are chained
     */
    void query(VkPhysicalDevice physical_device, uint32_t api_version = VK_API_VERSION_1_3) noexcept;

    /**
     * @brief pNext chain head for vkCreateDevice (enables the queried features)
     */
    [[nodiscard]] const void* device_pnext() const noexcept;
};

/**
 * @ingroup vulkan_init_utils
 * @brief print a startup summary of the collected device capabilities
 * @param capabilities the queried capabilities
 * @note prints driver/api version/device name and all enabled 1.1/1.2/1.3 features
 */
export void print_device_capabilities(device_capabilities const& capabilities);

/**
 * @ingroup vulkan_init_utils
 * @brief input data for creating a logical device
 */
export struct device_creation_info {
    queue_family_indices queue_families{};
    std::vector<const char*> extensions = {};
    std::vector<const char*> validation_layers = {};
    VkPhysicalDeviceFeatures device_features = {};
    const void* pNext = nullptr; // 用于Vulkan 1.1+的特性链
};

/**
 * @ingroup vulkan_init_utils
 * @brief Vulkan debug messenger callback, prints validation layer messages
 */
export [[maybe_unused]] VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(
    const VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    [[maybe_unused]] void* user_data) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief check whether the given validation layers are supported
 * @param validation_layers layer names to check
 * @return true if all layers are supported
 */
export bool check_validation_layer_support(const std::vector<const char*>& validation_layers) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief check whether the physical device supports the required device extensions
 * @param physical_device the device to check
 * @param required_extensions extension names that must be supported
 * @return true if all extensions are supported
 */
export bool check_device_extension_support(
    VkPhysicalDevice const& physical_device,
    std::vector<const char*> const& required_extensions) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief create a logical device and fetch its queues
 * @param physical_device the physical device to create from
 * @param create_info device features, extensions and validation layers
 * @return logical_device holding the device and queue handles on success; panics on failure
 */
export logical_device create_logical_device(
    VkPhysicalDevice physical_device,
    device_creation_info const& create_info) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief find graphics/present/compute/transfer queue family indices of a physical device
 * @param device the physical device to query
 * @param surface the presentation surface, may be VK_NULL_HANDLE
 * @return queue family indices, missing families are std::nullopt
 */
export queue_family_indices find_queue_families(VkPhysicalDevice const& device, VkSurfaceKHR const& surface) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief pick the most suitable physical device for the given surface
 * @param instance the vulkan instance
 * @param surface the presentation surface
 * @return the chosen physical device; panics if none is suitable
 */
export VkPhysicalDevice pick_suitable_device(VkInstance const& instance, VkSurfaceKHR surface) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief query swap chain support details of a physical device
 * @param device the physical device to query
 * @param surface the presentation surface
 * @return swap chain capabilities, formats and present modes
 */
export swap_chain_support_details query_swap_chain_support(VkPhysicalDevice const& device, VkSurfaceKHR const& surface) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief choose a present mode from the available ones
 * @param available_present_modes modes supported by the swap chain
 * @return the chosen present mode
 */
export VkPresentModeKHR choose_swap_present_mode(std::vector<VkPresentModeKHR> const& available_present_modes) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief choose a surface format from the available ones
 * @param available_formats formats supported by the swap chain
 * @return the chosen surface format
 */
export VkSurfaceFormatKHR choose_swap_surface_format(std::vector<VkSurfaceFormatKHR> const& available_formats) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief choose the swap chain extent, clamped to the surface capabilities
 * @param capabilities surface capabilities
 * @param window the GLFW window, used to query the framebuffer size
 * @return the chosen extent
 */
export VkExtent2D choose_swap_extent(VkSurfaceCapabilitiesKHR const& capabilities, GLFWwindow* window) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief find a memory type matching the filter and the required properties
 * @param type_filter bit mask of allowed memory types
 * @param properties required memory property flags
 * @param physical_device the physical device to query
 * @return the matching memory type index; panics if none matches
 */
export uint32_t find_memory_type(const uint32_t& type_filter, const VkMemoryPropertyFlags& properties, const VkPhysicalDevice& physical_device) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief find a supported depth format by preference order
 * @param physical_device the physical device to query
 * @return the chosen depth format; panics if none is supported
 */
export VkFormat find_depth_format(const VkPhysicalDevice& physical_device) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief get the maximum usable MSAA sample count of the physical device
 * @param physical_device the physical device to query
 * @return the maximum supported sample count flag
 */
export VkSampleCountFlagBits get_max_usable_sample_count(const VkPhysicalDevice& physical_device) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief create an image view for the given image
 * @param image the source image
 * @param format the image format
 * @param aspectFlags the image aspect mask
 * @param device the logical device
 * @return the created image view
 */
export VkImageView create_image_view(const VkImage& image, const VkFormat& format, const VkImageAspectFlags& aspectFlags, const VkDevice& device) noexcept;