module;

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

export module vulkan.core.init_utils;
export import vstd;

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
    // ---- Feature chain (shared by query and device creation): the sType of every member is
    //      fixed at construction (designated initializer), so query() only re-wires the pNext
    //      chain and issues the vkGetPhysicalDevice*2 calls each time. The remaining members of
    //      the Vulkan chain structs are deliberately left to zero-initialization (correct for
    //      query/creation), which -Wmissing-designated-field-initializers would otherwise flag.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
    VkPhysicalDeviceFeatures2 features_2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features features_1_1 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features features_1_2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features features_1_3 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan14Features features_1_4 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};

    // ---- Property chain (query only, for renderer decisions/diagnostics) ----
    VkPhysicalDeviceProperties2 properties_2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceDriverProperties driver_properties = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties subgroup_properties = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceDescriptorIndexingProperties descriptor_indexing_properties = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};
    VkPhysicalDeviceMaintenance4Properties maintenance4_properties = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
#pragma clang diagnostic pop

    /**
     * @brief query all features and properties of the physical device
     * @param physical_device the device to query
     * @param api_version the API version to target, decides which 1.x structs are chained
     */
    void query(VkPhysicalDevice physical_device, uint32_t api_version = VK_API_VERSION_1_3) noexcept;

    /**
     * @brief pNext chain head for vkCreateDevice (enables the queried features)
     */
    [[nodiscard]] void const* device_pnext() const noexcept;
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
    queue_family_indices queue_families = {};
    std::vector<char const*> extensions = {};
    std::vector<char const*> validation_layers = {};
    // Feature chain head for vkCreateDevice: must be a VkPhysicalDeviceFeatures2 struct (e.g.
    // device_capabilities::device_pnext()) whose .features carries the Vulkan 1.0 core features
    // and whose pNext links the 1.1/1.2/... feature structs. pEnabledFeatures stays NULL.
    void const* pNext = nullptr;
};

/**
 * @ingroup vulkan_init_utils
 * @brief Vulkan debug messenger callback, prints validation layer messages
 */
export [[maybe_unused]] VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT const message_severity,
    [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT message_type,
    VkDebugUtilsMessengerCallbackDataEXT const* callback_data,
    [[maybe_unused]] void* user_data) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief check whether the given validation layers are supported
 * @param validation_layers layer names to check
 * @return true if all layers are supported
 */
export bool check_validation_layer_support(std::vector<char const*> const& validation_layers) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief check whether the physical device supports the required device extensions
 * @param physical_device the device to check
 * @param required_extensions extension names that must be supported
 * @return true if all extensions are supported
 */
export bool check_device_extension_support(
    VkPhysicalDevice physical_device,
    std::vector<char const*> const& required_extensions) noexcept;

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
export queue_family_indices find_queue_families(VkPhysicalDevice device, VkSurfaceKHR surface) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief pick the most suitable physical device for the given surface
 * @param instance the vulkan instance
 * @param surface the presentation surface
 * @return the chosen physical device; panics if none is suitable
 */
export VkPhysicalDevice pick_suitable_device(VkInstance instance, VkSurfaceKHR surface) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief query swap chain support details of a physical device
 * @param device the physical device to query
 * @param surface the presentation surface
 * @return swap chain capabilities, formats and present modes
 */
export swap_chain_support_details query_swap_chain_support(VkPhysicalDevice device, VkSurfaceKHR surface) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief choose a present mode from the available ones
 * @param available_present_modes modes supported by the swap chain
 * @param vsync true prefers FIFO_KHR (vsync), false prefers MAILBOX_KHR (low latency)
 * @return the chosen present mode
 */
/**
 * @brief VK_PRESENT_MODE_FIFO_LATEST_READY (VK_EXT_present_mode_fifo_latest_ready, core in Vulkan 1.4)
 *
 * Spelled out numerically so the build does not depend on how new the Vulkan headers are: the value
 * is fixed by the extension. Like FIFO it is vsync-locked and never tears, but the presentation
 * engine shows the newest ready image at each vblank instead of draining a queue, so an application
 * that renders faster than the display does not pay FIFO's extra frame of latency.
 */
export constexpr VkPresentModeKHR present_mode_fifo_latest_ready = static_cast<VkPresentModeKHR>(1000361000);

export VkPresentModeKHR choose_swap_present_mode(std::vector<VkPresentModeKHR> const& available_present_modes, bool vsync = false) noexcept;

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
export VkExtent2D choose_swap_extent(VkSurfaceCapabilitiesKHR capabilities, GLFWwindow* window) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief find a memory type matching the filter and the required properties
 * @param type_filter bit mask of allowed memory types
 * @param properties required memory property flags
 * @param physical_device the physical device to query
 * @return the matching memory type index; panics if none matches
 */
export uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties, VkPhysicalDevice physical_device) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief find a supported depth format by preference order
 * @param physical_device the physical device to query
 * @return the chosen depth format; panics if none is supported
 */
export VkFormat find_depth_format(VkPhysicalDevice physical_device) noexcept;

/**
 * @ingroup vulkan_init_utils
 * @brief create an image view for the given image
 * @param image the source image
 * @param format the image format
 * @param aspect_flags the image aspect mask
 * @param device the logical device
 * @return the created image view
 */
export VkImageView create_image_view(VkImage image, VkFormat format, VkImageAspectFlags aspect_flags, VkDevice device) noexcept;