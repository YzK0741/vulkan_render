module;

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

module vulkan.core.init_utils;

import utility;

[[maybe_unused]] VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT const message_severity,
    [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT message_type,
    VkDebugUtilsMessengerCallbackDataEXT const* callback_data,
    [[maybe_unused]] void* user_data) noexcept {
    if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        // Errors go through error(): Debug prints red to stderr, Release writes to the log file
        utility::error(callback_data->pMessage);
    } else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        utility::log("[WARNING] {}", callback_data->pMessage);
    } else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        utility::log("[INFO] {}", callback_data->pMessage);
    } else {
        utility::log("[VERBOSE] {}", callback_data->pMessage);
    }

    return VK_FALSE; // VK_FALSE means not terminate this function call
}

bool check_validation_layer_support(std::vector<char const*> const& validation_layers) noexcept {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    std::unordered_set<std::string> available_names;
    for (auto const& layer : available_layers) {
        available_names.insert(layer.layerName);
    }

    return std::ranges::all_of(validation_layers,
                               [&](char const* name) { return available_names.contains(name); });
}

bool check_device_extension_support(
    VkPhysicalDevice const& physical_device,
    std::vector<char const*> const& required_extensions) noexcept {
    uint32_t extension_count;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count,
                                         available_extensions.data());

    std::set<std::string> required_set(required_extensions.begin(), required_extensions.end());
    for (auto const& [extension_name, spec_version] : available_extensions) {
        required_set.erase(extension_name);
    }

    return required_set.empty();
}

void device_capabilities::query(VkPhysicalDevice const physical_device, uint32_t const api_version) noexcept {
    // ---- Feature pNext chain: features_2 -> 1_1 -> 1_2 -> 1_3 -> 1_4 (truncated by api_version) ----
    features_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features_1_1.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features_1_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features_1_3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features_1_4.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;

    features_2.pNext = api_version >= VK_API_VERSION_1_1 ? &features_1_1 : nullptr;
    features_1_1.pNext = api_version >= VK_API_VERSION_1_2 ? &features_1_2 : nullptr;
    features_1_2.pNext = api_version >= VK_API_VERSION_1_3 ? &features_1_3 : nullptr;
    features_1_3.pNext = api_version >= VK_API_VERSION_1_4 ? &features_1_4 : nullptr;
    vkGetPhysicalDeviceFeatures2(physical_device, &features_2);

    // ---- Property pNext chain: properties_2 -> driver -> subgroup -> descriptor indexing -> maintenance4 ----
    properties_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    driver_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    descriptor_indexing_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES;
    maintenance4_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES;

    properties_2.pNext = &driver_properties;
    driver_properties.pNext = &subgroup_properties;
    subgroup_properties.pNext = &descriptor_indexing_properties;
    descriptor_indexing_properties.pNext = &maintenance4_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &properties_2);

    // ---- Feature policy: pass through driver support except explicitly disabled ones (take most features except ray tracing) ----
    features_1_1.protectedMemory = VK_FALSE; // protected memory not needed for now
}

void const* device_capabilities::device_pnext() const noexcept {
    return &this->features_1_1;
}

namespace {
    // Append names of enabled (VK_TRUE) members to the output string, return the enabled count
    template <typename T>
    size_t append_enabled_features(std::string& out, T const& features,
                                   std::initializer_list<std::pair<char const*, VkBool32 T::*>> const& entries) {
        size_t count = 0;
        for (auto const& [name, member] : entries) {
            if (features.*member == VK_TRUE) {
                if (!out.empty()) {
                    out += ", ";
                }
                out += name;
                ++count;
            }
        }
        return count;
    }

    // Wrap output at the given width, indenting continuation lines
    void print_wrapped(std::string const& text, int const width, std::string_view const indent) {
        std::string current(indent);
        size_t start = 0;
        while (start < text.size()) {
            size_t const comma = text.find(", ", start);
            size_t const token_end = comma == std::string::npos ? text.size() : comma;
            std::string_view const token(text.data() + start, token_end - start);
            if (current.size() > indent.size() && current.size() + token.size() + 2 > static_cast<size_t>(width)) {
                utility::log("{}", current);
                current = std::string(indent);
            }
            if (current.size() > indent.size()) {
                current += ", ";
            }
            current += token;
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 2;
        }
        if (!text.empty()) {
            utility::log("{}", current);
        }
    }
} // namespace

void print_device_capabilities(device_capabilities const& capabilities) {
    constexpr std::string_view box_line = "================================================";
    constexpr std::string_view sep_line = "------------------------------------------------";

    utility::log("{}", box_line);
    utility::log(" Vulkan device capabilities");
    utility::log("{}", box_line);

    static constexpr std::array<char const*, 5> device_type_names = {
        "other", "integrated gpu", "discrete gpu", "virtual gpu", "cpu"};
    uint32_t const device_type = static_cast<uint32_t>(capabilities.properties_2.properties.deviceType);
    char const* type_name = device_type < device_type_names.size() ? device_type_names[device_type] : "unknown";

    utility::log(" driver        : {} {}", capabilities.driver_properties.driverName, capabilities.driver_properties.driverInfo);
    utility::log(" api version   : {}.{}.{}",
                 VK_API_VERSION_MAJOR(capabilities.properties_2.properties.apiVersion),
                 VK_API_VERSION_MINOR(capabilities.properties_2.properties.apiVersion),
                 VK_API_VERSION_PATCH(capabilities.properties_2.properties.apiVersion));
    utility::log(" device        : {} ({})", capabilities.properties_2.properties.deviceName, type_name);
    utility::log("{}", sep_line);

    using feature_1_1 = VkPhysicalDeviceVulkan11Features;
    using feature_1_2 = VkPhysicalDeviceVulkan12Features;
    using feature_1_3 = VkPhysicalDeviceVulkan13Features;

    std::string enabled_1_1;
    size_t const count_1_1 = append_enabled_features(enabled_1_1, capabilities.features_1_1, {
                                                                                                 {"storageBuffer16BitAccess", &feature_1_1::storageBuffer16BitAccess},
                                                                                                 {"uniformAndStorageBuffer16BitAccess", &feature_1_1::uniformAndStorageBuffer16BitAccess},
                                                                                                 {"storagePushConstant16", &feature_1_1::storagePushConstant16},
                                                                                                 {"storageInputOutput16", &feature_1_1::storageInputOutput16},
                                                                                                 {"multiview", &feature_1_1::multiview},
                                                                                                 {"multiviewGeometryShader", &feature_1_1::multiviewGeometryShader},
                                                                                                 {"multiviewTessellationShader", &feature_1_1::multiviewTessellationShader},
                                                                                                 {"variablePointersStorageBuffer", &feature_1_1::variablePointersStorageBuffer},
                                                                                                 {"variablePointers", &feature_1_1::variablePointers},
                                                                                                 {"protectedMemory", &feature_1_1::protectedMemory},
                                                                                                 {"samplerYcbcrConversion", &feature_1_1::samplerYcbcrConversion},
                                                                                                 {"shaderDrawParameters", &feature_1_1::shaderDrawParameters},
                                                                                             });
    utility::log(" vulkan 1.1 features ({})", count_1_1);
    print_wrapped(enabled_1_1, 100, "   ");

    std::string enabled_1_2;
    size_t const count_1_2 = append_enabled_features(enabled_1_2, capabilities.features_1_2, {
                                                                                                 {"samplerMirrorClampToEdge", &feature_1_2::samplerMirrorClampToEdge},
                                                                                                 {"drawIndirectCount", &feature_1_2::drawIndirectCount},
                                                                                                 {"storageBuffer8BitAccess", &feature_1_2::storageBuffer8BitAccess},
                                                                                                 {"uniformAndStorageBuffer8BitAccess", &feature_1_2::uniformAndStorageBuffer8BitAccess},
                                                                                                 {"storagePushConstant8", &feature_1_2::storagePushConstant8},
                                                                                                 {"shaderBufferInt64Atomics", &feature_1_2::shaderBufferInt64Atomics},
                                                                                                 {"shaderSharedInt64Atomics", &feature_1_2::shaderSharedInt64Atomics},
                                                                                                 {"shaderFloat16", &feature_1_2::shaderFloat16},
                                                                                                 {"shaderInt8", &feature_1_2::shaderInt8},
                                                                                                 {"descriptorIndexing", &feature_1_2::descriptorIndexing},
                                                                                                 {"shaderInputAttachmentArrayDynamicIndexing", &feature_1_2::shaderInputAttachmentArrayDynamicIndexing},
                                                                                                 {"shaderUniformTexelBufferArrayDynamicIndexing", &feature_1_2::shaderUniformTexelBufferArrayDynamicIndexing},
                                                                                                 {"shaderStorageTexelBufferArrayDynamicIndexing", &feature_1_2::shaderStorageTexelBufferArrayDynamicIndexing},
                                                                                                 {"shaderUniformBufferArrayNonUniformIndexing", &feature_1_2::shaderUniformBufferArrayNonUniformIndexing},
                                                                                                 {"shaderSampledImageArrayNonUniformIndexing", &feature_1_2::shaderSampledImageArrayNonUniformIndexing},
                                                                                                 {"shaderStorageBufferArrayNonUniformIndexing", &feature_1_2::shaderStorageBufferArrayNonUniformIndexing},
                                                                                                 {"shaderStorageImageArrayNonUniformIndexing", &feature_1_2::shaderStorageImageArrayNonUniformIndexing},
                                                                                                 {"shaderInputAttachmentArrayNonUniformIndexing", &feature_1_2::shaderInputAttachmentArrayNonUniformIndexing},
                                                                                                 {"shaderUniformTexelBufferArrayNonUniformIndexing", &feature_1_2::shaderUniformTexelBufferArrayNonUniformIndexing},
                                                                                                 {"shaderStorageTexelBufferArrayNonUniformIndexing", &feature_1_2::shaderStorageTexelBufferArrayNonUniformIndexing},
                                                                                                 {"descriptorBindingUniformBufferUpdateAfterBind", &feature_1_2::descriptorBindingUniformBufferUpdateAfterBind},
                                                                                                 {"descriptorBindingSampledImageUpdateAfterBind", &feature_1_2::descriptorBindingSampledImageUpdateAfterBind},
                                                                                                 {"descriptorBindingStorageImageUpdateAfterBind", &feature_1_2::descriptorBindingStorageImageUpdateAfterBind},
                                                                                                 {"descriptorBindingStorageBufferUpdateAfterBind", &feature_1_2::descriptorBindingStorageBufferUpdateAfterBind},
                                                                                                 {"descriptorBindingUniformTexelBufferUpdateAfterBind", &feature_1_2::descriptorBindingUniformTexelBufferUpdateAfterBind},
                                                                                                 {"descriptorBindingStorageTexelBufferUpdateAfterBind", &feature_1_2::descriptorBindingStorageTexelBufferUpdateAfterBind},
                                                                                                 {"descriptorBindingUpdateUnusedWhilePending", &feature_1_2::descriptorBindingUpdateUnusedWhilePending},
                                                                                                 {"descriptorBindingPartiallyBound", &feature_1_2::descriptorBindingPartiallyBound},
                                                                                                 {"descriptorBindingVariableDescriptorCount", &feature_1_2::descriptorBindingVariableDescriptorCount},
                                                                                                 {"runtimeDescriptorArray", &feature_1_2::runtimeDescriptorArray},
                                                                                                 {"samplerFilterMinmax", &feature_1_2::samplerFilterMinmax},
                                                                                                 {"scalarBlockLayout", &feature_1_2::scalarBlockLayout},
                                                                                                 {"imagelessFramebuffer", &feature_1_2::imagelessFramebuffer},
                                                                                                 {"uniformBufferStandardLayout", &feature_1_2::uniformBufferStandardLayout},
                                                                                                 {"shaderSubgroupExtendedTypes", &feature_1_2::shaderSubgroupExtendedTypes},
                                                                                                 {"separateDepthStencilLayouts", &feature_1_2::separateDepthStencilLayouts},
                                                                                                 {"hostQueryReset", &feature_1_2::hostQueryReset},
                                                                                                 {"timelineSemaphore", &feature_1_2::timelineSemaphore},
                                                                                                 {"bufferDeviceAddress", &feature_1_2::bufferDeviceAddress},
                                                                                                 {"bufferDeviceAddressCaptureReplay", &feature_1_2::bufferDeviceAddressCaptureReplay},
                                                                                                 {"bufferDeviceAddressMultiDevice", &feature_1_2::bufferDeviceAddressMultiDevice},
                                                                                                 {"vulkanMemoryModel", &feature_1_2::vulkanMemoryModel},
                                                                                                 {"vulkanMemoryModelDeviceScope", &feature_1_2::vulkanMemoryModelDeviceScope},
                                                                                                 {"vulkanMemoryModelAvailabilityVisibilityChains", &feature_1_2::vulkanMemoryModelAvailabilityVisibilityChains},
                                                                                                 {"shaderOutputViewportIndex", &feature_1_2::shaderOutputViewportIndex},
                                                                                                 {"shaderOutputLayer", &feature_1_2::shaderOutputLayer},
                                                                                                 {"subgroupBroadcastDynamicId", &feature_1_2::subgroupBroadcastDynamicId},
                                                                                             });
    utility::log(" vulkan 1.2 features ({})", count_1_2);
    print_wrapped(enabled_1_2, 100, "   ");

    std::string enabled_1_3;
    size_t const count_1_3 = append_enabled_features(enabled_1_3, capabilities.features_1_3, {
                                                                                                 {"robustImageAccess", &feature_1_3::robustImageAccess},
                                                                                                 {"inlineUniformBlock", &feature_1_3::inlineUniformBlock},
                                                                                                 {"descriptorBindingInlineUniformBlockUpdateAfterBind", &feature_1_3::descriptorBindingInlineUniformBlockUpdateAfterBind},
                                                                                                 {"pipelineCreationCacheControl", &feature_1_3::pipelineCreationCacheControl},
                                                                                                 {"privateData", &feature_1_3::privateData},
                                                                                                 {"shaderDemoteToHelperInvocation", &feature_1_3::shaderDemoteToHelperInvocation},
                                                                                                 {"shaderTerminateInvocation", &feature_1_3::shaderTerminateInvocation},
                                                                                                 {"subgroupSizeControl", &feature_1_3::subgroupSizeControl},
                                                                                                 {"computeFullSubgroups", &feature_1_3::computeFullSubgroups},
                                                                                                 {"synchronization2", &feature_1_3::synchronization2},
                                                                                                 {"textureCompressionASTC_HDR", &feature_1_3::textureCompressionASTC_HDR},
                                                                                                 {"shaderZeroInitializeWorkgroupMemory", &feature_1_3::shaderZeroInitializeWorkgroupMemory},
                                                                                                 {"dynamicRendering", &feature_1_3::dynamicRendering},
                                                                                                 {"shaderIntegerDotProduct", &feature_1_3::shaderIntegerDotProduct},
                                                                                                 {"maintenance4", &feature_1_3::maintenance4},
                                                                                             });
    utility::log(" vulkan 1.3 features ({})", count_1_3);
    print_wrapped(enabled_1_3, 100, "   ");

    utility::log("{}", box_line);
}

logical_device create_logical_device(
    VkPhysicalDevice const physical_device, // NOLINT(*-misplaced-const)
    device_creation_info const& create_info) noexcept {
    if (!create_info.queue_families.is_complete()) {
        utility::error("Queue families not complete");
        utility::panic("Queue families not complete");
    }

    if (!create_info.queue_families.compute_family || !create_info.queue_families.graphics_family || !create_info.queue_families.present_family) {
        utility::panic("queue family is empty");
    }

    // Use a set to collect unique queue family indices
    std::set<uint32_t> unique_queue_families = {
        create_info.queue_families.graphics_family.value(),
        create_info.queue_families.present_family.value(),
    };

    // Optional: add more queue families
    if (create_info.queue_families.compute_family) {
        unique_queue_families.insert(create_info.queue_families.compute_family.value());
    }
    if (create_info.queue_families.transfer_family) {
        unique_queue_families.insert(create_info.queue_families.transfer_family.value());
    }

    // Create queue infos
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    constexpr float queue_priority = 1.0f;

    for (uint32_t const& queue_family : unique_queue_families) {
        VkDeviceQueueCreateInfo queue_info = {};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;
        queue_create_infos.push_back(queue_info);
    }

    VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR fifo_latest_ready_features = {};
    fifo_latest_ready_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_KHR;
    VkPhysicalDeviceFeatures2 physical_device_features = {};
    physical_device_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    physical_device_features.pNext = &fifo_latest_ready_features;
    vkGetPhysicalDeviceFeatures2(physical_device, &physical_device_features);

    // Create the device
    VkDeviceCreateInfo device_create_info = {};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    device_create_info.pQueueCreateInfos = queue_create_infos.data();
    // Features go through the pNext chain (headed by VkPhysicalDeviceFeatures2).
    // pEnabledFeatures must be NULL: with VkPhysicalDeviceFeatures2 / VkPhysicalDeviceVulkan11Features
    // in the chain, setting it violates VUID-VkDeviceCreateInfo-pNext-04748 / -02829.
    device_create_info.pEnabledFeatures = nullptr;

    // Extensions - must handle empty vectors correctly
    device_create_info.enabledExtensionCount = static_cast<uint32_t>(create_info.extensions.size());
    device_create_info.ppEnabledExtensionNames =
        create_info.extensions.empty() ? nullptr : create_info.extensions.data();

    // Validation layers - modern Vulkan usually doesn't enable them at device level
    device_create_info.enabledLayerCount = static_cast<uint32_t>(create_info.validation_layers.size());
    device_create_info.ppEnabledLayerNames =
        create_info.validation_layers.empty() ? nullptr : create_info.validation_layers.data();

    // Assemble the pNext chain: VkPhysicalDeviceFeatures2 (head, carries device_features) -> [FifoLatestReady] -> create_info.pNext (Vulkan11Features etc.)
    physical_device_features.features = create_info.device_features;
    if (fifo_latest_ready_features.presentModeFifoLatestReady == VK_TRUE) {
        fifo_latest_ready_features.pNext = const_cast<void*>(create_info.pNext);
        physical_device_features.pNext = &fifo_latest_ready_features;
    } else {
        physical_device_features.pNext = const_cast<void*>(create_info.pNext);
    }
    device_create_info.pNext = &physical_device_features;

    VkDevice device = {};
    VkResult const result = vkCreateDevice(physical_device, &device_create_info, nullptr, &device);
    if (result != VK_SUCCESS) {
        utility::panic(std::source_location::current(), "Failed to create logical device: {}", std::to_string(result));
    }

    // Get queues
    logical_device logical_device;
    logical_device.device = device;
    logical_device.graphics_family_index = create_info.queue_families.graphics_family.value();
    logical_device.present_family_index = create_info.queue_families.present_family.value();

    vkGetDeviceQueue(device, logical_device.graphics_family_index, 0,
                     &logical_device.graphics_queue);
    vkGetDeviceQueue(device, logical_device.present_family_index, 0,
                     &logical_device.present_queue);

    return logical_device;
}

queue_family_indices find_queue_families(VkPhysicalDevice const& device, VkSurfaceKHR const& surface) noexcept { // NOLINT(*-function-cognitive-complexity)
    queue_family_indices indices;

    // Get queue family properties
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

    // Find a suitable queue family
    for (uint32_t i = 0; i < queue_family_count; ++i) {
        auto const& queue_family = queue_families[i];

        // Check graphics support
        if ((queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) && !indices.graphics_family.has_value()) {
            indices.graphics_family = i;
        }

        // Check compute support (non-graphics queue)
        if ((queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !indices.compute_family.has_value() &&
            !(queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            indices.compute_family = i;
        }

        // Check transfer support (non-graphics/compute queue)
        if ((queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !indices.transfer_family.has_value() &&
            !(queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            !(queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            indices.transfer_family = i;
        }

        // Check present support (requires a surface)
        if (surface != VK_NULL_HANDLE) {
            VkBool32 present_support = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
            if (present_support && !indices.present_family.has_value()) {
                indices.present_family = i;
            }
        }

        // Early-exit if only graphics support is required
        if (surface == VK_NULL_HANDLE && indices.graphics_family.has_value()) {
            break;
        }

        // Early-exit once all required queues are found
        if (surface == VK_NULL_HANDLE) {
            if (indices.graphics_family.has_value()) {
                break;
            }
        } else if (indices.is_complete()) {
            break;
        }
    }

    // Fallback: if no dedicated compute/transfer queue was found, use the graphics queue
    if (!indices.compute_family.has_value() && indices.graphics_family.has_value()) {
        indices.compute_family = indices.graphics_family;
    }

    if (!indices.transfer_family.has_value()) {
        // Prefer the graphics queue; if none exists, use the first available queue
        if (indices.graphics_family.has_value()) {
            indices.transfer_family = indices.graphics_family;
        } else {
            if (queue_family_count > 0) {
                indices.transfer_family = {0};
            } else {
                indices.transfer_family = std::nullopt;
            }
        }
    }

    return indices;
}

VkPhysicalDevice pick_suitable_device(VkInstance const& instance, VkSurfaceKHR surface) noexcept {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0) {
        utility::panic("Failed to find GPUs with Vulkan support");
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    for (auto const& device : devices) {
        VkPhysicalDeviceProperties device_properties;
        VkPhysicalDeviceFeatures device_features;
        vkGetPhysicalDeviceProperties(device, &device_properties);
        vkGetPhysicalDeviceFeatures(device, &device_features);

        if (queue_family_indices indices = find_queue_families(device, surface); !indices.is_complete()) {
            continue;
        }

        // Check extension support
        std::vector<char const*> const required_extensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };
        if (!check_device_extension_support(device, required_extensions)) {
            continue;
        }

        return device; // suitable device found
    }

    utility::panic("Failed to find a suitable GPU!");
}

swap_chain_support_details query_swap_chain_support(VkPhysicalDevice const& device, VkSurfaceKHR const& surface) noexcept {
    swap_chain_support_details details = {};

    // 1. Query surface capabilities
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    // 2. Query surface formats
    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
    if (format_count != 0) {
        details.formats.resize(format_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
    }

    // 3. Query present modes
    uint32_t present_mode_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, nullptr);
    if (present_mode_count != 0) {
        details.present_modes.resize(present_mode_count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, details.present_modes.data());
    }

    utility::log("present modes count: {}", details.present_modes.size());

    return details;
}

VkPresentModeKHR choose_swap_present_mode(std::vector<VkPresentModeKHR> const& available_present_modes) noexcept {
    // Prefer MAILBOX (low latency), else fall back to FIFO (mandated by the Vulkan spec)
    if (std::ranges::find(available_present_modes, VK_PRESENT_MODE_MAILBOX_KHR) != available_present_modes.end()) {
        return VK_PRESENT_MODE_MAILBOX_KHR;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkSurfaceFormatKHR choose_swap_surface_format(std::vector<VkSurfaceFormatKHR> const& available_formats) noexcept {
    for (auto const& available_format : available_formats) {
        if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return available_format;
        }
    }
    return available_formats[0];
}

VkExtent2D choose_swap_extent(VkSurfaceCapabilitiesKHR const& capabilities, GLFWwindow* window) noexcept {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }
    int width;
    int height;
    glfwGetFramebufferSize(window, &width, &height);

    // Ensure width and height are non-zero
    width = std::max(width, 1);
    height = std::max(height, 1);

    VkExtent2D actual_extent = {
        .width = static_cast<uint32_t>(width),
        .height = static_cast<uint32_t>(height),
    };

    actual_extent.width = std::clamp(actual_extent.width,
                                     capabilities.minImageExtent.width,
                                     capabilities.maxImageExtent.width);
    actual_extent.height = std::clamp(actual_extent.height,
                                      capabilities.minImageExtent.height,
                                      capabilities.maxImageExtent.height);

    return actual_extent;
}

uint32_t find_memory_type(uint32_t const& type_filter, VkMemoryPropertyFlags const& properties, VkPhysicalDevice const& physical_device) noexcept {
    // Get the physical device's memory properties
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    // Iterate over all memory types
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        // Check whether the memory type satisfies the filter
        // type_filter is a bitmask; each bit corresponds to a memory type
        if ((type_filter & (1 << i)) &&
            // Check whether the memory properties meet the requirements
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i; // return the found memory type index
        }
    }
    utility::panic("failed to find suitable memory type");
}

VkFormat find_depth_format(VkPhysicalDevice const& physical_device) noexcept {
    // Try to find a supported depth format, in order of preference
    std::vector<VkFormat> const candidates = {
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D16_UNORM_S8_UINT,
        VK_FORMAT_D16_UNORM,
    };

    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

        // Check whether the format supports depth attachments
        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            return format;
        }
    }

    utility::panic("failed to find supported depth format!");
}

VkSampleCountFlagBits get_max_usable_sample_count(VkPhysicalDevice const& physical_device) noexcept {
    VkPhysicalDeviceProperties physical_device_properties;
    vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);

    VkSampleCountFlags const counts = physical_device_properties.limits.framebufferColorSampleCounts &
                                      physical_device_properties.limits.framebufferDepthSampleCounts;

    if (counts & VK_SAMPLE_COUNT_64_BIT) {
        return VK_SAMPLE_COUNT_64_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_32_BIT) {
        return VK_SAMPLE_COUNT_32_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_16_BIT) {
        return VK_SAMPLE_COUNT_16_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_8_BIT) {
        return VK_SAMPLE_COUNT_8_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_4_BIT) {
        return VK_SAMPLE_COUNT_4_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_2_BIT) {
        return VK_SAMPLE_COUNT_2_BIT;
    }

    return VK_SAMPLE_COUNT_1_BIT;
}

VkImageView create_image_view(VkImage const& image, VkFormat const& format, VkImageAspectFlags const& aspect_flags, VkDevice const& device) noexcept {
    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;

    // Component mapping (keep defaults)
    view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    // Subresource range (describes which part of the image is accessible)
    view_info.subresourceRange.aspectMask = aspect_flags;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    VkImageView image_view;
    if (vkCreateImageView(device, &view_info, nullptr, &image_view) != VK_SUCCESS) {
        utility::panic("failed to create image view!");
    }

    return image_view;
}