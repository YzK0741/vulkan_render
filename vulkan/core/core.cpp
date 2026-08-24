module;

#include <boost/stacktrace/stacktrace.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

module vulkan.core;
import vulkan.core.pipeline;
import vulkan.core.init_utils;

// core
namespace vulkan {
    core::core() {
        init_window(1080, 960);
        init_instance();
        init_surface();
        init_device_and_queue();
        init_swap_chain();
        init_image_views();
        create_depth_resources();
        color_format = swap_chain_image_format;
        create_color_resources();
        init_renderpass();
        create_frame_buffers();
        create_command_pool();
        create_descriptor_pool();
        create_sync_objects();

        vma.init(this->instance, this->device, this->physical_device, this->graphics_queue, this->graphics_family_index);
        this->register_cleanup([this] {
            vma.destroy();
        });
    };

    core::~core() {
        vkDeviceWaitIdle(this->device);
        this->do_cleanup();
    }

    void core::init_window(const int width, const int height, const std::string_view window_name) noexcept {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(
            width,
            height,
            window_name.data() ? window_name.data() : "vulkan",
            nullptr,
            nullptr);

        register_cleanup([this] {
            if (window) {
                glfwDestroyWindow(window);
            }
        });
    }

    void core::init_instance() noexcept {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "Hello Triangle";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "No Engine";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;

        // get and set GLFW required extensions
        uint32_t glfw_extension_count = 0;
        const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

        std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

#ifdef _DEBUG
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        std::println("add debug extension: {}", VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

        // print all extensions
        std::println("required instance extension ({}):", extensions.size());
        for (const auto& ext : extensions) {
            std::println("  - {}", ext);
        }

        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();

        // enable validation_layers
        std::vector<const char*> validation_layers;

#ifdef _DEBUG
        validation_layers = {
            "VK_LAYER_KHRONOS_validation"};

        // 检查验证层支持
        if (check_validation_layer_support(validation_layers)) {
            create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
            create_info.ppEnabledLayerNames = validation_layers.data();
            std::println("验证层已启用 ( {} )", validation_layers.size());
            for (const auto& layer : validation_layers) {
                std::println("  - {}", layer);
            }
        } else {
            std::println(stderr, "warning：VK_LAYER_KHRONOS_validation disabled");
            create_info.enabledLayerCount = 0;
        }
#else
        std::println("in release, validation layers disabled");
        create_info.enabledLayerCount = 0;
#endif

        if (VkResult result = vkCreateInstance(&create_info, nullptr, &this->instance); result != VK_SUCCESS) {
            // 提供更详细的错误信息
            std::string error_msg = "can not create vulkan instance，error code is: ";
            switch (result) {
            case VK_ERROR_OUT_OF_HOST_MEMORY:
                error_msg += "VK_ERROR_OUT_OF_HOST_MEMORY";
                break;
            case VK_ERROR_OUT_OF_DEVICE_MEMORY:
                error_msg += "VK_ERROR_OUT_OF_DEVICE_MEMORY";
                break;
            case VK_ERROR_INITIALIZATION_FAILED:
                error_msg += "VK_ERROR_INITIALIZATION_FAILED";
                break;
            case VK_ERROR_LAYER_NOT_PRESENT:
                error_msg += "VK_ERROR_LAYER_NOT_PRESENT";
                break;
            case VK_ERROR_EXTENSION_NOT_PRESENT:
                error_msg += "VK_ERROR_EXTENSION_NOT_PRESENT";
                break;
            case VK_ERROR_INCOMPATIBLE_DRIVER:
                error_msg += "VK_ERROR_INCOMPATIBLE_DRIVER";
                break;
            default:
                error_msg += std::to_string(static_cast<int>(result));
                break;
            }
            utility::panic(error_msg);
        }
        std::println("instance init succeeded");
        std::println("instance handler is 0x{:x}", reinterpret_cast<uint64_t>(this->instance));

#ifdef _DEBUG
        // get function pointer
        auto vkCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
            instance, "vkCreateDebugUtilsMessengerEXT"));
        auto vkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
            instance, "vkDestroyDebugUtilsMessengerEXT"));

        // 检查函数指针是否获取成功
        if (!vkCreateDebugUtilsMessengerEXT || !vkDestroyDebugUtilsMessengerEXT) {
            utility::panic("Failed to get debug utils function pointers");
        }
        VkDebugUtilsMessengerCreateInfoEXT debug_info{};
        debug_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debug_info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug_info.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug_info.pfnUserCallback = debug_callback; // 需要实现这个回调函数
        debug_info.pUserData = nullptr;

        if (vkCreateDebugUtilsMessengerEXT(instance, &debug_info, nullptr, &debug_messenger) != VK_SUCCESS) {
            std::println(stderr, "create debug messanger failed");
        } else {
            std::println(stderr, "create debug messanger succeeded");
        }
#endif
        this->register_cleanup([this] {
            vkDestroyInstance(this->instance, nullptr);
        });

#ifdef _DEBUG
        this->register_cleanup([vkDestroyDebugUtilsMessengerEXT, this] {
            vkDestroyDebugUtilsMessengerEXT(this->instance, this->debug_messenger, nullptr);
        });
#endif
    }

    void core::init_surface() noexcept {
        if (glfwCreateWindowSurface(this->instance, this->window, nullptr, &this->surface) != VK_SUCCESS) {
            utility::panic("can not init surface");
        }

        this->register_cleanup([this] {
            vkDestroySurfaceKHR(this->instance, this->surface, nullptr);
        });
    }

    void core::init_device_and_queue() noexcept {
        this->physical_device = pick_suitable_device(this->instance, this->surface);

        device_creation_info creation_info;

        creation_info.queue_families = find_queue_families(this->physical_device, this->surface);

        creation_info.extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        if (!check_device_extension_support(physical_device, creation_info.extensions)) {
            utility::panic("Required device extensions not supported");
        }

        VkPhysicalDeviceVulkan11Features vulkan11_features{};
        vulkan11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        vulkan11_features.storageBuffer16BitAccess = VK_TRUE;
        creation_info.pNext = &vulkan11_features;

        const auto [device, graphics_family_index, present_family_index, graphics_queue, present_queue] = create_logical_device(physical_device, creation_info); // NOLINT(*-misplaced-const)

        this->device = device;
        this->graphics_queue = graphics_queue;
        this->present_queue = present_queue;
        this->graphics_family_index = graphics_family_index;
        this->present_family_index = present_family_index;

        std::println("device and queue init succeeded");

        register_cleanup([this] {
            if (this->device != VK_NULL_HANDLE) {
                vkDestroyDevice(this->device, nullptr);
            }
        });
    }

    void core::init_swap_chain() noexcept {
        const auto [capabilities, formats, present_modes] = query_swap_chain_support(this->physical_device, this->surface);

        // 添加检查：
        if (formats.empty() || present_modes.empty()) {
            utility::panic("Swap chain not adequately supported");
        }

        const auto [format, colorSpace] = choose_swap_surface_format(formats);
        const VkPresentModeKHR present_mode = choose_swap_present_mode(present_modes);
        const VkExtent2D extent = choose_swap_extent(capabilities, this->window);

        uint32_t image_count = capabilities.minImageCount + 1;

        if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
            image_count = capabilities.maxImageCount;
        }

        image_count = std::max(image_count, capabilities.minImageCount);

        VkSwapchainCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = surface;

        create_info.minImageCount = image_count;
        create_info.imageFormat = format;
        create_info.imageColorSpace = colorSpace;
        create_info.imageExtent = extent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        const queue_family_indices indices = find_queue_families(this->physical_device, this->surface);
        const uint32_t queueFamilyIndices[] = {indices.graphics_family.value(), indices.present_family.value()};

        if (indices.graphics_family != indices.present_family) {
            create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = 2;
            create_info.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            create_info.queueFamilyIndexCount = 0;     // Optional
            create_info.pQueueFamilyIndices = nullptr; // Optional
        }

        create_info.preTransform = capabilities.currentTransform;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        create_info.presentMode = present_mode;
        create_info.clipped = VK_TRUE;
        create_info.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &create_info, nullptr, &this->swap_chain) != VK_SUCCESS) {
            utility::panic("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, this->swap_chain, &image_count, nullptr);
        this->swap_chain_images.resize(image_count);
        vkGetSwapchainImagesKHR(device, this->swap_chain, &image_count, this->swap_chain_images.data());

        this->swap_chain_image_format = format;
        this->swap_chain_extent = extent;

        register_cleanup([this] {
            if (swap_chain != VK_NULL_HANDLE) {
                vkDestroySwapchainKHR(device, swap_chain, nullptr);
                swap_chain = VK_NULL_HANDLE;
            }
        });
    }

    void core::init_image_views() noexcept {

        this->swap_chain_image_views.resize(this->swap_chain_images.size());
        for (size_t i = 0; i < this->swap_chain_images.size(); i++) {
            VkImageViewCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = this->swap_chain_images[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swap_chain_image_format;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &createInfo, nullptr, &this->swap_chain_image_views[i]) != VK_SUCCESS) {
                utility::panic("failed to create image views!");
            }
        }
        register_cleanup([this] {
            for (const auto& image_view : this->swap_chain_image_views) {
                vkDestroyImageView(device, image_view, nullptr);
            }
            this->swap_chain_image_views.clear();
        });
    }

    void core::create_depth_image(VkImage& image, VkDeviceMemory& imageMemory, VkImageView& image_view) const noexcept {
        // 使用类内的交换链尺寸
        const auto& [width, height] = this->swap_chain_extent;

        // 1. 创建图像
        VkImageCreateInfo image_info{};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.extent = {width, height, 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.format = this->depth_format;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        image_info.samples = this->msaa_samples;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS) {
            vkDestroyImage(device, image, nullptr);
            utility::panic("failed to create depth image!");
        }

        // 2. 分配内存
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = find_memory_type(memRequirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, this->physical_device);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            utility::panic("failed to allocate depth image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);

        // 3. 创建图像视图
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = this->depth_format;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &view_info, nullptr, &image_view) != VK_SUCCESS) {
            utility::panic("failed to create depth image view!");
        }
    }

    void core::create_depth_resources() noexcept {

        depth_format = find_depth_format(this->physical_device);
        msaa_samples = get_max_usable_sample_count(this->physical_device);

        // 先测试深度格式是否有效
        if (depth_format == VK_FORMAT_UNDEFINED) {
            utility::panic("can't find supported depth format");
        }

        depth_images.resize(swap_chain_image_views.size());
        depth_image_views.resize(swap_chain_image_views.size());
        depth_image_memories.resize(swap_chain_image_views.size()); // 确保分配内存

        for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
            // 直接调用 create_depth_image，但确保参数正确
            create_depth_image(depth_images[i], depth_image_memories[i], depth_image_views[i]);
        }

        register_cleanup([this] {
            for (const auto& view : depth_image_views) {
                vkDestroyImageView(device, view, nullptr);
            }
            for (const auto& image : depth_images) {
                vkDestroyImage(device, image, nullptr);
            }
            for (const auto& memory : depth_image_memories) {
                vkFreeMemory(device, memory, nullptr);
            }
            depth_image_views.clear();
            depth_images.clear();
            depth_image_memories.clear();
        });
    }

    void core::create_color_resources() {

        color_images.resize(swap_chain_image_views.size());
        color_image_memories.resize(swap_chain_image_views.size());
        color_image_views.resize(swap_chain_image_views.size());

        for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
            create_msaa_image(
                swap_chain_extent.width,
                swap_chain_extent.height,
                color_format,
                msaa_samples,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                color_images[i],
                color_image_memories[i]);

            color_image_views[i] = create_image_view(
                color_images[i],
                color_format,
                VK_IMAGE_ASPECT_COLOR_BIT,
                device);
        }

        register_cleanup([this] {
            for (const auto& view : color_image_views) {
                vkDestroyImageView(device, view, nullptr);
            }
            for (const auto& memory : color_image_memories) {
                vkFreeMemory(device, memory, nullptr);
            }
            for (const auto& image : color_images) {
                vkDestroyImage(device, image, nullptr);
            }
            color_image_views.clear();
            color_image_memories.clear();
            color_images.clear();
        });
    }

    void core::init_renderpass() noexcept {
        // 1. 颜色附件（现在是MSAA的）
        VkAttachmentDescription color_attachment = {};
        color_attachment.format = swap_chain_image_format;
        color_attachment.samples = msaa_samples; // 使用MSAA采样
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        // 关键：如果有MSAA，最终布局需要解析到呈现图像
        if (msaa_samples > VK_SAMPLE_COUNT_1_BIT) {
            color_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        } else {
            color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        }

        // 2. 深度附件（也需要MSAA）
        VkAttachmentDescription depth_attachment = {};
        depth_attachment.format = depth_format;
        depth_attachment.samples = msaa_samples; // 深度也要MSAA
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        // 3. 颜色解析附件（只在有MSAA时需要）
        VkAttachmentDescription color_resolve_attachment = {};
        if (msaa_samples > VK_SAMPLE_COUNT_1_BIT) {
            color_resolve_attachment.format = swap_chain_image_format;
            color_resolve_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
            color_resolve_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            color_resolve_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color_resolve_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            color_resolve_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            color_resolve_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            color_resolve_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        }

        VkAttachmentReference color_attachment_ref{};
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_attachment_ref{};
        depth_attachment_ref.attachment = 1;
        depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference color_resolve_ref{};
        if (msaa_samples > VK_SAMPLE_COUNT_1_BIT) {
            color_resolve_ref.attachment = 2;
            color_resolve_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment_ref;
        subpass.pDepthStencilAttachment = &depth_attachment_ref;

        // 关键：设置解析附件
        if (msaa_samples > VK_SAMPLE_COUNT_1_BIT) {
            subpass.pResolveAttachments = &color_resolve_ref;
        } else {
            subpass.pResolveAttachments = nullptr;
        }

        // 构建附件数组
        std::vector<VkAttachmentDescription> attachments;
        attachments.push_back(color_attachment);
        attachments.push_back(depth_attachment);

        if (msaa_samples > VK_SAMPLE_COUNT_1_BIT) {
            attachments.push_back(color_resolve_attachment);
        }

        VkRenderPassCreateInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
        render_pass_info.pAttachments = attachments.data();
        render_pass_info.subpassCount = 1;
        render_pass_info.pSubpasses = &subpass;

        // 添加子流程依赖
        std::array<VkSubpassDependency, 2> dependencies = {};
        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        render_pass_info.dependencyCount = static_cast<uint32_t>(dependencies.size());
        render_pass_info.pDependencies = dependencies.data();

        if (vkCreateRenderPass(device, &render_pass_info, nullptr, &renderpass) != VK_SUCCESS) {
            utility::panic("can't create renderpass");
        }
        register_cleanup(
            [this] {
                if (renderpass != VK_NULL_HANDLE) {
                    vkDestroyRenderPass(device, renderpass, nullptr);
                    renderpass = VK_NULL_HANDLE;
                }
            });
    }

    void core::create_frame_buffers() noexcept {

        swap_chain_framebuffers.resize(swap_chain_image_views.size());

        for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
            std::vector<VkImageView> attachments;

            // 如果有MSAA，第一个是MSAA颜色附件
            if (msaa_samples > VK_SAMPLE_COUNT_1_BIT) {
                attachments.push_back(color_image_views[i]); // MSAA颜色附件
            } else {
                attachments.push_back(swap_chain_image_views[i]); // 普通颜色附件
            }

            attachments.push_back(depth_image_views[i]); // 深度附件

            // 如果有MSAA，添加解析附件
            if (msaa_samples > VK_SAMPLE_COUNT_1_BIT) {
                attachments.push_back(swap_chain_image_views[i]); // 解析到交换链图像
            }

            VkFramebufferCreateInfo framebuffer_create_info{};
            framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer_create_info.renderPass = renderpass;
            framebuffer_create_info.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebuffer_create_info.pAttachments = attachments.data();
            framebuffer_create_info.width = swap_chain_extent.width;
            framebuffer_create_info.height = swap_chain_extent.height;
            framebuffer_create_info.layers = 1;

            if (vkCreateFramebuffer(this->device, &framebuffer_create_info, nullptr, &this->swap_chain_framebuffers[i]) != VK_SUCCESS) {
                utility::panic("can't create frame buffer");
            }
        }

        register_cleanup([this] {
            for (const auto& frame_buffer : swap_chain_framebuffers) {
                vkDestroyFramebuffer(device, frame_buffer, nullptr);
            }
            swap_chain_framebuffers.clear();
        });
    }

    void core::create_command_pool() noexcept {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = graphics_family_index;

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &command_pool) != VK_SUCCESS) {
            utility::panic("failed to create command pool");
        }

        register_cleanup([this] {
            if (command_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, command_pool, nullptr);
            }
        });
    }

    void core::create_msaa_image(
        const uint32_t& width,
        const uint32_t& height,
        const VkFormat& format,
        const VkSampleCountFlagBits& num_samples,
        const VkImageTiling& tiling,
        const VkImageUsageFlags& usage,
        const VkMemoryPropertyFlags& properties,
        VkImage& image,
        VkDeviceMemory& imageMemory) const noexcept {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = num_samples; // 关键：设置采样数
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            utility::panic("can't create msaa image");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = find_memory_type(
            memRequirements.memoryTypeBits,
            properties,
            physical_device);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            utility::panic("can't allocate msaa image memory");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    void core::create_descriptor_pool() noexcept {
        std::vector<VkDescriptorPoolSize> pool_sizes;
        constexpr int max_size = 1024;
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, max_size});

        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_size});

        VkDescriptorPoolCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        info.maxSets = max_size;
        info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
        info.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(this->device, &info, nullptr, &this->descriptor_pool)) {
            utility::panic("failed in creating descriptor pool");
        }
        register_cleanup([this] {
            if (descriptor_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(this->device, this->descriptor_pool, nullptr);
            }
        });
    }

    void core::create_sync_objects() {
        image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
        images_in_flight.resize(swap_chain_images.size(), VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT; // 初始为已触发状态

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphore_info, nullptr, &image_available_semaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphore_info, nullptr, &render_finished_semaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fence_info, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
                utility::panic("failed to create synchronization objects for a frame!");
            }
        }

        register_cleanup([this] {
            for (const auto& semaphore : image_available_semaphores) {
                vkDestroySemaphore(device, semaphore, nullptr);
            }

            for (const auto& semaphore : render_finished_semaphores) {
                vkDestroySemaphore(device, semaphore, nullptr);
            }

            for (const auto& fence : in_flight_fences) {
                vkDestroyFence(device, fence, nullptr);
            }
        });
    }

    vk_command_buffer core::make_command_buffer() {
        return ::vulkan::make_command_buffer(this->device, this->command_pool);
    }

    vk_descriptor_set core::make_descriptor_set(const VkDescriptorSetLayout layout) { // NOLINT(*-misplaced-const)
        return ::vulkan::make_descriptor_set(this->device, this->descriptor_pool, layout);
    }

    std::optional<vk_shader_module> core::make_shader_module(const std::span<unsigned char> shader) noexcept {
        return ::vulkan::make_shader_module(shader, this->device);
    }

    VkResult core::get_image_index(uint32_t& image_index) const {
        return vkAcquireNextImageKHR(
            this->device,
            this->swap_chain,
            UINT64_MAX,
            this->image_available_semaphores[this->current_frame],
            this->in_flight_fences[this->current_frame],
            &image_index);
    }

    void core::wait_usable_image(const uint32_t image_index) {
        if (images_in_flight[image_index] != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &images_in_flight[image_index], VK_TRUE, UINT64_MAX);
        }
        images_in_flight[image_index] = in_flight_fences[current_frame];
    }

    void core::reset_fence(const uint32_t frame_index) const {
        vkResetFences(device, 1, &in_flight_fences[frame_index]);
    }

    void core::to_next_frame() noexcept {
        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void core::recreate_swap_chain() {
        // 1. 等待设备空闲
        vkDeviceWaitIdle(device);

        // 2. 先销毁所有 Framebuffer
        for (const auto& frame_buffer : swap_chain_framebuffers) {
            vkDestroyFramebuffer(device, frame_buffer, nullptr);
        }
        swap_chain_framebuffers.clear();

        // 3. 销毁 RenderPass
        if (this->renderpass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(device, renderpass, nullptr);
            renderpass = VK_NULL_HANDLE;
        }

        // 4. 销毁 MSAA 颜色资源
        if (msaa_samples > VK_SAMPLE_COUNT_1_BIT) {
            for (const auto& view : color_image_views) {
                vkDestroyImageView(device, view, nullptr);
            }
            color_image_views.clear();

            for (const auto& image : color_images) {
                vkDestroyImage(device, image, nullptr);
            }
            color_images.clear();

            for (const auto& memory : color_image_memories) {
                vkFreeMemory(device, memory, nullptr);
            }
            color_image_memories.clear();
        }

        // 5. 销毁深度资源
        for (const auto& view : depth_image_views) {
            vkDestroyImageView(device, view, nullptr);
        }
        depth_image_views.clear();

        for (const auto& image : depth_images) {
            vkDestroyImage(device, image, nullptr);
        }
        depth_images.clear();

        for (const auto& memory : depth_image_memories) {
            vkFreeMemory(device, memory, nullptr);
        }
        depth_image_memories.clear();

        // 6. 销毁 SwapChain ImageViews
        for (const auto& image_view : swap_chain_image_views) {
            vkDestroyImageView(device, image_view, nullptr);
        }
        swap_chain_image_views.clear();

        // 7. 销毁 SwapChain 本身
        if (swap_chain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device, swap_chain, nullptr);
            swap_chain = VK_NULL_HANDLE;
        }

        // 8. 重新创建所有资源
        this->init_swap_chain();        // 重建 SwapChain
        this->init_image_views();       // 重建 ImageViews
        this->create_depth_resources(); // 重建深度资源

        if (msaa_samples > VK_SAMPLE_COUNT_1_BIT) {
            this->create_color_resources(); // 重建 MSAA 颜色资源
        }

        this->init_renderpass();      // 重建 RenderPass
        this->create_frame_buffers(); // 重建 Framebuffer

        // 按新图像数重建 per-image 追踪表，避免 wait_usable_image 越界
        images_in_flight.resize(swap_chain_images.size(), VK_NULL_HANDLE);
    }

    std::expected<vk_pipeline, std::string_view> core::make_pipeline(
        std::span<const unsigned char> vertex_shader_code,
        const std::span<const unsigned char> fragment_shader_code) {
        return vulkan::make_pipeline(
            this->device,
            this->renderpass,
            vertex_shader_code,
            fragment_shader_code,
            this->msaa_samples);
    }
} // namespace vulkan
