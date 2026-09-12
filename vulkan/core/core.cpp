module;

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

module vulkan.core;
import vulkan.core.pipeline;
import vulkan.core.init_utils;
import vulkan.constant_init;

// core
namespace vulkan {
    core::core()
        : core(core_create_info{}) {
    }

    core::core(core_create_info const& options)
        : create_options{options} {
        if (options.window.has_value()) {
            // caller-provided window: bind to it as-is - no glfwInit / glfwCreateWindow here and
            // no glfwDestroyWindow cleanup (ownership stays with the caller; see
            // core_create_info::window)
            window = *options.window;
        } else {
            init_window(options.window_width, options.window_height, options.window_title);
        }
        init_instance();
        init_surface();
        init_device_and_queue();
        init_swap_chain();
        init_image_views();
        create_depth_resources();
        color_format = swap_chain_image_format;
        create_render_targets(); // the scene's render targets: the post-process pass input
        create_command_pool();
        create_descriptor_pool();
        init_scene_layouts();
        create_sync_objects();
        create_timestamp_query_pool(); // GPU pass timings (a no-op on devices that cannot timestamp)

        vma.init(this->instance, this->device, this->physical_device, this->graphics_queue, this->graphics_family_index);
        this->register_cleanup([this] {
            vma.destroy();
        });
    };

    core::~core() {
        vkDeviceWaitIdle(this->device);
        this->do_cleanup();
    }

    // self-owned window path: only taken when core_create_info::window is empty (a caller-provided
    // window skips glfwInit/glfwCreateWindow entirely and registers no destroy cleanup)
    void core::init_window(int const width, int const height, std::string_view const window_name) noexcept {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // resizing recreates the swapchain via render_frame's OUT_OF_DATE handling

        window = glfwCreateWindow(
            width,
            height,
            window_name.empty() ? "vulkan" : window_name.data(),
            nullptr,
            nullptr);

        register_cleanup([this] {
            if (window) {
                glfwDestroyWindow(window);
            }
        });
    }

    void core::init_instance() noexcept {
        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "vulkan render";
        // version injected by CMake (project(VERSION) in CMakeLists.txt) - the single source
        app_info.applicationVersion = VK_MAKE_VERSION(VULKAN_RENDER_VERSION_MAJOR, VULKAN_RENDER_VERSION_MINOR, VULKAN_RENDER_VERSION_PATCH);
        app_info.pEngineName = "";
        app_info.engineVersion = VK_MAKE_VERSION(VULKAN_RENDER_VERSION_MAJOR, VULKAN_RENDER_VERSION_MINOR, VULKAN_RENDER_VERSION_PATCH);
        app_info.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;

        // get and set GLFW required extensions
        uint32_t glfw_extension_count = 0;
        char const** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

        std::vector<char const*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

        // validation layers + debug messenger: runtime switch from create_options (app_config's
        // [render] validation_layers; Debug defaults on, Release off - both overridable)
        bool const enable_validation = this->create_options.validation_layers;

        if (enable_validation) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            utility::log("add debug extension: {}", VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        // print all extensions
        utility::log("required instance extension ({}):", extensions.size());
        for (auto const& ext : extensions) {
            utility::log("  - {}", ext);
        }

        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();

        // enable validation_layers
        std::vector<char const*> validation_layers;

        if (enable_validation) {
            validation_layers = {
                "VK_LAYER_KHRONOS_validation",
            };

            // Check validation layer support
            if (check_validation_layer_support(validation_layers)) {
                create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
                create_info.ppEnabledLayerNames = validation_layers.data();
                utility::log("validation layers enabled ( {} )", validation_layers.size());
                for (auto const& layer : validation_layers) {
                    utility::log("  - {}", layer);
                }
            } else {
                utility::log("warning: VK_LAYER_KHRONOS_validation disabled");
                create_info.enabledLayerCount = 0;
            }
        } else {
            utility::log("validation layers disabled (config [render] validation_layers = false)");
            create_info.enabledLayerCount = 0;
        }

        if (VkResult result = vkCreateInstance(&create_info, nullptr, &this->instance); result != VK_SUCCESS) {
            // Provide more detailed error info
            std::string error_msg = "can not create vulkan instance, error code is: ";
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
        utility::log("instance init succeeded");
        utility::log("instance handler is 0x{:x}", reinterpret_cast<uint64_t>(this->instance));

        // register instance destruction first, then the messenger cleanup INSIDE the
        // validation branch below: cleanup runs LIFO, so the messenger (when created) is
        // destroyed before the instance it belongs to
        this->register_cleanup([this] {
            vkDestroyInstance(this->instance, nullptr);
        });

        if (enable_validation) {
            // get function pointer
            auto vkCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
                instance, "vkCreateDebugUtilsMessengerEXT"));
            auto vkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
                instance, "vkDestroyDebugUtilsMessengerEXT"));

            if ((vkCreateDebugUtilsMessengerEXT == nullptr) || (vkDestroyDebugUtilsMessengerEXT == nullptr)) {
                utility::panic("Failed to get debug utils function pointers");
            }
            VkDebugUtilsMessengerCreateInfoEXT debug_info = {};
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
            debug_info.pfnUserCallback = debug_callback;
            debug_info.pUserData = nullptr;

            if (vkCreateDebugUtilsMessengerEXT(instance, &debug_info, nullptr, &debug_messenger) != VK_SUCCESS) {
                utility::error("create debug messenger failed");
            } else {
                utility::log("create debug messenger succeeded");
            }
            this->register_cleanup([vkDestroyDebugUtilsMessengerEXT, this] {
                vkDestroyDebugUtilsMessengerEXT(this->instance, this->debug_messenger, nullptr);
            });
        }
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

        // Collect device capabilities: one query through the feature/property pNext chains (see device_capabilities)
        device_capabilities capabilities;
        capabilities.query(this->physical_device);
        print_device_capabilities(capabilities);
        // keep the plain properties around (limits such as timestampPeriod drive renderer
        // decisions like the GPU pass timings) - the capabilities query already fetched them
        this->device_properties = capabilities.properties_2.properties;

        // Dynamic rendering (Vulkan 1.3 core) is mandatory: pick_suitable_device only accepts
        // apiVersion >= 1.3 devices, and frames always record through vkCmdBeginRendering - no
        // render pass / framebuffer objects exist. Double-check the feature bit anyway (a
        // conformant 1.3 driver must expose it).
        if (capabilities.features_1_3.dynamicRendering != VK_TRUE) {
            utility::panic("dynamic rendering (Vulkan 1.3) is required but not supported by the device");
        }

        device_creation_info creation_info;

        creation_info.queue_families = find_queue_families(this->physical_device, this->surface);

        // Spelled as a string rather than the header macro so the build does not depend on how new the
        // Vulkan headers are; it is only ever pushed when the device advertises it.
        constexpr char const* fifo_latest_ready_extension = "VK_EXT_present_mode_fifo_latest_ready";

        creation_info.extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        // VK_EXT_present_mode_fifo_latest_ready is asked for only when the device has it, so the
        // extension is never enabled blindly and the present mode that needs it is only ever chosen
        // from the surface's own list (see choose_swap_present_mode).
        if (check_device_extension_support(physical_device, std::vector<char const*>{fifo_latest_ready_extension})) {
            creation_info.extensions.push_back(fifo_latest_ready_extension);
        }

        if (!check_device_extension_support(physical_device, creation_info.extensions)) {
            utility::panic("Required device extensions not supported");
        }

        // Features go through the pNext chain (device_capabilities query result, incl. all 1.1/1.2/1.3 supported features)
        creation_info.pNext = capabilities.device_pnext();

        auto const [device, graphics_family_index, present_family_index, graphics_queue, present_queue] = create_logical_device(physical_device, creation_info); // NOLINT(*-misplaced-const)

        this->device = device;
        this->graphics_queue = graphics_queue;
        this->present_queue = present_queue;
        this->graphics_family_index = graphics_family_index;
        this->present_family_index = present_family_index;

        utility::log("device and queue init succeeded");

        register_cleanup([this] {
            if (this->device != VK_NULL_HANDLE) {
                vkDestroyDevice(this->device, nullptr);
            }
        });
    }

    void core::init_swap_chain() noexcept {
        auto const [capabilities, formats, present_modes] = query_swap_chain_support(this->physical_device, this->surface);

        // Add checks:
        if (formats.empty() || present_modes.empty()) {
            utility::panic("Swap chain not adequately supported");
        }

        auto const [format, color_space] = choose_swap_surface_format(formats);
        VkPresentModeKHR const present_mode = choose_swap_present_mode(present_modes, this->create_options.vsync);
        {
            char const* name = "other";
            if (present_mode == present_mode_fifo_latest_ready) {
                name = "fifo-latest-ready";
            } else if (present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
                name = "mailbox";
            } else if (present_mode == VK_PRESENT_MODE_FIFO_KHR) {
                name = "fifo";
            } else if (present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                name = "immediate";
            }
            // Logged because it decides whether the fps counter can be believed: with a vsync mode the
            // presented rate is the display's, not the renderer's.
            utility::log("swapchain: present mode {} (vsync {}, {} modes offered)", name, this->create_options.vsync, present_modes.size());
        }

        VkExtent2D const extent = choose_swap_extent(capabilities, this->window);

        uint32_t image_count = capabilities.minImageCount + 1;

        if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
            image_count = capabilities.maxImageCount;
        }

        image_count = std::max(image_count, capabilities.minImageCount);

        VkSwapchainCreateInfoKHR create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = surface;

        create_info.minImageCount = image_count;
        create_info.imageFormat = format;
        create_info.imageColorSpace = color_space;
        create_info.imageExtent = extent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        // The F12 screenshot read-back copies FROM a swapchain image (vkCmdCopyImageToBuffer), which
        // requires the image to have been created with TRANSFER_SRC usage
        // (VUID-vkCmdCopyImageToBuffer-srcImage-00186). Ask for it when the surface supports it -
        // imageUsage has to stay a subset of the surface's supportedUsageFlags - and remember the
        // answer so the screenshot path can disable itself instead of performing an illegal copy on a
        // surface that does not.
        this->swapchain_transfer_src_supported = (capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
        if (this->swapchain_transfer_src_supported) {
            create_info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        } else {
            utility::log("swapchain: surface does not support VK_IMAGE_USAGE_TRANSFER_SRC_BIT - screenshots disabled");
        }

        queue_family_indices const indices = find_queue_families(this->physical_device, this->surface);
        if (!indices.compute_family || !indices.graphics_family || !indices.present_family) {
            utility::panic("find queue family index failed");
        }

        uint32_t const queue_family_indices[] = {indices.graphics_family.value(), indices.present_family.value()};

        if (indices.graphics_family != indices.present_family) {
            create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = 2;
            create_info.pQueueFamilyIndices = queue_family_indices;
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
            VkImageViewCreateInfo const create_info = make_image_view_info(this->swap_chain_images[i], swap_chain_image_format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1);
            if (vkCreateImageView(device, &create_info, nullptr, &this->swap_chain_image_views[i]) != VK_SUCCESS) {
                utility::panic("failed to create image views!");
            }
        }
        register_cleanup([this] {
            for (auto const& image_view : this->swap_chain_image_views) {
                vkDestroyImageView(device, image_view, nullptr);
            }
            this->swap_chain_image_views.clear();
        });
    }

    void core::create_depth_image(VkImage& image, VkDeviceMemory& image_memory, VkImageView& image_view) const noexcept {
        // Use the swapchain size stored in the class
        auto const& [width, height] = this->swap_chain_extent;

        // 1. Create images
        VkImageCreateInfo image_info = {};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.extent = {width, height, 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.format = this->depth_format;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS) {
            vkDestroyImage(device, image, nullptr);
            utility::panic("failed to create depth image!");
        }

        // 2. Allocate memory
        VkMemoryRequirements mem_requirements;
        vkGetImageMemoryRequirements(device, image, &mem_requirements);

        VkMemoryAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_requirements.size;
        alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits,
                                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, this->physical_device);

        if (vkAllocateMemory(device, &alloc_info, nullptr, &image_memory) != VK_SUCCESS) {
            utility::panic("failed to allocate depth image memory!");
        }

        vkBindImageMemory(device, image, image_memory, 0);

        // 3. Create image views
        VkImageViewCreateInfo const view_info = make_image_view_info(image, this->depth_format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, 1, 1);

        if (vkCreateImageView(device, &view_info, nullptr, &image_view) != VK_SUCCESS) {
            utility::panic("failed to create depth image view!");
        }
    }

    void core::create_depth_resources() noexcept {

        depth_format = find_depth_format(this->physical_device);

        // First test whether the depth format is valid
        if (depth_format == VK_FORMAT_UNDEFINED) {
            utility::panic("can't find supported depth format");
        }

        // Single-sampled, always: the scene renders into a 1x G-buffer whose depth is its own, and
        // this main depth image is what the no-G-buffer fallback instance (and nothing else) uses.
        // There is no sample count to resolve here - the [render] msaa setting went with the forward
        // path, which was its only consumer.

        depth_images.resize(swap_chain_image_views.size());
        depth_image_views.resize(swap_chain_image_views.size());
        depth_image_memories.resize(swap_chain_image_views.size()); // make sure memory is allocated

        for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
            // Call create_depth_image directly, but make sure the arguments are correct
            create_depth_image(depth_images[i], depth_image_memories[i], depth_image_views[i]);
        }

        register_cleanup([this] {
            for (auto const& view : depth_image_views) {
                vkDestroyImageView(device, view, nullptr);
            }
            for (auto const& image : depth_images) {
                vkDestroyImage(device, image, nullptr);
            }
            for (auto const& memory : depth_image_memories) {
                vkFreeMemory(device, memory, nullptr);
            }
            depth_image_views.clear();
            depth_images.clear();
            depth_image_memories.clear();
        });
    }

    void core::create_render_targets() {
        // One HDR scene target per swapchain image: the lighting stage (or the TAA resolve, when TAA
        // is on) writes it and the post-process pass samples it. TRANSFER_SRC as well, because the TAA
        // resolve copies the frame it wrote here into the history image (vkCmdCopyImage requires the
        // source to carry the usage flag).
        hdr_images.resize(swap_chain_image_views.size());
        hdr_image_memories.resize(swap_chain_image_views.size());
        hdr_image_views.resize(swap_chain_image_views.size());

        for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
            create_target_image(
                swap_chain_extent.width,
                swap_chain_extent.height,
                hdr_format,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                hdr_images[i],
                hdr_image_memories[i]);

            hdr_image_views[i] = create_image_view(
                hdr_images[i],
                hdr_format,
                VK_IMAGE_ASPECT_COLOR_BIT,
                device);
        }

        // Display-referred (LDR) targets, one per swapchain image, same size and lifetime as the
        // HDR ones: the post composite renders into them when FXAA is enabled and the FXAA pass
        // samples them. Format is hdr_format (R16F) on purpose - the values are display range but
        // stored gamma-encoded so FXAA can threshold them without a per-tap decode.
        ldr_images.resize(swap_chain_image_views.size());
        ldr_image_memories.resize(swap_chain_image_views.size());
        ldr_image_views.resize(swap_chain_image_views.size());
        for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
            create_target_image(
                swap_chain_extent.width,
                swap_chain_extent.height,
                hdr_format,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                ldr_images[i],
                ldr_image_memories[i]);

            ldr_image_views[i] = create_image_view(
                ldr_images[i],
                hdr_format,
                VK_IMAGE_ASPECT_COLOR_BIT,
                device);
        }

        // ---- G-buffer targets (see gbuffer_formats) + the pass's own 1x depth image ----
        // One set per swapchain image, single-sampled: a G-buffer cannot be multisampled without
        // per-sample shading, which is the trade that makes TAA the engine's anti-aliasing.
        // COLOR_ATTACHMENT | SAMPLED because the pass writes them as attachments and the
        // lighting/transparent/TAA/debug passes sample them.
        for (uint32_t target = 0; target < gbuffer_target_count; ++target) {
            std::vector<VkImage>& target_images = gbuffer_images[target];
            std::vector<VkDeviceMemory>& target_memories = gbuffer_image_memories[target];
            std::vector<VkImageView>& target_views = gbuffer_image_views[target];
            target_images.resize(swap_chain_image_views.size());
            target_memories.resize(swap_chain_image_views.size());
            target_views.resize(swap_chain_image_views.size());

            for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
                create_target_image(
                    swap_chain_extent.width,
                    swap_chain_extent.height,
                    gbuffer_formats[target],
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    target_images[i],
                    target_memories[i]);

                target_views[i] = create_image_view(
                    target_images[i],
                    gbuffer_formats[target],
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    device);
            }
        }

        // Motion vectors + the TAA working image (the scene color the resolve reads): same extent and
        // lifetime as the G-buffer targets, single-sampled, written as attachments and sampled
        // afterwards.
        auto const create_sampled_target = [this](std::vector<VkImage>& images, std::vector<VkDeviceMemory>& memories, std::vector<VkImageView>& views, VkFormat const format) {
            images.resize(swap_chain_image_views.size());
            memories.resize(swap_chain_image_views.size());
            views.resize(swap_chain_image_views.size());
            for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
                create_target_image(
                    swap_chain_extent.width,
                    swap_chain_extent.height,
                    format,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    images[i],
                    memories[i]);
                views[i] = create_image_view(images[i], format, VK_IMAGE_ASPECT_COLOR_BIT, device);
            }
        };
        create_sampled_target(velocity_images, velocity_image_memories, velocity_image_views, gbuffer_velocity_format);
        create_sampled_target(scene_color_images, scene_color_image_memories, scene_color_image_views, hdr_format);

        // The resolved-history image only needs TRANSFER_DST (the runtime copies the resolved frame
        // into it) and SAMPLED (the next frame's resolve reads it).
        taa_history_images.resize(swap_chain_image_views.size());
        taa_history_image_memories.resize(swap_chain_image_views.size());
        taa_history_image_views.resize(swap_chain_image_views.size());
        for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
            create_target_image(
                swap_chain_extent.width,
                swap_chain_extent.height,
                hdr_format,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                taa_history_images[i],
                taa_history_image_memories[i]);

            taa_history_image_views[i] = create_image_view(taa_history_images[i], hdr_format, VK_IMAGE_ASPECT_COLOR_BIT, device);
        }

        gbuffer_depth_images.resize(swap_chain_image_views.size());
        gbuffer_depth_image_memories.resize(swap_chain_image_views.size());
        gbuffer_depth_image_views.resize(swap_chain_image_views.size());
        for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
            create_target_image(
                swap_chain_extent.width,
                swap_chain_extent.height,
                depth_format,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                gbuffer_depth_images[i],
                gbuffer_depth_image_memories[i]);

            // the DEPTH aspect (a color view of a depth image is invalid) - same raw-view
            // convention as the other per-image targets, whose destruction is registered below
            gbuffer_depth_image_views[i] = create_image_view(gbuffer_depth_images[i], depth_format, VK_IMAGE_ASPECT_DEPTH_BIT, device);
        }

        // bloom targets: the 4-level chain (halved per level, min 1x1), same lifetime as the HDR
        // targets; each level gets one target per swapchain image
        for (uint32_t level = 0; level < bloom_level_count; ++level) {
            std::vector<VkImage>& level_images = bloom_images[level];
            std::vector<VkDeviceMemory>& level_memories = bloom_image_memories[level];
            std::vector<VkImageView>& level_views = bloom_image_views[level];
            uint32_t const level_width = std::max(1u, swap_chain_extent.width >> (level + 1u));
            uint32_t const level_height = std::max(1u, swap_chain_extent.height >> (level + 1u));

            level_images.resize(swap_chain_image_views.size());
            level_memories.resize(swap_chain_image_views.size());
            level_views.resize(swap_chain_image_views.size());
            for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
                create_target_image(
                    level_width,
                    level_height,
                    hdr_format,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    level_images[i],
                    level_memories[i]);

                level_views[i] = create_image_view(
                    level_images[i],
                    hdr_format,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    device);
            }
        }

        // The teardown is registered ONCE, not once per swapchain generation: this function reruns on
        // every recreate_swap_chain(), and register_cleanup() *pushes* (LIFO), so registering
        // unconditionally grew the cleanup stack by one identical lambda per resize. The single
        // lambda destroys whatever the vectors hold at destruction time, which is what we want.
        if (this->resolve_cleanup_registered) {
            return;
        }
        this->resolve_cleanup_registered = true;

        register_cleanup([this] {
            for (auto const& view : hdr_image_views) {
                vkDestroyImageView(device, view, nullptr);
            }
            for (auto const& memory : hdr_image_memories) {
                vkFreeMemory(device, memory, nullptr);
            }
            for (auto const& image : hdr_images) {
                vkDestroyImage(device, image, nullptr);
            }
            hdr_image_views.clear();
            hdr_image_memories.clear();
            hdr_images.clear();
            // the LDR (FXAA input) targets share this lifetime: they are (re)created with the
            // swapchain in exactly the same way, so they belong to the same cleanup registration
            for (auto const& view : ldr_image_views) {
                vkDestroyImageView(device, view, nullptr);
            }
            for (auto const& memory : ldr_image_memories) {
                vkFreeMemory(device, memory, nullptr);
            }
            for (auto const& image : ldr_images) {
                vkDestroyImage(device, image, nullptr);
            }
            ldr_image_views.clear();
            ldr_image_memories.clear();
            ldr_images.clear();
            // the G-buffer targets + the pass's own depth image share this lifetime too
            for (auto const& target_views : gbuffer_image_views) {
                for (auto const& view : target_views) {
                    vkDestroyImageView(device, view, nullptr);
                }
            }
            for (auto const& target_memories : gbuffer_image_memories) {
                for (auto const& memory : target_memories) {
                    vkFreeMemory(device, memory, nullptr);
                }
            }
            for (auto const& target_images : gbuffer_images) {
                for (auto const& image : target_images) {
                    vkDestroyImage(device, image, nullptr);
                }
            }
            gbuffer_image_views = {};
            gbuffer_image_memories = {};
            gbuffer_images = {};
            for (auto const& view : gbuffer_depth_image_views) {
                vkDestroyImageView(device, view, nullptr);
            }
            gbuffer_depth_image_views.clear();
            for (auto const& memory : gbuffer_depth_image_memories) {
                vkFreeMemory(device, memory, nullptr);
            }
            gbuffer_depth_image_memories.clear();
            for (auto const& image : gbuffer_depth_images) {
                vkDestroyImage(device, image, nullptr);
            }
            gbuffer_depth_images.clear();
            // motion vectors + the TAA working images share the same lifetime (see
            // create_render_targets)
            auto const destroy_images = [this](std::vector<VkImage>& images, std::vector<VkDeviceMemory>& memories, std::vector<VkImageView>& views) {
                for (auto const& view : views) {
                    vkDestroyImageView(device, view, nullptr);
                }
                for (auto const& memory : memories) {
                    vkFreeMemory(device, memory, nullptr);
                }
                for (auto const& image : images) {
                    vkDestroyImage(device, image, nullptr);
                }
                views.clear();
                memories.clear();
                images.clear();
            };
            destroy_images(velocity_images, velocity_image_memories, velocity_image_views);
            destroy_images(scene_color_images, scene_color_image_memories, scene_color_image_views);
            destroy_images(taa_history_images, taa_history_image_memories, taa_history_image_views);
            for (auto const& level_views : bloom_image_views) {
                for (auto const& view : level_views) {
                    vkDestroyImageView(device, view, nullptr);
                }
            }
            for (auto const& level_memories : bloom_image_memories) {
                for (auto const& memory : level_memories) {
                    vkFreeMemory(device, memory, nullptr);
                }
            }
            for (auto const& level_images : bloom_images) {
                for (auto const& image : level_images) {
                    vkDestroyImage(device, image, nullptr);
                }
            }
            bloom_image_views = {};
            bloom_image_memories = {};
            bloom_images = {};
        });
    }

    void core::create_command_pool() noexcept {
        VkCommandPoolCreateInfo const pool_info = make_command_pool_info(graphics_family_index);

        if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
            utility::panic("failed to create command pool");
        }

        register_cleanup([this] {
            if (command_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, command_pool, nullptr);
            }
        });
    }

    void core::create_target_image(
        uint32_t width,
        uint32_t height,
        VkFormat format,
        VkImageTiling tiling,
        VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkImage& image,
        VkDeviceMemory& image_memory) const noexcept {
        VkImageCreateInfo image_info = {};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.extent.width = width;
        image_info.extent.height = height;
        image_info.extent.depth = 1;
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.format = format;
        image_info.tiling = tiling;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.usage = usage;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS) {
            utility::panic("can't create target image");
        }

        VkMemoryRequirements mem_requirements;
        vkGetImageMemoryRequirements(device, image, &mem_requirements);

        VkMemoryAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_requirements.size;
        alloc_info.memoryTypeIndex = find_memory_type(
            mem_requirements.memoryTypeBits,
            properties,
            physical_device);

        if (vkAllocateMemory(device, &alloc_info, nullptr, &image_memory) != VK_SUCCESS) {
            utility::panic("can't allocate target image memory");
        }

        vkBindImageMemory(device, image, image_memory, 0);
    }

    void core::create_descriptor_pool() noexcept {
        std::vector<VkDescriptorPoolSize> pool_sizes;
        // Uniform buffers: one camera UBO + one light UBO binding per scene set
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 128});
        // Combined image samplers: the texture array (scene_texture_capacity per scene set)
        // dominates; plus the IBL bindings and the shadow map per set
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 * scene_texture_capacity});
        // Material table + instance transform storage buffers: two per scene set
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128});

        VkDescriptorPoolCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        // No UPDATE_AFTER_BIND: the scene descriptor sets are static per frame slot (each slot's
        // set always points at its own camera/shadow/skin/morph resources) and the shared
        // bindings (textures/IBL/materials/instances/light) are written before the render loop
        // starts. The texture array keeps PARTIALLY_BOUND so unwritten entries stay valid.
        info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        info.maxSets = 64;
        info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
        info.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(this->device, &info, nullptr, &this->descriptor_pool) != 0) {
            utility::panic("failed in creating descriptor pool");
        }
        register_cleanup([this] {
            if (descriptor_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(this->device, this->descriptor_pool, nullptr);
            }
        });
    }

    void core::init_scene_layouts() noexcept {
        // ---- 1. Fixed flat descriptor set layout (see the convention docs in core.cppm) ----
        std::array<VkDescriptorSetLayoutBinding, 13> bindings = {};
        bindings[0] = {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = nullptr};
        bindings[1] = {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = scene_texture_capacity, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .pImmutableSamplers = nullptr};
        bindings[2] = {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .pImmutableSamplers = nullptr};
        bindings[3] = {.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .pImmutableSamplers = nullptr};
        bindings[4] = {.binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .pImmutableSamplers = nullptr};
        // material table: per-material texture indices + factors (see material_record in vulkan/scene_tree/scene_tree.cppm)
        bindings[5] = {.binding = 5, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .pImmutableSamplers = nullptr};
        // per-instance world transforms for instanced draws (mat4 per instance, read in pbr.vert)
        bindings[6] = {.binding = 6, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .pImmutableSamplers = nullptr};
        // light UBO: directional sun (light-space view-proj + direction) + BRDF model ids +
        // the punctual-light count/array (read by shadow.vert and pbr.frag)
        bindings[7] = {.binding = 7, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = nullptr};
        // shadow map depth texture: LINEAR min/mag with compareEnable = VK_TRUE, so one
        // sampler2DArrayShadow texture() tap already returns the lit fraction of its own 2x2 texel
        // footprint - the hardware does the comparison (shading.glsl averages a 3x3 grid of taps)
        bindings[8] = {.binding = 8, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .pImmutableSamplers = nullptr};
        // per-joint skin matrices (mat4 per joint; indices 0-3 are the identity block for
        // unskinned draws; read in pbr.vert / shadow.vert, filled per frame by set_skin_matrices)
        bindings[9] = {.binding = 9, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .pImmutableSamplers = nullptr};
        // morph data (floats): per-morphable-primitive delta + weight blocks; written by the
        // caller through the runtime's morph scratch memory (set once + per frame for weights)
        bindings[10] = {.binding = 10, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .pImmutableSamplers = nullptr};
        // clustered light culling (M5): the per-cluster light count and the fixed-capacity index
        // rows. Written by the cluster COMPUTE pass, read by the fragment stage - hence both stages
        // in the flags (a binding is only usable from a stage that declares it here).
        bindings[11] = {.binding = 11, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = nullptr};
        bindings[12] = {.binding = 12, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = nullptr};

        std::array<VkDescriptorBindingFlags, 13> binding_flags = {};
        // texture array: only written entries are valid, appended before the render loop starts;
        // non-uniform indexing itself is a device feature, not a layout flag
        binding_flags[1] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

        VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info = {};
        flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        flags_info.bindingCount = static_cast<uint32_t>(binding_flags.size());
        flags_info.pBindingFlags = binding_flags.data();

        VkDescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        // No UPDATE_AFTER_BIND_POOL: the only binding flag left is PARTIALLY_BOUND (binding 1), which
        // does not require the update-after-bind pool flag. The layout flag only means anything
        // together with the pool flag - a set layout created with UPDATE_AFTER_BIND_POOL has to be
        // allocated from a pool created with UPDATE_AFTER_BIND - and neither is needed here.
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        layout_info.pNext = &flags_info;
        if (vkCreateDescriptorSetLayout(this->device, &layout_info, nullptr, &this->scene_descriptor_set_layout) != VK_SUCCESS) {
            utility::panic("failed to create scene descriptor set layout");
        }

        // ---- 2. Fixed pipeline layout: the scene set + the agreed push constant block ----
        // ONE range: the 96-byte per-primitive material block plus the pass-wide cascade index that
        // follows it (see core::scene_cascade_push_offset). One range rather than two because GLSL
        // allows a single push_constant block per stage: shadow.vert declares one block whose last
        // member is the cascade index, and that block has to fit inside a matching range. Shaders that
        // only need the material fields (pbr.vert/frag) still declare their 96-byte block, which is
        // contained in this one.
        VkPushConstantRange push_range = {};
        // VERTEX|FRAGMENT only - NOT compute: the cluster compute shader declares no push_constant
        // block, and every stage listed here must also be passed by each vkCmdPushConstants that
        // touches the range (VUID-vkCmdPushConstants-offset-01796), which the graphics pushes do not.
        push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push_range.offset = 0;
        push_range.size = scene_push_constant_size + scene_cascade_push_size;
        static_assert(scene_push_constant_size + scene_cascade_push_size <= 128, "the shared push constant range must fit the 128 bytes every Vulkan implementation guarantees");

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &this->scene_descriptor_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(this->device, &pipeline_layout_info, nullptr, &this->scene_pipeline_layout) != VK_SUCCESS) {
            utility::panic("failed to create scene pipeline layout");
        }

        register_cleanup([this] {
            if (this->scene_pipeline_layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(this->device, this->scene_pipeline_layout, nullptr);
                this->scene_pipeline_layout = VK_NULL_HANDLE;
            }
            if (this->scene_descriptor_set_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(this->device, this->scene_descriptor_set_layout, nullptr);
                this->scene_descriptor_set_layout = VK_NULL_HANDLE;
            }
        });
    }

    void core::create_sync_objects() {
        // Per frame slot: one BINARY image-available semaphore (vkAcquireNextImageKHR requires
        // binary) + one TIMELINE semaphore that replaces the old per-slot fences and counts
        // completed submissions. vkQueuePresentKHR also requires a binary wait, but present may
        // run on a separate queue, so the present-ready gates are ONE BINARY PER SWAPCHAIN IMAGE
        // (an image is only re-acquired after its present finished, which keeps the per-image
        // gate safe across queues).
        image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        present_ready_semaphores.resize(swap_chain_images.size());
        frame_done_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        frame_done_values.assign(MAX_FRAMES_IN_FLIGHT, 0);

        VkSemaphoreTypeCreateInfo timeline_type = make_timeline_semaphore_type_info();
        VkSemaphoreCreateInfo binary_info = make_binary_semaphore_info();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            if (vkCreateSemaphore(device, &binary_info, nullptr, &image_available_semaphores[i]) != VK_SUCCESS) {
                utility::panic("failed to create image-available semaphore");
            }
            VkSemaphoreCreateInfo timeline_info = binary_info;
            timeline_info.pNext = &timeline_type;
            if (vkCreateSemaphore(device, &timeline_info, nullptr, &frame_done_semaphores[i]) != VK_SUCCESS) {
                utility::panic("failed to create frame-done timeline semaphore");
            }
        }
        for (auto& semaphore : present_ready_semaphores) {
            if (vkCreateSemaphore(device, &binary_info, nullptr, &semaphore) != VK_SUCCESS) {
                utility::panic("failed to create present-ready semaphore");
            }
        }

        register_cleanup([this] {
            for (auto const& semaphore : image_available_semaphores) {
                vkDestroySemaphore(device, semaphore, nullptr);
            }
            for (auto const& semaphore : present_ready_semaphores) {
                vkDestroySemaphore(device, semaphore, nullptr);
            }
            for (auto const& semaphore : frame_done_semaphores) {
                vkDestroySemaphore(device, semaphore, nullptr);
            }
        });
    }

    void core::create_timestamp_query_pool() noexcept {
        // Two device facts decide whether pass timings are possible: how wide the timestamp
        // counter of a graphics queue is (timestampValidBits - a family that cannot write
        // timestamps reports 0) and how long one tick takes (timestampPeriod, ns/tick).
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(this->physical_device, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(this->physical_device, &family_count, families.data());
        if (this->graphics_family_index < families.size()) {
            this->timestamp_valid_bits = families[this->graphics_family_index].timestampValidBits;
        }
        this->timestamp_period_ns = this->device_properties.limits.timestampPeriod;

        if (this->timestamp_valid_bits == 0 || this->timestamp_period_ns <= 0.0f) {
            utility::log("gpu timing: unavailable on this queue ({} valid bits, {} ns/tick) - pass timings are off",
                         this->timestamp_valid_bits,
                         static_cast<double>(this->timestamp_period_ns));
            return;
        }

        VkQueryPoolCreateInfo const pool_info = make_query_pool_info(VK_QUERY_TYPE_TIMESTAMP, static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * gpu_timing_mark_capacity);
        if (vkCreateQueryPool(this->device, &pool_info, nullptr, &this->timestamp_query_pool) != VK_SUCCESS) {
            utility::log("gpu timing: timestamp query pool creation failed - pass timings are off");
            this->timestamp_query_pool = VK_NULL_HANDLE;
            return;
        }
        this->gpu_timing_supported = true;
        utility::log("gpu timing: {} marks/frame available ({} ns/tick, {} valid bits)",
                     gpu_timing_mark_capacity,
                     static_cast<double>(this->timestamp_period_ns),
                     this->timestamp_valid_bits);

        register_cleanup([this] {
            if (this->timestamp_query_pool != VK_NULL_HANDLE) {
                vkDestroyQueryPool(this->device, this->timestamp_query_pool, nullptr);
            }
        });
    }

    void core::begin_gpu_timing(VkCommandBuffer const command_buffer, uint32_t const slot) noexcept {
        this->gpu_timing_marks[slot] = 0;
        if (!this->gpu_timing_supported) {
            return;
        }
        // Reset on the GPU timeline: the query range may still be "in use" from the host's point of
        // view, and a recorded reset is ordered against the writes that follow it in the same
        // command buffer - a host-side vkResetQueryPool would need the slot to be idle, which is a
        // constraint the caller would have to remember on every path.
        vkCmdResetQueryPool(command_buffer, this->timestamp_query_pool, slot * gpu_timing_mark_capacity, gpu_timing_mark_capacity);
    }

    void core::mark_gpu_timing(VkCommandBuffer const command_buffer, uint32_t const slot, VkPipelineStageFlagBits const stage) noexcept {
        if (!this->gpu_timing_supported || this->gpu_timing_marks[slot] >= gpu_timing_mark_capacity) {
            return;
        }
        vkCmdWriteTimestamp(command_buffer, stage, this->timestamp_query_pool, slot * gpu_timing_mark_capacity + this->gpu_timing_marks[slot]);
        ++this->gpu_timing_marks[slot];
    }

    gpu_timing_result core::read_gpu_timings(uint32_t const slot) {
        gpu_timing_result result = {};
        if (!this->gpu_timing_supported) {
            return result;
        }
        // Only read a submitted slot, and only once per submission: the caller paced the slot, so
        // the queries of its last submission are complete, while a slot whose frame failed before
        // recording has nothing new to report.
        uint64_t const submitted = this->frame_done_values[slot];
        if (submitted == 0 || submitted <= this->gpu_timing_read_value[slot]) {
            return result;
        }
        this->gpu_timing_read_value[slot] = submitted; // this submission is now accounted for
        uint32_t const marks = this->gpu_timing_marks[slot];
        if (marks < 2) {
            return result; // a single mark has no interval to report
        }

        std::array<uint64_t, gpu_timing_mark_capacity> ticks = {};
        VkResult const status = vkGetQueryPoolResults(this->device,
                                                      this->timestamp_query_pool,
                                                      slot * gpu_timing_mark_capacity,
                                                      marks,
                                                      sizeof(uint64_t) * marks,
                                                      ticks.data(),
                                                      sizeof(uint64_t),
                                                      VK_QUERY_RESULT_64_BIT);
        if (status != VK_SUCCESS) {
            // VK_NOT_READY (or a lost pool): report "no measurement" rather than waiting - a frame
            // without a timing line is fine, a stalled frame is not.
            return result;
        }

        result.mark_count = marks;
        for (uint32_t mark = 0; mark + 1 < marks; ++mark) {
            result.milliseconds[mark] = utility::timestamp_delta_milliseconds(ticks[mark], ticks[mark + 1], this->timestamp_valid_bits, this->timestamp_period_ns);
        }
        return result;
    }

    vk_command_buffer core::make_command_buffer() const {
        return ::vulkan::make_command_buffer(this->device, this->command_pool);
    }

    vk_command_buffer core::make_secondary_command_buffer() const {
        return ::vulkan::make_secondary_command_buffer(this->device, this->command_pool);
    }

    vk_command_buffer core::make_secondary_command_buffer(VkCommandPool const pool) const {
        return ::vulkan::make_secondary_command_buffer(this->device, pool);
    }

    VkCommandPool core::make_command_pool() {
        VkCommandPoolCreateInfo pool_info = make_command_pool_info(this->graphics_family_index);

        VkCommandPool pool = VK_NULL_HANDLE;
        if (vkCreateCommandPool(this->device, &pool_info, nullptr, &pool) != VK_SUCCESS) {
            utility::panic("failed to create extra command pool");
        }
        // lifetime tied to this core: the pool is destroyed by the registered cleanup (LIFO,
        // after every command buffer allocated from it was freed by its RAII owner)
        this->register_cleanup([this, pool] {
            if (pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(this->device, pool, nullptr);
            }
        });
        return pool;
    }

    vk_descriptor_set core::make_descriptor_set(VkDescriptorSetLayout const layout) const { // NOLINT(*-misplaced-const)
        return ::vulkan::make_descriptor_set(this->device, this->descriptor_pool, layout);
    }

    vk_image_view core::make_image_view(VkImage const image, VkFormat const format, VkImageViewType const type) const {
        VkImageViewCreateInfo const view_info = make_image_view_info(image, format, type, VK_IMAGE_ASPECT_COLOR_BIT, VK_REMAINING_MIP_LEVELS, VK_REMAINING_ARRAY_LAYERS);
        VkImageView view = VK_NULL_HANDLE;
        vkCreateImageView(this->device, &view_info, nullptr, &view);
        return vk_image_view(view, this->device);
    }

    vk_sampler core::make_sampler(VkSamplerAddressMode const address_mode, float const max_lod) const {
        VkSamplerCreateInfo info = make_texture_sampler_info(address_mode, max_lod);
        VkSampler sampler = VK_NULL_HANDLE;
        vkCreateSampler(this->device, &info, nullptr, &sampler);
        return vk_sampler(sampler, this->device);
    }

    std::optional<vk_shader_module> core::make_shader_module(std::span<unsigned char> const shader) const noexcept {
        return ::vulkan::make_shader_module(shader, this->device);
    }

    void core::wait_frame_slot(uint32_t const slot) const {
        // Host pacing: wait until this slot's last submission (its timeline value) completed.
        // Value 0 means the slot was never submitted — nothing to wait for.
        uint64_t const value = this->frame_done_values[slot];
        if (value == 0) {
            return;
        }
        VkSemaphoreWaitInfo wait_info = {};
        wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &this->frame_done_semaphores[slot];
        wait_info.pValues = &value;
        vkWaitSemaphores(this->device, &wait_info, UINT64_MAX);
    }

    void core::to_next_frame() noexcept {
        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    VkResult core::submit(VkCommandBuffer const command_buffer, uint32_t const image_index) {
        // Signal this frame slot's TIMELINE to the next value (GPU completion + host pacing,
        // see wait_frame_slot) and the image's binary present-ready semaphore (vkQueuePresentKHR
        // requires a binary wait; per-image so a separate present queue cannot race a re-signal).
        // VUID-VkSubmitInfo-pNext-03241: with a timeline in the signal list the value count must
        // equal the semaphore count (the value for the binary is ignored).
        uint32_t const slot = static_cast<uint32_t>(this->current_frame);
        // The value this submission asks the slot's timeline to take. It is recorded only once
        // vkQueueSubmit has accepted the submission (below): wait_frame_slot() waits on the RECORDED
        // value, so recording one that no submission will ever signal would block this slot forever.
        uint64_t const signal_value = this->frame_done_values[slot] + 1;

        constexpr VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSemaphore signal_semaphores[2] = {this->frame_done_semaphores[slot], this->present_ready_semaphores[image_index]};
        uint64_t signal_values[2] = {signal_value, 0};
        VkTimelineSemaphoreSubmitInfo timeline_info = {};
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.signalSemaphoreValueCount = 2;
        timeline_info.pSignalSemaphoreValues = signal_values;

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pNext = &timeline_info;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &this->image_available_semaphores[slot];
        submit_info.pWaitDstStageMask = &wait_stage;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        submit_info.signalSemaphoreCount = 2;
        submit_info.pSignalSemaphores = signal_semaphores;
        VkResult const result = vkQueueSubmit(this->graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
        if (result == VK_SUCCESS) {
            this->frame_done_values[slot] = signal_value;
        }
        return result;
    }

    VkResult core::present(uint32_t const image_index) const {
        // Wait this image's binary present-ready semaphore (signaled by submit() above);
        // vkQueuePresentKHR requires binary wait semaphores (VUID-vkQueuePresentKHR-pWaitSemaphores-03267).
        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &this->present_ready_semaphores[image_index];
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &this->swap_chain;
        present_info.pImageIndices = &image_index;
        return vkQueuePresentKHR(this->present_queue, &present_info);
    }

    void core::recreate_swap_chain() {
        // 0. A minimized (or otherwise not-yet-sized) window reports currentExtent (0, 0). Building a
        //    swapchain and the per-image targets from that is invalid - vkCreateSwapchainKHR
        //    (VUID-VkSwapchainCreateInfoKHR-imageExtent-01689) and every vkCreateImage
        //    (VUID-VkImageCreateInfo-extent-00944/-00945) reject a zero extent - and there is nothing
        //    to render into anyway. Keep the current generation untouched and let the caller retry:
        //    the frame loop already skips frames whose swapchain extent is zero
        //    (runtime::pace_and_acquire), and a restore / resize produces a sized window shortly.
        swap_chain_support_details const support = query_swap_chain_support(this->physical_device, this->surface);
        if (support.capabilities.currentExtent.width == 0 || support.capabilities.currentExtent.height == 0) {
            if (!this->zero_extent_recreation_logged) {
                this->zero_extent_recreation_logged = true;
                utility::log("swapchain recreation deferred: the window has no drawable size yet (minimized / live resize)");
            }
            return;
        }
        this->zero_extent_recreation_logged = false;

        // 1. Wait for the device to be idle
        vkDeviceWaitIdle(device);

        // 2. Destroy the scene targets (the G-buffer/velocity/HDR/LDR/bloom set is rebuilt below)
        for (auto const& view : hdr_image_views) {
            vkDestroyImageView(device, view, nullptr);
        }
        hdr_image_views.clear();
        for (auto const& image : hdr_images) {
            vkDestroyImage(device, image, nullptr);
        }
        hdr_images.clear();
        for (auto const& memory : hdr_image_memories) {
            vkFreeMemory(device, memory, nullptr);
        }
        hdr_image_memories.clear();

        // 2c. Destroy the display-referred (FXAA input) targets
        for (auto const& view : ldr_image_views) {
            vkDestroyImageView(device, view, nullptr);
        }
        ldr_image_views.clear();
        for (auto const& image : ldr_images) {
            vkDestroyImage(device, image, nullptr);
        }
        ldr_images.clear();
        for (auto const& memory : ldr_image_memories) {
            vkFreeMemory(device, memory, nullptr);
        }
        ldr_image_memories.clear();

        // 2d-2. Destroy the G-buffer targets + the G-buffer pass's own depth image
        for (auto const& target_views : gbuffer_image_views) {
            for (auto const& view : target_views) {
                vkDestroyImageView(device, view, nullptr);
            }
        }
        gbuffer_image_views = {};
        for (auto const& target_images : gbuffer_images) {
            for (auto const& image : target_images) {
                vkDestroyImage(device, image, nullptr);
            }
        }
        gbuffer_images = {};
        for (auto const& target_memories : gbuffer_image_memories) {
            for (auto const& memory : target_memories) {
                vkFreeMemory(device, memory, nullptr);
            }
        }
        gbuffer_image_memories = {};
        for (auto const& view : gbuffer_depth_image_views) {
            vkDestroyImageView(device, view, nullptr);
        }
        gbuffer_depth_image_views.clear();
        for (auto const& image : gbuffer_depth_images) {
            vkDestroyImage(device, image, nullptr);
        }
        gbuffer_depth_images.clear();
        for (auto const& memory : gbuffer_depth_image_memories) {
            vkFreeMemory(device, memory, nullptr);
        }
        gbuffer_depth_image_memories.clear();

        // 2d-3. Destroy the motion-vector / TAA working images (same lifetime as the G-buffer)
        auto const destroy_target_set = [this](std::vector<VkImage>& images, std::vector<VkDeviceMemory>& memories, std::vector<VkImageView>& views) {
            for (auto const& view : views) {
                vkDestroyImageView(device, view, nullptr);
            }
            views.clear();
            for (auto const& image : images) {
                vkDestroyImage(device, image, nullptr);
            }
            images.clear();
            for (auto const& memory : memories) {
                vkFreeMemory(device, memory, nullptr);
            }
            memories.clear();
        };
        destroy_target_set(velocity_images, velocity_image_memories, velocity_image_views);
        destroy_target_set(scene_color_images, scene_color_image_memories, scene_color_image_views);
        destroy_target_set(taa_history_images, taa_history_image_memories, taa_history_image_views);

        // 2d. Destroy the bloom targets (all levels)
        for (auto const& level_views : bloom_image_views) {
            for (auto const& view : level_views) {
                vkDestroyImageView(device, view, nullptr);
            }
        }
        bloom_image_views = {};
        for (auto const& level_images : bloom_images) {
            for (auto const& image : level_images) {
                vkDestroyImage(device, image, nullptr);
            }
        }
        bloom_images = {};
        for (auto const& level_memories : bloom_image_memories) {
            for (auto const& memory : level_memories) {
                vkFreeMemory(device, memory, nullptr);
            }
        }
        bloom_image_memories = {};

        // 3. Destroy depth resources
        for (auto const& view : depth_image_views) {
            vkDestroyImageView(device, view, nullptr);
        }
        depth_image_views.clear();

        for (auto const& image : depth_images) {
            vkDestroyImage(device, image, nullptr);
        }
        depth_images.clear();

        for (auto const& memory : depth_image_memories) {
            vkFreeMemory(device, memory, nullptr);
        }
        depth_image_memories.clear();

        // 4. Destroy swapchain image views
        for (auto const& image_view : swap_chain_image_views) {
            vkDestroyImageView(device, image_view, nullptr);
        }
        swap_chain_image_views.clear();

        // 5. Destroy the swapchain itself
        if (swap_chain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device, swap_chain, nullptr);
            swap_chain = VK_NULL_HANDLE;
        }

        // 6. Recreate all resources
        this->init_swap_chain();        // rebuild swapchain
        this->init_image_views();       // rebuild image views
        this->create_depth_resources(); // rebuild depth resources
        this->create_render_targets();  // rebuild the scene targets

        // Present-ready semaphores are allocated per image index; destroy and rebuild when the
        // count changes (device is idle here). The per-slot timeline + binary acquire
        // semaphores are independent of the image count and survive untouched.
        for (auto const& semaphore : present_ready_semaphores) {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        present_ready_semaphores.resize(swap_chain_images.size());
        VkSemaphoreCreateInfo binary_info = make_binary_semaphore_info();
        for (auto& semaphore : present_ready_semaphores) {
            if (vkCreateSemaphore(device, &binary_info, nullptr, &semaphore) != VK_SUCCESS) {
                utility::panic("failed to recreate present-ready semaphore!");
            }
        }
    }

    vk_image_view core::make_depth_image_view(VkImage const image, VkFormat const format) const {
        VkImageViewCreateInfo const view_info = make_image_view_info(image, format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, VK_REMAINING_MIP_LEVELS, VK_REMAINING_ARRAY_LAYERS);
        VkImageView view = VK_NULL_HANDLE;
        vkCreateImageView(this->device, &view_info, nullptr, &view);
        return vk_image_view(view, this->device);
    }

    vk_image_view core::make_depth_array_view(VkImage const image, VkFormat const format) const {
        // every layer in one view: this is what sample2DArrayShadow reads (see make_depth_layer_view
        // for the per-layer views the shadow pass renders into)
        VkImageViewCreateInfo const view_info = make_image_view_info(image, format, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_ASPECT_DEPTH_BIT, VK_REMAINING_MIP_LEVELS, VK_REMAINING_ARRAY_LAYERS);
        VkImageView view = VK_NULL_HANDLE;
        vkCreateImageView(this->device, &view_info, nullptr, &view);
        return vk_image_view(view, this->device);
    }

    vk_image_view core::make_depth_layer_view(VkImage const image, VkFormat const format, uint32_t const layer) const {
        // one layer, as a plain 2D depth view: a dynamic rendering instance renders into exactly one
        // cascade, and a 2D view keeps that pass identical to the single-shadow-map one
        VkImageViewCreateInfo view_info = make_image_view_info(image, format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, VK_REMAINING_MIP_LEVELS, 1);
        view_info.subresourceRange.baseArrayLayer = layer;
        VkImageView view = VK_NULL_HANDLE;
        vkCreateImageView(this->device, &view_info, nullptr, &view);
        return vk_image_view(view, this->device);
    }

    vk_sampler core::make_shadow_sampler() const {
        // Shadow map sampler: depth-compare + LINEAR filtering gives HARDWARE percentage-closer
        // filtering - one texture() in pbr.frag (sampler2DShadow with a reference depth) returns
        // the lit fraction of the 2x2 texel neighborhood, so the shader no longer hand-loops a
        // 3x3 PCF. compareOp matches pbr.frag's test: lit when the fragment is not deeper than
        // the stored depth (ref <= stored).
        VkSamplerCreateInfo info = make_shadow_sampler_info();
        VkSampler sampler = VK_NULL_HANDLE;
        vkCreateSampler(this->device, &info, nullptr, &sampler);
        return vk_sampler(sampler, this->device);
    }

    std::expected<vk_pipeline, std::string_view> core::make_pipeline(
        std::span<unsigned char const> vertex_shader_code,
        std::span<unsigned char const> const fragment_shader_code,
        bool const depth_test_enabled) const {
        auto result = vulkan::make_pipeline(
            this->device,
            this->scene_pipeline_layout,
            this->swap_chain_image_format,
            this->depth_format,
            vertex_shader_code,
            fragment_shader_code,
            // 1x, always: the scene renders into a single-sampled G-buffer, so no pipeline this
            // engine builds can rasterize multisampled. The builders keep the parameter (it is a
            // pipeline property, not an engine setting), and this is the only value passed.
            VK_SAMPLE_COUNT_1_BIT,
            depth_test_enabled);
        if (result) {
            // Save the fullscreen viewport/scissor for the current swapchain size, used directly before draw
            result->viewport = {
                0.0f,
                0.0f,
                static_cast<float>(this->swap_chain_extent.width),
                static_cast<float>(this->swap_chain_extent.height),
                0.0f,
                1.0f,
            };
            result->scissor = {{0, 0}, this->swap_chain_extent};
        }
        return result;
    }

    std::expected<vk_pipeline, std::string_view> core::make_gbuffer_pipeline(
        std::span<unsigned char const> const vertex_shader_code,
        std::span<unsigned char const> const fragment_shader_code) const {
        // Five color targets: the three surface targets, the motion vectors, and the scene color the
        // pass ADDS the emissive term into (lighting-independent, and it needs the emissive texture and
        // the UVs the G-buffer does not store - see core::gbuffer_pass_attachment_count). The first
        // four are overwritten, the scene color accumulates, so the blend states differ per attachment.
        std::array<VkFormat, gbuffer_pass_attachment_count> const formats = {
            gbuffer_formats[0],
            gbuffer_formats[1],
            gbuffer_formats[2],
            gbuffer_velocity_format,
            hdr_format,
        };
        std::array<VkPipelineColorBlendAttachmentState, gbuffer_pass_attachment_count> const blends = {
            make_color_blend_attachment_opaque(),
            make_color_blend_attachment_opaque(),
            make_color_blend_attachment_opaque(),
            make_color_blend_attachment_opaque(), // motion vectors are data, not coverage
            make_color_blend_attachment_additive(),
        };
        auto result = vulkan::make_pipeline(
            this->device,
            this->scene_pipeline_layout,
            std::span<VkFormat const>(formats),
            this->depth_format,
            vertex_shader_code,
            fragment_shader_code,
            VK_SAMPLE_COUNT_1_BIT, // a G-buffer is never multisampled (see gbuffer_formats)
            true,                  // depth test + write: opaque geometry, and the lighting pass needs depth
            0.0f,
            0.0f,
            0.0f,
            std::span<VkPipelineColorBlendAttachmentState const>(blends));
        if (result) {
            // same fullscreen viewport/scissor default as the forward pipelines (the frame path
            // re-syncs it on every swapchain recreation)
            result->viewport = {
                0.0f,
                0.0f,
                static_cast<float>(this->swap_chain_extent.width),
                static_cast<float>(this->swap_chain_extent.height),
                0.0f,
                1.0f,
            };
            result->scissor = {{0, 0}, this->swap_chain_extent};
        }
        return result;
    }

    std::expected<vk_pipeline, std::string_view> core::make_depth_pipeline(
        std::span<unsigned char const> vertex_shader_code,
        std::span<unsigned char const> const fragment_shader_code,
        VkFormat const depth_format,
        float const depth_bias_constant_factor,
        float const depth_bias_slope_factor,
        float const depth_bias_clamp) const {
        auto result = vulkan::make_pipeline(
            this->device,
            this->scene_pipeline_layout,
            VK_FORMAT_UNDEFINED, // no color attachment
            depth_format,
            vertex_shader_code,
            fragment_shader_code,
            VK_SAMPLE_COUNT_1_BIT, // the shadow map is single-sampled
            true,                  // depth test + write
            false,                 // no color attachment
            depth_bias_constant_factor,
            depth_bias_slope_factor,
            depth_bias_clamp);
        // viewport/scissor are dynamic states set by the caller before drawing (the shadow map
        // is a fixed-size target, so core::make_pipeline's swapchain-size defaults do not apply)
        return result;
    }

    std::expected<vk_pipeline, std::string_view> core::make_cluster_pipeline(std::span<unsigned char const> const compute_shader_code) const {
        using fail = std::unexpected<std::string_view>;
        std::optional<vk_shader_module> const module = vulkan::make_shader_module(compute_shader_code, this->device);
        if (!module.has_value()) {
            return fail("failed to create the cluster compute shader module");
        }
        VkPipelineShaderStageCreateInfo stage_info = {};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = **module;
        stage_info.pName = "main"; // the SPIR-V entry point, as in vulkan::make_pipeline

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage_info;
        // the shared scene layout: the cluster pass reads the camera + light UBOs and writes the
        // per-cluster buffers through the same set 0 every graphics pipeline uses
        pipeline_info.layout = this->scene_pipeline_layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(this->device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("vkCreateComputePipelines failed");
        }
        return vk_pipeline(pipeline, this->scene_pipeline_layout, this->device);
    }

    void core::wait_idle() const noexcept {
        vkDeviceWaitIdle(this->device);
    }

    void core::set_window_title(std::string_view const title) const noexcept {
        if (this->window != nullptr) {
            glfwSetWindowTitle(this->window, title.data());
        }
    }
} // namespace vulkan
