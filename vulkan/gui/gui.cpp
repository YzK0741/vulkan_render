module;

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

module vulkan.gui;

import utility;

namespace vulkan {
    gui_content::~gui_content() {
        this->shutdown();
    }

    bool gui_content::init(gui_create_info const& info) {
        if (this->active) {
            return true; // idempotent
        }
        // ImGui draws into the runtime's OPEN main rendering instance. That instance uses
        // dynamic rendering (the runtime's shadow pass already requires Vulkan 1.3 dynamic
        // rendering, so the classic fallback cannot host a usable overlay anyway).
        if (info.device == VK_NULL_HANDLE || info.graphics_queue == VK_NULL_HANDLE || info.window == nullptr) {
            utility::log("gui_content: init skipped (incomplete gui_create_info)");
            return false;
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.IniFilename = nullptr; // do not write imgui.ini next to the executable

        // Platform backend. install_callbacks=true makes imgui chain-call the runtime's own
        // GLFW callbacks (mouse/scroll -> orbit camera) which were registered earlier.
        if (!ImGui_ImplGlfw_InitForVulkan(info.window, /*install_callbacks=*/true)) {
            utility::log("gui_content: ImGui_ImplGlfw_InitForVulkan failed");
            ImGui::DestroyContext();
            return false;
        }

        ImGui_ImplVulkan_InitInfo backend_info = {};
        backend_info.ApiVersion = VK_API_VERSION_1_3;
        backend_info.Instance = info.instance;
        backend_info.PhysicalDevice = info.physical_device;
        backend_info.Device = info.device;
        backend_info.QueueFamily = info.graphics_queue_family;
        backend_info.Queue = info.graphics_queue;
        // backend creates its own descriptor pool (we must not share the runtime's scene pool)
        backend_info.DescriptorPoolSize = 8;
        // frames in flight: the runtime advances one slot per rendered frame, matching the
        // backend's per-frame render-buffer ring
        backend_info.MinImageCount = info.frames_in_flight;
        backend_info.ImageCount = info.frames_in_flight;
        backend_info.UseDynamicRendering = true;
        backend_info.PipelineInfoMain.MSAASamples = info.msaa_samples;
        VkPipelineRenderingCreateInfo rendering_info = {};
        rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachmentFormats = &info.color_format;
        backend_info.PipelineInfoMain.PipelineRenderingCreateInfo = rendering_info;
        backend_info.CheckVkResultFn = [](VkResult const err) {
            if (err != VK_SUCCESS) {
                utility::log("imgui vulkan backend error: {}", static_cast<int>(err));
            }
        };

        if (!ImGui_ImplVulkan_Init(&backend_info)) {
            utility::log("gui_content: ImGui_ImplVulkan_Init failed");
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            return false;
        }

        this->active = true;
        this->frames_in_flight = info.frames_in_flight;
        utility::log("gui_content: ImGui overlay initialized (dynamic rendering, MSAA {})", static_cast<int>(info.msaa_samples));
        return true;
    }

    void gui_content::shutdown() {
        if (!this->active) {
            return;
        }
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        this->active = false;
        this->ui_builder = {};
    }

    bool gui_content::is_active() const noexcept {
        return this->active;
    }

    void gui_content::set_ui_builder(std::function<void()> builder) {
        this->ui_builder = std::move(builder);
    }

    void gui_content::new_frame() {
        if (!this->active) {
            return;
        }
        ImGui_ImplGlfw_NewFrame();
        ImGui_ImplVulkan_NewFrame();
        ImGui::NewFrame();
    }

    void gui_content::record(VkCommandBuffer const cmd) {
        if (!this->active) {
            return;
        }
        if (this->ui_builder) {
            this->ui_builder(); // app draws its windows/widgets (ImGui API)
        }
        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    }

    void gui_content::on_swapchain_recreated() {
        if (!this->active) {
            return;
        }
        // swapchain image count may have changed; the backend's per-frame buffers stay sized to
        // the frames-in-flight count captured at init(), so only sync the min-image hint.
        ImGui_ImplVulkan_SetMinImageCount(this->frames_in_flight);
    }
} // namespace vulkan
