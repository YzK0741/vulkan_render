module;

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

module vulkan.gui;

import utility;

namespace vulkan::gui {
    // ---- gui_content lifecycle (see the module docs: ImGui state lives in ImGui's globals) ----

    gui_content::~gui_content() {
        this->shutdown();
    }

    bool gui_content::init(gui_create_info const& info) {
        if (this->active) {
            return true; // idempotent
        }
        if (info.device == VK_NULL_HANDLE || info.graphics_queue == VK_NULL_HANDLE || info.window == nullptr) {
            utility::log("gui_content: init skipped (incomplete gui_create_info)");
            return false;
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        // Persist window layout (position/size/open) to imgui_layout.ini in the working
        // directory: dragging a panel to a comfortable size records it there, and the recorded
        // size can be baked into the panel defaults later. DestroyContext() saves on shutdown.
        io.IniFilename = "imgui_layout.ini";

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
        // Save the window layout (position/size/open) before tearing the backends down, so a
        // dragged panel size survives into imgui_layout.ini for the next run.
        if (ImGui::GetIO().WantSaveIniSettings) {
            ImGui::SaveIniSettingsToDisk(ImGui::GetIO().IniFilename);
        }
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        this->active = false;
        this->panels.clear();
    }

    bool gui_content::is_active() const noexcept {
        return this->active;
    }

    // ---- panel management ----

    debug_panel& gui_content::add_panel(std::string title) {
        this->panels.push_back(std::make_unique<debug_panel>(std::move(title)));
        return *this->panels.back();
    }

    void gui_content::remove_panel(debug_panel const& panel) {
        std::erase_if(this->panels, [&panel](std::unique_ptr<debug_panel> const& p) { return p.get() == &panel; });
    }

    void gui_content::set_panel_visible(debug_panel const& panel, bool const visible) {
        // debug_panel stores its open flag in is_open; find the panel and update it
        for (auto const& p : this->panels) {
            if (p.get() == &panel) {
                p->set_open(visible);
                return;
            }
        }
    }

    std::size_t gui_content::panel_count() const noexcept {
        return this->panels.size();
    }

    void gui_content::new_frame() const {
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
        for (auto const& panel : this->panels) {
            if (panel->open()) {
                panel->draw();
            }
        }
        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    }

    void gui_content::on_swapchain_recreated() const {
        if (!this->active) {
            return;
        }
        // swapchain image count may have changed; the backend's per-frame buffers stay sized to
        // the frames-in-flight count captured at init(), so only sync the min-image hint.
        ImGui_ImplVulkan_SetMinImageCount(this->frames_in_flight);
    }

    // ---- debug_panel ----

    debug_panel::debug_panel(std::string title)
        : title{std::move(title)} {
    }

    void debug_panel::push_back(std::unique_ptr<widget> item) {
        this->items.push_back(std::move(item));
    }

    void debug_panel::pop_back() {
        if (!this->items.empty()) {
            this->items.pop_back();
        }
    }

    void debug_panel::clear() {
        this->items.clear();
    }

    bool debug_panel::empty() const noexcept {
        return this->items.empty();
    }

    std::size_t debug_panel::size() const noexcept {
        return this->items.size();
    }

    std::string const& debug_panel::get_title() const noexcept {
        return this->title;
    }

    void debug_panel::set_open(bool const open) noexcept {
        this->is_open = open;
    }

    bool debug_panel::open() const noexcept {
        return this->is_open;
    }

    void debug_panel::set_default_size(float const width, float const height) noexcept {
        this->default_width = width;
        this->default_height = height;
    }

    void debug_panel::draw() {
        if (this->default_width > 0.0f && this->default_height > 0.0f) {
            ImGui::SetNextWindowSize(ImVec2(this->default_width, this->default_height), ImGuiCond_FirstUseEver);
        }
        if (!ImGui::Begin(this->title.c_str(), &this->is_open)) {
            ImGui::End();
            return;
        }
        for (auto const& item : this->items) {
            if (item != nullptr) {
                item->draw();
            }
        }
        ImGui::End();
    }

    // ---- widgets ----

    label_widget::label_widget(std::string text)
        : text_fn{[t = std::move(text)]() { return t; }} { // copies per frame (stable text)
    }

    label_widget::label_widget(std::function<std::string()> text_fn)
        : text_fn{std::move(text_fn)} {
    }

    void label_widget::draw() {
        if (this->text_fn) {
            ImGui::TextUnformatted(this->text_fn().c_str());
        }
    }

    checkbox_widget::checkbox_widget(std::string label, bool* value, std::function<void(bool)> on_change)
        : label{std::move(label)}
        , value{value}
        , on_change{std::move(on_change)} {
    }

    void checkbox_widget::draw() {
        if (this->value == nullptr) {
            return;
        }
        bool const before = *this->value;
        if (ImGui::Checkbox(this->label.c_str(), this->value) && before != *this->value && this->on_change) {
            this->on_change(*this->value);
        }
    }

    slider_widget::slider_widget(std::string label, float* value, float min, float max, std::function<void(float)> on_change)
        : label{std::move(label)}
        , value{value}
        , min{min}
        , max{max}
        , on_change{std::move(on_change)} {
    }

    void slider_widget::draw() {
        if (this->value == nullptr) {
            return;
        }
        float const before = *this->value;
        if (ImGui::SliderFloat(this->label.c_str(), this->value, this->min, this->max) && before != *this->value && this->on_change) {
            this->on_change(*this->value);
        }
    }

    vec3_widget::vec3_widget(std::string label, float* value, float const speed, std::function<void()> on_change)
        : label{std::move(label)}
        , value{value}
        , speed{speed}
        , on_change{std::move(on_change)} {
    }

    void vec3_widget::draw() {
        if (this->value == nullptr) {
            return;
        }
        // snapshot the xyz triplet to detect whether the drag actually changed anything
        std::array<float, 3> const before = {this->value[0], this->value[1], this->value[2]};
        if (ImGui::DragFloat3(this->label.c_str(), this->value, this->speed)) {
            bool changed = false;
            for (int i = 0; i < 3; ++i) {
                changed = changed || this->value[i] != before[static_cast<std::size_t>(i)];
            }
            if (changed && this->on_change) {
                this->on_change();
            }
        }
    }

    combo_widget::combo_widget(std::string label, std::vector<std::string> items, int* current_item, std::function<void(int)> on_change)
        : label{std::move(label)}
        , items{std::move(items)}
        , current_item{current_item}
        , on_change{std::move(on_change)} {
    }

    void combo_widget::draw() {
        if (this->current_item == nullptr || this->items.empty()) {
            return;
        }
        *this->current_item = std::clamp(*this->current_item, 0, static_cast<int>(this->items.size()) - 1);
        int const before = *this->current_item;
        if (ImGui::BeginCombo(this->label.c_str(), this->items[static_cast<std::size_t>(*this->current_item)].c_str())) {
            for (int i = 0; i < static_cast<int>(this->items.size()); ++i) {
                bool const selected = i == *this->current_item;
                if (ImGui::Selectable(this->items[static_cast<std::size_t>(i)].c_str(), selected)) {
                    *this->current_item = i;
                }
            }
            ImGui::EndCombo();
        }
        if (before != *this->current_item && this->on_change) {
            this->on_change(*this->current_item);
        }
    }
} // namespace vulkan::gui
