module;

#include <vulkan/vulkan.h>

// GLFWwindow is used as an opaque pointer in gui_create_info; glfw3.h itself is only included
// in gui.cpp (the platform backend calls). A forward declaration keeps this interface light.
struct GLFWwindow;

export module vulkan.gui;
export import std;

/**
 * @file gui.cppm
 * @defgroup vulkan_gui Vulkan Debug GUI
 * @brief Dear ImGui integration for the vulkan runtime: owns the ImGui context, the GLFW +
 *        Vulkan backends and the per-frame recording slot, and exposes a small widget/panel
 *        API so the application builds its debug UI without touching Vulkan or GLFW directly.
 * @note
 *      - plain classes with direct members, like the rest of the vulkan module (no pimpl): the
 *        ImGui context and backend state live in ImGui's own global storage, so the gui_content
 *        only tracks whether the overlay is initialized, the registered panels and their widgets
 *      - widgets follow the scene_tree::primitive inheritance pattern: a widget base class with
 *        a virtual draw() and one subclass per control (label / checkbox / slider ...), so a new
 *        control is just a new subclass; debug_panel holds its widgets in a stack-style vector
 *        (push_back appends = more space, pop_back removes the top) and draws them in order
 *      - the runtime owns a gui_content member (optional, enabled via enable_debug_gui());
 *        when active, the runtime calls new_frame() before recording and record() after
 *        record_main_drawcalls() while the main rendering instance is still open, so the UI
 *        draws on top of the scene inside the same pass (and the same MSAA resolve)
 */
namespace vulkan {
    /**
     * @ingroup vulkan_gui
     * @brief everything gui_content needs to initialize its ImGui backends, decoupled from the
     *        vulkan::core object (the runtime assembles this from its core)
     */
    export struct gui_create_info {
        // ---- GLFW platform backend ----
        GLFWwindow* window = nullptr; // the window the overlay attaches to (callbacks chain)

        // ---- Vulkan renderer backend ----
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        uint32_t graphics_queue_family = 0; // queue family of @p graphics_queue
        VkQueue graphics_queue = VK_NULL_HANDLE;
        // the overlay draws into the OPEN main rendering instance, so its pipeline must match
        // the frame's color attachment: swapchain format + the scene's MSAA sample count
        VkFormat color_format = VK_FORMAT_UNDEFINED;
        VkSampleCountFlagBits msaa_samples = VK_SAMPLE_COUNT_1_BIT;
        // frames in flight (the backend sizes its per-frame render-buffer ring to this)
        uint32_t frames_in_flight = 2;
    };

    /**
     * @ingroup vulkan_gui
     * @brief base class of every debug widget: declares the draw strategy. Derived classes
     *        implement one control each (text / checkbox / slider ...), so the panel just does
     *        "for each widget: widget->draw()" — new controls only add a subclass
     *        (same pattern as scene_tree::primitive / normal_draw_primitive).
     * @note draw() is called once per frame inside debug_panel::draw(); implementations call
     *       the ImGui API directly (see gui.cpp)
     */
    export class widget {
    public:
        widget() = default;
        virtual ~widget() = default;
        widget(widget const&) = delete;
        widget& operator=(widget const&) = delete;
        widget(widget&&) noexcept = default;
        widget& operator=(widget&&) noexcept = default;

        /** @brief draw this widget at the current ImGui cursor position */
        virtual void draw() = 0;
    };

    /**
     * @ingroup vulkan_gui
     * @brief a text line; the text may be a fixed string or a std::function evaluated every
     *        frame (for live values such as fps)
     */
    export class label_widget final : public widget {
    public:
        explicit label_widget(std::string text);
        explicit label_widget(std::function<std::string()> text_fn);
        void draw() override;

    private:
        std::function<std::string()> text_fn; // wraps a fixed string or a per-frame callback
    };

    /**
     * @ingroup vulkan_gui
     * @brief a checkbox bound to an external bool; @p on_change fires when the user toggles it
     *        (e.g. to forward the new value into the runtime)
     */
    export class checkbox_widget final : public widget {
    public:
        checkbox_widget(std::string label, bool* value, std::function<void(bool)> on_change = {});
        void draw() override;

    private:
        std::string label;
        bool* value = nullptr; // external state the widget reads/writes
        std::function<void(bool)> on_change;
    };

    /**
     * @ingroup vulkan_gui
     * @brief a float slider bound to an external float; @p on_change fires on user drags
     */
    export class slider_widget final : public widget {
    public:
        slider_widget(std::string label, float* value, float min, float max, std::function<void(float)> on_change = {});
        void draw() override;

    private:
        std::string label;
        float* value = nullptr;
        float min = 0.0f;
        float max = 1.0f;
        std::function<void(float)> on_change;
    };

    /**
     * @ingroup vulkan_gui
     * @brief one debug window: a title, an open flag and a stack of widgets. Widgets are stored
     *        in a stack-style vector — push_back() appends (grows the panel), pop_back() removes
     *        the last one, so add/remove only ever touch the end. draw() opens the ImGui window
     *        and draws every widget in order.
     * @note the window close button flips @p open, so callers can honor it via open() / set_open()
     */
    export class debug_panel {
    public:
        explicit debug_panel(std::string title);

        // ---- widget stack (add/remove only affects the end) ----
        void push_back(std::unique_ptr<widget> item);
        void pop_back();
        void clear();
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] std::size_t size() const noexcept;

        // ---- window state ----
        [[nodiscard]] std::string const& get_title() const noexcept;
        void set_open(bool open) noexcept;
        [[nodiscard]] bool open() const noexcept;

        /** @brief draw the ImGui window and all widgets (called by gui_content::record) */
        void draw();

    private:
        std::string title;
        std::vector<std::unique_ptr<widget>> items; // widget stack (tail = top)
        bool is_open = true;
    };

    /**
     * @ingroup vulkan_gui
     * @brief owned Dear ImGui overlay for one window. Holds the registered debug panels and
     *        draws them every frame; external code obtains the instance from the runtime
     *        (runtime::debug_gui()) and adds/removes its own panels.
     * @note non-copyable (a second copy would fight over the single global ImGui context);
     *       the runtime holds exactly one as a direct member
     */
    export class gui_content {
    public:
        gui_content() noexcept = default;
        ~gui_content(); // shuts the overlay down (see gui.cpp); safe on a default-constructed object
        gui_content(gui_content const&) = delete;
        gui_content& operator=(gui_content const&) = delete;

        /**
         * @ingroup vulkan_gui
         * @brief create the ImGui context and initialize the GLFW + Vulkan backends from @p info
         * @return false when initialization failed (the overlay stays inactive)
         * @note requires the handles in @p info to be fully initialized and to outlive the
         *       gui_content (the runtime guarantees this)
         */
        bool init(gui_create_info const& info);

        /**
         * @ingroup vulkan_gui
         * @brief destroy the Vulkan/GLFW backends and the ImGui context; safe to call when not
         *        initialized (no-op) and idempotent
         */
        void shutdown();

        /** @brief true after a successful init() and before shutdown() */
        [[nodiscard]] bool is_active() const noexcept;

        // ---- panel management (the external participation surface) ----
        /**
         * @ingroup vulkan_gui
         * @brief create a debug panel with the given title and register it (drawn every frame
         *        in registration order)
         * @return reference to the new panel (stable: panels are heap-held); caller pushes
         *         widgets onto it and may remove it later via remove_panel()
         */
        debug_panel& add_panel(std::string title);

        /**
         * @ingroup vulkan_gui
         * @brief unregister and destroy @p panel (must be one returned by add_panel())
         */
        void remove_panel(debug_panel const& panel);

        /**
         * @ingroup vulkan_gui
         * @brief show / hide @p panel without destroying it
         */
        void set_panel_visible(debug_panel const& panel, bool visible);

        /** @brief number of registered panels */
        [[nodiscard]] std::size_t panel_count() const noexcept;

        /**
         * @ingroup vulkan_gui
         * @brief begin a new ImGui frame (glfw + vulkan + ImGui NewFrame). Call once per frame
         *        after the frame's swapchain image was acquired and before any ImGui widgets.
         */
        void new_frame();

        /**
         * @ingroup vulkan_gui
         * @brief run the registered UI builder (if any), draw every visible panel, then render
         *        the ImGui draw data into @p cmd
         * @param cmd command buffer currently being recorded, inside an OPEN rendering instance
         *        whose color attachment matches the overlay's pipeline (the runtime's main pass)
         * @note no-op when inactive
         */
        void record(VkCommandBuffer cmd);

        /**
         * @ingroup vulkan_gui
         * @brief called by the runtime after a swapchain recreation so the backend can adapt
         *        (e.g. min image count)
         */
        void on_swapchain_recreated();

    private:
        std::vector<std::unique_ptr<debug_panel>> panels; // registered panels, drawn in order
        // true between a successful init() and shutdown(). The ImGui context and backend data
        // themselves live in ImGui's global storage, not here.
        bool active = false;
        // frames in flight captured at init() (the backend's min-image hint on recreation)
        uint32_t frames_in_flight = 2;
    };
} // namespace vulkan
