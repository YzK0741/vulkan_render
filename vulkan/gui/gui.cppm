module;

#include <vulkan/vulkan.h>

export module vulkan.gui;
export import std;
import vulkan.core;

/**
 * @file gui.cppm
 * @defgroup vulkan_gui Vulkan Debug GUI
 * @brief Dear ImGui integration for the vulkan runtime: owns the ImGui context, the GLFW +
 *        Vulkan backends and the per-frame recording slot, and exposes a small layout/content
 *        API so the application builds its debug UI without touching Vulkan or GLFW directly.
 * @note
 *      - plain class with direct members, like the rest of the vulkan module (no pimpl): the
 *        ImGui context and backend state live in ImGui's own global storage, so this object
 *        only tracks whether the overlay is initialized and which UI content to draw
 *      - the runtime owns a gui_content member (optional, enabled via enable_debug_gui());
 *        when active, the runtime calls new_frame() before recording and record() after
 *        record_main_drawcalls() while the main rendering instance is still open, so the UI
 *        draws on top of the scene inside the same pass (and the same MSAA resolve)
 */
namespace vulkan {
    /**
     * @ingroup vulkan_gui
     * @brief owned Dear ImGui overlay for one vulkan::core window
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
         * @brief create the ImGui context and initialize the GLFW + Vulkan backends for @p core
         * @return false when initialization failed (the overlay stays inactive)
         * @note requires the core's window/device/queues to be fully initialized; the core must
         *       outlive the gui_content (the runtime guarantees this by member order)
         */
        bool init(core& core);

        /**
         * @ingroup vulkan_gui
         * @brief destroy the Vulkan/GLFW backends and the ImGui context; safe to call when not
         *        initialized (no-op) and idempotent
         */
        void shutdown();

        /** @brief true after a successful init() and before shutdown() */
        [[nodiscard]] bool is_active() const noexcept;

        /**
         * @ingroup vulkan_gui
         * @brief register the per-frame UI content builder; called every frame right before
         *        record() so the application may open ImGui windows / draw widgets
         * @param builder draws the UI (may use the ImGui API directly); empty clears it
         * @note the builder runs on the render thread inside the frame; keep it cheap
         */
        void set_ui_builder(std::function<void()> builder);

        /**
         * @ingroup vulkan_gui
         * @brief begin a new ImGui frame (glfw + vulkan + ImGui NewFrame). Call once per frame
         *        after the frame's swapchain image was acquired and before any ImGui widgets.
         */
        void new_frame();

        /**
         * @ingroup vulkan_gui
         * @brief run the registered UI builder, then render the ImGui draw data into @p cmd
         * @param cmd command buffer currently being recorded, inside an OPEN rendering instance
         *        whose color attachment matches the overlay's pipeline (the runtime's main pass)
         * @note no-op when inactive or when the UI builder produced nothing
         */
        void record(VkCommandBuffer cmd);

        /**
         * @ingroup vulkan_gui
         * @brief called by the runtime after a swapchain recreation so the backend can adapt
         *        (e.g. min image count)
         */
        void on_swapchain_recreated();

    private:
        // UI content builder (app code, drawn once per frame inside record()).
        std::function<void()> ui_builder;
        // true between a successful init() and shutdown(). The ImGui context and backend data
        // themselves live in ImGui's global storage, not here.
        bool active = false;
    };
} // namespace vulkan
