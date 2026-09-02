module;

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

export module vulkan.core.filter;
export import std;
export import vulkan.core;

/**
 * @file filter.cppm
 * @defgroup vulkan_core_filter Vulkan Core Filter
 * @brief filtered view over a vulkan::core: forwards only the methods and objects suitable
 *        for external consumers, hiding the initialization internals
 */
namespace vulkan {
    /**
     * @ingroup vulkan_core_filter
     * @brief filtered view over a core: forwards only the methods and objects suitable for
     *        external consumers, hiding the initialization internals
     * @note
     *      - holds a non-owning pointer to a core; the core must outlive the filter
     *      - the runtime exposes it via operator-> so external code never sees the raw core
     *      - frame management (acquire/submit/present) and pipeline/model creation are
     *        deliberately not forwarded: they belong to runtime / model
     */
    export class core_filter {
        core* vk_core = nullptr;

    public:
        explicit core_filter(core& core) noexcept;

        // ---- read-only access to objects external code may safely touch ----
        [[nodiscard]] VkDevice get_device() const noexcept;
        [[nodiscard]] GLFWwindow* get_window() const noexcept;
        [[nodiscard]] VkExtent2D get_swap_chain_extent() const noexcept;
        [[nodiscard]] VkFormat get_swap_chain_image_format() const noexcept;
        [[nodiscard]] uint32_t get_current_frame() const noexcept;
        static constexpr int max_frames_in_flight = core::MAX_FRAMES_IN_FLIGHT;

        // ---- facade operations (forwarded from core so callers need no raw API) ----
        void wait_idle() const noexcept;
        void set_window_title(std::string_view title) const noexcept;

        // ---- safe factory operations ----
        vk_command_buffer make_command_buffer() const;
        vk_descriptor_set make_descriptor_set(VkDescriptorSetLayout layout) const;
        std::optional<vk_shader_module> make_shader_module(std::span<unsigned char> shader) const noexcept;
        vk_image_view make_image_view(VkImage image, VkFormat format, VkImageViewType type) const;
        vk_sampler make_sampler(VkSamplerAddressMode address_mode, float max_lod) const;

        // ---- GPU allocation ----
        /**
         * @ingroup vulkan_core_filter
         * @brief access the VMA allocator for GPU buffer / image allocation
         * @return reference to the core's vma_allocator (thread-safe for creation and lookup)
         * @note non-const accessor: allocating GPU memory mutates the allocator, so this is
         *       only available on a non-const filter — a const runtime cannot allocate
         */
        vma_allocator& get_vma() noexcept;

        // ---- swapchain handling ----
        void recreate_swap_chain() const;
    };
} // namespace vulkan
