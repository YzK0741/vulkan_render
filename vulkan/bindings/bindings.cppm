// module version: 0.1.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/bindings/bindings.cppm
 * @brief The descriptor-set side of the runtime: the per-swapchain-image set families and the
 *        per-frame-slot scene sets, with the pool lifetime rule that belongs to them.
 * @ingroup vulkan_bindings
 *
 * Extracted from vulkan.runtime. What made this worth its own module is that the three per-image
 * families - the post chain's, the G-buffer debug view's and TAA's - are THE SAME ALGORITHM read line
 * by line, differing only in how many sets each image needs (5 for post: the prefilter, three
 * downsamples and the composite; 1 for the other two), which views each set points at, and which
 * sampler it uses. So they become one class instead of three functions, and the property that keeps
 * descriptor writes off the per-frame path - "if the bound views are unchanged, do nothing" - moves
 * from being an incidental early return in each of them into the structure of that class.
 *
 * The pool lifetime rule travels with the pools, because it is only about them: a pool must not be
 * destroyed while a recorded command buffer still names one of its sets (VUID-vkDestroyDescriptorPool
 * -descriptorPool-00303), so a pool that is replaced is RETIRED rather than destroyed. Callers retire
 * their families when the swapchain is recreated (runtime::on_swapchain_recreated).
 *
 * The interface takes handles, never the runtime: the core's per-image views and the runtime's buffers,
 * images and samplers arrive as parameters, so nothing here can reach into either.
 */

module;

#include <cstdint>
#include <functional>
#include <span>
#include <vector>
#include <vulkan/vulkan.h>

export module vulkan.bindings;

import vulkan.core;

namespace vulkan::bindings {
    /**
     * @brief the descriptor sets of one per-image family (post chain, G-buffer debug view, TAA)
     * @ingroup vulkan_bindings
     *
     * ensure() is called once per frame and is a no-op unless the views it would bind have changed
     * (a swapchain recreation, a resize, or the first call after creation): it compares the signature
     * view of every family member against what it bound last time. That comparison is what keeps a
     * vkUpdateDescriptorSets off the frame path, so it is deliberately part of the class rather than a
     * caller's responsibility.
     *
     * @note the sets are flattened: set(image, which) is the @c which -th set of @c image, with
     *       @c sets_per_image entries per image (post: 0=prefilter, 1..3=downsamples, 4=composite).
     */
    export class image_set_family {
    public:
        /// describes one image's sets to the family: the caller writes them (bindings depend on the
        /// pass's own view layout, which only the call site knows)
        using write_sets_fn = std::function<void(core const& vk, uint32_t image_index, std::span<VkDescriptorSet const> sets)>;

        /**
         * @brief make sure this family's sets exist, are allocated for @p signature_views.size() images
         *        and point at @p signature_views; returns false when the inputs are not usable yet
         * @param vk the core (device, and the per-image views the caller passes in the callback)
         * @param layout the set layout the family allocates from (owned by the pass, see vulkan.pipelines)
         * @param sets_per_image how many sets each image needs (1 unless a chain like post's needs more)
         * @param signature_views one view per image whose identity decides whether a rebind is needed
         * @param write called for each image that needs (re)binding, with that image's sets
         */
        [[nodiscard]] bool ensure(core const& vk,
                                  VkDescriptorSetLayout layout,
                                  uint32_t sets_per_image,
                                  std::span<VkImageView const> signature_views,
                                  write_sets_fn const& write);

        /// every set of image @p image_index, or an empty span when the family is not ready
        [[nodiscard]] std::span<VkDescriptorSet const> sets(uint32_t image_index) const noexcept;

        /// the @c which -th set of image @p image_index (VK_NULL_HANDLE when out of range)
        [[nodiscard]] VkDescriptorSet set(uint32_t image_index, uint32_t which) const noexcept;

        /// whether the sets are allocated and bound
        [[nodiscard]] bool ready() const noexcept;

        /// how many images the family is currently allocated for (0 before the first ensure)
        [[nodiscard]] uint32_t image_count() const noexcept;

        /// retire the current pool instead of destroying it: a pool must outlive every recorded command
        /// buffer that names one of its sets (see the class comment). Called on swapchain recreation.
        void retire_all();

    private:
        VkDescriptorPool pool = VK_NULL_HANDLE;
        uint32_t pool_capacity = 0;
        uint32_t sets_per_image = 0;
        std::vector<VkDescriptorSet> flat_sets = {};   // sets_per_image entries per image
        std::vector<VkImageView> bound_signature = {}; // what the current sets point at
        std::vector<VkDescriptorPool> retired = {};    // destroyed with the runtime, never earlier
    };
} // namespace vulkan::bindings