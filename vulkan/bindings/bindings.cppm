// module version: 0.2.0  (independent of the app version in CMakeLists project(VERSION))

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

#include <algorithm>
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
         * @param descriptors_per_set how many combined-image-sampler descriptors one of those sets holds
         *        (the caller decides the pool size, so a wrong count shows up as an allocation failure)
         * @param signature_views one view per image whose identity decides whether a rebind is needed
         * @param write called for each image that needs (re)binding, with that image's sets
         */
        [[nodiscard]] bool ensure(core const& vk,
                                  VkDescriptorSetLayout layout,
                                  uint32_t sets_per_image,
                                  uint32_t descriptors_per_set,
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

        /// destroys every pool this family owns, the retired ones included: they outlived their
        /// generation on purpose, so they may only go away once the runtime itself does
        ~image_set_family();

    private:
        VkDevice device = VK_NULL_HANDLE; // remembered by ensure() so the destructor can destroy the pools
        VkDescriptorPool pool = VK_NULL_HANDLE;
        uint32_t pool_capacity = 0;
        uint32_t sets_per_image = 0;
        std::vector<VkDescriptorSet> flat_sets = {};   // sets_per_image entries per image
        std::vector<VkImageView> bound_signature = {}; // what the current sets point at
        std::vector<VkDescriptorPool> retired = {};    // destroyed with the runtime, never earlier
    };
    // ---- image_set_family --------------------------------------------------------------------------
    // The algorithm the three per-image families shared: ready check, image count check, signature
    // comparison, pool (re)creation with retirement, allocation, then the caller's writes. Everything a
    // family knows that the class cannot is a parameter: how many sets an image needs, how many
    // descriptors a set holds (which sizes the pool), which views decide a rebind, and what to write.
    namespace {
        [[nodiscard]] bool same_views(std::vector<VkImageView> const& bound, std::span<VkImageView const> const wanted) {
            return bound.size() == wanted.size() && std::equal(bound.begin(), bound.end(), wanted.begin());
        }
    } // namespace

    image_set_family::~image_set_family() {
        for (VkDescriptorPool const pooled : this->retired) {
            vkDestroyDescriptorPool(this->device, pooled, nullptr);
        }
        this->retired.clear();
        if (this->pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(this->device, this->pool, nullptr);
            this->pool = VK_NULL_HANDLE;
        }
    }

    bool image_set_family::ensure(core const& vk, VkDescriptorSetLayout const layout, uint32_t const per_image, uint32_t const descriptors_per_set,
                                  std::span<VkImageView const> const signature_views, write_sets_fn const& write) {
        this->device = vk.device;
        std::size_t const image_count = signature_views.size();
        if (layout == VK_NULL_HANDLE || !static_cast<bool>(write) || image_count == 0 || per_image == 0) {
            return false;
        }
        // The property that keeps descriptor writes off the frame path: the sets stay allocated, and their
        // contents are rewritten only when the views they point at actually change.
        if (this->sets_per_image == per_image && this->flat_sets.size() == image_count * per_image && same_views(this->bound_signature, signature_views)) {
            return true;
        }

        if (this->pool == VK_NULL_HANDLE || this->pool_capacity != image_count || this->sets_per_image != per_image) {
            // A pool cannot grow and its sets are still allocated, so a different image count needs a new
            // pool; the old one is RETIRED rather than destroyed, because recorded command buffers still
            // name its sets (VUID-vkDestroyDescriptorPool-descriptorPool-00303).
            if (this->pool != VK_NULL_HANDLE) {
                this->retired.push_back(this->pool);
                this->pool = VK_NULL_HANDLE;
            }
            VkDescriptorPoolSize pool_size = {};
            pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            pool_size.descriptorCount = static_cast<uint32_t>(image_count * per_image * descriptors_per_set);
            VkDescriptorPoolCreateInfo pool_info = {};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = static_cast<uint32_t>(image_count * per_image);
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            if (vkCreateDescriptorPool(this->device, &pool_info, nullptr, &this->pool) != VK_SUCCESS) {
                this->pool = VK_NULL_HANDLE;
                this->pool_capacity = 0;
                this->sets_per_image = 0;
                this->flat_sets.clear();
                this->bound_signature.clear();
                return false;
            }
            this->pool_capacity = static_cast<uint32_t>(image_count);
            this->sets_per_image = per_image;
            this->flat_sets.clear();
            this->bound_signature.clear();
        }

        if (this->flat_sets.size() != image_count * per_image) {
            std::vector<VkDescriptorSetLayout> const layouts(image_count * per_image, layout);
            this->flat_sets.assign(image_count * per_image, VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo allocate_info = {};
            allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate_info.descriptorPool = this->pool;
            allocate_info.descriptorSetCount = static_cast<uint32_t>(layouts.size());
            allocate_info.pSetLayouts = layouts.data();
            if (vkAllocateDescriptorSets(this->device, &allocate_info, this->flat_sets.data()) != VK_SUCCESS) {
                this->flat_sets.clear();
                this->bound_signature.clear();
                return false;
            }
        }

        for (std::size_t i = 0; i < image_count; ++i) {
            write(vk, static_cast<uint32_t>(i), std::span<VkDescriptorSet const>(this->flat_sets.data() + i * per_image, per_image));
        }
        this->bound_signature.assign(signature_views.begin(), signature_views.end());
        return true;
    }

    std::span<VkDescriptorSet const> image_set_family::sets(uint32_t const image_index) const noexcept {
        if (this->sets_per_image == 0) {
            return {};
        }
        std::size_t const first = static_cast<std::size_t>(image_index) * this->sets_per_image;
        if (first + this->sets_per_image > this->flat_sets.size()) {
            return {};
        }
        return {this->flat_sets.data() + first, this->sets_per_image};
    }

    VkDescriptorSet image_set_family::set(uint32_t const image_index, uint32_t const which) const noexcept {
        std::span<VkDescriptorSet const> const all = this->sets(image_index);
        return which < all.size() ? all[which] : VK_NULL_HANDLE;
    }

    bool image_set_family::ready() const noexcept {
        return this->sets_per_image != 0 && this->flat_sets.size() == this->bound_signature.size() * this->sets_per_image;
    }

    uint32_t image_set_family::image_count() const noexcept {
        return this->sets_per_image == 0 ? 0u : static_cast<uint32_t>(this->flat_sets.size() / this->sets_per_image);
    }

    void image_set_family::retire_all() {
        if (this->pool != VK_NULL_HANDLE) {
            this->retired.push_back(this->pool);
            this->pool = VK_NULL_HANDLE;
        }
        this->pool_capacity = 0;
        this->sets_per_image = 0;
        this->flat_sets.clear();
        this->bound_signature.clear();
    }
} // namespace vulkan::bindings