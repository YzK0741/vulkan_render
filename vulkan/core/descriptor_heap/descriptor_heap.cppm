module; // the macro-using Vulkan header must not be imported into a module purview

#include <cstdint>
#include <span>
#include <vulkan/vulkan.h>

/**
 * @file vulkan/core/descriptor_heap/descriptor_heap.cppm
 * @brief VK_EXT_descriptor_heap as this renderer's binding model: descriptors in a buffer the application
 *        writes, instead of descriptor sets, layouts and pools.
 * @defgroup vulkan_core_descriptor_heap Descriptor Heap
 *
 * WHY IT IS A SUBMODULE OF core AND NOT A PASS OR A TOP-LEVEL MODULE: a heap is DEVICE-WIDE state. The
 * extension's own guidance is to bind one heap for the application's lifetime, because binding a new one costs
 * a pipeline flush; the descriptors in it are the same contract the shared scene set already describes; and
 * every pass reads them. So it lives beside the vma allocator as something `core` owns and hands down, and it
 * deliberately knows NOTHING about what a binding means - set 0 binding 7 being the light UBO is core's
 * business, not this file's.
 *
 * WHAT IT OWNS: the two heap buffers (resources and samplers, which are SEPARATE heaps in this API), the
 * layout arithmetic that the device's properties dictate, and the three calls that make the model work -
 * vkWriteResourceDescriptorsEXT to write a descriptor, vkCmdBindResourceHeapEXT/vkCmdBindSamplerHeapEXT to
 * bind the heaps, and VkDescriptorSetAndBindingMappingInfoEXT to let a shader keep naming `set N, binding M`.
 *
 * THE MAPPING IS WHAT MAKES THIS A MIGRATION RATHER THAN A REWRITE: it is chained into
 * VkPipelineShaderStageCreateInfo (not only into VkShaderCreateInfoEXT), so the shaders keep their existing
 * `layout(set = ..., binding = ...)` declarations and the heap is what those numbers resolve to. A combined
 * image sampler is expressed with an EMBEDDED SAMPLER in the resource heap
 * (VkDescriptorMappingSourceConstantOffsetEXT::pEmbeddedSampler, whose sampler part is taken from the reserved
 * sampler range) - which is why a heap binding must respect minSamplerHeapReservedRangeWithEmbedded.
 */

export module vulkan.core.descriptor_heap;

import vulkan.core.vma;         // the allocator the heaps are allocated from
import vulkan.core.vma.handles; // vk_buffer, which is what a heap buffer is

namespace vulkan {
    /**
     * @ingroup vulkan_core_descriptor_heap
     * @brief the device's heap numbers, copied out of VkPhysicalDeviceDescriptorHeapPropertiesEXT at init
     * @note it is a struct of its own rather than a reference to the queried properties so that this module does
     *       not depend on the capability layer's header: the caller copies what a layout needs, and the heap
     *       never reaches back into device state it does not own.
     */
    export struct heap_limits {
        VkDeviceSize max_resource_size = 0;
        VkDeviceSize max_sampler_size = 0;
        /// what a heap binding's device address must be a multiple of
        VkDeviceSize resource_alignment = 0;
        VkDeviceSize sampler_alignment = 0;
        /// how much of each heap an implementation wants reserved before descriptors may follow
        VkDeviceSize resource_reserved = 0;
        VkDeviceSize sampler_reserved_with_embedded = 0;
        /// the strides a descriptor of each kind occupies in the resource heap
        uint32_t buffer_descriptor_size = 0;
        uint32_t image_descriptor_size = 0;
        uint32_t sampler_descriptor_size = 0;
        /// the push-data window (vkCmdPushDataEXT) and how many embedded samplers fit
        uint32_t max_push_data = 0;
        uint32_t max_embedded_samplers = 0;
    };

    /**
     * @ingroup vulkan_core_descriptor_heap
     * @brief one resource heap and one sampler heap, with the calls that write and bind them
     */
    export class descriptor_heap {
    public:
        descriptor_heap() = default;
        descriptor_heap(descriptor_heap const&) = delete;
        descriptor_heap& operator=(descriptor_heap const&) = delete;
        descriptor_heap(descriptor_heap&&) = delete;
        descriptor_heap& operator=(descriptor_heap&&) = delete;
        ~descriptor_heap();

        /**
         * @brief allocate both heaps and load the entry points
         * @param allocator the device's allocator, which the heaps are buffers from
         * @param device the logical device
         * @param limits the device's heap properties (see heap_limits)
         * @return whether both heaps exist and every entry point was published; a false leaves nothing behind,
         *         so a caller can keep running with descriptor sets exactly as before
         * @note the sizes asked for are the WORKING sizes this renderer needs rather than the device's maxima:
         *       the scene set's descriptors are counted in hundreds of bytes, and the sampler heap's working
         *       size is the reserved range the embedded-sampler path requires.
         */
        [[nodiscard]] bool init(vma_allocator& allocator, VkDevice device, heap_limits const& limits) noexcept;
        /// @brief release both heaps (the buffers free themselves; this drops the references)
        void destroy() noexcept;

        [[nodiscard]] bool ready() const noexcept {
            return this->write_descriptors_ != nullptr && this->bind_resource_heap != nullptr && this->bind_sampler_heap != nullptr && this->resource_address_ != 0;
        }
        /// @brief where the resource heap begins, i.e. what a mapping's heapOffset is relative to
        [[nodiscard]] VkDeviceAddress resource_address() const noexcept {
            return this->resource_address_;
        }
        [[nodiscard]] VkDeviceAddress sampler_address() const noexcept {
            return this->sampler_address_;
        }
        [[nodiscard]] VkDeviceSize resource_size() const noexcept {
            return this->resource_size_;
        }
        [[nodiscard]] VkDeviceSize sampler_size() const noexcept {
            return this->sampler_size_;
        }
        [[nodiscard]] heap_limits const& limits() const noexcept {
            return this->limits_;
        }

        /**
         * @brief write descriptors into the resource heap
         * @param descriptors_offset the byte offset into the resource heap the first descriptor lands at
         * @param infos one VkResourceDescriptorInfoEXT per descriptor (a buffer is written as an address range)
         * @return whether the write happened (false when the heap is not ready or the range would overflow it)
         * @note the host range handed to vkWriteResourceDescriptorsEXT is the heap buffer itself, which is why
         *       the heap is HOST_VISIBLE: writing a descriptor is a memcpy into memory the driver also reads.
         */
        [[nodiscard]] bool write_descriptors(VkDeviceSize descriptors_offset, std::span<VkResourceDescriptorInfoEXT const> infos) noexcept;

        /**
         * @brief bind both heaps for the command buffer being recorded
         * @note the reserved range is what makes embedded samplers legal: a combined image sampler whose sampler
         *       lives in the resource heap needs the sampler heap's reserved window declared here.
         */
        void record_bind(VkCommandBuffer command_buffer) const noexcept;

        /// @brief the heap offset of the descriptor KIND @p type occupies, or 0 when the heap cannot hold it
        [[nodiscard]] uint32_t descriptor_stride(VkDescriptorType type) const noexcept;

        /// @brief the byte offset of descriptor @p index of the block that starts at @p block_offset
        [[nodiscard]] VkDeviceSize descriptor_offset(VkDeviceSize block_offset, uint32_t index, VkDescriptorType type) const noexcept {
            return block_offset + static_cast<VkDeviceSize>(index) * this->descriptor_stride(type);
        }

    private:
        VkDevice device = VK_NULL_HANDLE;
        vk_buffer resource_heap_ = {};
        vk_buffer sampler_heap_ = {};
        VkDeviceAddress resource_address_ = 0;
        VkDeviceAddress sampler_address_ = 0;
        VkDeviceSize resource_size_ = 0;
        VkDeviceSize sampler_size_ = 0;
        heap_limits limits_ = {};
        PFN_vkWriteResourceDescriptorsEXT write_descriptors_ = nullptr;
        PFN_vkCmdBindResourceHeapEXT bind_resource_heap = nullptr;
        PFN_vkCmdBindSamplerHeapEXT bind_sampler_heap = nullptr;
    };
} // namespace vulkan
