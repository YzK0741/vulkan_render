module; // the macro-using Vulkan header must not be imported into a module purview

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
 *
 * AND THE UNIT OF THAT MIGRATION IS THE FRAME, NOT THE PASS - measured, not assumed. Binding the heap is
 * command-buffer state that takes over EVERY stage recorded after it: with the bind recorded at the start of the
 * frame and every mapping switched off (so the heap held nothing any shader had asked for), all nine reference
 * scenarios came back as the SAME frame - hash DC5F6D66428C26D8, mean 0.00 against the unlit reference's 88.1 -
 * with validation SILENT, because a stage whose descriptors came from a set reads the heap instead once one is
 * bound. Two further rules cost one gate run each and are not optional: a mapping is silently IGNORED unless the
 * pipeline is created with VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT (a flags2 bit, so it arrives through
 * VkPipelineCreateFlags2CreateInfo), and that flag REQUIRES a null VkPipelineLayout - the layout is precisely what
 * the mapping replaces. So a renderer maps every stage of a frame or none of them: a half-migrated frame does not
 * render a half-right picture, it renders nothing.
 */

export module vulkan.core.descriptor_heap;

import vstd;
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
         * @note THE DESTINATION IS A HOST ADDRESS, not the heap's device address:
         *       vkWriteResourceDescriptorsEXT takes VkHostAddressRangeEXT, i.e. a pointer into memory the host has
         *       mapped, and this class keeps the heap's mapped pointer for exactly that. (Writing through the
         *       DEVICE address - which the first version of this did - crashes the process the moment a driver
         *       accepts the descriptor and dereferences it, and validation says nothing about it because the
         *       address is just a void* to it.)
         */
        [[nodiscard]] bool write_descriptors(VkDeviceSize descriptors_offset, std::span<VkResourceDescriptorInfoEXT const> infos) noexcept;

        /**
         * @brief write SAMPLER descriptors into the sampler heap
         * @param descriptors_offset the byte offset in the SAMPLER heap the first descriptor lands at
         * @param samplers the VkSamplerCreateInfo of each sampler, in the order they are written
         * @return whether the write happened (false when the heap is not ready or the range would overflow it)
         * @note SAMPLERS ARE THE OTHER HEAP, and this is not a detail: the API splits resources and samplers into
         *       two heaps, so a shader that names a heap sampler at all needs the sampler heap bound as well (see
         *       record_bind). The create info is what a heap descriptor carries - exactly as a heap IMAGE
         *       descriptor carries a view create info - because the driver creates the object inside the heap.
         */
        [[nodiscard]] bool write_samplers(VkDeviceSize descriptors_offset, std::span<VkSamplerCreateInfo const> samplers) noexcept;

        /**
         * @brief bind both heaps for the command buffer being recorded
         * @note the reserved range is what makes embedded samplers legal: a combined image sampler whose sampler
         *       lives in the resource heap needs the sampler heap's reserved window declared here.
         */
        void record_bind(VkCommandBuffer command_buffer) const noexcept;

        /// @brief the heap offset of the descriptor KIND @p type occupies, or 0 when the heap cannot hold it
        [[nodiscard]] uint32_t descriptor_stride(VkDescriptorType type) const noexcept;

        /**
         * @brief write ONE image descriptor (a SAMPLED image) into the resource heap
         * @param offset_bytes the byte offset in the heap, e.g. `descriptor_offset(block, index, type)`
         * @param view the VIEW TO CREATE - and this is the model difference worth knowing: a heap image
         *        descriptor carries a VkImageViewCreateInfo, not an existing VkImageView, because the driver
         *        creates the view inside the descriptor. A descriptor-set path that already holds a view has to
         *        keep the create info it made that view from in order to write the same binding here.
         * @param layout the layout the image will be in when sampled
         * @param type VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE or VK_DESCRIPTOR_TYPE_STORAGE_IMAGE - and NOT
         *        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, which the resource heap does not accept at all
         *        (VUID-VkResourceDescriptorInfoEXT-type-11210 lists the kinds a heap descriptor may be, and a
         *        combined image sampler is not among them): the heap holds the IMAGE, and the sampler comes from
         *        the mapping - an embedded sampler, or a sampler-heap offset - which is what
         *        minSamplerHeapReservedRangeWithEmbedded exists for.
         */
        [[nodiscard]] bool write_image(VkDeviceSize offset_bytes, VkImageViewCreateInfo const& view, VkImageLayout layout, VkDescriptorType type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) noexcept;
        /// @brief write ONE buffer descriptor (its device address range) into the resource heap
        [[nodiscard]] bool write_buffer(VkDeviceSize offset_bytes, VkDeviceAddress address, VkDeviceSize size, VkDescriptorType type) noexcept;

        /**
         * @brief build a mapping that makes EXISTING shaders read @p first_binding .. of @p set from the heap
         * @param mapping the output entry
         * @param set the descriptor set number the shaders already declare
         * @param first_binding the first binding of the range
         * @param binding_count how many consecutive bindings the range covers
         * @param heap_offset the byte offset in the resource heap the range starts at
         * @param array_stride the stride between elements of a descriptor ARRAY, or 0 for a single descriptor
         * @param embedded_sampler the sampler a combined image sampler uses (may be null for image-only kinds)
         * @return whether the entry was written
         * @note this is what keeps the GLSL untouched: the mapping is chained into each
         *       VkPipelineShaderStageCreateInfo and resolves `layout(set = set, binding = ...)` to heap memory.
         */
        [[nodiscard]] bool make_mapping(VkDescriptorSetAndBindingMappingEXT& mapping,
                                        uint32_t set,
                                        uint32_t first_binding,
                                        uint32_t binding_count,
                                        uint32_t heap_offset,
                                        uint32_t array_stride,
                                        VkSpirvResourceTypeFlagsEXT resource_mask,
                                        VkSamplerCreateInfo const* embedded_sampler) const noexcept;

        /**
         * @brief reserve @p count descriptors of @p type in the resource heap and return their byte offset
         * @return the offset, or VK_WHOLE_SIZE when the heap is not ready or the reservation does not fit
         *
         * @note THIS IS WHAT KEEPS THE LAYOUT AND THE MAPPINGS FROM DISAGREEING. A caller that computes an offset
         *       by hand and a mapping built from another number is the silent failure this design has - the heap
         *       reads memory that was never written, validation says nothing, and the picture is simply wrong. So
         *       the offsets are RESERVED here, once, and both the write and the mapping use the reserved number.
         */
        [[nodiscard]] VkDeviceSize reserve(uint32_t count, VkDescriptorType type) noexcept;

        /**
         * @brief reserve @p bytes of the resource heap and return the offset, aligned to @p alignment
         * @return the offset, or VK_WHOLE_SIZE when the heap is not ready or the reservation does not fit
         * @note this is the form a MIXED block needs: a scene set holds descriptors of several kinds - buffers at
         *       one stride, images at another - so a block is laid out by hand and reserved as bytes, while the
         *       single-kind case above stays the convenient one.
         */
        [[nodiscard]] VkDeviceSize reserve_bytes(VkDeviceSize bytes, VkDeviceSize alignment) noexcept;

        /// @brief the byte offset of descriptor @p index of the block that starts at @p block_offset
        [[nodiscard]] VkDeviceSize descriptor_offset(VkDeviceSize block_offset, uint32_t index, VkDescriptorType type) const noexcept {
            return block_offset + static_cast<VkDeviceSize>(index) * this->descriptor_stride(type);
        }

        /**
         * @brief where a caller may START placing its own descriptors: past the implementation's reserved window
         *
         * @note THIS IS NOT OPTIONAL BOOKKEEPING. minResourceHeapReservedRange is memory the implementation reserves
         *       for itself (the embedded-sampler machinery lives in it), and descriptors written INSIDE it are not
         *       read back as written - which is not a validation error, it is a wrong picture. The first version of
         *       this renderer's texture array was written at offset 0 and every one of the nine reference frames
         *       changed, with validation silent; the mapping's heapOffset and the write offset both come from here
         *       for that reason, and they must stay the same number.
         */
        [[nodiscard]] VkDeviceSize usable_offset() const noexcept {
            VkDeviceSize const alignment = this->limits_.resource_alignment != 0 ? this->limits_.resource_alignment : 1u;
            return ((this->limits_.resource_reserved + alignment - 1u) / alignment) * alignment;
        }

    private:
        VkDevice device = VK_NULL_HANDLE;
        vk_buffer resource_heap_ = {};
        vk_buffer sampler_heap_ = {};
        /// the MAPPED pointer of the resource heap: vkWriteResourceDescriptorsEXT writes through a HOST address
        void* resource_mapped_ = nullptr;
        /// the same for the sampler heap, whose descriptors are written by vkWriteSamplerDescriptorsEXT
        void* sampler_mapped_ = nullptr;
        VkDeviceAddress resource_address_ = 0;
        VkDeviceAddress sampler_address_ = 0;
        VkDeviceSize resource_size_ = 0;
        VkDeviceSize sampler_size_ = 0;
        /// the bump pointer for reserve(), which starts past the implementation's reserved window
        VkDeviceSize next_free_ = 0;
        heap_limits limits_ = {};
        PFN_vkWriteResourceDescriptorsEXT write_descriptors_ = nullptr;
        PFN_vkWriteSamplerDescriptorsEXT write_samplers_ = nullptr;
        PFN_vkCmdBindResourceHeapEXT bind_resource_heap = nullptr;
        PFN_vkCmdBindSamplerHeapEXT bind_sampler_heap = nullptr;
    };
} // namespace vulkan
