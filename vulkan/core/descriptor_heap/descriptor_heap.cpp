module; // the macro-using Vulkan header must not be imported into a module purview

#include <vulkan/vulkan.h>

// LOAD-BEARING, for the reason chores.cpp documents at length: with -fno-exceptions and the vendored std
// module, a TU that instantiates std::vector sees TWO 'operator new(size_t, align_val_t)' declarations - module
// std's and the textual libc++ copy baked into utility.data_block.pcm - and resolves neither. This file
// allocates (the zero-filled heap contents, and utility::log's formatting), and it died with an access
// violation inside the FIRST allocation it made until this include was added, with no log line, no validation
// message and no allocation error: exactly the shape of the ambiguous-operator-new failure, one step further
// along. Textually including glm merges the two copies. Do not remove this include to "clean up".
#include <fstream>
#include <glm/glm.hpp>

module vulkan.core.descriptor_heap;

import utility;

namespace vulkan {
    namespace {
        /// The working sizes. THIS RENDERER'S LAYOUT DECIDES THEM, not the device: the slot grid every
        /// heap-native shader addresses (see docs/descriptor_heap_migration.md) starts at a FIXED 1 MiB, so the
        /// resource heap has to be able to hold that base plus its slots - the first version asked for 256 KiB,
        /// which could not even fit the base, and the grid was refused with the heap left unused (measured). The
        /// sampler heap is capped at 128 KiB by the API, so its grid base is 64 KiB instead, which has to leave
        /// room for the samplers themselves - 64 KiB was entirely the reserved window the embedded-sampler path
        /// requires, i.e. no usable sampler space at all.
        constexpr VkDeviceSize resource_working_size = (1024u + 64u) * 1024u; // the 1 MiB grid base + its 64 KiB
        constexpr VkDeviceSize sampler_working_size = 128u * 1024u;           // the API's maximum (64 reserved + 64 usable)

        /// the VkBuffer behind a vk_buffer (the wrapper holds a VMA handle, not the Vulkan one)
        VkBuffer buffer_of(vma_allocator& allocator, vk_buffer const& buffer) noexcept {
            auto const* const detail = buffer.valid() ? allocator.get_buffer_detail(buffer.handle()) : nullptr;
            return detail != nullptr ? detail->buffer : VK_NULL_HANDLE;
        }

        VkDeviceAddress address_of(vma_allocator& allocator, VkDevice device, vk_buffer const& buffer) noexcept {
            VkBuffer const handle = buffer_of(allocator, buffer);
            if (handle == VK_NULL_HANDLE) {
                return 0;
            }
            VkBufferDeviceAddressInfo const info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = handle};
            return vkGetBufferDeviceAddress(device, &info);
        }
    } // namespace

    descriptor_heap::~descriptor_heap() {
        // The buffers are RAII, so there is nothing to free here - but the class is non-copyable and non-movable
        // on purpose (a heap is device state with an address other structures encode), and this definition is
        // where that intent is stated.
    }

    bool descriptor_heap::init(vma_allocator& allocator, VkDevice const device, heap_limits const& limits) noexcept {
        if (device == VK_NULL_HANDLE || limits.max_resource_size == 0 || limits.max_sampler_size == 0) {
            return false; // no device, or a device that published no heap limits: nothing to lay out
        }
        this->write_descriptors_ = reinterpret_cast<PFN_vkWriteResourceDescriptorsEXT>(vkGetDeviceProcAddr(device, "vkWriteResourceDescriptorsEXT"));
        this->bind_resource_heap = reinterpret_cast<PFN_vkCmdBindResourceHeapEXT>(vkGetDeviceProcAddr(device, "vkCmdBindResourceHeapEXT"));
        this->bind_sampler_heap = reinterpret_cast<PFN_vkCmdBindSamplerHeapEXT>(vkGetDeviceProcAddr(device, "vkCmdBindSamplerHeapEXT"));
        if (this->write_descriptors_ == nullptr || this->bind_resource_heap == nullptr || this->bind_sampler_heap == nullptr) {
            // The extension entry points come from the device rather than from the link line, the same rule the
            // acceleration-structure module follows: vulkan-1's import library exports no extension command.
            utility::log("descriptor heap: the device did not publish the heap entry points, so descriptor sets stay the binding model");
            return false;
        }
        this->limits_ = limits;
        this->device = device;

        // BOTH HEAPS ARE HOST_VISIBLE, and that is the model rather than a shortcut: a descriptor is written by
        // the application (vkWriteResourceDescriptorsEXT writes through a host range) and read by the device.
        // DESCRIPTOR_HEAP_BIT_EXT is what makes the buffer a legal heap at all, and the device address is what a
        // mapping's heapOffset is relative to.
        VkBufferUsageFlags const heap_usage = VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        this->resource_size_ = limits.max_resource_size < resource_working_size ? limits.max_resource_size : resource_working_size;
        this->sampler_size_ = limits.max_sampler_size < sampler_working_size ? limits.max_sampler_size : sampler_working_size;
        // ZERO-FILLED CONTENTS, for the ordinary reason that a heap holds no descriptors until one is written.
        // (The crash that led to this line was NOT the data pointer: it was ORDERING. A heap is two buffers from
        // the allocator, and core creates it after vma.init() for that reason - create_buffer before the
        // allocator exists is an access violation with no log line, no validation message and no allocation
        // error, which is why this took so long to find. See vulkan/core/core.cpp.)
        std::vector<unsigned char> const zeroed_resource(static_cast<std::size_t>(this->resource_size_), 0u);
        std::vector<unsigned char> const zeroed_sampler(static_cast<std::size_t>(this->sampler_size_), 0u);
        this->resource_heap_ = allocator.create_buffer(zeroed_resource.data(), zeroed_resource.size(), buffer_type::storage_coherent, heap_usage);
        this->sampler_heap_ = allocator.create_buffer(zeroed_sampler.data(), zeroed_sampler.size(), buffer_type::storage_coherent, heap_usage);
        if (!this->resource_heap_.valid() || !this->sampler_heap_.valid()) {
            utility::log("descriptor heap: the heap allocations failed, so descriptor sets stay the binding model");
            this->destroy();
            return false;
        }
        this->resource_address_ = address_of(allocator, device, this->resource_heap_);
        // THE MAPPED POINTER, which is NOT the address above: a descriptor is written through a HOST address
        // (VkHostAddressRangeEXT), so this is what write_descriptors needs. A heap that is not mapped cannot be
        // written by the host at all, so a missing mapping disables the heap instead of crashing on first write.
        auto const* const resource_detail = this->resource_heap_.valid() ? allocator.get_buffer_detail(this->resource_heap_.handle()) : nullptr;
        this->resource_mapped_ = resource_detail != nullptr ? resource_detail->allocation_info.pMappedData : nullptr;
        if (this->resource_mapped_ == nullptr) {
            utility::log("descriptor heap: the resource heap is not mapped, so descriptor sets stay the binding model");
            this->destroy();
            return false;
        }
        this->sampler_address_ = address_of(allocator, device, this->sampler_heap_);
        if (this->resource_address_ == 0 || this->sampler_address_ == 0) {
            utility::log("descriptor heap: the heap buffers have no device address, so descriptor sets stay the binding model");
            this->destroy();
            return false;
        }
        // The alignment is a property of the ADDRESS, not of the buffer, exactly as the micromap's data address
        // was: a heap binding whose address is not a multiple of resourceHeapAlignment is invalid. VMA's
        // device-side allocations are normally 256-byte aligned, which satisfies both numbers here (64 and 32),
        // so this is a check rather than a fixup - and a failed check disables the heap instead of binding it.
        if (limits.resource_alignment != 0 && (this->resource_address_ % limits.resource_alignment) != 0) {
            utility::log("descriptor heap: the resource heap address {} is not a multiple of the required alignment {}", this->resource_address_, limits.resource_alignment);
            this->destroy();
            return false;
        }
        if (limits.sampler_alignment != 0 && (this->sampler_address_ % limits.sampler_alignment) != 0) {
            utility::log("descriptor heap: the sampler heap address {} is not a multiple of the required alignment {}", this->sampler_address_, limits.sampler_alignment);
            this->destroy();
            return false;
        }

        utility::log("SUCCESS: descriptor heap created (resource {} KiB at 0x{:x}, sampler {} KiB at 0x{:x}; strides buffer {} B, image {} B, sampler {} B)",
                     this->resource_size_ / 1024,
                     this->resource_address_,
                     this->sampler_size_ / 1024,
                     this->sampler_address_,
                     limits.buffer_descriptor_size,
                     limits.image_descriptor_size,
                     limits.sampler_descriptor_size);
        return true;
    }

    void descriptor_heap::destroy() noexcept {
        this->resource_heap_ = {};
        this->sampler_heap_ = {};
        this->resource_address_ = 0;
        this->sampler_address_ = 0;
        this->resource_size_ = 0;
        this->sampler_size_ = 0;
        this->next_free_ = 0;
    }

    bool descriptor_heap::write_descriptors(VkDeviceSize const descriptors_offset, std::span<VkResourceDescriptorInfoEXT const> const infos) noexcept {
        if (!this->ready() || infos.empty() || descriptors_offset >= this->resource_size_) {
            return false;
        }
        // The host range is the heap's own memory at the offset the descriptors go to: the call writes
        // descriptorCount descriptors of the declared types there, which is why the destination must be big
        // enough for the sum of their strides.
        VkDeviceSize needed = 0;
        for (VkResourceDescriptorInfoEXT const& info : infos) {
            needed += this->descriptor_stride(info.type);
        }
        if (descriptors_offset + needed > this->resource_size_) {
            utility::log("descriptor heap: a write of {} descriptors ({} B) at offset {} does not fit the {} B resource heap", infos.size(), needed, descriptors_offset, this->resource_size_);
            return false;
        }
        // THE HOST RANGE IS A HOST POINTER, i.e. the heap's MAPPED memory at the offset the descriptors go to - not
        // the heap's device address re-cast as a pointer, which is what this did first and which crashes the
        // process as soon as a driver accepts the descriptor and writes through it.
        VkHostAddressRangeEXT host_range = {};
        host_range.address = static_cast<unsigned char*>(this->resource_mapped_) + descriptors_offset;
        host_range.size = static_cast<std::size_t>(needed);
        return this->write_descriptors_(this->device, static_cast<uint32_t>(infos.size()), infos.data(), &host_range) == VK_SUCCESS;
    }

    void descriptor_heap::record_bind(VkCommandBuffer const command_buffer) const noexcept {
        if (!this->ready() || command_buffer == VK_NULL_HANDLE) {
            return;
        }
        VkBindHeapInfoEXT resource_info = {};
        resource_info.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT;
        resource_info.heapRange.address = this->resource_address_;
        resource_info.heapRange.size = this->resource_size_;
        resource_info.reservedRangeOffset = 0;
        resource_info.reservedRangeSize = this->limits_.resource_reserved;
        this->bind_resource_heap(command_buffer, &resource_info);

        VkBindHeapInfoEXT sampler_info = {};
        sampler_info.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT;
        sampler_info.heapRange.address = this->sampler_address_;
        sampler_info.heapRange.size = this->sampler_size_;
        sampler_info.reservedRangeOffset = 0;
        // The reserved window is the reason a combined image sampler can live in the resource heap with its
        // sampler taken from here: minSamplerHeapReservedRangeWithEmbedded is the floor for exactly that.
        sampler_info.reservedRangeSize = this->limits_.sampler_reserved_with_embedded;
        this->bind_sampler_heap(command_buffer, &sampler_info);
    }

    uint32_t descriptor_heap::descriptor_stride(VkDescriptorType const type) const noexcept {
        switch (type) { // NOLINT(*-switch-missing-default-case) - the default below is what an unhandled type gets
        case VK_DESCRIPTOR_TYPE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            return this->limits_.image_descriptor_size;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            return this->limits_.buffer_descriptor_size;
        default:
            return this->limits_.image_descriptor_size; // textures dominate this renderer's bindings
        }
    }

    bool descriptor_heap::write_image(VkDeviceSize const offset_bytes, VkImageViewCreateInfo const& view, VkImageLayout const layout, VkDescriptorType const type) noexcept {
        VkImageDescriptorInfoEXT image_info = {};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT;
        image_info.pNext = nullptr;
        image_info.pView = &view; // the VIEW TO CREATE, not an existing VkImageView (see the header's note)
        image_info.layout = layout;
        VkResourceDescriptorInfoEXT const info = {.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
                                                  .pNext = nullptr,
                                                  .type = type,
                                                  .data = {.pImage = &image_info}};
        return this->write_descriptors(offset_bytes, std::span<VkResourceDescriptorInfoEXT const>(&info, 1));
    }

    bool descriptor_heap::write_buffer(VkDeviceSize const offset_bytes, VkDeviceAddress const address, VkDeviceSize const size, VkDescriptorType const type) noexcept {
        VkDeviceAddressRangeEXT const range = {.address = address, .size = size};
        VkResourceDescriptorInfoEXT const info = {.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
                                                  .pNext = nullptr,
                                                  .type = type,
                                                  .data = {.pAddressRange = &range}};
        return this->write_descriptors(offset_bytes, std::span<VkResourceDescriptorInfoEXT const>(&info, 1));
    }

    bool descriptor_heap::make_mapping(VkDescriptorSetAndBindingMappingEXT& mapping,
                                       uint32_t const set,
                                       uint32_t const first_binding,
                                       uint32_t const binding_count,
                                       uint32_t const heap_offset,
                                       uint32_t const array_stride,
                                       VkSpirvResourceTypeFlagsEXT const resource_mask,
                                       VkSamplerCreateInfo const* const embedded_sampler) const noexcept {
        if (!this->ready()) {
            return false;
        }
        mapping = VkDescriptorSetAndBindingMappingEXT{};
        mapping.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT;
        mapping.pNext = nullptr;
        mapping.descriptorSet = set;
        mapping.firstBinding = first_binding;
        mapping.bindingCount = binding_count;
        // The resource mask names the SHADER RESOURCE KINDS the range covers. It is the caller's because only the
        // caller knows what its shader declares there: a `sampler2D` binding is a combined sampled image, while a
        // `readonly buffer` is a read-only storage buffer - and the valid usage only forbids two mappings from
        // overlapping in BOTH range and mask, so the precise mask is what lets a later step map another kind over
        // the same range without a conflict.
        mapping.resourceMask = resource_mask;
        mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
        mapping.sourceData.constantOffset.heapOffset = heap_offset;
        mapping.sourceData.constantOffset.heapArrayStride = array_stride;
        // A combined image sampler takes its sampler from here (an EMBEDDED sampler) or from the sampler heap at
        // samplerHeapOffset; the sampler heap's reserved-with-embedded window is what makes the first form legal.
        mapping.sourceData.constantOffset.pEmbeddedSampler = embedded_sampler;
        mapping.sourceData.constantOffset.samplerHeapOffset = 0;
        mapping.sourceData.constantOffset.samplerHeapArrayStride = 0;
        return true;
    }

    VkDeviceSize descriptor_heap::reserve(uint32_t const count, VkDescriptorType const type) noexcept {
        if (!this->ready() || count == 0) {
            return VK_WHOLE_SIZE;
        }
        VkDeviceSize const stride = this->descriptor_stride(type);
        VkDeviceSize const offset = this->next_free_ != 0 ? this->next_free_ : this->usable_offset();
        VkDeviceSize const end = offset + stride * count;
        if (end > this->resource_size_) {
            utility::log("descriptor heap: a reservation of {} descriptors ({} B) does not fit the {} B resource heap", count, stride * count, this->resource_size_);
            return VK_WHOLE_SIZE;
        }
        this->next_free_ = end;
        return offset;
    }

    VkDeviceSize descriptor_heap::reserve_bytes(VkDeviceSize const bytes, VkDeviceSize const alignment) noexcept {
        if (!this->ready() || bytes == 0) {
            return VK_WHOLE_SIZE;
        }
        VkDeviceSize const step = alignment != 0 ? alignment : 1u;
        VkDeviceSize const cursor = this->next_free_ != 0 ? this->next_free_ : this->usable_offset();
        VkDeviceSize const offset = ((cursor + step - 1u) / step) * step;
        if (offset + bytes > this->resource_size_) {
            utility::log("descriptor heap: a reservation of {} B does not fit the {} B resource heap", bytes, this->resource_size_);
            return VK_WHOLE_SIZE;
        }
        this->next_free_ = offset + bytes;
        return offset;
    }
} // namespace vulkan
