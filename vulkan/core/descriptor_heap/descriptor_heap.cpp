module; // the macro-using Vulkan header must not be imported into a module purview

#include <vulkan/vulkan.h>

module vulkan.core.descriptor_heap;

import utility;

namespace vulkan {
    namespace {
        /// the working sizes: the scene set's descriptors are hundreds of bytes, and the sampler heap's
        /// meaningful floor is the reserved window the embedded-sampler path requires
        constexpr VkDeviceSize resource_working_size = 256u * 1024u;
        constexpr VkDeviceSize sampler_working_size = 64u * 1024u;

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
        this->resource_heap_ = allocator.create_buffer(nullptr, this->resource_size_, buffer_type::storage_coherent, heap_usage);
        this->sampler_heap_ = allocator.create_buffer(nullptr, this->sampler_size_, buffer_type::storage_coherent, heap_usage);
        if (!this->resource_heap_.valid() || !this->sampler_heap_.valid()) {
            utility::log("descriptor heap: the heap allocations failed, so descriptor sets stay the binding model");
            this->destroy();
            return false;
        }
        this->resource_address_ = address_of(allocator, device, this->resource_heap_);
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
        VkHostAddressRangeEXT host_range = {}; // the heap's own memory: a descriptor write is a memcpy the driver validates
        host_range.address = reinterpret_cast<void*>(static_cast<std::uintptr_t>(this->resource_address_ + descriptors_offset));
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
} // namespace vulkan
