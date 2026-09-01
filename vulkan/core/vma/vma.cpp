module;

#include <vulkan/vulkan.h>
#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

module vulkan.core.vma;

namespace {
    constexpr uint32_t sizeof_vk_format(VkFormat const format) {
        switch (format) {
        // 8-bit single channel
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_SNORM:
        case VK_FORMAT_R8_USCALED:
        case VK_FORMAT_R8_SSCALED:
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8_SINT:
        case VK_FORMAT_R8_SRGB:
            return 1;

        // 16-bit single channel / 8-bit dual channel
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R16_SNORM:
        case VK_FORMAT_R16_USCALED:
        case VK_FORMAT_R16_SSCALED:
        case VK_FORMAT_R16_UINT:
        case VK_FORMAT_R16_SINT:
        case VK_FORMAT_R16_SFLOAT:
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8_SNORM:
        case VK_FORMAT_R8G8_USCALED:
        case VK_FORMAT_R8G8_SSCALED:
        case VK_FORMAT_R8G8_UINT:
        case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R8G8_SRGB:
            return 2;

        // 24-bit
        case VK_FORMAT_R8G8B8_UNORM:
        case VK_FORMAT_R8G8B8_SNORM:
        case VK_FORMAT_R8G8B8_USCALED:
        case VK_FORMAT_R8G8B8_SSCALED:
        case VK_FORMAT_R8G8B8_UINT:
        case VK_FORMAT_R8G8B8_SINT:
        case VK_FORMAT_R8G8B8_SRGB:
            return 3;

        // 32-bit
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R32_SINT:
        case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_R16G16_UNORM:
        case VK_FORMAT_R16G16_SNORM:
        case VK_FORMAT_R16G16_USCALED:
        case VK_FORMAT_R16G16_SSCALED:
        case VK_FORMAT_R16G16_UINT:
        case VK_FORMAT_R16G16_SINT:
        case VK_FORMAT_R16G16_SFLOAT:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_R8G8B8A8_USCALED:
        case VK_FORMAT_R8G8B8A8_SSCALED:
        case VK_FORMAT_R8G8B8A8_UINT:
        case VK_FORMAT_R8G8B8A8_SINT:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SNORM:
        case VK_FORMAT_B8G8R8A8_USCALED:
        case VK_FORMAT_B8G8R8A8_SSCALED:
        case VK_FORMAT_B8G8R8A8_UINT:
        case VK_FORMAT_B8G8R8A8_SINT:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_A2B10G10R10_UINT_PACK32:
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        case VK_FORMAT_A2R10G10B10_UINT_PACK32:
            return 4;

        // 64-bit
        case VK_FORMAT_R64_UINT:
        case VK_FORMAT_R64_SINT:
        case VK_FORMAT_R64_SFLOAT:
        case VK_FORMAT_R32G32_UINT:
        case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_R32G32_SFLOAT:
        case VK_FORMAT_R16G16B16A16_UNORM:
        case VK_FORMAT_R16G16B16A16_SNORM:
        case VK_FORMAT_R16G16B16A16_USCALED:
        case VK_FORMAT_R16G16B16A16_SSCALED:
        case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            return 8;

        // 96-bit
        case VK_FORMAT_R32G32B32_UINT:
        case VK_FORMAT_R32G32B32_SINT:
        case VK_FORMAT_R32G32B32_SFLOAT:
            return 12;

        // 128-bit
        case VK_FORMAT_R64G64_UINT:
        case VK_FORMAT_R64G64_SINT:
        case VK_FORMAT_R64G64_SFLOAT:
        case VK_FORMAT_R32G32B32A32_UINT:
        case VK_FORMAT_R32G32B32A32_SINT:
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            return 16;

        // Depth/stencil
        case VK_FORMAT_D16_UNORM:
            return 2;
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT:
            return 4;
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
        case VK_FORMAT_S8_UINT:
        // BC compressed formats
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
            return 8;

        case VK_FORMAT_BC2_UNORM_BLOCK:
        case VK_FORMAT_BC2_SRGB_BLOCK:
        case VK_FORMAT_BC3_UNORM_BLOCK:
        case VK_FORMAT_BC3_SRGB_BLOCK:
        case VK_FORMAT_BC4_UNORM_BLOCK:
        case VK_FORMAT_BC4_SNORM_BLOCK:
        case VK_FORMAT_BC5_UNORM_BLOCK:
        case VK_FORMAT_BC5_SNORM_BLOCK:
        case VK_FORMAT_BC6H_UFLOAT_BLOCK:
        case VK_FORMAT_BC6H_SFLOAT_BLOCK:
        case VK_FORMAT_BC7_UNORM_BLOCK:
        case VK_FORMAT_BC7_SRGB_BLOCK:
        // ASTC compressed formats (all are 16 bytes/block)
        case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
        case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:
        case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:
        case VK_FORMAT_ASTC_5x4_SRGB_BLOCK:
        case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_5x5_SRGB_BLOCK:
        case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_6x5_SRGB_BLOCK:
        case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:
        case VK_FORMAT_ASTC_6x6_SRGB_BLOCK:
        case VK_FORMAT_ASTC_8x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_8x5_SRGB_BLOCK:
        case VK_FORMAT_ASTC_8x6_UNORM_BLOCK:
        case VK_FORMAT_ASTC_8x6_SRGB_BLOCK:
        case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:
        case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:
        case VK_FORMAT_ASTC_10x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x5_SRGB_BLOCK:
        case VK_FORMAT_ASTC_10x6_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x6_SRGB_BLOCK:
        case VK_FORMAT_ASTC_10x8_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x8_SRGB_BLOCK:
        case VK_FORMAT_ASTC_10x10_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x10_SRGB_BLOCK:
        case VK_FORMAT_ASTC_12x10_UNORM_BLOCK:
        case VK_FORMAT_ASTC_12x10_SRGB_BLOCK:
        case VK_FORMAT_ASTC_12x12_UNORM_BLOCK:
        case VK_FORMAT_ASTC_12x12_SRGB_BLOCK:
            return 16;

        // ETC2 / EAC compressed formats (8 bytes/block)
        case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
        case VK_FORMAT_EAC_R11_UNORM_BLOCK:
        case VK_FORMAT_EAC_R11_SNORM_BLOCK:
        case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
        case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
            return 8;

        default:
            return 0;
        }
    }

    constexpr VmaAllocationCreateInfo get_allocation_info_from_type(vulkan::buffer_type const type) {
        VmaAllocationCreateInfo info = {};
        switch (type) {
        case vulkan::buffer_type::vertex:
            [[fallthrough]];
        case vulkan::buffer_type::index:
            [[fallthrough]];
        case vulkan::buffer_type::uniform_gpu_only: {
            info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            break;
        }
        case vulkan::buffer_type::uniform_coherent: {
            info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            break;
        }
        case vulkan::buffer_type::uniform_cached: {
            info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
            info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            break;
        }
        case vulkan::buffer_type::storage_coherent: {
            info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            break;
        }
        }
        return info;
    }

    constexpr VkBufferCreateInfo get_create_info_from_type(vulkan::buffer_type const type) {
        VkBufferCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

        switch (type) {
        case vulkan::buffer_type::vertex: {
            info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            break;
        }
        case vulkan::buffer_type::index: {
            info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            break;
        }
        case vulkan::buffer_type::uniform_gpu_only: {
            info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            break;
        }
        case vulkan::buffer_type::uniform_coherent:
        case vulkan::buffer_type::uniform_cached: {
            info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            break;
        }
        case vulkan::buffer_type::storage_coherent: {
            info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            break;
        }
        }
        return info;
    }

    constexpr VmaAllocationCreateInfo get_image_allocation_info_from_type(vulkan::image_type const type) {
        VmaAllocationCreateInfo info = {};

        switch (type) {
        case vulkan::image_type::texture_2d:
        case vulkan::image_type::texture_2d_color:
        case vulkan::image_type::texture_2d_depth:
        case vulkan::image_type::texture_cubemap:
        case vulkan::image_type::render_target: {
            info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            break;
        }
        case vulkan::image_type::texture_2d_staging: {
            info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            break;
        }
        }
        return info;
    }

    constexpr VkImageCreateInfo get_image_create_info_from_type(
        vulkan::image_type const type,
        vulkan::image_create_info const& info) {
        VkImageCreateInfo image_info = {};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.extent.width = info.width;
        image_info.extent.height = info.height;
        image_info.extent.depth = 1;
        image_info.mipLevels = info.mip_levels;
        image_info.arrayLayers = info.array_layers;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.format = info.format;

        switch (type) {
        case vulkan::image_type::texture_2d:
        case vulkan::image_type::texture_2d_staging:
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT |
                               info.extra_usage;
            break;

        case vulkan::image_type::texture_2d_color:
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT |
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                               info.extra_usage;
            break;

        case vulkan::image_type::texture_2d_depth:
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                               info.extra_usage;
            break;

        case vulkan::image_type::texture_cubemap:
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
            image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT |
                               info.extra_usage;
            break;

        case vulkan::image_type::render_target:
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT |
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                               info.extra_usage;
            break;
        }

        return image_info;
    }
} // namespace

namespace vulkan {
    void vma_allocator::init(
        VkInstance const instance,              // NOLINT(*-misplaced-const)
        VkDevice const device,                  // NOLINT(*-misplaced-const)
        VkPhysicalDevice const physical_device, // NOLINT(*-misplaced-const)
        VkQueue const queue,                    // NOLINT(*-misplaced-const)
        uint32_t const queue_family_index) {

        std::lock_guard guard(this->access_mutex);
        if (this->allocator != VK_NULL_HANDLE) {
            return;
        }

        VmaAllocatorCreateInfo vma_allocator_create_info = {};
        vma_allocator_create_info.instance = instance;
        vma_allocator_create_info.device = device;
        vma_allocator_create_info.physicalDevice = physical_device;
        vma_allocator_create_info.flags = 0;

        vmaCreateAllocator(&vma_allocator_create_info, &this->allocator);

        this->device = device;
        this->queue = queue;
        this->queue_family_index = queue_family_index;
        this->command_cache.push_back(this->create_command_pair());
    }

    void vma_allocator::destroy() {
        if (this->allocator != VK_NULL_HANDLE) {
            std::lock_guard guard(this->access_mutex);
            for (auto const& buffer : buffers | std::ranges::views::values) {
                vmaDestroyBuffer(this->allocator, buffer.buffer, buffer.allocation);
            }
            this->buffers.clear();

            for (auto const& image : images | std::ranges::views::values) {
                vmaDestroyImage(this->allocator, image.image, image.allocation);
            }
            this->images.clear();

            for (auto const& fence : this->fence_cache) {
                vkDestroyFence(this->device, fence, nullptr);
            }
            this->fence_cache.clear();
            for (auto& command_pool : this->command_cache | std::views::keys) {
                vkDestroyCommandPool(this->device, command_pool, nullptr);
                command_pool = VK_NULL_HANDLE;
            }
            // Release the cached staging buffer
            if (this->staging_cache.buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(this->allocator, this->staging_cache.buffer, this->staging_cache.allocation);
                this->staging_cache = {};
                this->buffer_size = 0;
            }

            vmaDestroyAllocator(this->allocator);
            this->allocator = VK_NULL_HANDLE;
        }
    }

    VkFence vma_allocator::create_fence() const {

        VkFence fence = VK_NULL_HANDLE;

        VkFenceCreateInfo fence_create_info = {};
        fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(this->device, &fence_create_info, nullptr, &fence);

        return fence;
    }

    bool vma_allocator::ensure_staging_buffer(VkDeviceSize const size, VkBuffer& buffer, VmaAllocation& allocation, VmaAllocationInfo& info) {
        // Reuse if the cached capacity is sufficient
        if (this->staging_cache.buffer != VK_NULL_HANDLE && this->buffer_size >= size) {
            buffer = this->staging_cache.buffer;
            allocation = this->staging_cache.allocation;
            info = this->staging_cache.allocation_info;
            return true;
        }

        // Not enough capacity: destroy the current cache and re-create
        if (this->staging_cache.buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(this->allocator, this->staging_cache.buffer, this->staging_cache.allocation);
            this->staging_cache = {};
            this->buffer_size = 0;
        }

        VkBufferCreateInfo staging_create_info = {};
        staging_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        staging_create_info.size = size;
        staging_create_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo staging_alloc_info = {};
        staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        staging_alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                                   VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VkResult const result = vmaCreateBuffer(
            this->allocator,
            &staging_create_info,
            &staging_alloc_info,
            &buffer,
            &allocation,
            &info);

        if (result != VK_SUCCESS) {
            utility::error("Failed to create staging buffer: {}", static_cast<int>(result));
            return false;
        }

        this->staging_cache = {buffer, allocation, info};
        this->buffer_size = size;
        return true;
    }

    bool vma_allocator::direct_upload(VmaAllocation const& allocation, VmaAllocationInfo& allocation_info, void const* data, VkDeviceSize const size) const {
        void* mapped_data = nullptr;
        if (VkResult const result = vmaMapMemory(this->allocator, allocation, &mapped_data); result != VK_SUCCESS) {
            utility::error("Failed to map memory: {}", static_cast<int>(result));
            return false;
        }

        memcpy(mapped_data, data, size);

        vmaGetAllocationInfo(this->allocator, allocation, &allocation_info);
        if ((allocation_info.memoryType & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            vmaFlushAllocation(this->allocator, allocation, 0, size);
        }

        vmaUnmapMemory(this->allocator, allocation);
        return true;
    }

    bool vma_allocator::staging_upload(VkBuffer const dst_buffer, void const* data, VkDeviceSize const size) { // NOLINT(*-misplaced-const)
        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VmaAllocation staging_allocation = VK_NULL_HANDLE;
        VmaAllocationInfo staging_info = {};

        // The staging buffer is shared; it must be held exclusively during writes + GPU copies
        std::lock_guard staging_guard(this->staging_mutex);

        // Reuse the cached staging buffer, rebuilding automatically when too small
        if (!this->ensure_staging_buffer(size, staging_buffer, staging_allocation, staging_info)) {
            return false;
        }

        // Copy data
        if (staging_info.pMappedData) {
            memcpy(staging_info.pMappedData, data, size);
            if ((staging_info.memoryType & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
                vmaFlushAllocation(this->allocator, staging_allocation, 0, size);
            }
        } else {
            return false;
        }
        // Execute the copy command (cache access guarded by cache_mutex)
        std::pair<VkCommandPool, VkCommandBuffer> command_pair;
        VkFence fence = VK_NULL_HANDLE;
        {
            std::lock_guard guard(this->cache_mutex);
            if (!this->command_cache.empty()) {
                command_pair = this->command_cache.back();
                this->command_cache.pop_back();
            } else {
                command_pair = this->create_command_pair();
            }

            if (!this->fence_cache.empty()) {
                fence = this->fence_cache.back();
                this->fence_cache.pop_back();
            } else {
                fence = this->create_fence();
            }
        }

        VkCommandBuffer command_buffer = command_pair.second;

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(command_buffer, &begin_info);

        VkBufferCopy copy_region = {};
        copy_region.size = size;
        vkCmdCopyBuffer(command_buffer, staging_buffer, dst_buffer, 1, &copy_region);

        vkEndCommandBuffer(command_buffer);

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        {
            // VkQueue is externally synchronized; submits must be serialized
            std::lock_guard guard(this->queue_mutex);
            vkQueueSubmit(this->queue, 1, &submit_info, fence);
        }

        vkWaitForFences(this->device, 1, &fence, VK_TRUE, UINT64_MAX);

        // Cleanup
        vkResetFences(this->device, 1, &fence);
        vkResetCommandBuffer(command_buffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
        {
            std::lock_guard guard(this->cache_mutex);
            this->fence_cache.push_back(fence);
            this->command_cache.push_back(command_pair);
        }

        return true;
    }

    bool vma_allocator::direct_image_upload(VmaAllocation const allocation, void const* data, VkDeviceSize const size) const { // NOLINT(*-misplaced-const)
        void* mapped_data = nullptr;
        if (VkResult const result = vmaMapMemory(this->allocator, allocation, &mapped_data); result != VK_SUCCESS) {
            utility::error("Failed to map image memory: {}", static_cast<int>(result));
            return false;
        }

        memcpy(mapped_data, data, size);

        VmaAllocationInfo alloc_info;
        vmaGetAllocationInfo(this->allocator, allocation, &alloc_info);
        if ((alloc_info.memoryType & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            vmaFlushAllocation(this->allocator, allocation, 0, size);
        }

        vmaUnmapMemory(this->allocator, allocation);
        return true;
    }

    bool vma_allocator::staging_image_upload(VkImage dst_image, void const* data, VkDeviceSize size, image_create_info const& info) {
        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VmaAllocation staging_allocation = VK_NULL_HANDLE;
        VmaAllocationInfo staging_info = {};

        // The staging buffer is shared; it must be held exclusively during writes + GPU copies
        std::lock_guard staging_guard(this->staging_mutex);

        // Reuse the cached staging buffer, rebuilding automatically when too small
        if (!this->ensure_staging_buffer(size, staging_buffer, staging_allocation, staging_info)) {
            return false;
        }

        // Copy data into the staging buffer
        if (staging_info.pMappedData) {
            memcpy(staging_info.pMappedData, data, size);
            if ((staging_info.memoryType & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
                vmaFlushAllocation(this->allocator, staging_allocation, 0, size);
            }
        } else {
            return false;
        }
        // Execute the copy command (cache access guarded by cache_mutex)
        std::pair<VkCommandPool, VkCommandBuffer> command_pair;
        VkFence fence = VK_NULL_HANDLE;
        {
            std::lock_guard guard(this->cache_mutex);
            if (!this->command_cache.empty()) {
                command_pair = this->command_cache.back();
                this->command_cache.pop_back();
            } else {
                command_pair = this->create_command_pair();
            }

            if (!this->fence_cache.empty()) {
                fence = this->fence_cache.back();
                this->fence_cache.pop_back();
            } else {
                fence = this->create_fence();
            }
        }
        VkCommandBuffer command_buffer = command_pair.second;

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(command_buffer, &begin_info);

        // ========== Fix point 1: correct layout transition order ==========
        // Step 1: UNDEFINED -> TRANSFER_DST_OPTIMAL
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = dst_image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = info.mip_levels;
        barrier.subresourceRange.layerCount = info.array_layers;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        // Step 2: copy per mip.
        // Data is laid out in mip-major order (all layers of mip0 -> all layers of mip1 -> ...),
        // with layers contiguous within each mip (face0, face1, ...), located via bufferOffset.
        uint32_t const bytes_per_pixel = sizeof_vk_format(info.format);
        std::vector<VkBufferImageCopy> regions;
        regions.reserve(info.mip_levels);
        VkDeviceSize buffer_offset = 0;
        for (uint32_t mip = 0; mip < info.mip_levels; ++mip) {
            uint32_t const mip_width = std::max(1u, info.width >> mip);
            uint32_t const mip_height = std::max(1u, info.height >> mip);

            VkBufferImageCopy region = {};
            region.bufferOffset = buffer_offset;
            region.bufferRowLength = 0; // 0 means tightly packed
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = mip;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = info.array_layers;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {mip_width, mip_height, 1};
            regions.push_back(region);

            buffer_offset += static_cast<VkDeviceSize>(mip_width) * mip_height * info.array_layers * bytes_per_pixel;
        }

        vkCmdCopyBufferToImage(
            command_buffer,
            staging_buffer,
            dst_image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<uint32_t>(regions.size()),
            regions.data());

        // Step 3: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        vkEndCommandBuffer(command_buffer);

        // Submit and wait for completion
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        {
            // VkQueue is externally synchronized; submits must be serialized
            std::lock_guard guard(this->queue_mutex);
            vkQueueSubmit(this->queue, 1, &submit_info, fence);
        }

        vkWaitForFences(this->device, 1, &fence, VK_TRUE, UINT64_MAX);

        // Cleanup
        vkResetFences(this->device, 1, &fence);
        vkResetCommandBuffer(command_buffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
        {
            std::lock_guard guard(this->cache_mutex);
            this->fence_cache.push_back(fence);
            this->command_cache.push_back(command_pair);
        }

        return true;
    }

    uint64_t vma_allocator::create_buffer(unsigned char const* data, uint64_t const size_byte, buffer_type const type) {
        uint64_t handle = 0;
        // distribute() locks internally (enable_handle_distribute::access_mutex); no outer lock needed
        if (auto const result = this->distribute(); result) {
            handle = result.value();
        } else {
            return handle;
        }

        auto const allocation_create_info = get_allocation_info_from_type(type);
        auto buffer_create_info = get_create_info_from_type(type);
        buffer_create_info.size = size_byte;

        VmaAllocation allocation = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocationInfo alloc_info = {};

        // VMA is internally thread-safe; allocation and upload need no access_mutex
        VkResult const result = vmaCreateBuffer(
            this->allocator,
            &buffer_create_info,
            &allocation_create_info,
            &buffer,
            &allocation,
            &alloc_info);

        if (result != VK_SUCCESS) {
            utility::error("Failed to create buffer: {}", static_cast<int>(result));
            this->recycle(handle);
            return 0;
        }

        bool upload_success = false;
        switch (type) {
        case buffer_type::uniform_coherent:
        case buffer_type::uniform_cached:
        case buffer_type::storage_coherent:
            // Upload via direct mapping
            upload_success = direct_upload(allocation, alloc_info, data, size_byte);
            break;

        case buffer_type::vertex:
        case buffer_type::index:
        case buffer_type::uniform_gpu_only:
            // Use a staging buffer
            upload_success = staging_upload(buffer, data, size_byte);
            break;
        }

        if (!upload_success) {
            vmaDestroyBuffer(this->allocator, buffer, allocation);
            this->recycle(handle);
            utility::panic("Failed to upload buffer");
        }

        {
            std::lock_guard guard(this->access_mutex);
            // built on the stack, then moved in (use_count keeps its default 1)
            buffer_detail detail;
            detail.buffer = buffer;
            detail.allocation = allocation;
            detail.allocation_info = alloc_info;
            this->buffers.emplace(handle, std::move(detail));
        }
        return handle;
    }

    uint64_t vma_allocator::create_image(unsigned char const* data, uint64_t const size_byte, image_create_info const& create_info, image_type const type) {
        // sha256 is pure CPU work; keep it outside the critical section. It is computed before
        // allocating so a content hit can reuse an existing image without any allocation or upload.
        auto const digest = utility::sha256(std::span(data, size_byte));
        if (!digest) {
            utility::panic("sha256 failed");
        }

        // Only immutable, data-uploaded textures are shareable: depth / staging / render targets
        // must stay distinct even with identical parameters.
        constexpr auto is_dedupable = [](image_type const t) {
            return t == image_type::texture_2d || t == image_type::texture_2d_color || t == image_type::texture_cubemap;
        };
        if (is_dedupable(type)) {
            std::lock_guard guard(this->access_mutex);
            for (auto& [existing_handle, detail] : this->images) {
                if (detail.type == type && detail.create_info == create_info && detail.digest == digest.value()) {
                    detail.use_count.fetch_add(1); // shared: bump the reference count and reuse
                    return existing_handle;
                }
            }
        }

        uint64_t handle = 0;
        // distribute() locks internally (enable_handle_distribute::access_mutex); no outer lock needed
        if (auto const result = this->distribute(); result) {
            handle = result.value();
        } else {
            return handle;
        }

        VkDeviceSize const image_size = size_byte;

        // Expected size = array_layers * sum of all mip sizes * bytes per pixel
        VkDeviceSize expected_size = 0;
        for (uint32_t mip = 0; mip < create_info.mip_levels; ++mip) {
            expected_size += static_cast<VkDeviceSize>(std::max(1u, create_info.width >> mip)) *
                             std::max(1u, create_info.height >> mip) *
                             sizeof_vk_format(create_info.format);
        }
        expected_size *= create_info.array_layers;
        if (expected_size != image_size) {
            utility::log("incorrect image size [{}], expected [{}]", image_size, expected_size);
        }

        auto const alloc_info = get_image_allocation_info_from_type(type);
        auto image_create_info = get_image_create_info_from_type(type, create_info);
        image_create_info.mipLevels = create_info.mip_levels;
        image_create_info.extent.width = create_info.width;
        image_create_info.extent.height = create_info.height;
        image_create_info.extent.depth = 1;
        image_create_info.arrayLayers = create_info.array_layers;

        VmaAllocation allocation = VK_NULL_HANDLE;
        VkImage image = VK_NULL_HANDLE;
        VmaAllocationInfo alloc_detail = {};

        VkResult const vk_result = vmaCreateImage(
            this->allocator,
            &image_create_info,
            &alloc_info,
            &image,
            &allocation,
            &alloc_detail);

        if (vk_result != VK_SUCCESS) {
            utility::error("Failed to create image: {}", static_cast<int>(vk_result));
            this->recycle(handle);
            return 0;
        }

        // Pick the upload path based on the type
        bool upload_success = false;
        switch (type) {
        case image_type::texture_2d_staging:
            upload_success = direct_image_upload(allocation, data, image_size);
            break;

        case image_type::texture_2d:
        case image_type::texture_2d_color:
        case image_type::texture_cubemap:
        case image_type::render_target:
            upload_success = staging_image_upload(image, data, image_size, create_info);
            break;

        case image_type::texture_2d_depth:
            upload_success = true;
            break;
        }

        if (!upload_success) {
            vmaDestroyImage(this->allocator, image, allocation);
            this->recycle(handle);
            return 0;
        }

        {
            std::lock_guard guard(this->access_mutex);
            // built on the stack, then moved in (use_count keeps its default 1)
            image_detail detail;
            detail.image = image;
            detail.allocation = allocation;
            detail.allocation_info = alloc_detail;
            detail.digest = digest.value();
            detail.create_info = create_info;
            detail.type = type;
            this->images.emplace(handle, std::move(detail));
        }
        return handle;
    }

    buffer_detail const* vma_allocator::get_buffer_detail(uint64_t const handle) {
        std::lock_guard guard(this->access_mutex);
        if (this->buffers.contains(handle)) {
            return &this->buffers[handle];
        }
        return nullptr;
    }

    image_detail const* vma_allocator::get_image_detail(uint64_t const handle) {
        std::lock_guard guard(this->access_mutex);
        if (this->images.contains(handle)) {
            return &this->images[handle];
        }
        return nullptr;
    }

    void vma_allocator::free_buffer(uint64_t const handle) {
        std::lock_guard guard(this->access_mutex);
        auto const it = this->buffers.find(handle);
        if (it == this->buffers.end()) {
            return;
        }
        // shared resources are freed by reference count: only the last free really destroys
        if (it->second.use_count.fetch_sub(1) > 1) {
            return;
        }
        vmaDestroyBuffer(this->allocator, it->second.buffer, it->second.allocation);
        this->buffers.erase(it);
    }

    void vma_allocator::free_image(uint64_t const handle) {
        std::lock_guard guard(this->access_mutex);
        auto const it = this->images.find(handle);
        if (it == this->images.end()) {
            return;
        }
        // shared images are freed by reference count: only the last free really destroys
        if (it->second.use_count.fetch_sub(1) > 1) {
            return;
        }
        vmaDestroyImage(this->allocator, it->second.image, it->second.allocation);
        this->images.erase(it);
    }

    std::pair<VkCommandPool, VkCommandBuffer> vma_allocator::create_command_pair() const {
        VkCommandPoolCreateInfo command_pool_create_info = {};

        command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        command_pool_create_info.queueFamilyIndex = this->queue_family_index;

        VkCommandPool command_pool;
        if (vkCreateCommandPool(device, &command_pool_create_info, nullptr, &command_pool) != VK_SUCCESS) {
            utility::panic("Failed to create command pool");
        }

        VkCommandBuffer command_buffer;
        VkCommandBufferAllocateInfo buffer_allocate_info = {};
        buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        buffer_allocate_info.commandBufferCount = 1;
        buffer_allocate_info.commandPool = command_pool;
        if (vkAllocateCommandBuffers(this->device, &buffer_allocate_info, &command_buffer) != VK_SUCCESS) {
            utility::panic("Failed to create command buffer");
        }
        return {command_pool, command_buffer};
    }
} // namespace vulkan
