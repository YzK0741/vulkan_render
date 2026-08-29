module;

#include <vulkan/vulkan.h>
#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

module vulkan.vma;

namespace {
    constexpr uint32_t sizeof_vk_format(const VkFormat format) {
        switch (format) {
        // 8-bit 单通道
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_SNORM:
        case VK_FORMAT_R8_USCALED:
        case VK_FORMAT_R8_SSCALED:
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8_SINT:
        case VK_FORMAT_R8_SRGB:
            return 1;

        // 16-bit 单通道 / 8-bit 双通道
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

        // 深度/模板
        case VK_FORMAT_D16_UNORM:
            return 2;
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT:
            return 4;
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
        case VK_FORMAT_S8_UINT:
        // BC 压缩格式
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
        // ASTC 压缩格式（所有格式都是16字节/块）
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

        // ETC2 / EAC 压缩格式（8字节/块）
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

    constexpr VmaAllocationCreateInfo get_allocation_info_from_type(const vulkan::buffer_type type) {
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
        }
        return info;
    }

    constexpr VkBufferCreateInfo get_create_info_from_type(const vulkan::buffer_type type) {
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
        }
        return info;
    }

    constexpr VmaAllocationCreateInfo get_image_allocation_info_from_type(const vulkan::image_type type) {
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
        const vulkan::image_type type,
        const vulkan::image_create_info& info) {
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
        const VkInstance instance,              // NOLINT(*-misplaced-const)
        const VkDevice device,                  // NOLINT(*-misplaced-const)
        const VkPhysicalDevice physical_device, // NOLINT(*-misplaced-const)
        const VkQueue queue,                    // NOLINT(*-misplaced-const)
        const uint32_t queue_family_index) {

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
            // 释放缓存的 staging buffer
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

    bool vma_allocator::ensure_staging_buffer(const VkDeviceSize size, VkBuffer& buffer, VmaAllocation& allocation, VmaAllocationInfo& info) {
        // 缓存容量足够则复用
        if (this->staging_cache.buffer != VK_NULL_HANDLE && this->buffer_size >= size) {
            buffer = this->staging_cache.buffer;
            allocation = this->staging_cache.allocation;
            info = this->staging_cache.allocation_info;
            return true;
        }

        // 容量不足：销毁当前缓存并重新创建
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

        const VkResult result = vmaCreateBuffer(
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

    bool vma_allocator::direct_upload(VmaAllocation const& allocation, VmaAllocationInfo& allocation_info, const void* data, const VkDeviceSize size) const {
        void* mapped_data = nullptr;
        if (const VkResult result = vmaMapMemory(this->allocator, allocation, &mapped_data); result != VK_SUCCESS) {
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

    bool vma_allocator::staging_upload(const VkBuffer dst_buffer, const void* data, const VkDeviceSize size) { // NOLINT(*-misplaced-const)
        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VmaAllocation staging_allocation = VK_NULL_HANDLE;
        VmaAllocationInfo staging_info = {};

        // staging buffer 是共享资源，写入 + GPU 拷贝完成期间必须互斥占用
        std::lock_guard staging_guard(this->staging_mutex);

        // 复用缓存中的 staging buffer，容量不足时自动重建
        if (!this->ensure_staging_buffer(size, staging_buffer, staging_allocation, staging_info)) {
            return false;
        }

        // 拷贝数据
        if (staging_info.pMappedData) {
            memcpy(staging_info.pMappedData, data, size);
            if ((staging_info.memoryType & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
                vmaFlushAllocation(this->allocator, staging_allocation, 0, size);
            }
        } else {
            return false;
        }
        // 执行拷贝命令（缓存访问由 cache_mutex 保护）
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
            // VkQueue 是外部同步对象，submit 必须串行化
            std::lock_guard guard(this->queue_mutex);
            vkQueueSubmit(this->queue, 1, &submit_info, fence);
        }

        vkWaitForFences(this->device, 1, &fence, VK_TRUE, UINT64_MAX);

        // 清理
        vkResetFences(this->device, 1, &fence);
        vkResetCommandBuffer(command_buffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
        {
            std::lock_guard guard(this->cache_mutex);
            this->fence_cache.push_back(fence);
            this->command_cache.push_back(command_pair);
        }

        return true;
    }

    bool vma_allocator::direct_image_upload(const VmaAllocation allocation, const void* data, const VkDeviceSize size) const { // NOLINT(*-misplaced-const)
        void* mapped_data = nullptr;
        if (const VkResult result = vmaMapMemory(this->allocator, allocation, &mapped_data); result != VK_SUCCESS) {
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

    bool vma_allocator::staging_image_upload(VkImage dst_image, const void* data, VkDeviceSize size, const image_create_info& info) {
        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VmaAllocation staging_allocation = VK_NULL_HANDLE;
        VmaAllocationInfo staging_info = {};

        // staging buffer 是共享资源，写入 + GPU 拷贝完成期间必须互斥占用
        std::lock_guard staging_guard(this->staging_mutex);

        // 复用缓存中的 staging buffer，容量不足时自动重建
        if (!this->ensure_staging_buffer(size, staging_buffer, staging_allocation, staging_info)) {
            return false;
        }

        // 拷贝数据到 staging buffer
        if (staging_info.pMappedData) {
            memcpy(staging_info.pMappedData, data, size);
            if ((staging_info.memoryType & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
                vmaFlushAllocation(this->allocator, staging_allocation, 0, size);
            }
        } else {
            return false;
        }
        // 执行拷贝命令（缓存访问由 cache_mutex 保护）
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

        // ========== 修复点 1：正确的布局转换顺序 ==========
        // 第一步：UNDEFINED -> TRANSFER_DST_OPTIMAL
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

        // 第二步：逐 mip 拷贝。
        // 数据按 mip 主序排列（mip0 全部 layer → mip1 全部 layer → ...），
        // 每个 mip 内各 layer 连续（face0, face1, ...），由 bufferOffset 定位。
        const uint32_t bytes_per_pixel = sizeof_vk_format(info.format);
        std::vector<VkBufferImageCopy> regions;
        regions.reserve(info.mip_levels);
        VkDeviceSize buffer_offset = 0;
        for (uint32_t mip = 0; mip < info.mip_levels; ++mip) {
            const uint32_t mip_width = std::max(1u, info.width >> mip);
            const uint32_t mip_height = std::max(1u, info.height >> mip);

            VkBufferImageCopy region = {};
            region.bufferOffset = buffer_offset;
            region.bufferRowLength = 0; // 0 表示紧密排列
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

        // 第三步：TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
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

        // 提交并等待完成
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        {
            // VkQueue 是外部同步对象，submit 必须串行化
            std::lock_guard guard(this->queue_mutex);
            vkQueueSubmit(this->queue, 1, &submit_info, fence);
        }

        vkWaitForFences(this->device, 1, &fence, VK_TRUE, UINT64_MAX);

        // 清理
        vkResetFences(this->device, 1, &fence);
        vkResetCommandBuffer(command_buffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
        {
            std::lock_guard guard(this->cache_mutex);
            this->fence_cache.push_back(fence);
            this->command_cache.push_back(command_pair);
        }

        return true;
    }

    uint64_t vma_allocator::create_buffer(const unsigned char* data, const uint64_t size_byte, const buffer_type type) {
        uint64_t handle = 0;
        // distribute() 内部自带锁（enable_handle_distribute::access_mutex），无需外层持锁
        if (const auto result = this->distribute(); result) {
            handle = result.value();
        } else {
            return handle;
        }

        const auto allocation_create_info = get_allocation_info_from_type(type);
        auto buffer_create_info = get_create_info_from_type(type);
        buffer_create_info.size = size_byte;

        VmaAllocation allocation = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocationInfo alloc_info = {};

        // VMA 内部线程安全，分配与上传都无需持有 access_mutex
        const VkResult result = vmaCreateBuffer(
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
            // 直接映射上传
            upload_success = direct_upload(allocation, alloc_info, data, size_byte);
            break;

        case buffer_type::vertex:
        case buffer_type::index:
        case buffer_type::uniform_gpu_only:
            // 使用 staging buffer
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
            this->buffers[handle] = {.buffer = buffer, .allocation = allocation, .allocation_info = alloc_info};
        }
        return handle;
    }

    uint64_t vma_allocator::create_image(const unsigned char* data, const uint64_t size_byte, image_create_info const& create_info, const image_type type) {
        uint64_t handle = 0;
        // distribute() 内部自带锁（enable_handle_distribute::access_mutex），无需外层持锁
        if (const auto result = this->distribute(); result) {
            handle = result.value();
        } else {
            return handle;
        }

        const VkDeviceSize image_size = size_byte;

        // 期望大小 = array_layers × 所有 mip 尺寸之和 × 每像素字节数
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

        const auto alloc_info = get_image_allocation_info_from_type(type);
        auto image_create_info = get_image_create_info_from_type(type, create_info);
        image_create_info.mipLevels = create_info.mip_levels;
        image_create_info.extent.width = create_info.width;
        image_create_info.extent.height = create_info.height;
        image_create_info.extent.depth = 1;
        image_create_info.arrayLayers = create_info.array_layers;

        VmaAllocation allocation = VK_NULL_HANDLE;
        VkImage image = VK_NULL_HANDLE;
        VmaAllocationInfo alloc_detail = {};

        const VkResult vk_result = vmaCreateImage(
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

        // 根据类型选择上传方式
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

        // sha256 是纯 CPU 计算，放在临界区之外
        const auto digest = utility::sha256(std::span(data, size_byte));

        if (!digest) {
            utility::panic("sha256 failed");
        }

        {
            std::lock_guard guard(this->access_mutex);
            this->images[handle] = {.image = image, .allocation = allocation, .allocation_info = alloc_detail, .digest = digest.value()};
        }
        return handle;
    }

    const buffer_detail* vma_allocator::get_buffer_detail(const uint64_t handle) {
        std::lock_guard guard(this->access_mutex);
        if (this->buffers.contains(handle)) {
            return &this->buffers[handle];
        }
        return nullptr;
    }

    const image_detail* vma_allocator::get_image_detail(const uint64_t handle) {
        std::lock_guard guard(this->access_mutex);
        if (this->images.contains(handle)) {
            return &this->images[handle];
        }
        return nullptr;
    }

    void vma_allocator::free_buffer(const uint64_t handle) {
        std::lock_guard guard(this->access_mutex);
        if (this->buffers.contains(handle)) {
            const auto& info = this->buffers[handle];
            vmaDestroyBuffer(this->allocator, info.buffer, info.allocation);
            this->buffers.erase(handle);
        }
    }

    void vma_allocator::free_image(const uint64_t handle) {
        std::lock_guard guard(this->access_mutex);
        if (this->images.contains(handle)) {
            const auto& info = this->images[handle];
            vmaDestroyImage(this->allocator, info.image, info.allocation);
            this->images.erase(handle);
        }
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
