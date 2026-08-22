//
// Created by 小叶 on 2026/7/28.
//
module;

#include <vma/vk_mem_alloc.h>
#include <map>
#include <vector>
#include <mutex>
#include <span>

export module vulkan.vma;
import utility;

namespace vulkan {

    export struct buffer_detail {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocation_info = {};
    };

    export struct image_detail {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocation_info = {};
        utility::sha256_digest digest = {};
    };

    export enum class buffer_type {
        vertex,              // GPU_ONLY, 需要 staging buffer
        index,               // GPU_ONLY, 需要 staging buffer
        uniform_gpu_only,    // GPU_ONLY, 适合不需要频繁更新的 uniform
        uniform_coherent,    // HOST_VISIBLE | HOST_COHERENT, 适合每帧更新的 uniform
        uniform_cached       // HOST_VISIBLE | HOST_CACHED, 适合 read-back
        };

    export enum class image_type {
        texture_2d,           // GPU_ONLY, 普通纹理，需要 staging
        texture_2d_color,     // GPU_ONLY, 带颜色格式的纹理
        texture_2d_depth,     // GPU_ONLY, 深度纹理
        texture_2d_staging,   // HOST_VISIBLE, 用于动态更新的纹理
        texture_cubemap,      // GPU_ONLY, 立方体贴图
        render_target         // GPU_ONLY, 渲染目标 (可读写)
    };

    // Image 的辅助结构
    export struct image_create_info {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t mip_levels = 1;
        uint32_t array_layers = 1;
        VkFormat format = {};
        VkImageUsageFlags extra_usage = 0;
    };

    export class vma_allocator : utility::enable_handle_distribute {
        VmaAllocator allocator = {};
        VkDevice device = VK_NULL_HANDLE;
        VkQueue queue = VK_NULL_HANDLE;
        std::vector<std::pair<VkCommandPool, VkCommandBuffer>> command_cache;
        std::vector<VkFence> fence_cache = {};
        std::map<uint64_t, buffer_detail> buffers = {};
        std::map<uint64_t, image_detail> images = {};
        std::mutex access_mutex = {};
        uint32_t queue_family_index = 0;

        [[nodiscard]] VkFence create_fence() const;
        bool direct_upload(VmaAllocation const &allocation, VmaAllocationInfo &allocation_info, const void *data, uint64_t size) const;
        bool staging_upload(VkBuffer dst_buffer, const void *data, VkDeviceSize size);
        bool direct_image_upload(VmaAllocation allocation, const void* data, VkDeviceSize size) const;
        bool staging_image_upload(VkImage dst_image, const void* data, VkDeviceSize size, const image_create_info& info);
        [[nodiscard]] std::pair<VkCommandPool, VkCommandBuffer> create_command_pair() const;


    public:

        void init(VkInstance instance, VkDevice device, VkPhysicalDevice physical_device, VkQueue queue, uint32_t queue_family_index);

        void destroy();

        uint64_t create_buffer(const unsigned char *data, uint64_t size_byte, buffer_type type);

        template <typename T>
        uint64_t create_buffer(std::span<T> data, const buffer_type type) {
            return this->create_buffer(reinterpret_cast<unsigned char*>(data.data()), data.size_bytes(), type);
        }

        template <typename T, std::size_t N>
        uint64_t create_buffer(std::span<T, N> data, const buffer_type type) {
            return this->create_buffer(reinterpret_cast<unsigned char*>(data.data()), data.size_bytes(), type);
        }

        uint64_t create_image(const unsigned char *data, uint64_t size_byte, image_create_info const &create_info, image_type type);

        template <typename T>
        uint64_t create_image(std::span<T> data, image_create_info create_info, const image_type type) {
            return this->create_image(reinterpret_cast<unsigned char*>(data.data()), create_info, data.size_bytes(), type);
        }

        template <typename T, size_t N>
        uint64_t create_image(std::span<T, N> data, image_create_info create_info, const image_type type) {
            return this->create_image(reinterpret_cast<unsigned char*>(data.data()), create_info, data.size_bytes(), type);
        }

        [[nodiscard]] const buffer_detail *get_buffer_detail(uint64_t handle);

        [[nodiscard]] const image_detail *get_image_detail(uint64_t handle);

        void free_buffer(uint64_t handle);

        void free_image(uint64_t handle);
    };
}

