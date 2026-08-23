//
// Created by 小叶 on 2026/7/28.
//
module;

#include <vma/vk_mem_alloc.h>

export module vulkan.vma;
export import std;
import utility;

/**
 * @file vma.cppm
 * @defgroup vulkan_vma Vulkan VMA Allocator
 * @brief handle-based GPU buffer/image allocator built on VMA
 * @note
 *      - buffers/images are referenced by uint64_t handles
 *      - thread-safe for creation and lookup
 */
namespace vulkan {

    /**
     * @ingroup vulkan_vma
     * @brief detail of a created buffer: the buffer handle, its allocation and allocation info
     */
    export struct buffer_detail {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocation_info = {};
    };

    /**
     * @ingroup vulkan_vma
     * @brief detail of a created image: the image handle, its allocation and allocation info
     */
    export struct image_detail {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocation_info = {};
        utility::sha256_digest digest = {};
    };

    /**
     * @ingroup vulkan_vma
     * @brief buffer usage type, decides memory properties and upload strategy
     */
    export enum class buffer_type {
        vertex,              // GPU_ONLY, 需要 staging buffer
        index,               // GPU_ONLY, 需要 staging buffer
        uniform_gpu_only,    // GPU_ONLY, 适合不需要频繁更新的 uniform
        uniform_coherent,    // HOST_VISIBLE | HOST_COHERENT, 适合每帧更新的 uniform
        uniform_cached       // HOST_VISIBLE | HOST_CACHED, 适合 read-back
        };

    /**
     * @ingroup vulkan_vma
     * @brief image usage type, decides memory properties and upload strategy
     */
    export enum class image_type {
        texture_2d,           // GPU_ONLY, 普通纹理，需要 staging
        texture_2d_color,     // GPU_ONLY, 带颜色格式的纹理
        texture_2d_depth,     // GPU_ONLY, 深度纹理
        texture_2d_staging,   // HOST_VISIBLE, 用于动态更新的纹理
        texture_cubemap,      // GPU_ONLY, 立方体贴图
        render_target         // GPU_ONLY, 渲染目标 (可读写)
    };

    /**
     * @ingroup vulkan_vma
     * @brief auxiliary data for image creation
     */
    // Image 的辅助结构
    export struct image_create_info {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t mip_levels = 1;
        uint32_t array_layers = 1;
        VkFormat format = {};
        VkImageUsageFlags extra_usage = 0;
    };

    /**
     * @ingroup vulkan_vma
     * @brief handle-based GPU buffer/image allocator built on VMA
     * @note
     *      - buffers/images are referenced by uint64_t handles
     *      - thread-safe: creation and lookup are protected by an internal mutex
     *      - call destroy() to release the underlying VmaAllocator
     */
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

        /**
         * @ingroup vulkan_vma
         * @brief initialize the VMA allocator with the given vulkan objects
         * @param instance the vulkan instance
         * @param device the logical device
         * @param physical_device the physical device
         * @param queue a queue used for staging uploads
         * @param queue_family_index the queue family of the given queue
         */
        void init(VkInstance instance, VkDevice device, VkPhysicalDevice physical_device, VkQueue queue, uint32_t queue_family_index);

        /**
         * @ingroup vulkan_vma
         * @brief release the underlying VmaAllocator and cached upload resources
         */
        void destroy();

        /**
         * @ingroup vulkan_vma
         * @brief create a GPU buffer of the given type and upload the data
         * @param data source bytes to upload
         * @param size_byte byte size of the data
         * @param type buffer usage type
         * @return the buffer handle
         */
        uint64_t create_buffer(const unsigned char *data, uint64_t size_byte, buffer_type type);

        /**
         * @ingroup vulkan_vma
         * @brief create a GPU buffer from a typed span
         * @tparam T element type of the span
         * @param data source data to upload
         * @param type buffer usage type
         * @return the buffer handle
         */
        template <typename T>
        uint64_t create_buffer(std::span<T> data, const buffer_type type) {
            return this->create_buffer(reinterpret_cast<unsigned char*>(data.data()), data.size_bytes(), type);
        }

        /**
         * @ingroup vulkan_vma
         * @brief create a GPU buffer from a fixed-size typed span
         * @tparam T element type of the span
         * @tparam N element count of the span
         * @param data source data to upload
         * @param type buffer usage type
         * @return the buffer handle
         */
        template <typename T, std::size_t N>
        uint64_t create_buffer(std::span<T, N> data, const buffer_type type) {
            return this->create_buffer(reinterpret_cast<unsigned char*>(data.data()), data.size_bytes(), type);
        }

        /**
         * @ingroup vulkan_vma
         * @brief create a GPU image of the given type and upload the data
         * @param data source bytes to upload
         * @param size_byte byte size of the data
         * @param create_info image width/height/format/mip levels etc.
         * @param type image usage type
         * @return the image handle
         */
        uint64_t create_image(const unsigned char *data, uint64_t size_byte, image_create_info const &create_info, image_type type);

        /**
         * @ingroup vulkan_vma
         * @brief create a GPU image from a typed span
         * @tparam T element type of the span
         * @param data source data to upload
         * @param create_info image width/height/format/mip levels etc.
         * @param type image usage type
         * @return the image handle
         */
        template <typename T>
        uint64_t create_image(std::span<T> data, image_create_info create_info, const image_type type) {
            return this->create_image(reinterpret_cast<unsigned char*>(data.data()), create_info, data.size_bytes(), type);
        }

        /**
         * @ingroup vulkan_vma
         * @brief create a GPU image from a fixed-size typed span
         * @tparam T element type of the span
         * @tparam N element count of the span
         * @param data source data to upload
         * @param create_info image width/height/format/mip levels etc.
         * @param type image usage type
         * @return the image handle
         */
        template <typename T, size_t N>
        uint64_t create_image(std::span<T, N> data, image_create_info create_info, const image_type type) {
            return this->create_image(reinterpret_cast<unsigned char*>(data.data()), create_info, data.size_bytes(), type);
        }

        /**
         * @ingroup vulkan_vma
         * @brief get the detail of a buffer by its handle
         * @param handle the buffer handle
         * @return pointer to the buffer detail, or nullptr if the handle is invalid
         */
        [[nodiscard]] const buffer_detail *get_buffer_detail(uint64_t handle);

        /**
         * @ingroup vulkan_vma
         * @brief get the detail of an image by its handle
         * @param handle the image handle
         * @return pointer to the image detail, or nullptr if the handle is invalid
         */
        [[nodiscard]] const image_detail *get_image_detail(uint64_t handle);

        /**
         * @ingroup vulkan_vma
         * @brief free a buffer by its handle, no-op if the handle is invalid
         * @param handle the buffer handle
         */
        void free_buffer(uint64_t handle);

        /**
         * @ingroup vulkan_vma
         * @brief free an image by its handle, no-op if the handle is invalid
         * @param handle the image handle
         */
        void free_image(uint64_t handle);
    };
}

