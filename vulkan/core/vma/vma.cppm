module;

#include <vma/vk_mem_alloc.h>

export module vulkan.core.vma;
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
        // reference count for shared resources; buffers have no digest today, so this stays 1
        std::atomic<uint32_t> use_count = 1;

        buffer_detail() = default;
        // std::atomic is neither copyable nor movable: a custom move constructor carries the
        // reference count by loading it (the source is discarded right after)
        buffer_detail(buffer_detail&& other) noexcept
            : buffer(other.buffer)
            , allocation(other.allocation)
            , allocation_info(other.allocation_info)
            , use_count(other.use_count.load()) {
            other.buffer = VK_NULL_HANDLE;
            other.allocation = VK_NULL_HANDLE;
        }
        buffer_detail(buffer_detail const&) = delete;
        buffer_detail& operator=(buffer_detail const&) = delete;
        buffer_detail& operator=(buffer_detail&&) = delete;
    };

    /**
     * @ingroup vulkan_vma
     * @brief buffer usage type, decides memory properties and upload strategy
     */
    export enum class buffer_type {
        vertex,           // GPU_ONLY, requires a staging buffer
        index,            // GPU_ONLY, requires a staging buffer
        uniform_gpu_only, // GPU_ONLY, suited for uniforms updated infrequently
        uniform_coherent, // HOST_VISIBLE | HOST_COHERENT, suited for per-frame uniforms
        uniform_cached,   // HOST_VISIBLE | HOST_CACHED, suited for read-back
        storage_coherent, // HOST_VISIBLE | HOST_COHERENT storage buffer (e.g. GPU-visible material table)
    };

    /**
     * @ingroup vulkan_vma
     * @brief image usage type, decides memory properties and upload strategy
     */
    export enum class image_type {
        texture_2d,         // GPU_ONLY, regular texture, needs staging
        texture_2d_color,   // GPU_ONLY, texture with a color format
        texture_2d_depth,   // GPU_ONLY, depth texture
        texture_2d_staging, // HOST_VISIBLE, for dynamically updated textures
        texture_cubemap,    // GPU_ONLY, cubemap
        render_target,      // GPU_ONLY, render target (readable/writable)
    };

    /**
     * @ingroup vulkan_vma
     * @brief auxiliary data for image creation
     */
    // Helper structure for Image
    export struct image_create_info {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t mip_levels = 1;
        uint32_t array_layers = 1;
        VkFormat format = {};
        VkImageUsageFlags extra_usage = 0;
        bool operator==(image_create_info const&) const = default;
    };

    /**
     * @ingroup vulkan_vma
     * @brief detail of a created image: the image handle, its allocation and allocation info
     * @note immutable, data-uploaded textures (texture_2d / texture_2d_color / texture_cubemap)
     *       are deduplicated by content: identical bytes + identical create parameters share one
     *       GPU image and one handle, tracked by use_count
     */
    export struct image_detail {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocation_info = {};
        // XXH3_64bits content digest (data_block<8>, hex-formatable); all-zero = no content digest
        // (never deduplicated)
        utility::xxh3_digest digest = {};
        // creation parameters, kept so a digest hit only reuses an identical image
        image_create_info create_info = {};
        image_type type = image_type::texture_2d;
        // reference count for shared images; free_image() only really destroys at zero
        std::atomic<uint32_t> use_count = 1;

        image_detail() = default;
        // std::atomic is neither copyable nor movable: a custom move constructor carries the
        // reference count by loading it (the source is discarded right after)
        image_detail(image_detail&& other) noexcept
            : image(other.image)
            , allocation(other.allocation)
            , allocation_info(other.allocation_info)
            , digest(other.digest)
            , create_info(std::move(other.create_info))
            , type(other.type)
            , use_count(other.use_count.load()) {
            other.image = VK_NULL_HANDLE;
            other.allocation = VK_NULL_HANDLE;
        }
        image_detail(image_detail const&) = delete;
        image_detail& operator=(image_detail const&) = delete;
        image_detail& operator=(image_detail&&) = delete;
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
        // Guards command_cache / fence_cache so uploads can safely reuse them outside access_mutex
        std::mutex cache_mutex = {};
        // VkQueue is an externally synchronized object; vkQueueSubmit must be serialized
        std::mutex queue_mutex = {};
        // Staging buffer cache: avoids creating/destroying large staging memory on every upload
        struct staging_buffer_cache {
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VmaAllocationInfo allocation_info = {};
        };
        staging_buffer_cache staging_cache = {};
        // Capacity (bytes) of the currently cached staging buffer
        VkDeviceSize buffer_size = 0;
        // The staging buffer is shared; writes and GPU copies must hold it exclusively
        std::mutex staging_mutex = {};
        uint32_t queue_family_index = 0;

        [[nodiscard]] VkFence create_fence() const;
        // Called while holding staging_mutex: reuse the cached buffer if large enough, else destroy and rebuild
        bool ensure_staging_buffer(VkDeviceSize size, VkBuffer& buffer, VmaAllocation& allocation, VmaAllocationInfo& info);
        bool direct_upload(VmaAllocation const& allocation, VmaAllocationInfo& allocation_info, void const* data, uint64_t size) const;
        bool staging_upload(VkBuffer dst_buffer, void const* data, VkDeviceSize size);
        bool direct_image_upload(VmaAllocation allocation, void const* data, VkDeviceSize size) const;
        bool staging_image_upload(VkImage dst_image, void const* data, VkDeviceSize size, image_create_info const& info);
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
        uint64_t create_buffer(unsigned char const* data, uint64_t size_byte, buffer_type type);

        /**
         * @ingroup vulkan_vma
         * @brief create a GPU buffer from a typed span
         * @tparam T element type of the span
         * @param data source data to upload
         * @param type buffer usage type
         * @return the buffer handle
         */
        template <typename T>
        uint64_t create_buffer(std::span<T> data, buffer_type const type) {
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
        uint64_t create_buffer(std::span<T, N> data, buffer_type const type) {
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
        uint64_t create_image(unsigned char const* data, uint64_t size_byte, image_create_info const& create_info, image_type type);

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
        uint64_t create_image(std::span<T> data, image_create_info create_info, image_type const type) {
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
        uint64_t create_image(std::span<T, N> data, image_create_info create_info, image_type const type) {
            return this->create_image(reinterpret_cast<unsigned char*>(data.data()), create_info, data.size_bytes(), type);
        }

        /**
         * @ingroup vulkan_vma
         * @brief get the detail of a buffer by its handle
         * @param handle the buffer handle
         * @return pointer to the buffer detail, or nullptr if the handle is invalid
         */
        [[nodiscard]] buffer_detail const* get_buffer_detail(uint64_t handle);

        /**
         * @ingroup vulkan_vma
         * @brief get the detail of an image by its handle
         * @param handle the image handle
         * @return pointer to the image detail, or nullptr if the handle is invalid
         */
        [[nodiscard]] image_detail const* get_image_detail(uint64_t handle);

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
} // namespace vulkan
