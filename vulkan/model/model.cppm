//
// Created by 小叶 on 2026/7/30.
//

module;

#include <vulkan/vulkan.h>

export module vulkan.model;
export import std;
export import vulkan.core.handles;
export import vulkan.vma;

/**
 * @file model.cppm
 * @defgroup vulkan_model Vulkan Model
 * @brief GPU model management: vertex/index buffers, descriptor set and draw command
 * @note
 *      - holds the GPU resources of a single model
 *      - resources are created externally, release them via destroy()
 */
namespace vulkan {
    /**
     * @ingroup vulkan_model
     * @brief model class managing vertex/index buffers and the descriptor set
     * @note
     *      - draw() binds the descriptor set and vertex/index buffers, then issues a draw call
     *      - destroy() frees the buffers and the descriptor set, pass the owning device/pool/vma
     */
    export class model {
        uint32_t vertex_buffer_handle = 0;
        VkBuffer vertex_buffer = VK_NULL_HANDLE;
        uint32_t index_buffer_handle = 0;
        VkBuffer index_buffer = VK_NULL_HANDLE;
        uint32_t index_count = 0;

        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

    public:
        void draw(VkCommandBuffer command_buffer, VkPipelineLayout pipeline_layout) const;
        void destroy(VkDevice device, VkDescriptorPool descriptor_pool, vma_allocator& vma);
    };
}
