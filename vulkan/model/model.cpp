//
// Created by 小叶 on 2026/7/30.
//

module;

#include <vulkan/vulkan.h>

module vulkan.model;

namespace vulkan {
    void model::draw(const VkCommandBuffer command_buffer, const VkPipelineLayout pipeline_layout) const { // NOLINT(*-misplaced-const)
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &this->vertex_buffer, nullptr);
        vkCmdBindIndexBuffer(command_buffer, this->index_buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(command_buffer, this->index_count, 1, 0, 0, 0);
    }

    void model::destroy(const VkDevice device, const VkDescriptorPool descriptor_pool, vma_allocator& vma) { // NOLINT(*-misplaced-const)
        vma.free_buffer(vertex_buffer_handle);
        vma.free_buffer(index_buffer_handle);
        vkFreeDescriptorSets(device, descriptor_pool, 1, &this->descriptor_set);
        this->vertex_buffer = VK_NULL_HANDLE;
        this->index_buffer = VK_NULL_HANDLE;
        this->descriptor_set = VK_NULL_HANDLE;
        this->vertex_buffer_handle = 0;
        this->index_buffer_handle = 0;
    }
}
