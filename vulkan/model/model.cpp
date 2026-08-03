//
// Created by 小叶 on 2026/7/30.
//

module;

#include <vulkan/vulkan.h>

module vulkan.model;

namespace vulkan {
    void model::draw(const VkCommandBuffer command_buffer, const VkPipelineLayout pipeline_layout) const {
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &this->vertex_buffer, nullptr);
        vkCmdBindIndexBuffer(command_buffer, this->index_buffer, this->index_count, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(command_buffer, this->index_count, 1, 0, 0, 0);
    }
}
