//
// Created by 小叶 on 2026/7/30.
//

module;

#include <cstdint>
#include <vulkan/vulkan.h>

export module vulkan.model;

export import vulkan.core.handles;

namespace vulkan {
    export class model {
        uint32_t vertex_buffer_handle = 0;
        VkBuffer vertex_buffer = VK_NULL_HANDLE;
        uint32_t index_buffer_handle = 0;
        VkBuffer index_buffer = VK_NULL_HANDLE;
        uint32_t index_count = 0;

        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

    public:
        void draw(VkCommandBuffer command_buffer, VkPipelineLayout pipeline_layout) const;
    };
}
