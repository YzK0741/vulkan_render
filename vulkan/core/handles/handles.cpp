//
// Created by 小叶 on 2026/7/29.
//

module;

#include <vulkan/vulkan.h>

module vulkan.core.handles;

// vk_command_buffer
namespace vulkan {
    vk_command_buffer::vk_command_buffer(const VkCommandBuffer command_buffer, VkDevice& device, VkCommandPool& pool) noexcept { // NOLINT(*-misplaced-const)
        this->command_buffer = command_buffer;
        this->device = &device;
        this->command_pool = &pool;
    }

    VkCommandBuffer const &vk_command_buffer::get() const noexcept {
        return this->command_buffer;
    }

    VkCommandBuffer const &vk_command_buffer::operator*() const noexcept {
        return this->get();
    }

    void vk_command_buffer::release() noexcept {
        if (this->command_buffer != VK_NULL_HANDLE && this->command_pool != nullptr) {
            vkFreeCommandBuffers(*this->device, *this->command_pool, 1, &this->command_buffer);
        }

        this->device = nullptr;
        this->command_pool = nullptr;
        this->command_buffer = VK_NULL_HANDLE;
    }

    vk_command_buffer::~vk_command_buffer() noexcept {
        this->release();
    }

    vk_command_buffer::vk_command_buffer(vk_command_buffer &&other) noexcept {
        this->command_buffer = other.command_buffer;
        this->device = other.device;
        this->command_pool = other.command_pool;
        other.device = nullptr;
        other.command_pool = nullptr;
        other.command_buffer = VK_NULL_HANDLE;
    }

    vk_command_buffer &vk_command_buffer::operator=(vk_command_buffer &&other) noexcept {
        if (this == &other) {
            return *this;
        }

        this->release();
        this->command_buffer = other.command_buffer;
        this->device = other.device;
        this->command_pool = other.command_pool;
        other.device = nullptr;
        other.command_pool = nullptr;
        other.command_buffer = VK_NULL_HANDLE;
        return *this;
    }

    vk_command_buffer make_command_buffer(VkDevice &device, VkCommandPool &command_pool) noexcept {
        VkCommandBuffer buffer;

        VkCommandBufferAllocateInfo allocate_info;
        allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate_info.commandPool = command_pool;
        allocate_info.commandBufferCount = 1;

        vkAllocateCommandBuffers(device, &allocate_info, &buffer);

        return vk_command_buffer(buffer, device, command_pool);
    }
}


// vk_descriptor_set
namespace vulkan {
    vk_descriptor_set::vk_descriptor_set(const VkDescriptorSet descriptor_set, VkDevice &device, VkDescriptorPool &descriptor_pool) noexcept { // NOLINT(*-misplaced-const)
        this->descriptor_set = descriptor_set;
        this->device = &device;
        this->descriptor_pool = &descriptor_pool;
    }

    vk_descriptor_set::vk_descriptor_set(vk_descriptor_set&& other) noexcept {
        this->descriptor_set = other.descriptor_set;
        this->device = other.device;
        this->descriptor_pool = other.descriptor_pool;
        other.descriptor_set = VK_NULL_HANDLE;
        other.device = nullptr;
        other.descriptor_pool = nullptr;
    }

    vk_descriptor_set &vk_descriptor_set::operator=(vk_descriptor_set &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        this->release();
        this->descriptor_set = other.descriptor_set;
        this->device = other.device;
        this->descriptor_pool = other.descriptor_pool;
        other.descriptor_set = VK_NULL_HANDLE;
        other.device = nullptr;
        other.descriptor_pool = nullptr;
        return *this;
    }

    vk_descriptor_set::~vk_descriptor_set() noexcept {
        this->release();
    }

    VkDescriptorSet const &vk_descriptor_set::get() const noexcept {
        return this->descriptor_set;
    }

    VkDescriptorSet const &vk_descriptor_set::operator*() const noexcept {
        return this->descriptor_set;
    }

    void vk_descriptor_set::release() noexcept {
        if (this->descriptor_set != VK_NULL_HANDLE && this->device && this->descriptor_pool) {
            vkFreeDescriptorSets(*this->device, *this->descriptor_pool, 1, &this->descriptor_set);
        }
        this->descriptor_set = VK_NULL_HANDLE;
        this->device = nullptr;
        this->descriptor_pool = nullptr;
    }

    vk_descriptor_set make_descriptor_set(VkDevice &device, VkDescriptorPool &descriptor_pool, VkDescriptorSetLayout const &layout) noexcept {
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo allocate_info = {};
        allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate_info.descriptorPool = descriptor_pool;
        allocate_info.descriptorSetCount = 1;
        allocate_info.pSetLayouts = &layout;
        allocate_info.descriptorSetCount = 1;
        vkAllocateDescriptorSets(device, &allocate_info, &descriptor_set);

        return vk_descriptor_set(descriptor_set, device, descriptor_pool);
    }
}

// vk_shader_module
namespace vulkan {
    vk_shader_module::vk_shader_module(const VkShaderModule &shader_module, VkDevice &device) noexcept {
        this->shader_module = shader_module;
        this->device = &device;
    }

    vk_shader_module::~vk_shader_module() noexcept {
        this->release();
    }

    VkShaderModule const &vk_shader_module::get() const noexcept {
        return this->shader_module;
    }

    VkShaderModule const &vk_shader_module::operator*() const noexcept {
        return this->shader_module;
    }

    vk_shader_module::vk_shader_module(vk_shader_module &&other) noexcept {
        this->shader_module = other.shader_module;
        this->device = other.device;
        other.shader_module = VK_NULL_HANDLE;
        other.device = nullptr;
    }

    vk_shader_module &vk_shader_module::operator=(vk_shader_module &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        this->shader_module = other.shader_module;
        this->device = other.device;
        other.shader_module = VK_NULL_HANDLE;
        other.device = nullptr;
        return *this;
    }

    void vk_shader_module::release() {
        if (this->shader_module != VK_NULL_HANDLE && this->device != nullptr) {
            vkDestroyShaderModule(*this->device, this->shader_module, nullptr);
        }
        this->shader_module = VK_NULL_HANDLE;
        this->device = nullptr;
    }

    std::optional<vk_shader_module> make_shader_module(const std::span<const unsigned char> shader, VkDevice &device) noexcept {
        VkShaderModule shader_module = {};

        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = shader.size_bytes();
        create_info.pCode = reinterpret_cast<const uint32_t*>(shader.data());

        if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
            return std::nullopt;
        }

        return vk_shader_module(shader_module, device);
    }
}

// vk_pipeline
namespace vulkan {
    vk_pipeline::vk_pipeline(const VkPipeline pipeline, const VkPipelineLayout pipeline_layout, std::vector<VkDescriptorSetLayout> const& descriptor_set_layouts, VkDevice& device) noexcept { // NOLINT(*-misplaced-const)
        this->pipeline = pipeline;
        this->pipeline_layout = pipeline_layout;
        this->descriptor_set_layouts = descriptor_set_layouts;
        this->device = &device;
    }

    vk_pipeline::vk_pipeline(vk_pipeline &&other) noexcept {
        this->device = other.device;
        this->pipeline = other.pipeline;
        this->descriptor_set_layouts = other.descriptor_set_layouts;
        this->pipeline_layout = other.pipeline_layout;
        other.device = nullptr;
        other.pipeline = nullptr;
        other.descriptor_set_layouts.clear();
        other.pipeline_layout = VK_NULL_HANDLE;
    }

    vk_pipeline &vk_pipeline::operator=(vk_pipeline &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        this->device = other.device;
        this->pipeline = other.pipeline;
        this->descriptor_set_layouts = other.descriptor_set_layouts;
        this->pipeline_layout = other.pipeline_layout;
        other.device = nullptr;
        other.pipeline = nullptr;
        other.descriptor_set_layouts.clear();
        other.pipeline_layout = VK_NULL_HANDLE;
        return *this;
    }

    vk_pipeline::~vk_pipeline() {
        this->release();
    }

    VkPipeline vk_pipeline::get_pipeline() const noexcept {
        return this->pipeline;
    }

    VkPipelineLayout vk_pipeline::get_pipeline_layout() const noexcept {
        return pipeline_layout;
    }

    std::vector<VkDescriptorSetLayout> vk_pipeline::get_descriptor_set_layouts() const noexcept {
        return this->descriptor_set_layouts;
    }

    void vk_pipeline::release() noexcept {
        if (device) {
            if (!this->descriptor_set_layouts.empty()) {
                for (auto& set_layout : this->descriptor_set_layouts) {
                    vkDestroyDescriptorSetLayout(*this->device, set_layout, nullptr);
                }
            }

            if (this->pipeline_layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(*this->device, this->pipeline_layout, nullptr);
            }

            if (this->pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(*this->device, this->pipeline, nullptr);
            }
        }
        this->device = nullptr;
        this->descriptor_set_layouts.clear();
        this->pipeline_layout = VK_NULL_HANDLE;
        this->pipeline = VK_NULL_HANDLE;
    }
}
