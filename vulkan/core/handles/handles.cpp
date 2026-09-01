module;

#include <vulkan/vulkan.h>

module vulkan.core.handles;

import utility;

// vk_command_buffer
namespace vulkan {
    vk_command_buffer::vk_command_buffer(VkCommandBuffer const command_buffer, VkDevice const device, VkCommandPool const pool) noexcept { // NOLINT(*-misplaced-const)
        this->command_buffer = command_buffer;
        this->device = device;
        this->command_pool = pool;
    }

    VkCommandBuffer const& vk_command_buffer::get() const noexcept {
        return this->command_buffer;
    }

    VkCommandBuffer const& vk_command_buffer::operator*() const noexcept {
        return this->get();
    }

    void vk_command_buffer::release() noexcept {
        if (this->command_buffer != VK_NULL_HANDLE && this->device != VK_NULL_HANDLE && this->command_pool != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(this->device, this->command_pool, 1, &this->command_buffer);
        }

        this->device = VK_NULL_HANDLE;
        this->command_pool = VK_NULL_HANDLE;
        this->command_buffer = VK_NULL_HANDLE;
    }

    vk_command_buffer::~vk_command_buffer() noexcept {
        this->release();
    }

    vk_command_buffer::vk_command_buffer(vk_command_buffer&& other) noexcept {
        this->command_buffer = other.command_buffer;
        this->device = other.device;
        this->command_pool = other.command_pool;
        other.device = VK_NULL_HANDLE;
        other.command_pool = VK_NULL_HANDLE;
        other.command_buffer = VK_NULL_HANDLE;
    }

    vk_command_buffer& vk_command_buffer::operator=(vk_command_buffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        this->release();
        this->command_buffer = other.command_buffer;
        this->device = other.device;
        this->command_pool = other.command_pool;
        other.device = VK_NULL_HANDLE;
        other.command_pool = VK_NULL_HANDLE;
        other.command_buffer = VK_NULL_HANDLE;
        return *this;
    }

    vk_command_buffer make_command_buffer(VkDevice const device, VkCommandPool const command_pool) noexcept {
        VkCommandBuffer buffer = VK_NULL_HANDLE;

        VkCommandBufferAllocateInfo allocate_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        if (vkAllocateCommandBuffers(device, &allocate_info, &buffer) != VK_SUCCESS) {
            utility::panic("failed to allocate command buffer");
        }

        return vk_command_buffer(buffer, device, command_pool);
    }
} // namespace vulkan

// vk_descriptor_set
namespace vulkan {
    vk_descriptor_set::vk_descriptor_set(VkDescriptorSet const descriptor_set, VkDevice const device, VkDescriptorPool const descriptor_pool) noexcept { // NOLINT(*-misplaced-const)
        this->descriptor_set = descriptor_set;
        this->device = device;
        this->descriptor_pool = descriptor_pool;
    }

    vk_descriptor_set::vk_descriptor_set(vk_descriptor_set&& other) noexcept {
        this->descriptor_set = other.descriptor_set;
        this->device = other.device;
        this->descriptor_pool = other.descriptor_pool;
        other.descriptor_set = VK_NULL_HANDLE;
        other.device = VK_NULL_HANDLE;
        other.descriptor_pool = VK_NULL_HANDLE;
    }

    vk_descriptor_set& vk_descriptor_set::operator=(vk_descriptor_set&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        this->release();
        this->descriptor_set = other.descriptor_set;
        this->device = other.device;
        this->descriptor_pool = other.descriptor_pool;
        other.descriptor_set = VK_NULL_HANDLE;
        other.device = VK_NULL_HANDLE;
        other.descriptor_pool = VK_NULL_HANDLE;
        return *this;
    }

    vk_descriptor_set::~vk_descriptor_set() noexcept {
        this->release();
    }

    VkDescriptorSet const& vk_descriptor_set::get() const noexcept {
        return this->descriptor_set;
    }

    VkDescriptorSet const& vk_descriptor_set::operator*() const noexcept {
        return this->descriptor_set;
    }

    void vk_descriptor_set::release() noexcept {
        if (this->descriptor_set != VK_NULL_HANDLE && this->device != VK_NULL_HANDLE && this->descriptor_pool != VK_NULL_HANDLE) {
            vkFreeDescriptorSets(this->device, this->descriptor_pool, 1, &this->descriptor_set);
        }
        this->descriptor_set = VK_NULL_HANDLE;
        this->device = VK_NULL_HANDLE;
        this->descriptor_pool = VK_NULL_HANDLE;
    }

    vk_descriptor_set make_descriptor_set(VkDevice const device, VkDescriptorPool const descriptor_pool, VkDescriptorSetLayout const& layout) noexcept {
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo allocate_info = {};
        allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate_info.descriptorPool = descriptor_pool;
        allocate_info.descriptorSetCount = 1;
        allocate_info.pSetLayouts = &layout;
        vkAllocateDescriptorSets(device, &allocate_info, &descriptor_set);

        return vk_descriptor_set(descriptor_set, device, descriptor_pool);
    }
} // namespace vulkan

// vk_shader_module
namespace vulkan {
    vk_shader_module::vk_shader_module(VkShaderModule const& shader_module, VkDevice const device) noexcept {
        this->shader_module = shader_module;
        this->device = device;
    }

    vk_shader_module::~vk_shader_module() noexcept {
        this->release();
    }

    VkShaderModule const& vk_shader_module::get() const noexcept {
        return this->shader_module;
    }

    VkShaderModule const& vk_shader_module::operator*() const noexcept {
        return this->shader_module;
    }

    vk_shader_module::vk_shader_module(vk_shader_module&& other) noexcept {
        this->shader_module = other.shader_module;
        this->device = other.device;
        other.shader_module = VK_NULL_HANDLE;
        other.device = VK_NULL_HANDLE;
    }

    vk_shader_module& vk_shader_module::operator=(vk_shader_module&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        this->release();
        this->shader_module = other.shader_module;
        this->device = other.device;
        other.shader_module = VK_NULL_HANDLE;
        other.device = VK_NULL_HANDLE;
        return *this;
    }

    void vk_shader_module::release() {
        if (this->shader_module != VK_NULL_HANDLE && this->device != VK_NULL_HANDLE) {
            vkDestroyShaderModule(this->device, this->shader_module, nullptr);
        }
        this->shader_module = VK_NULL_HANDLE;
        this->device = VK_NULL_HANDLE;
    }

    std::optional<vk_shader_module> make_shader_module(std::span<unsigned char const> const shader, VkDevice const device) noexcept {
        VkShaderModule shader_module = {};

        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = shader.size_bytes();
        create_info.pCode = reinterpret_cast<uint32_t const*>(shader.data());

        if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
            return std::nullopt;
        }

        return vk_shader_module(shader_module, device);
    }
} // namespace vulkan

// vk_pipeline
namespace vulkan {
    vk_pipeline::vk_pipeline(VkPipeline const pipeline, VkPipelineLayout const pipeline_layout, VkDevice const device) noexcept { // NOLINT(*-misplaced-const)
        this->pipeline = pipeline;
        this->pipeline_layout = pipeline_layout;
        this->device = device;
    }

    vk_pipeline::vk_pipeline(vk_pipeline&& other) noexcept {
        this->device = other.device;
        this->pipeline = other.pipeline;
        this->pipeline_layout = other.pipeline_layout;
        this->viewport = other.viewport;
        this->scissor = other.scissor;
        other.device = VK_NULL_HANDLE;
        other.pipeline = VK_NULL_HANDLE;
        other.pipeline_layout = VK_NULL_HANDLE;
    }

    vk_pipeline& vk_pipeline::operator=(vk_pipeline&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        this->release();
        this->device = other.device;
        this->pipeline = other.pipeline;
        this->pipeline_layout = other.pipeline_layout;
        this->viewport = other.viewport;
        this->scissor = other.scissor;
        other.device = VK_NULL_HANDLE;
        other.pipeline = VK_NULL_HANDLE;
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

    void vk_pipeline::begin_pipeline(VkCommandBuffer const command_buffer) const {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, this->pipeline);
        vkCmdSetViewport(command_buffer, 0, 1, &this->viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &this->scissor);
    }

    void vk_pipeline::release() noexcept {
        if (this->device != VK_NULL_HANDLE) {
            if (this->pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(this->device, this->pipeline, nullptr);
            }
        }
        this->device = VK_NULL_HANDLE;
        this->pipeline = VK_NULL_HANDLE;
        this->pipeline_layout = VK_NULL_HANDLE;
    }

    // vk_image_view
    vk_image_view::vk_image_view(VkImageView const image_view, VkDevice const device) noexcept {
        this->image_view = image_view;
        this->device = device;
    }

    vk_image_view::~vk_image_view() noexcept {
        this->release();
    }

    VkImageView const& vk_image_view::get() const noexcept {
        return this->image_view;
    }

    VkImageView const& vk_image_view::operator*() const noexcept {
        return this->image_view;
    }

    void vk_image_view::release() noexcept {
        if (this->device != VK_NULL_HANDLE && this->image_view != VK_NULL_HANDLE) {
            vkDestroyImageView(this->device, this->image_view, nullptr);
        }
        this->image_view = VK_NULL_HANDLE;
        this->device = VK_NULL_HANDLE;
    }

    vk_image_view::vk_image_view(vk_image_view&& other) noexcept {
        this->image_view = other.image_view;
        this->device = other.device;
        other.image_view = VK_NULL_HANDLE;
        other.device = VK_NULL_HANDLE;
    }

    vk_image_view& vk_image_view::operator=(vk_image_view&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        this->release();
        this->image_view = other.image_view;
        this->device = other.device;
        other.image_view = VK_NULL_HANDLE;
        other.device = VK_NULL_HANDLE;
        return *this;
    }

    // vk_sampler
    vk_sampler::vk_sampler(VkSampler const sampler, VkDevice const device) noexcept {
        this->sampler = sampler;
        this->device = device;
    }

    vk_sampler::~vk_sampler() noexcept {
        this->release();
    }

    VkSampler const& vk_sampler::get() const noexcept {
        return this->sampler;
    }

    VkSampler const& vk_sampler::operator*() const noexcept {
        return this->sampler;
    }

    void vk_sampler::release() noexcept {
        if (this->device != VK_NULL_HANDLE && this->sampler != VK_NULL_HANDLE) {
            vkDestroySampler(this->device, this->sampler, nullptr);
        }
        this->sampler = VK_NULL_HANDLE;
        this->device = VK_NULL_HANDLE;
    }

    vk_sampler::vk_sampler(vk_sampler&& other) noexcept {
        this->sampler = other.sampler;
        this->device = other.device;
        other.sampler = VK_NULL_HANDLE;
        other.device = VK_NULL_HANDLE;
    }

    vk_sampler& vk_sampler::operator=(vk_sampler&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        this->release();
        this->sampler = other.sampler;
        this->device = other.device;
        other.sampler = VK_NULL_HANDLE;
        other.device = VK_NULL_HANDLE;
        return *this;
    }
} // namespace vulkan
