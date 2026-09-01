module;

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

module vulkan.core.filter;

namespace vulkan {
    core_filter::core_filter(core& core) noexcept
        : vk_core(std::addressof(core)) {
    }

    VkDevice core_filter::get_device() const noexcept {
        return this->vk_core->device;
    }

    GLFWwindow* core_filter::get_window() const noexcept {
        return this->vk_core->window;
    }

    VkExtent2D core_filter::get_swap_chain_extent() const noexcept {
        return this->vk_core->swap_chain_extent;
    }

    VkFormat core_filter::get_swap_chain_image_format() const noexcept {
        return this->vk_core->swap_chain_image_format;
    }

    uint32_t core_filter::get_current_frame() const noexcept {
        return static_cast<uint32_t>(this->vk_core->current_frame);
    }

    vk_command_buffer core_filter::make_command_buffer() const {
        return this->vk_core->make_command_buffer();
    }

    vk_descriptor_set core_filter::make_descriptor_set(VkDescriptorSetLayout const layout) const {
        return this->vk_core->make_descriptor_set(layout);
    }

    std::optional<vk_shader_module> core_filter::make_shader_module(std::span<unsigned char> const shader) const noexcept {
        return this->vk_core->make_shader_module(shader);
    }

    vk_image_view core_filter::make_image_view(VkImage const image, VkFormat const format, VkImageViewType const type) const {
        return this->vk_core->make_image_view(image, format, type);
    }

    vk_sampler core_filter::make_sampler(VkSamplerAddressMode const address_mode, float const max_lod) const {
        return this->vk_core->make_sampler(address_mode, max_lod);
    }

    // Deliberately non-const: returning a mutable vma_allocator from a const method would break
    // the const contract (a const runtime must not allocate). Some toolchains still suggest
    // adding const here, hence the suppression.
    vma_allocator& core_filter::get_vma() noexcept { // NOLINT
        return this->vk_core->vma;
    }

    void core_filter::recreate_swap_chain() const {
        this->vk_core->recreate_swap_chain();
    }
} // namespace vulkan
