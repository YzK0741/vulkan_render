module;

#include <vulkan/vulkan.h>

export module vulkan.core.handles;

export import std;

/**
 * @defgroup vulkan_handles Vulkan Main Handles' RAII Wrapper
 * @file handles.cppm
 */
namespace vulkan {
    /**
     * @ingroup vulkan_handles
     * @brief raii wrapper VkCommandBuffer
     * @note
     *     - use operator* or get() to get naked handle
     *     - sole ownership
     */
    export class vk_command_buffer {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkDevice* device = nullptr;
        VkCommandPool* command_pool = nullptr;

    public:

        [[nodiscard]] VkCommandBuffer const& get() const noexcept;
        [[nodiscard]] VkCommandBuffer const& operator*() const noexcept;
        void release() noexcept;
        explicit vk_command_buffer(VkCommandBuffer command_buffer, VkDevice& device, VkCommandPool& pool) noexcept;
        ~vk_command_buffer() noexcept;

        explicit vk_command_buffer(vk_command_buffer& command_buffer) = delete;
        vk_command_buffer(vk_command_buffer&& other) noexcept ;
        vk_command_buffer& operator=(vk_command_buffer& other) = delete;
        vk_command_buffer& operator=(vk_command_buffer&& other) noexcept;
    };

    /**
     * @ingroup vulkan_handles
     * @param device valid VkDevice
     * @param command_pool valid VkCommandPool
     * @return raii VkCommandBuffer wrapper
     */
    export vk_command_buffer make_command_buffer(VkDevice &device, VkCommandPool &command_pool) noexcept;

    /**
     * @ingroup vulkan_handles
     * @brief raii wrapper of VkDescriptorSet
     * @note
     *     - use operator* or get() to get naked handle
     *     - sole ownership
     */
    export class vk_descriptor_set {
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VkDevice* device = nullptr;
        VkDescriptorPool* descriptor_pool = nullptr;

    public:

        [[nodiscard]] VkDescriptorSet const& get() const noexcept;
        [[nodiscard]] VkDescriptorSet const& operator*() const noexcept;
        void release() noexcept;
        explicit vk_descriptor_set(VkDescriptorSet descriptor_set, VkDevice& device, VkDescriptorPool& descriptor_pool) noexcept;
        ~vk_descriptor_set() noexcept;

        explicit vk_descriptor_set(vk_descriptor_set& descriptor_set) = delete;
        vk_descriptor_set(vk_descriptor_set&& other) noexcept ;
        vk_descriptor_set& operator=(vk_descriptor_set& other) = delete;
        vk_descriptor_set& operator=(vk_descriptor_set&& other) noexcept;
    };

    /**
     * @ingroup vulkan_handles
     * @param device valid VkDevice
     * @param descriptor_pool valid VkDescriptorPool
     * @param layout valid VkDescriptorSetLayout
     * @return raii wrapper of VkDescriptorSet
     */
    export vk_descriptor_set make_descriptor_set(VkDevice &device, VkDescriptorPool &descriptor_pool, VkDescriptorSetLayout const &layout) noexcept;

    /**
     * @ingroup vulkan_handles
     * @brief raii wrapper of VkShaderModule
     * @note
     *     - use operator* or get() to get naked handle
     *     - sole ownership
     */
    export class vk_shader_module {
        VkShaderModule shader_module = VK_NULL_HANDLE;
        VkDevice* device = nullptr;

    public:
        [[nodiscard]] VkShaderModule const& get() const noexcept;
        [[nodiscard]] VkShaderModule const& operator*() const noexcept;
        void release();
        explicit vk_shader_module(const VkShaderModule &shader_module, VkDevice &device) noexcept;
        ~vk_shader_module() noexcept;

        explicit vk_shader_module(vk_shader_module& shader_module) = delete;
        vk_shader_module(vk_shader_module&& other) noexcept;
        vk_shader_module& operator=(vk_shader_module& other) = delete;
        vk_shader_module& operator=(vk_shader_module&& other) noexcept;
    };

    /**
     * @ingroup vulkan_handles
     * @param shader binary shader code
     * @param device valid VkDevice
     * @return success: raii wrapper of VkShaderModule
     *     fail: std::nullopt
     */
    export std::optional<vk_shader_module> make_shader_module(std::span<const unsigned char> shader,
                                                              VkDevice &device) noexcept;

    /**
     * @ingroup vulkan_handles
     * @brief raii wrapper of VkPipeline
     * @note
     *     - use operator* or get() to get naked handle
     *     - sole ownership
     */
    export struct vk_pipeline {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

        std::vector<VkDescriptorSetLayout> descriptor_set_layouts = {};
        VkDevice* device = nullptr;

        explicit vk_pipeline(VkPipeline pipeline, VkPipelineLayout pipeline_layout, std::vector<VkDescriptorSetLayout> const& descriptor_set_layouts, VkDevice& device) noexcept; // NOLINT(*-avoid-const-params-in-decls)
        ~vk_pipeline();
        [[nodiscard]] VkPipeline get_pipeline() const noexcept;
        [[nodiscard]] VkPipelineLayout get_pipeline_layout() const noexcept;
        [[nodiscard]] std::vector<VkDescriptorSetLayout> get_descriptor_set_layouts() const noexcept;

        vk_pipeline(vk_pipeline&) = delete;
        vk_pipeline(vk_pipeline&& other) noexcept;
        vk_pipeline& operator=(vk_pipeline& other) = delete;
        vk_pipeline& operator=(vk_pipeline&& other) noexcept;
        void release() noexcept;
    };
}
