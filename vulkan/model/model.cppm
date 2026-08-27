module;

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

export module vulkan.model;
export import std;
export import vulkan.core.handles;
export import vulkan.vma;

/**
 * @file model.cppm
 * @defgroup vulkan_model Vulkan Model
 * @brief GPU model management: vertex/index buffers, descriptor set, draw command,
 *        plus camera/MVP helpers and CPU-side IBL precomputation (environment mip chains)
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
    export struct model {
        uint32_t vertex_buffer_handle = 0;
        VkBuffer vertex_buffer = VK_NULL_HANDLE;
        uint32_t index_buffer_handle = 0;
        VkBuffer index_buffer = VK_NULL_HANDLE;
        uint32_t index_count = 0;

        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

        void draw(VkCommandBuffer command_buffer, VkPipelineLayout pipeline_layout) const;
        void destroy(VkDevice device, VkDescriptorPool descriptor_pool, vma_allocator& vma);
        [[nodiscard]] bool is_valid() const noexcept;
    };

    /**
     * @ingroup vulkan_model
     * @brief camera UBO content, layout matches the CameraUBO block in pbr.frag
     */
    export struct camera_ubo {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
        glm::vec3 camera_pos;
        float padding = 0.0f;
    };

    /**
     * @ingroup vulkan_model
     * @brief build the camera UBO from orbit camera state (the model is assumed centered at the origin)
     * @param yaw yaw angle in radians (see vulkan::runtime::camera)
     * @param pitch pitch angle in radians
     * @param distance camera distance from the origin
     * @param model the model matrix (fit scale + centering)
     * @param aspect swapchain width / height
     * @return camera UBO with view/proj/camera_pos filled in; model is copied through
     * @note proj uses perspectiveRH_ZO with a Y flip to match Vulkan's y-down framebuffer
     */
    export camera_ubo make_orbit_camera_ubo(
        float yaw,
        float pitch,
        float distance,
        const glm::mat4& model,
        float aspect);

    // ---- CPU 端 IBL 预计算（预过滤 mip 链 / 辐照度 / BRDF LUT / 半精度转换） ----

    /**
     * @ingroup vulkan_model
     * @brief generate a procedural HDR environment cubemap (RGBA32F, 6 faces packed)
     */
    export std::vector<float> generate_environment_cubemap(int size);

    /**
     * @ingroup vulkan_model
     * @brief GGX importance-sampled prefilter of the environment into a mip chain
     * @param env the base environment cubemap from generate_environment_cubemap()
     * @param env_size base cubemap size
     * @param mip_count number of mip levels
     * @return mip-major RGBA32F data (mip0 all faces, then mip1, ...), ready for a cubemap upload
     */
    export std::vector<float> prefilter_environment(const std::vector<float>& env, int env_size, int mip_count);

    /**
     * @ingroup vulkan_model
     * @brief cosine-weighted hemisphere convolution for the diffuse irradiance cubemap
     */
    export std::vector<float> generate_irradiance_map(const std::vector<float>& env, int env_size, int irr_size);

    /**
     * @ingroup vulkan_model
     * @brief BRDF integration LUT filled with the Frostbite analytic approximation (RG32F: scale, bias)
     */
    export std::vector<float> generate_brdf_lut(int size);

    /**
     * @ingroup vulkan_model
     * @brief convert 4-channel float data to a packed RGBA16F byte stream
     */
    export std::vector<unsigned char> to_half_rgba(const std::vector<float>& data);

    /**
     * @ingroup vulkan_model
     * @brief convert 2-channel float data to a packed RG16F byte stream
     */
    export std::vector<unsigned char> to_half_rg(const std::vector<float>& data);
} // namespace vulkan
