module;

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

export module vulkan.model;
export import std;
export import vulkan.core;

/**
 * @file model.cppm
 * @defgroup vulkan_model Vulkan Model
 * @brief GPU model management: geometry buffers, material descriptor set, per-frame UBO,
 *        plus camera/MVP helpers and CPU-side IBL precomputation (environment mip chains)
 * @note
 *      - a model owns every GPU resource it needs to draw itself
 *      - create via vulkan::make_model() (wrapped by vulkan::runtime::make_model), release via destroy()
 */
namespace vulkan {
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
     * @brief RGBA texture pixels ready for GPU upload (already converted to the target format)
     * @note valid == false means "missing texture", the model falls back to a 1x1 white image
     */
    export struct texture_input {
        std::span<const unsigned char> data = {};
        uint32_t width = 0;
        uint32_t height = 0;
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        bool valid = false;
    };

    /**
     * @ingroup vulkan_model
     * @brief precomputed split-sum IBL resources as half-float bytes, ready for upload
     * @note env_size == 0 disables the IBL bindings
     */
    export struct ibl_input {
        std::span<const unsigned char> prefiltered_env = {};
        std::span<const unsigned char> irradiance = {};
        std::span<const unsigned char> brdf_lut = {};
        uint32_t env_size = 0;
        uint32_t env_mip_count = 0;
        uint32_t irr_size = 0;
        uint32_t lut_size = 0;
    };

    /**
     * @ingroup vulkan_model
     * @brief everything make_model() needs: geometry + material textures + IBL
     */
    export struct model_create_info {
        std::span<const unsigned char> vertex_data = {};
        uint32_t vertex_stride = 0;
        uint32_t vertex_count = 0;
        std::span<const unsigned char> index_data = {};
        VkIndexType index_type = VK_INDEX_TYPE_UINT32;
        uint32_t index_count = 0;

        texture_input albedo = {};
        texture_input metallic_roughness = {};
        texture_input normal = {};
        texture_input occlusion = {};
        texture_input emissive = {};
        ibl_input ibl = {};

        // world transform applied to the geometry (e.g. fit-scale + centering from the bounding box)
        glm::mat4 model_matrix = glm::mat4(1.0f);
    };

    /**
     * @ingroup vulkan_model
     * @brief material parameters pushed at draw time, layout matches pbr.frag's PushConstants (48 bytes)
     */
    export struct material_push_constants {
        glm::vec4 base_color_factor = glm::vec4(1.0f);
        glm::vec4 emissive_factor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        float metallic_factor = 1.0f;
        float roughness_factor = 1.0f;
        float normal_scale = 1.0f;
        uint32_t flags = 0; // bit0: normal map, bit1: occlusion map, bit2: emissive map
    };
    static_assert(sizeof(material_push_constants) == 48);

    /**
     * @ingroup vulkan_model
     * @brief a GPU model: geometry buffers, material descriptor set (set 1) and
     *        per-frame camera UBO descriptor sets (set 0)
     * @note
     *      - owns every GPU resource it needs to draw itself, including its pipeline reference
     *        and material push constants
     *      - draw() binds the pipeline, both descriptor sets, geometry and push constants, then draws
     *      - destroy() frees the vma buffers/images; RAII views/sets/samplers free themselves
     */
    export struct model {
        // geometry: vma handles for release, detail pointers for access (no raw Vulkan objects)
        uint64_t vertex_buffer_handle = 0;
        const buffer_detail* vertex_detail = nullptr;
        uint64_t index_buffer_handle = 0;
        const buffer_detail* index_detail = nullptr;
        VkIndexType index_type = VK_INDEX_TYPE_UINT32;
        uint32_t index_count = 0;
        uint32_t vertex_count = 0;

        // pipeline the model binds against (points into the runtime's pipeline cache; valid for
        // the runtime's lifetime), the world transform and the material push constants
        const vk_pipeline* pipeline = nullptr;
        glm::mat4 model_matrix = glm::mat4(1.0f);
        material_push_constants push = {};

        // set 1: material textures + IBL images (vma handles to free; views are RAII)
        std::vector<uint64_t> image_handles = {};
        std::vector<vk_image_view> image_views = {};
        vk_descriptor_set material_set = {};
        vk_sampler texture_sampler = {};
        vk_sampler env_sampler = {};

        // set 0: per-frame camera UBO
        std::vector<uint64_t> ubo_buffer_handles = {};
        std::vector<void*> ubo_mapped = {};
        std::vector<vk_descriptor_set> ubo_sets = {};
        size_t ubo_size = 0;

        /**
         * @brief record the model's draw commands; the pipeline and push constants come from the model
         * @param command_buffer the command buffer being recorded
         * @param frame_slot the frame slot whose camera UBO set to bind (see vulkan::core::MAX_FRAMES_IN_FLIGHT)
         */
        void draw(VkCommandBuffer command_buffer, uint32_t frame_slot) const;
        void update_camera_ubo(uint32_t frame_slot, const camera_ubo& ubo) noexcept;
        void destroy(vma_allocator& vma) noexcept;
        [[nodiscard]] bool is_valid() const noexcept;
    };

    /**
     * @ingroup vulkan_model
     * @brief build a model (geometry + material + IBL + per-frame UBOs) from plain data
     * @param core the vulkan core (vma / device / descriptor pool / view & sampler factories)
     * @param pipeline the pipeline whose descriptor set layouts the model binds against
     * @param info geometry, material textures and IBL bytes
     * @return the created model, owning all of its GPU resources
     * @note panics on allocation failure (consistent with the rest of the engine)
     */
    export model make_model(vulkan::core& core, const vk_pipeline& pipeline, const model_create_info& info);

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

    // ---- CPU-side IBL precomputation (prefiltered mip chain / irradiance / BRDF LUT / half-float conversion) ----

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
