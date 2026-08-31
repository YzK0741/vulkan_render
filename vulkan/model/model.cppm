module;

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

export module vulkan.model;
export import std;
export import vulkan.core;

/**
 * @file model.cppm
 * @defgroup vulkan_model Vulkan Model
 * @brief GPU model management: geometry buffers, material push constants, plus camera/MVP
 *        helpers (CPU-side IBL precomputation lives in the vulkan.math module)
 * @note
 *      - a model owns its geometry and the material push constants; it does NOT own descriptor
 *        sets — the runtime owns the single flat scene descriptor set (camera UBO + texture
 *        array + IBL, see core::init_scene_layouts) and binds it once per frame
 *      - create via vulkan::runtime::make_model(), release via destroy()
 */
namespace vulkan {
    /**
     * @ingroup vulkan_model
     * @brief camera UBO content, layout matches the CameraUBO block in pbr.frag (no model
     *        matrix: the per-model world transform lives in the push constants instead, so the
     *        camera UBO can be shared by every model)
     */
    export struct camera_ubo {
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
     * @brief everything runtime::make_model() needs: geometry + material textures
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

        // world transform applied to the geometry (e.g. fit-scale + centering from the bounding box);
        // pushed per model (the shared camera UBO carries no model matrix)
        glm::mat4 model_matrix = glm::mat4(1.0f);
    };

    /**
     * @ingroup vulkan_model
     * @brief material parameters pushed at draw time, layout matches pbr.frag's PushConstants (128 bytes)
     * @note
     *      - texture_base indexes into the runtime's shared scene texture array
     *        (albedo +0, metallic-roughness +1, normal +2, occlusion +3, emissive +4)
     *      - model is the per-model world transform (the shared camera UBO carries no model matrix)
     */
    export struct material_push_constants {
        glm::vec4 base_color_factor = glm::vec4(1.0f);
        glm::vec4 emissive_factor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        float metallic_factor = 1.0f;
        float roughness_factor = 1.0f;
        float normal_scale = 1.0f;
        uint32_t flags = 0; // bit0: normal map, bit1: occlusion map, bit2: emissive map
        uint32_t texture_base = 0;
        // glm::mat4 is only 4-byte aligned by default, but GLSL std430 aligns mat4 to 16 bytes
        // (offset 64 in the block): align explicitly so the CPU layout matches the shader
        alignas(16) glm::mat4 model = glm::mat4(1.0f);
    };
    static_assert(sizeof(material_push_constants) == scene_push_constant_size);

    /**
     * @ingroup vulkan_model
     * @brief a GPU model: geometry buffers + material push constants + texture array base index
     * @note
     *      - owns only its geometry (vma buffers); textures live in the runtime's shared texture
     *        array and descriptor sets are owned by the runtime (single scene set, bound once)
     *      - draw() binds geometry, pushes the material constants and draws; the runtime binds
     *        the pipeline and the scene set before the model loop
     *      - destroy() frees the geometry buffers
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
        // the runtime's lifetime) and the material push constants (includes texture_base + model)
        const vk_pipeline* pipeline = nullptr;
        material_push_constants push = {};

        /**
         * @brief record the model's draw commands (geometry + push constants + indexed draw)
         * @param command_buffer the command buffer being recorded
         * @note the pipeline and the shared scene descriptor set are bound by the runtime
         */
        void draw(VkCommandBuffer command_buffer) const;
        void destroy(vma_allocator& vma) noexcept;
        [[nodiscard]] bool is_valid() const noexcept;
    };

    /**
     * @ingroup vulkan_model
     * @brief build the camera UBO from orbit camera state (the scene is assumed centered at the origin)
     * @param yaw yaw angle in radians (see vulkan::runtime::camera)
     * @param pitch pitch angle in radians
     * @param distance camera distance from the origin
     * @param aspect swapchain width / height
     * @return camera UBO with view/proj/camera_pos filled in
     * @note proj uses perspectiveRH_ZO with a Y flip to match Vulkan's y-down framebuffer
     */
    export camera_ubo make_orbit_camera_ubo(
        float yaw,
        float pitch,
        float distance,
        float aspect);
} // namespace vulkan
