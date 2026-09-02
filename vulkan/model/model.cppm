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
        // RGBA8 bytes of mip level 0 (or the whole mip chain, mip-major: mip0, mip1, ..., when
        // mip_levels > 1); the vma upload path copies each level at a computed buffer offset
        std::span<unsigned char const> data = {};
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t mip_levels = 1;
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        bool valid = false;
    };

    /**
     * @ingroup vulkan_model
     * @brief precomputed split-sum IBL resources as half-float bytes, ready for upload
     * @note env_size == 0 disables the IBL bindings
     */
    export struct ibl_input {
        std::span<unsigned char const> prefiltered_env = {};
        std::span<unsigned char const> irradiance = {};
        std::span<unsigned char const> brdf_lut = {};
        uint32_t env_size = 0;
        uint32_t env_mip_count = 0;
        uint32_t irr_size = 0;
        uint32_t lut_size = 0;
    };

    /**
     * @ingroup vulkan_model
     * @brief PBR material factors, mirrored into the GPU material table (material_record)
     * @note the same shape as gltf::material_factors, converted by the scene builder
     */
    export struct material_factors {
        glm::vec4 base_color_factor = glm::vec4(1.0f);
        glm::vec4 emissive_factor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        float metallic_factor = 1.0f;
        float roughness_factor = 1.0f;
        float normal_scale = 1.0f;
    };

    /**
     * @ingroup vulkan_model
     * @brief result of one runtime::import_scene() batch import
     */
    export struct scene_import_result {
        uint32_t primitive_count = 0;
        uint32_t material_count = 0;
    };

    // The scene_drawable_iterator concept is STRUCTURAL over the getters' result shapes, so a
    // scene iterator can satisfy it with its own pure-CPU types (e.g. the glTF loader's) — no
    // shared type identity with vulkan.model is required. The runtime template converts the
    // read values (spans / widths / factors) into its internal types.

    /** @brief a vertex source: interleaved byte span + stride + vertex count */
    export template <class T>
    concept vertex_source = requires(T const& v) {
        { v.data } -> std::convertible_to<std::span<unsigned char const>>;
        { v.stride } -> std::convertible_to<uint32_t>;
        { v.count } -> std::convertible_to<uint32_t>;
    };

    /** @brief an index source: byte span + bytes-per-index (2 or 4) + index count */
    export template <class T>
    concept index_source = requires(T const& v) {
        { v.data } -> std::convertible_to<std::span<unsigned char const>>;
        { v.width } -> std::convertible_to<unsigned char>;
        { v.count } -> std::convertible_to<uint32_t>;
    };

    /** @brief a texture source: mip-major RGBA8 byte span + dimensions + validity */
    export template <class T>
    concept image_source = requires(T const& v) {
        { v.data } -> std::convertible_to<std::span<unsigned char const>>;
        { v.width } -> std::convertible_to<uint32_t>;
        { v.height } -> std::convertible_to<uint32_t>;
        { v.mip_levels } -> std::convertible_to<uint32_t>;
        { v.valid } -> std::convertible_to<bool>;
    };

    /** @brief a PBR factors source: the same field names/shapes as material_factors */
    export template <class T>
    concept factors_source = requires(T const& v) {
        { v.base_color_factor } -> std::convertible_to<glm::vec4>;
        { v.emissive_factor } -> std::convertible_to<glm::vec4>;
        { v.metallic_factor } -> std::convertible_to<float>;
        { v.roughness_factor } -> std::convertible_to<float>;
        { v.normal_scale } -> std::convertible_to<float>;
    };

    /**
     * @ingroup vulkan_model
     * @brief concept for a scene-traversal iterator the runtime can consume directly:
     *        ++ moves to the next drawable, then geometry/material are read through the
     *        getters (vertex/index/transform + one getter per material slot). The getters
     *        return pure CPU values (byte spans etc., see the *_source concepts above);
     *        the runtime template converts them to its internal types (formats, index type).
     *        A missing material slot is reported through image_source::valid == false and the
     *        runtime falls back to its white texture.
     */
    export template <class I>
    concept scene_drawable_iterator = requires(I& it, I const& end) {
        { ++it } -> std::same_as<I&>;
        { it != end } -> std::convertible_to<bool>;
        { it.get_vertex() } -> vertex_source;
        { it.get_index() } -> index_source;
        { it.get_transform() } -> std::convertible_to<glm::mat4>;
        { it.get_albedo() } -> image_source;
        { it.get_metallic_roughness() } -> image_source;
        { it.get_normal() } -> image_source;
        { it.get_occlusion() } -> image_source;
        { it.get_emissive() } -> image_source;
        { it.get_factors() } -> factors_source;
    };

    /**
     * @ingroup vulkan_model
     * @brief everything runtime::make_model() needs: geometry + material textures + factors
     */
    export struct model_create_info {
        std::span<unsigned char const> vertex_data = {};
        uint32_t vertex_stride = 0;
        uint32_t vertex_count = 0;
        std::span<unsigned char const> index_data = {};
        VkIndexType index_type = VK_INDEX_TYPE_UINT32;
        uint32_t index_count = 0;

        texture_input albedo = {};
        texture_input metallic_roughness = {};
        texture_input normal = {};
        texture_input occlusion = {};
        texture_input emissive = {};

        // PBR factors, stored in the model's material_record
        material_factors factors = {};

        // world transform applied to the geometry (e.g. fit-scale + centering from the bounding box);
        // pushed per model (the shared camera UBO carries no model matrix)
        glm::mat4 model_matrix = glm::mat4(1.0f);
    };

    /**
     * @ingroup vulkan_model
     * @brief one entry of the scene's GPU-side material table (set 0 binding 5, a storage buffer):
     *        the 5 texture array indices + all material parameters. Models only push a
     *        material_index and the shader reads the record — material data lives in one
     *        GPU-visible place and is shareable between models
     * @note layout matches the Material struct in pbr.frag (std430, 80 bytes)
     */
    export struct material_record {
        glm::uvec4 tex_indices = {}; // albedo, metallic-roughness, normal, occlusion (indices into the texture array)
        uint32_t emissive_index = 0; // emissive texture index
        uint32_t _pad[3] = {};       // keep the vec4 members 16-byte aligned (std430)
        glm::vec4 base_color_factor = glm::vec4(1.0f);
        glm::vec4 emissive_factor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        float metallic_factor = 1.0f;
        float roughness_factor = 1.0f;
        float normal_scale = 1.0f;
        uint32_t flags = 0; // bit0: normal map, bit1: occlusion map, bit2: emissive map
    };
    static_assert(sizeof(material_record) == 80);

    /**
     * @ingroup vulkan_model
     * @brief max entries of the GPU material table
     */
    export constexpr uint32_t material_capacity = 256;

    /**
     * @ingroup vulkan_model
     * @brief max per-instance transforms of an instanced draw (set 0 binding 6 storage buffer)
     */
    export constexpr uint32_t instance_capacity = 8192;

    /**
     * @ingroup vulkan_model
     * @brief per-draw push constants, layout matches pbr.frag's PushConstants (80 bytes)
     * @note material data (texture indices, factors, flags) lives in the material table
     *       (set 0 binding 5), so the push block only carries the material reference and the
     *       per-model world transform
     */
    export struct material_push_constants {
        uint32_t material_index = 0; // index into the scene's material table
        uint32_t flags = 0;          // bit0: instanced draw -> model matrix comes from the
                                     //       instance transform buffer (set 0 binding 6)
        // glm::mat4 is only 4-byte aligned by default, but GLSL std430 aligns mat4 to 16 bytes
        // (offset 16 in the block): align explicitly so the CPU layout matches the shader
        alignas(16) glm::mat4 model = glm::mat4(1.0f);
    };
    static_assert(sizeof(material_push_constants) == scene_push_constant_size);

    /**
     * @ingroup vulkan_model
     * @brief base class of every drawable: owns geometry buffers + material push constants and
     *        declares the draw strategy interface. Derived classes implement how the geometry
     *        is drawn (single draw, instanced grid, ...), so the runtime's frame loop stays a
     *        generic "for each model: model->draw()" — new strategies only add a subclass.
     * @note
     *      - owns only its geometry (vma buffers); textures live in the runtime's shared texture
     *        array and descriptor sets are owned by the runtime (single scene set, bound once)
     *      - the runtime binds the pipeline and the scene set before calling draw()
     *      - destroy() frees whatever the instance owns (vma buffers); call it before teardown
     */
    export class model {
    public:
        virtual ~model() = default;

        // geometry: vma handles for release, detail pointers for access (no raw Vulkan objects)
        uint64_t vertex_buffer_handle = 0;
        buffer_detail const* vertex_detail = nullptr;
        uint64_t index_buffer_handle = 0;
        buffer_detail const* index_detail = nullptr;
        VkIndexType index_type = VK_INDEX_TYPE_UINT32;
        uint32_t index_count = 0;
        uint32_t vertex_count = 0;

        // pipeline the model binds against (points into the runtime's pipeline cache; valid for
        // the runtime's lifetime) and the material push constants (material_index + model)
        vk_pipeline const* pipeline = nullptr;
        material_push_constants push = {};

        /**
         * @brief record the model's draw commands (the runtime already bound the pipeline and
         *        the shared scene descriptor set)
         * @param command_buffer the command buffer being recorded
         */
        virtual void draw(VkCommandBuffer command_buffer) const = 0;
        virtual void destroy(vma_allocator& vma) noexcept = 0;
        [[nodiscard]] virtual bool is_valid() const noexcept = 0;

    protected:
        // shared recording: bind this object's geometry buffers and push the push constants
        void bind_geometry_and_push(VkCommandBuffer command_buffer) const;
    };

    /**
     * @ingroup vulkan_model
     * @brief the standard drawable: one indexed draw of its own geometry (push.model places it)
     */
    export class normal_draw_model final : public model {
    public:
        void draw(VkCommandBuffer command_buffer) const override;
        void destroy(vma_allocator& vma) noexcept override;
        [[nodiscard]] bool is_valid() const noexcept override;
    };

    /**
     * @ingroup vulkan_model
     * @brief instanced drawable: draws the geometry of another model (source) instance_count
     *        times in ONE draw call; per-instance world transforms come from the runtime's
     *        instance transform buffer (scene set binding 6, push flag bit0). Owns nothing:
     *        geometry belongs to source, destroy() is a no-op, source must outlive this model.
     */
    export class instanced_draw_model final : public model {
    public:
        model const* source = nullptr;
        uint32_t instance_count = 0;

        void draw(VkCommandBuffer command_buffer) const override;
        void destroy(vma_allocator& vma) noexcept override;
        [[nodiscard]] bool is_valid() const noexcept override;
    };

    /**
     * @ingroup vulkan_model
     * @brief build the camera UBO from orbit camera state (the camera orbits the target point)
     * @param yaw yaw angle in radians (see vulkan::runtime::camera)
     * @param pitch pitch angle in radians
     * @param distance camera distance from the target
     * @param target the point the camera looks at and orbits around (e.g. the centered scene origin,
     *        or the scene sink so the camera follows the model)
     * @param aspect swapchain width / height
     * @return camera UBO with view/proj/camera_pos filled in
     * @note proj uses perspectiveRH_ZO with a Y flip to match Vulkan's y-down framebuffer
     */
    export camera_ubo make_orbit_camera_ubo(
        float yaw,
        float pitch,
        float distance,
        glm::vec3 const& target,
        float aspect);
} // namespace vulkan
