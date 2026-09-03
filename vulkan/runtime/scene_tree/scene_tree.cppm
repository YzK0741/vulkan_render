module;

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

export module vulkan.runtime.scene_tree;
export import std;
export import vulkan.core;

/**
 * @file scene_tree.cppm
 * @defgroup vulkan_runtime_scene_tree Vulkan Runtime Scene Tree
 * @brief scene organization: a transform hierarchy of scene_node objects with
 *        primitive leaves, plus the GPU primitives that live in those leaves.
 *
 * One module for the whole scene-tree concept (absorbed vulkan.model):
 *   - storage: scene / scene_node { name, local, children, primitive_leaf } and the
 *     abstract leaf interface scene_tree::primitive (pure virtual set_world); the
 *     CPU-side walkers update_world() / visit_primitives() accumulate world
 *     transforms every frame
 *   - GPU primitives: vulkan::primitive (owns geometry buffers + material push
 *     constants, implements scene_tree::primitive) and its draw strategies
 *     normal_draw_primitive / instanced_draw_primitive
 *   - GPU material / UBO records (material_record, material_push_constants,
 *     camera_ubo, light_ubo) and the structural iterator concepts
 *     (scene_drawable_iterator / scene_node_iterator) the import templates drive
 * @note imports vulkan.core for the GPU types (vk_pipeline, vma handles,
 *       buffer_detail); the module/header used to be split as vulkan.model and was
 *       merged here so the scene-tree concept owns its primitives in one place
 */
namespace vulkan::scene_tree {
    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief abstract primitive leaf of a scene node (the interface the GPU primitives
     *        below implement: normal_draw_primitive / instanced_draw_primitive)
     * @note pure interface: implementations own their GPU geometry and record
     *       their draw commands; scene_tree only feeds them their accumulated
     *       world transform every frame
     */
    export class primitive {
    public:
        virtual ~primitive() = default;

        /**
         * @brief store the accumulated world transform of the owning node
         * @param world parent_world * node.local (computed by update_world)
         */
        virtual void set_world(glm::mat4 const& world) = 0;
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief one node of the scene tree: a local transform, child nodes and an
     *        optional primitive leaf. Value semantics: children are owned inline
     *        (copying a node copies its subtree).
     */
    export struct scene_node {
        std::string name = {};             // debugging / future animation lookup
        glm::mat4 local = glm::mat4(1.0f); // local transform (T*R*S or full matrix)
        std::vector<scene_node> children = {};
        std::unique_ptr<primitive> primitive_leaf = {}; // null for transform-only nodes

        scene_node() = default;
        // unique_ptr makes the node non-copyable; define an explicit clone for subtree copies
        scene_node(scene_node&&) noexcept = default;
        scene_node& operator=(scene_node&&) noexcept = default;
        scene_node(scene_node const&) = delete;
        scene_node& operator=(scene_node const&) = delete;

        /** @brief deep-copy this subtree (children and all) */
        [[nodiscard]] scene_node clone() const;
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief a named scene: a list of root nodes (mirrors gltf::scene's shape)
     */
    export struct scene {
        std::string name = {};
        std::vector<scene_node> roots = {};
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief depth-first walk that accumulates world transforms and pushes them
     *        into every primitive leaf: world(child) = world(parent) * child.local
     * @param node subtree root to walk (call once per scene root with mat4(1))
     * @param parent_world accumulated world of this node's parent
     */
    export void update_world(scene_node& node, glm::mat4 const& parent_world);

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief walk the subtree and call @p visit on every primitive leaf
     * @tparam F invocable(scene_node const&, glm::mat4 const& world)
     */
    export template <class F>
    void visit_primitives(scene_node const& node, glm::mat4 const& parent_world, F&& visit) {
        glm::mat4 const world = parent_world * node.local;
        if (node.primitive_leaf) {
            visit(node, world);
        }
        for (scene_node const& child : node.children) {
            visit_primitives(child, world, visit);
        }
    }
} // namespace vulkan::scene_tree

namespace vulkan {
    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief camera UBO content, layout matches the CameraUBO block in pbr.frag (no model
     *        matrix: the per-primitive world transform lives in the push constants instead,
     *        so the camera UBO can be shared by every primitive)
     */
    export struct camera_ubo {
        glm::mat4 view;
        glm::mat4 proj;
        glm::vec3 camera_pos;
        float padding = 0.0f;
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief directional light UBO content, layout matches the LightUBO block in pbr.frag /
     *        shadow.vert (scene set binding 7): light-space view-proj plus the light direction
     * @note single fixed directional sun: the runtime builds it from the scene bounds and both
     *       shader stages read it, so the sun disc, the PBR direct light and the shadow map
     *       sampling all agree on one direction
     */
    export struct light_ubo {
        glm::mat4 light_view_proj; // world -> light clip space (orthographic)
        glm::vec4 light_dir;       // xyz: normalized light direction (sun)
    };
    static_assert(sizeof(light_ubo) == 80);

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief RGBA texture pixels ready for GPU upload (already converted to the target format)
     * @note valid == false means "missing texture", the primitive falls back to a 1x1 white image
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
     * @ingroup vulkan_runtime_scene_tree
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
     * @ingroup vulkan_runtime_scene_tree
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
     * @ingroup vulkan_runtime_scene_tree
     * @brief result of one runtime::import_scene() batch import
     */
    export struct scene_import_result {
        uint32_t primitive_count = 0;
        uint32_t material_count = 0;
    };

    // The scene_drawable_iterator concept is STRUCTURAL over the getters' result shapes, so a
    // scene iterator can satisfy it with its own pure-CPU types (e.g. the glTF loader's) — no
    // shared type identity is required. The runtime template converts the read values (spans /
    // widths / factors) into its internal types.

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
     * @ingroup vulkan_runtime_scene_tree
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
        { it.get_double_sided() } -> std::convertible_to<bool>;
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief concept for a scene-tree structural iterator: DFS pre-order over the retained
     *        node hierarchy (transform-only nodes included), so a consumer can rebuild the
     *        parent/child edges with an explicit stack. ++ moves to the next node, then the
     *        node's identity is read through get_name() / get_local_transform() / get_depth()
     *        and its drawable load through get_drawable_count() (the number of drawables of
     *        this node; 0 = transform-only node). The paired drawable stream (a
     *        scene_drawable_iterator over the same pool) stays aligned node-for-node.
     */
    export template <class I>
    concept scene_node_iterator = requires(I& it, I const& end) {
        { ++it } -> std::same_as<I&>;
        { it != end } -> std::convertible_to<bool>;
        { it.get_name() } -> std::convertible_to<std::string_view>;
        { it.get_local_transform() } -> std::convertible_to<glm::mat4>;
        { it.get_depth() } -> std::convertible_to<std::size_t>;
        { it.get_drawable_count() } -> std::convertible_to<std::size_t>;
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief everything runtime::make_primitive() needs: geometry + material textures + factors
     */
    export struct primitive_create_info {
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

        // PBR factors, stored in the primitive's material_record
        material_factors factors = {};

        // glTF doubleSided: render back faces and flip their normals (cull mode + record flag)
        bool double_sided = false;

        // world transform applied to the geometry (e.g. fit-scale + centering from the bounding box);
        // pushed per primitive (the shared camera UBO carries no model matrix)
        glm::mat4 model_matrix = glm::mat4(1.0f);
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief one entry of the scene's GPU-side material table (set 0 binding 5, a storage buffer):
     *        the 5 texture array indices + all material parameters. Primitives only push a
     *        material_index and the shader reads the record — material data lives in one
     *        GPU-visible place and is shareable between primitives
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
        uint32_t flags = 0; // bit0: normal map, bit1: occlusion map, bit2: emissive map, bit3: double-sided
    };
    static_assert(sizeof(material_record) == 80);

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief max entries of the GPU material table
     */
    export constexpr uint32_t material_capacity = 256;

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief max per-instance transforms of an instanced draw (set 0 binding 6 storage buffer)
     */
    export constexpr uint32_t instance_capacity = 8192;

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief per-draw push constants, layout matches pbr.frag's PushConstants (80 bytes)
     * @note material data (texture indices, factors, flags) lives in the material table
     *       (set 0 binding 5), so the push block only carries the material reference and the
     *       per-primitive world transform
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
     * @ingroup vulkan_runtime_scene_tree
     * @brief base class of every GPU primitive: owns geometry buffers + material push constants
     *        and declares the draw strategy interface. Derived classes implement how the
     *        geometry is drawn (single draw, instanced grid, ...), so the runtime's frame loop
     *        stays a generic "for each primitive: primitive->draw()" — new strategies only add
     *        a subclass. Implements the scene tree's leaf concept (scene_tree::primitive).
     * @note
     *      - owns only its geometry (vma buffers); textures live in the runtime's shared texture
     *        array and descriptor sets are owned by the runtime (single scene set, bound once)
     *      - the runtime binds the pipeline and the scene set before calling draw()
     *      - destroy() frees whatever the instance owns (vma buffers); call it before teardown
     *      - a scene tree node holds one of these as its primitive_leaf and update_world() feeds
     *        the accumulated world matrix straight into push.model (the push block layout is
     *        shared, so draw() keeps working unchanged)
     */
    export class primitive : public vulkan::scene_tree::primitive {
    public:
        virtual ~primitive() = default;

        // geometry: vma handles for release, detail pointers for access (no raw Vulkan objects)
        uint64_t vertex_buffer_handle = 0;
        buffer_detail const* vertex_detail = nullptr;
        uint64_t index_buffer_handle = 0;
        buffer_detail const* index_detail = nullptr;
        VkIndexType index_type = VK_INDEX_TYPE_UINT32;
        uint32_t index_count = 0;
        uint32_t vertex_count = 0;

        // pipeline the primitive binds against (points into the runtime's pipeline cache; valid
        // for the runtime's lifetime) and the material push constants (material_index + model)
        vk_pipeline const* pipeline = nullptr;
        material_push_constants push = {};
        bool double_sided = false; // glTF doubleSided: disable back-face culling (per draw)

        /**
         * @brief store the world transform accumulated by the owning scene tree node
         * @param world the node's world matrix (parent_world * local)
         * @note the primitive's world transform lives in push.model, which draw() pushes as-is
         */
        void set_world(glm::mat4 const& world) override;

        /**
         * @brief record the primitive's draw commands (the runtime already bound the pipeline
         *        and the shared scene descriptor set)
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
     * @ingroup vulkan_runtime_scene_tree
     * @brief the standard primitive: one indexed draw of its own geometry (push.model places it)
     */
    export class normal_draw_primitive final : public primitive {
    public:
        void draw(VkCommandBuffer command_buffer) const override;
        void destroy(vma_allocator& vma) noexcept override;
        [[nodiscard]] bool is_valid() const noexcept override;
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief instanced primitive: draws the geometry of another primitive (source)
     *        instance_count times in ONE draw call; per-instance world transforms come from the
     *        runtime's instance transform buffer (scene set binding 6, push flag bit0). Owns
     *        nothing: geometry belongs to source, destroy() is a no-op, source must outlive it.
     */
    export class instanced_draw_primitive final : public primitive {
    public:
        primitive const* source = nullptr;
        uint32_t instance_count = 0;

        void draw(VkCommandBuffer command_buffer) const override;
        void destroy(vma_allocator& vma) noexcept override;
        [[nodiscard]] bool is_valid() const noexcept override;
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
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

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief build the directional light UBO (light-space view-proj + direction) for shadow
     *        mapping. The light direction matches the analytic sky sun (see skybox.frag), so
     *        shadows, the PBR direct light and the visible sun disc all agree.
     * @param scene_center world-space center of the shadow frustum (e.g. the imported scene
     *        bounds center after the scene offset is applied)
     * @param scene_radius conservative radius covering the shadow casters
     * @return light UBO with an orthographic view-proj framing the scene bounds
     * @note ortho box sized to cover a sphere of the given radius around scene_center; the light
     *       looks down the (0.3, 1.0, 0.5) direction (the same sun as the skybox)
     */
    export light_ubo make_directional_light_ubo(glm::vec3 const& scene_center, float scene_radius);
} // namespace vulkan
