module;

#include <cstddef> // offsetof (layout guard below)
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

export module vulkan.runtime.scene_tree;
export import vstd;
export import vulkan.core;
export import vulkan.render_environment;

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
        // identity of the source node this runtime node was rebuilt from: import_scene records
        // the structural iterator's get_source_index() here (the glTF loader stores the asset
        // node index), so callers can map e.g. animation channel targets onto the live tree
        std::size_t source_index = 0;
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

        /**
         * @ingroup vulkan_runtime_scene_tree
         * @brief append a new child node (empty: name "", identity local) and return it, so the
         *        caller fills it in place: add_child().name = ...; add_child().local = ...;
         * @return the appended child (reference valid until the next structural mutation of
         *         this node's children vector - pushing more children may reallocate)
         */
        scene_node& add_child();

        /**
         * @ingroup vulkan_runtime_scene_tree
         * @brief move @p child (and its whole subtree) into this node's children
         * @return the appended child (reference valid until the next structural mutation)
         */
        scene_node& add_child(scene_node child);

        /**
         * @ingroup vulkan_runtime_scene_tree
         * @brief attach an already-built primitive as this node's primitive leaf (replaces any
         *        existing leaf). Building happens through runtime::create_primitive(), which
         *        returns the primitive WITHOUT attaching it; this call places it under this
         *        node, so programmatic scenes can group primitives under transform nodes
         *        instead of only appending root leaves.
         * @param leaf the primitive to attach (ownership moves into this node)
         * @return the attached primitive (stable: it lives on the heap, unlike this node)
         * @note the node becomes a leaf node; any children it already had stay siblings below it
         * @note structural change: call runtime::scene_changed() afterwards so the culling BVH
         *       rebuilds (attach does not know about the runtime)
         */
        primitive* attach(std::unique_ptr<primitive> leaf);

        /**
         * @ingroup vulkan_runtime_scene_tree
         * @brief depth-first search this subtree for the first node with the given name
         *        (pre-order, matching scene_iterator's order); returns nullptr when absent.
         *        Empty names never match (they are the unnamed-node default).
         */
        [[nodiscard]] scene_node* find_node(std::string_view name) noexcept;
        [[nodiscard]] scene_node const* find_node(std::string_view name) const noexcept;
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief a named scene: a list of root nodes (mirrors gltf::scene's shape)
     */
    export struct scene {
        std::string name = {};
        std::vector<scene_node> roots = {};

        /**
         * @ingroup vulkan_runtime_scene_tree
         * @brief append a new root node (empty: name "", identity local) and return it
         * @return the appended root (reference valid until the next structural mutation of
         *         roots - pushing more roots may reallocate)
         */
        scene_node& add_root();

        /**
         * @ingroup vulkan_runtime_scene_tree
         * @brief move @p root (and its whole subtree) in as a new root
         * @return the appended root (reference valid until the next structural mutation)
         */
        scene_node& add_root(scene_node root);

        /**
         * @ingroup vulkan_runtime_scene_tree
         * @brief depth-first search every root for the first node named @p name (pre-order);
         *        returns nullptr when absent. Empty names never match.
         */
        [[nodiscard]] scene_node* find_node(std::string_view name) noexcept;
        [[nodiscard]] scene_node const* find_node(std::string_view name) const noexcept;
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

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief single-pass forward iterator over a scene's node tree: DFS pre-order across every
     *        root, INCLUDING transform-only (mesh-less) nodes, yielding each scene_node.
     *
     * Lets callers walk the live runtime tree with a range-for instead of hand-written
     * recursive lambdas:
     * @code {.cpp}
     * for (scene_node& node : scene) {          // scene.begin()/scene.end()
     *     // node.name / node.local / node.source_index / node.primitive_leaf
     * }
     * @endcode
     *
     * @note structural-frozen tree only: the iterator holds a stack of node addresses, so the
     *       tree must not be restructured (no make/import/clear, no push_back into a
     *       node's children) while an iterator is alive. Editing node.local or the leaf
     *       contents in place is fine - that is what the demo's per-frame animation writes do.
     * @note not a std iterator category (no reference typedefs): deliberately minimal - only
     *       ++ / != / * / -> , enough for range-for and manual loops
     */
    export class scene_iterator {
    public:
        scene_iterator() = default; // end()
        explicit scene_iterator(scene& owner);

        [[nodiscard]] scene_node& operator*() const noexcept;
        [[nodiscard]] scene_node* operator->() const noexcept;
        scene_iterator& operator++();

        /** @brief depth of the current node (0 = a scene root); lets callers tell roots apart */
        [[nodiscard]] std::size_t depth() const noexcept {
            return this->stack.empty() ? 0 : this->stack.back().second;
        }

        friend bool operator==(scene_iterator const& a, scene_iterator const& b) noexcept {
            if (a.exhausted || b.exhausted) {
                return a.exhausted == b.exhausted;
            }
            return a.stack == b.stack;
        }
        friend bool operator!=(scene_iterator const& a, scene_iterator const& b) noexcept {
            return !(a == b);
        }

    private:
        void descend(); // move to the next node in DFS pre-order, or set exhausted

        std::vector<std::pair<scene_node*, std::size_t>> stack = {}; // {node, depth}; back() = current
        bool exhausted = true;                                       // default = end(); begin() clears it
    };

    /** @brief begin()/end() of a scene's node tree (DFS pre-order, every root) */
    export inline scene_iterator begin(scene& owner) noexcept {
        return scene_iterator{owner};
    }
    /** @brief end() sentinel of a scene's node-tree range */
    export inline scene_iterator end(scene&) noexcept {
        return scene_iterator{};
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
     * @brief one user-configurable punctual light (API surface of runtime::set_point_lights).
     *        Point lights are omni-directional; a spot light additionally restricts its cone to
     *        @p spot_direction with a soft edge whose outer half-angle cosine is
     *        @p spot_outer_cos (the shader derives the soft INNER cone as mix(outer, 1, 0.6)).
     * @note intensity/range are ARTISTIC units, not physical: the shader uses inverse-square
     *       falloff 1/(1+d^2) (well-behaved at zero distance) with a smooth range fade
     *       (1-(d/r)^2)^2 - both differ from the physical/Khronos forms (1/d^2,
     *       (1-(d/r)^4)^2), which are unbounded/too harsh for the demo's scales.
     * @note the spot inner cone is derived in-shader; a per-light innerConeAngle (glTF
     *       KHR_lights_punctual) is not surfaced yet, so spot support is stub-level.
     */
    export struct punctual_light {
        glm::vec3 position = glm::vec3(0.0f);                    // world position
        float range = 10.0f;                                     // 0 = infinite falloff, otherwise smooth cutoff
        glm::vec3 color = glm::vec3(1.0f);                       // linear light color
        float intensity = 1.0f;                                  // radiance scale (color * intensity)
        bool spot = false;                                       // false = point light (omni)
        glm::vec3 spot_direction = glm::vec3(0.0f, -1.0f, 0.0f); // spot axis (normalized when spot)
        float spot_outer_cos = -0.2f;                            // cos of the outer cone half-angle (spot only)
    };
    /** @brief max simultaneous punctual lights (LightUBO.punctual_lights / GLSL PunctualLight array) */
    export constexpr uint32_t max_punctual_lights = 2;
    /** @brief one punctual light in the GPU light UBO (std140, 64 bytes; mirror PunctualLight in pbr.frag) */
    export struct point_light {
        glm::vec4 position = {}; // xyz: world position (w unused)
        glm::vec4 color = {};    // xyz: linear color * intensity (w unused)
        glm::vec4 spot_dir = {}; // xyz: spot axis, normalized when the light is a spot (w unused)
        glm::vec4 params = {};   // x = range (0 = infinite), y = 0 point / 1 spot, z = cos(outer cone), w = unused
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief light UBO content, layout matches the LightUBO block in pbr.frag / shadow.vert
     *        (scene set binding 7): light-space view-proj + the light direction, then the
     *        active punctual light count and the punctual light array
     * @note the directional sun is built from the scene bounds (enable_shadows) and the shadow
     *       map samples agree on its direction; punctual lights never cast shadows and ride the
     *       same block after the directional header
     */
    export struct light_ubo {
        glm::mat4 light_view_proj; // world -> light clip space (orthographic)
        glm::vec4 light_dir;       // xyz: normalized light direction (sun)
        float shadow_enabled;      // 1.0 samples the shadow map, 0.0 skips shadows
        // Selectable BRDF models (set via runtime::set_brdf_model / set_diffuse_model, gui
        // combos). Rides the std140 padding of this block - the shader reads them as floats:
        //   brdf_model:   0 = GGX + joint Smith (default), 1 = GGX + height-correlated Smith,
        //                 2 = Beckmann + Smith, 3 = Blinn-Phong + Smith
        //   diffuse_model: 0 = Lambert (default), 1 = Oren-Nayar
        float brdf_model = 0.0f;
        float diffuse_model = 0.0f;
        float pad = 0.0f;
        glm::vec4 light_count = {}; // x = number of active punctual lights (GLSL reads it as uint + vec3 pad)
        std::array<point_light, max_punctual_lights> punctual_lights = {};
    };
    // std140 layout guard against the GLSL LightUBO in pbr.frag: light_count is a glm::vec4
    // (16 B @96, its x carries the count the shader reads as uint) and the punctual light array
    // must sit at byte 112 with a 240-byte block (a vec3 pad on the GLSL side would push the
    // array to 128 and shift every light by 16 bytes - see pbr.frag's layout comment).
    static_assert(sizeof(light_ubo) == 240);
    static_assert(offsetof(light_ubo, punctual_lights) == 112);
    static_assert(sizeof(point_light) == 64);

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
        float occlusion_strength = 1.0f; // occlusion map influence: mix(1, sampled AO, strength)
        float alpha_cutoff = 0.5f;       // alphaMode MASK threshold (fragment discard below it)
        bool alpha_mask = false;         // alphaMode == MASK
        bool alpha_blend = false;        // alphaMode == BLEND (alpha-blended / transparent)
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
        { v.occlusion_strength } -> std::convertible_to<float>;
        { v.alpha_cutoff } -> std::convertible_to<float>;
        { v.alpha_mask } -> std::convertible_to<bool>;
        { v.alpha_blend } -> std::convertible_to<bool>;
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
        { it.get_source_index() } -> std::convertible_to<std::size_t>;
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
        glm::uvec4 tex_indices = {};     // albedo, metallic-roughness, normal, occlusion (indices into the texture array)
        uint32_t emissive_index = 0;     // emissive texture index
        float alpha_cutoff = 0.5f;       // alphaMode MASK threshold (fragment discard below it)
        float occlusion_strength = 1.0f; // occlusion map influence: mix(1, sampled AO, strength)
        uint32_t _pad = 0;               // keep the vec4 members 16-byte aligned (std430)
        glm::vec4 base_color_factor = glm::vec4(1.0f);
        glm::vec4 emissive_factor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        float metallic_factor = 1.0f;
        float roughness_factor = 1.0f;
        float normal_scale = 1.0f;
        uint32_t flags = 0; // bit0: normal map, bit1: occlusion map, bit2: emissive map, bit3: double-sided, bit4: alphaMode MASK, bit5: alphaMode BLEND
    };
    static_assert(sizeof(material_record) == 80);

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief max entries of the GPU material table
     * @note sized for the heaviest glTF stress sample (NodePerformanceTest: 10000 rocks, each
     *       with its own material record - factors differ per rock, so content dedup cannot
     *       collapse them). 16384 x 80 B = 1.3 MiB storage buffer, negligible. The runtime
     *       dedups byte-identical materials (register_material) and reserves index 0 as the
     *       default material; registrations past the capacity degrade to it with a logged
     *       warning instead of failing the whole scene.
     */
    export constexpr uint32_t material_capacity = 16384;

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief max per-instance transforms of an instanced draw (set 0 binding 6 storage buffer)
     */
    export constexpr uint32_t instance_capacity = 8192;

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief max skin matrices of the scene skin buffer (set 0 binding 9 storage buffer), in
     *        mat4s. Indices 0-3 are the identity block (the fallback for unskinned draws:
     *        skin_base = 0), the per-skin joint blocks follow at 4.
     * @note sized for the heavy recursive-skeleton sample (RecursiveSkeletons: 84 skins x 10
     *       joints = 840 joint matrices + identity); the buffer is 2048 x 64 B = 128 KiB per
     *       frame slot, negligible against the 8 MiB morph buffer
     */
    export constexpr uint32_t scene_skin_capacity = 2048;

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief byte capacity of the scene morph buffer (set 0 binding 10 storage buffer, floats):
     *        per-morphable-primitive blocks of vertex deltas + morph weights, laid out by the
     *        caller (see the material_push_constants morph fields); 0 = no morph buffer
     */
    export constexpr std::size_t scene_morph_capacity = std::size_t{8u} * 1024u * 1024u; // 8 MiB of floats

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief per-draw push constants, layout matches the shaders' PushConstants (96 bytes)
     * @note material data lives in the material table (set 0 binding 5), so the push block only
     *       carries the material reference, the skin-matrix block start, the morph block start
     *       and the per-primitive world transform. morph_targets == 0 means "not morphable" and
     *       the vertex shader skips the blend; morph_base is a FLOAT index into binding 10.
     */
    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief index of one material in the runtime's material table (scene set binding 5).
     *        Strongly typed on the CPU side so it cannot be confused with the other GPU-table
     *        indices (instance/skin/morph bases); it is a single uint32_t, so push-constant /
     *        material-record byte layout is unchanged (memcpy/push use the raw bytes).
     */
    export struct material_id {
        uint32_t value = 0;
    };

    export struct material_push_constants {
        material_id material_index = {}; // index into the scene's material table
        uint32_t flags = 0;              // bit0: instanced draw -> model matrix comes from the
                                         //       instance transform buffer (set 0 binding 6)
        // index into the scene skin-matrix buffer (binding 9) where this primitive's joint
        // matrices start; 0 = the identity block (unskinned). The vertex shader reads
        // matrices[skin_base + in_joints.x] etc. — set once per primitive after import
        uint32_t skin_base = 0;
        // morph blend (binding 10): float index of this primitive's morph block (deltas first:
        // per vertex per target pos-delta/nrm-delta, then the per-target weights); morph_targets /
        // morph_vertices describe the block stride. All three stay 0 for non-morphable draws.
        uint32_t morph_base = 0;
        uint32_t morph_targets = 0;  // number of morph targets (0 = no morph)
        uint32_t morph_vertices = 0; // vertex count of this primitive (block stride)
        // mat4 start of THIS instanced primitive's transforms in the shared instance buffer
        // (set 0 binding 6): the vertex shader reads instances.transforms[instance_base +
        // gl_InstanceIndex]. Only meaningful when flag bit0 is set; other draw strategies keep 0.
        uint32_t instance_base = 0;
        // glm::mat4 is only 4-byte aligned by default, but GLSL std430 aligns mat4 to 16 bytes
        // (offset 32 in the block): align explicitly so the CPU layout matches the shader
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
        ~primitive() override = default;

        // geometry: RAII owners (vk_buffer) release the GPU buffers on destruction; the detail
        // pointers are cached accessors for binding (the allocator's objects outlive the tree)
        vk_buffer vertex_buffer = {};
        buffer_detail const* vertex_detail = nullptr;
        vk_buffer index_buffer = {};
        buffer_detail const* index_detail = nullptr;
        VkIndexType index_type = VK_INDEX_TYPE_UINT32;
        uint32_t index_count = 0;
        uint32_t vertex_count = 0;

        // Pipeline the primitive draws with. Empty = DEFAULT semantics: the primitive does not
        // care which pipeline records it, it asks the draw-time render_environment to bind that
        // session's default (normal / instanced / static draws all work this way - they draw
        // with whatever default the recording pass set). Non-empty = an explicit pipeline name
        // the primitive requests through render_environment::bind_pipeline (custom draw
        // strategies). Never a vk_pipeline pointer: pipelines live in the runtime's cache and
        // are reached by name through the environment, so the scene tree stays independent of
        // pipeline objects.
        std::string_view pipeline_name = {};
        // the material push constants (material_index + model)
        material_push_constants push = {};
        bool double_sided = false; // glTF doubleSided: disable back-face culling (per draw)
        // alphaMode BLEND: drawn alpha-blended in the transparent pass (depth write off,
        // back-to-front order). The GPU material record also carries the flag; this mirror on
        // the primitive lets draw() pick the depth-write state without a GPU readback.
        bool transparent = false;
        // alphaMode MASK: fragment discard below alpha_cutoff in the main shader. Mirrored here
        // so the shadow pass can skip masked leaves - the depth-only shadow shader has no alpha
        // test and would otherwise cast solid shadows.
        bool alpha_masked = false;

        // local-space AABB of this primitive's geometry (model space, i.e. before push.model);
        // filled by the runtime when the geometry is uploaded. has_bounds == false means "no
        // single world AABB" (e.g. an instanced primitive spreads over many transforms) and the
        // primitive is never frustum-culled.
        glm::vec3 local_aabb_min = glm::vec3(0.0f);
        glm::vec3 local_aabb_max = glm::vec3(0.0f);
        bool has_bounds = false;

        /**
         * @brief transform the local AABB by the primitive's current world matrix (push.model,
         *        written by update_world -> set_world) into a world-space AABB
         * @return world AABB; {0,0,0}..{0,0,0} when has_bounds == false
         */
        [[nodiscard]] std::pair<glm::vec3, glm::vec3> world_aabb() const noexcept {
            if (!this->has_bounds) {
                return {};
            }
            glm::vec3 wmin = glm::vec3(std::numeric_limits<float>::infinity());
            glm::vec3 wmax = glm::vec3(-std::numeric_limits<float>::infinity());
            for (int i = 0; i < 8; ++i) {
                glm::vec3 const corner{
                    (i & 1) ? this->local_aabb_max.x : this->local_aabb_min.x,
                    (i & 2) ? this->local_aabb_max.y : this->local_aabb_min.y,
                    (i & 4) ? this->local_aabb_max.z : this->local_aabb_min.z,
                };
                glm::vec4 const world = this->push.model * glm::vec4(corner, 1.0f);
                wmin = glm::min(wmin, glm::vec3(world));
                wmax = glm::max(wmax, glm::vec3(world));
            }
            return {wmin, wmax};
        }

        /**
         * @brief store the world transform accumulated by the owning scene tree node
         * @param world the node's world matrix (parent_world * local)
         * @note the primitive's world transform lives in push.model, which draw() pushes as-is
         */
        void set_world(glm::mat4 const& world) override;

        /**
         * @brief record the primitive's draw commands (the shared scene descriptor set is bound
         *        by the caller; the pipeline the primitive draws with and the command buffer to
         *        record into both come from @p env)
         * @param env the recording session's render environment: the session command buffer,
         *        default / named pipeline binding (deduplicated) + the shared push-constant
         *        layout. One instance per recording thread, never shared across workers.
         */
        virtual void draw(render_environment& env) const = 0;
        virtual void destroy(vma_allocator& vma) noexcept = 0;
        [[nodiscard]] virtual bool is_valid() const noexcept = 0;

    protected:
        // shared recording: bind this object's geometry buffers and push the push constants
        // onto the environment's command buffer (the push-constant layout is the environment's
        // shared scene layout, valid for every pipeline - push does not depend on which pipeline
        // is currently bound)
        void bind_geometry_and_push(render_environment const& env) const;
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief the standard primitive: one indexed draw of its own geometry (push.model places it)
     */
    export class normal_draw_primitive final : public primitive {
    public:
        void draw(render_environment& env) const override;
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

        void draw(render_environment& env) const override;
        void destroy(vma_allocator& vma) noexcept override;
        [[nodiscard]] bool is_valid() const noexcept override;
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief one chunk of a static_draw: an index sub-range of the merged buffer plus the
     *        material this chunk draws with (each chunk may bind a different material, so one
     *        merged buffer can hold many sub-meshes with distinct materials)
     */
    export struct static_draw_chunk {
        uint32_t first_index = 0;   // first index of this chunk in the merged index buffer
        uint32_t index_count = 0;   // number of indices this chunk draws
        uint32_t vertex_offset = 0; // base vertex into the merged vertex buffer (chunks past the
                                    // first when the packer did not remap indices; 0 otherwise)
        // per-chunk material (same shape as primitive_create_info's material fields)
        texture_input albedo = {};
        texture_input metallic_roughness = {};
        texture_input normal = {};
        texture_input occlusion = {};
        texture_input emissive = {};
        material_factors factors = {};
        bool double_sided = false;
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief build description for runtime::make_static_draw(): ONE merged vertex/index buffer
     *        (the packer's output) plus the chunk table over it. Each chunk is drawn as a
     *        single offset draw call after ONE buffer bind, so N static sub-meshes cost 1 bind
     *        + N draws instead of N binds + N draws.
     * @note the chunk table is REQUIRED (a non-empty, validated list): every chunk's index
     *       window and vertex references are checked against the merged buffers at
     *       make_static_draw() time - out-of-range chunks are logged and skipped.
     */
    export struct static_draw_create_info {
        std::span<unsigned char const> vertex_data = {};
        uint32_t vertex_stride = 0;
        uint32_t vertex_count = 0;
        std::span<unsigned char const> index_data = {};
        VkIndexType index_type = VK_INDEX_TYPE_UINT32;
        uint32_t index_count = 0; // whole merged index count (the chunk table covers a subset)
        std::vector<static_draw_chunk> chunks = {};
        glm::mat4 model_matrix = glm::mat4(1.0f);
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief static batch primitive: OWNS one merged vertex/index buffer and draws a chunk
     *        table over it — every chunk shares the single buffer bind, each chunk is one
     *        offset draw with its own material (push.material_index). Self-contained: no
     *        source primitive to outlive, destroy() releases the owned buffers like a normal
     *        draw. This is the primitive-level form of a static scene: one buffer, one bind,
     *        N offset draws. Placement works like every other leaf: the node's local
     *        transform (set from static_draw_create_info::model_matrix by make_static_draw)
     *        becomes push.model via update_world, so the whole batch shares one world
     *        transform; per-chunk placement needs separate batches or per-chunk model baking
     *        later.
     * @note AABB: one local box over the whole merged geometry (batch-level frustum culling);
     *       per-chunk AABBs would need chunk-level culling, deferred.
     */
    export class static_draw_primitive final : public primitive {
    public:
        // merged geometry: owned (RAII vk_buffer releases on destroy, like normal_draw)
        vk_buffer vertex_buffer = {};
        buffer_detail const* vertex_detail = nullptr;
        vk_buffer index_buffer = {};
        buffer_detail const* index_detail = nullptr;
        VkIndexType index_type = VK_INDEX_TYPE_UINT32;
        uint32_t index_count = 0; // whole merged index count (upper bound for chunk validation)
        uint32_t vertex_count = 0;
        // chunk table over the merged buffer; each entry draws once after the single bind.
        // Material identity lives in material_index; double_sided is per chunk (cull mode).
        // Always non-empty after make_static_draw() succeeds (chunks are validated there).
        struct chunk_record {
            uint32_t first_index = 0;
            uint32_t index_count = 0;
            uint32_t vertex_offset = 0;
            material_id material_index = {};
            bool double_sided = false;
        };
        std::vector<chunk_record> chunks = {};

        void draw(render_environment& env) const override;
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
     * @param scene_radius conservative radius of the scene around @p target (bounds radius); the
     *        projection far plane always covers target + scene_radius so zooming in never clips
     *        the far side of the scene
     * @param aspect swapchain width / height
     * @return camera UBO with view/proj/camera_pos filled in
     * @note proj uses perspectiveRH_ZO with a Y flip to match Vulkan's y-down framebuffer
     */
    export camera_ubo make_orbit_camera_ubo(
        float yaw,
        float pitch,
        float distance,
        glm::vec3 const& target,
        float scene_radius,
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
