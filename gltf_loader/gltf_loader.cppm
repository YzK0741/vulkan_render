module;

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module gltf_loader;
export import std;
/**
 * @file gltf_loader.cppm
 * @defgroup gltf_loader glTF Loader
 * @brief load glTF/GLB files into pure CPU-side data structures, no Vulkan dependency.
 *        Besides the raw scene data it provides renderer-ready drawable iteration
 *        (resolved_material / drawable_iterator) in pure CPU types; the runtime's
 *        scene_drawable_iterator concept is structural over them, so no Vulkan type is
 *        needed here.
 * @note
 *      - built on fastgltf
 *      - load_model() returns std::expected, failures are reported via error_code
 *      - drawable_iterator models vulkan::scene_drawable_iterator and can be fed directly
 *        to vulkan::runtime::import_scene()
 */
namespace gltf {

    /**
     * @ingroup gltf_loader
     * @brief component type of an accessor element, values are the glTF OpenGL constants
     */
    export enum class component_type : int {
        byte_t = 5120,
        unsigned_byte_t = 5121,
        short_t = 5122,
        unsigned_short_t = 5123,
        int_t = 5124,
        unsigned_int_t = 5125,
        float_t = 5126,
        double_t = 5130,
        unknown = 0,
    };

    /**
     * @ingroup gltf_loader
     * @brief convert a glTF component type constant to component_type
     * @param gltf_constant the OpenGL constant (5120..5130)
     * @return the mapped component_type, unknown for unrecognized values
     */
    export constexpr component_type to_component_type(int const gltf_constant) {
        switch (gltf_constant) {
        case 5120:
            return component_type::byte_t;
        case 5121:
            return component_type::unsigned_byte_t;
        case 5122:
            return component_type::short_t;
        case 5123:
            return component_type::unsigned_short_t;
        case 5124:
            return component_type::int_t;
        case 5125:
            return component_type::unsigned_int_t;
        case 5126:
            return component_type::float_t;
        case 5130:
            return component_type::double_t;
        default:
            return component_type::unknown;
        }
    }

    /**
     * @ingroup gltf_loader
     * @brief byte size of a single component of the given component type
     * @param type the component type
     * @return byte size, 0 for unknown
     */
    export constexpr uint8_t get_component_size(component_type const type) {
        switch (type) {
        case component_type::byte_t:
            [[fallthrough]];
        case component_type::unsigned_byte_t:
            return 1;
        case component_type::short_t:
            [[fallthrough]];
        case component_type::unsigned_short_t:
            return 2;
        case component_type::int_t:
            [[fallthrough]];
        case component_type::unsigned_int_t:
            [[fallthrough]];
        case component_type::float_t:
            return 4;
        case component_type::double_t:
            return 8;
        case component_type::unknown:
            return 0;
        }
        std::unreachable();
    }

    /**
     * @ingroup gltf_loader
     * @brief convert component_type back to the glTF OpenGL constant
     * @param type the component type
     * @return the glTF OpenGL constant, -1 for unknown
     */
    export constexpr int to_gltf_macro_type(component_type const type) {
        if (type == component_type::unknown) {
            return -1;
        }
        return static_cast<int>(type);
    }

    /**
     * @ingroup gltf_loader
     * @brief element type of an accessor (scalar/vector/matrix)
     */
    export enum class element_type {
        scale,
        vec2,
        vec3,
        vec4,
        mat2,
        mat3,
        mat4,
        unknown,
    };

    /**
     * @ingroup gltf_loader
     * @brief convert a glTF type constant to element_type
     * @param type the glTF type constant (0..6)
     * @return the mapped element_type
     */
    export constexpr element_type to_element_type(int const type) {
        switch (type) {
        case 0:
            return element_type::scale;
        case 1:
            return element_type::vec2;
        case 2:
            return element_type::vec3;
        case 3:
            return element_type::vec4;
        case 4:
            return element_type::mat2;
        case 5:
            return element_type::mat3;
        case 6:
            return element_type::mat4;
        default:
            break;
        }
        std::unreachable();
    }

    /**
     * @ingroup gltf_loader
     * @brief element count of the given element type
     * @param type the element type
     * @return element count, 0 for unknown
     */
    export constexpr uint8_t get_element_size(element_type const type) {
        switch (type) {
        case element_type::scale:
            return 1;
        case element_type::vec2:
            return 2;
        case element_type::vec3:
            return 3;
        case element_type::vec4:
            [[fallthrough]];
        case element_type::mat2:
            return 4;
        case element_type::mat3:
            return 9;
        case element_type::mat4:
            return 16;
        case element_type::unknown:
            return 0;
        }
        std::unreachable();
    }

    /**
     * @ingroup gltf_loader
     * @brief error codes returned by load_model
     */
    export enum class error_code {
        file_not_found,
        file_type_error,
        file_load_failed,
    };

    /**
     * @ingroup gltf_loader
     * @brief decoded image data of a texture
     */
    export struct texture_data {
        std::vector<unsigned char> data;
        uint32_t width = 0;
        uint32_t height = 0;
        uint8_t component = 0; // aka. channels
    };

    /**
     * @ingroup gltf_loader
     * @brief raw vertex data portion together with its component type
     */
    export struct vertex_portion {
        std::vector<unsigned char> data;
        component_type component;
    };

    /**
     * @ingroup gltf_loader
     * @brief metallic-roughness material factors, defaults follow the glTF spec
     */
    export struct material_factors {
        glm::vec4 base_color_factor = glm::vec4(1.0f);
        glm::vec3 emissive_factor = glm::vec3(0.0f);
        float metallic_factor = 1.0f;
        float roughness_factor = 1.0f;
        float normal_scale = 1.0f;
    };

    /**
     * @ingroup gltf_loader
     * @brief a glTF material: factors plus the texture slots it uses
     * @note texture_indices maps slot names ("albedo" / "metallic_roughness" / "normal" /
     *       "occlusion" / "emissive") to indices into scenes::textures
     */
    export struct material {
        material_factors factors = {};
        std::map<std::string, uint16_t> texture_indices = {};
        bool double_sided = false; // glTF doubleSided: back faces are rendered, normals flipped
    };

    /**
     * @ingroup gltf_loader
     * @brief a drawable primitive: vertex attributes, index data and its material reference
     */
    export struct primitive {
        std::map<std::string, vertex_portion> vertex;
        std::vector<unsigned char> index;
        component_type index_component_type;
        // index into scenes::materials; std::numeric_limits<uint32_t>::max() when the primitive
        // has no material (render with default factors and no textures)
        uint32_t material_index = std::numeric_limits<uint32_t>::max();
    };

    /**
     * @ingroup gltf_loader
     * @brief a mesh composed of primitives
     */
    export struct mesh {
        std::vector<primitive> primitives;
    };

    // ---- animation (glTF keyframe animation, exported decoded; playback is a consumer concern) ----

    /**
     * @ingroup gltf_loader
     * @brief interpolation mode of one animation sampler (glTF "interpolation")
     */
    export enum class animation_interpolation : int {
        linear = 0,       // glTF LINEAR: blend between consecutive keyframes (slerp for rotations)
        step = 1,         // glTF STEP: hold the previous keyframe's value until the next keyframe
        cubic_spline = 2, // glTF CUBICSPLINE: Hermite spline with per-key in/out tangents
    };

    /**
     * @ingroup gltf_loader
     * @brief animated node property of one animation channel (glTF "path")
     * @note "weights" (morph targets) are not exported: morph targets are unsupported
     */
    export enum class animation_path : int {
        translation = 1, // values are xyz triplets (one per keyframe)
        rotation = 2,    // values are xyzw quaternions (w scalar, one per keyframe)
        scale = 3,       // values are xyz triplets (one per keyframe)
    };

    /**
     * @ingroup gltf_loader
     * @brief one decoded animation sampler: keyframe times + flat output values
     * @note
     *      - times: one float per keyframe, in seconds, monotonically non-decreasing (as stored)
     *      - values: flat floats. LINEAR / STEP hold key_count * components floats
     *        (3 for translation/scale, 4 for rotation). CUBICSPLINE holds key_count *
     *        components * 3 floats, grouped per keyframe in glTF order: in-tangent, value,
     *        out-tangent. Whether a sampler was decoded from valid accessors is not tracked —
     *        a broken/unsupported sampler simply has empty times/values and must be skipped.
     *      - glTF requires float input; other numeric component types are converted to float
     *        when present (so the loader stays robust against non-conforming files)
     */
    export struct animation_sampler {
        std::vector<float> times = {};
        std::vector<float> values = {};
        animation_interpolation interpolation = animation_interpolation::linear;
    };

    /**
     * @ingroup gltf_loader
     * @brief one animation channel: animate one TRS property of a node from a sampler
     * @note
     *      - sampler indexes the owning animation's samplers (glTF semantics)
     *      - target_node is the animated node's index in the glTF asset's node table. Locate
     *        the matching node of a scene's pool by gltf::node::source_index (animations are
     *        file-scoped; a node may be reachable from several scenes)
     */
    export struct animation_channel {
        animation_path path = animation_path::translation;
        std::size_t sampler = 0;
        std::size_t target_node = 0;
    };

    /**
     * @ingroup gltf_loader
     * @brief one glTF animation: channels over samplers, file-scoped (not tied to one scene)
     */
    export struct animation {
        std::string name = {};
        std::vector<animation_sampler> samplers = {};
        std::vector<animation_channel> channels = {};
    };

    /**
     * @ingroup gltf_loader
     * @brief a glTF skin: the joints driving a skinned mesh and their inverse bind matrices
     * @note
     *      - joints are asset node indices (resolve against a scene's pool through
     *        gltf::node::source_index, like animation_channel::target_node)
     *      - inverse_bind_matrices holds one mat4 per joint (joint order), decoded from the
     *        asset's IBM accessor; when the asset omits it (or it is broken) identity matrices
     *        are filled in — the glTF default
     *      - a skinned node's mesh vertices carry JOINTS_0 (joint indices into @p joints) and
     *        WEIGHTS_0 attributes; the drawable vertex layout keeps both (identity for
     *        non-skinned meshes), so a consumer only needs the per-frame joint matrices
     */
    export struct skin {
        std::string name = {};
        std::vector<std::size_t> joints = {};              // asset node indices, in joint order
        std::vector<glm::mat4> inverse_bind_matrices = {}; // one per joint (identity when omitted)
    };

    /**
     * @ingroup gltf_loader
     * @brief evaluated value of one animation channel at a point in time: a vec3 for the
     *        translation/scale paths or a quaternion for the rotation path
     * @note valid == false means the sampler had no keyframes (or broken values); nothing
     *       could be sampled
     */
    export struct channel_sample {
        bool valid = false;
        glm::vec3 vec3 = glm::vec3(0.0f);                   // translation / scale paths
        glm::quat quat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // rotation path (normalized, w scalar)
    };

    /**
     * @ingroup gltf_loader
     * @brief evaluate the channel's sampler at @p t seconds (glTF keyframe sampling)
     * @param sampler the sampler to evaluate (see animation_sampler for the values layout)
     * @param path the property the sampler's values encode — determines the component count
     *        per key (3 for translation/scale, 4 for rotation) and which member of the
     *        returned channel_sample is filled
     * @param t playback time in seconds; clamped to the sampler's keyframe range
     * @return channel_sample with valid == false when the sampler has no usable keyframes
     * @note interpolation follows the sampler's mode: LINEAR lerps translations/scales and
     *       slerps rotations along the shortest arc; STEP holds the previous keyframe's
     *       value; CUBICSPLINE evaluates the Hermite spline from the per-key in/out tangents
     *       (rotation results are normalized afterwards, per the glTF spec)
     */
    export channel_sample sample_channel(animation_sampler const& sampler, animation_path path, float t);

    /**
     * @ingroup gltf_loader
     * @brief one node's animated local pose at a point in time: the TRS base pose of the node
     *        (see gltf::node) overridden by every channel of @p animation that targets it
     * @note the caller picks the animated node(s) per scene by matching
     *       gltf::node::source_index against animation_channel::target_node, evaluates the
     *       per-node pose through this function, then composes T * R * S to write the node's
     *       local transform (see §8 of docs/gltf_loader_usage.md)
     */
    export struct node_pose {
        bool any_channel = false;                // true when at least one channel applied
        glm::vec3 translation = glm::vec3(0.0f); // base pose, overridden per channel path
        glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 scale = glm::vec3(1.0f);
    };

    /**
     * @ingroup gltf_loader
     * @brief evaluate every channel of @p animation targeting @p target_node (an asset node
     *        index, see animation_channel::target_node) at time @p t and merge the results
     *        onto the node's TRS base pose
     * @param animation the animation to play
     * @param target_node asset node index of the animated node
     * @param base the node's TRS base pose (gltf::node translation/rotation/scale)
     * @param t playback time in seconds (clamped per sampler)
     * @return merged pose: node_pose::any_channel == true when at least one channel of the
     *         animation targeted this node and evaluated successfully
     */
    export node_pose sample_node(animation const& animation, std::size_t target_node, node_pose const& base, float t);

    /**
     * @ingroup gltf_loader
     * @brief a scene node: local transform + meshes + child links
     * @note
     *      - the node hierarchy is KEPT: scene.nodes holds the DFS pre-order of every reachable
     *        node (including intermediate transform-only nodes), and each node records its
     *        children as indices into the same scene.nodes list. Callers that need the full
     *        parent->child structure walk children[]; the drawable iterators (scenes::begin /
     *        drawable_iterator) simply flatten the same list and skip mesh-less nodes.
     *      - local_transform is the node's own transform relative to its parent (TRS-composed
     *        matrix, or the node's matrix when the asset stored a raw matrix); transform_matrix
     *        keeps the accumulated world matrix for backward compatibility (same value as the
     *        loader used to expose before the tree was retained)
     *      - source_index links the pool entry back to the glTF asset's node table (animation
     *        channels reference nodes by that index). translation / rotation / scale hold the
     *        TRS base pose when the file declared the node as TRS — the only form the glTF
     *        spec allows animation to target — and stay identity for matrix nodes (which are
     *        never animatable)
     */
    export struct node {
        std::string name = {};                       // glTF node name (empty when unnamed)
        glm::mat4 local_transform = glm::mat4(1.0f); // transform relative to the parent node
        std::vector<mesh> meshes = {};               // meshes attached at this node (may be empty)
        // indices into the owning scene.nodes (DFS pre-order): the direct children of this node
        std::vector<std::size_t> children = {};       // empty for leaves / transform-only nodes
        glm::mat4 transform_matrix = glm::mat4(1.0f); // world matrix (parent * local), kept for compatibility
        // animation substrate: asset node index + TRS base pose (see the @note above)
        std::size_t source_index = 0;                           // index in the glTF asset's node table
        glm::vec3 translation = glm::vec3(0.0f);                // TRS base pose; identity for matrix nodes
        glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // identity quaternion (w scalar)
        glm::vec3 scale = glm::vec3(1.0f);                      // TRS base pose; identity for matrix nodes
        // index into scenes::skins when this node's mesh is skinned (glTF node.skin); the
        // drawable's JOINTS_0 indices reference the skin's joint list
        std::optional<std::size_t> skin_index = std::nullopt;
    };

    /**
     * @ingroup gltf_loader
     * @brief a named scene containing nodes
     * @note nodes holds every node reachable from the scene roots in DFS pre-order (roots
     *       first); root_indices lists the indices of the scene's root nodes inside nodes
     */
    export struct scene {
        std::string name;
        std::vector<node> nodes;
        std::vector<std::size_t> root_indices = {}; // indices into nodes: the scene roots
    };

    export struct scenes; // forward declaration (defined below; scene_iterator only holds a pointer)

    /**
     * @ingroup gltf_loader
     * @brief one drawable primitive of the scene hierarchy as seen by the iterator:
     *        the primitive plus the world transform of the node that owns it
     * @note the primitive carries its own material_index (primitive::material_index)
     */
    export struct drawable_ref {
        primitive const* primitive = nullptr;
        glm::mat4 transform_matrix = glm::mat4(1.0f);
    };

    /**
     * @ingroup gltf_loader
     * @brief single-pass input iterator over every drawable primitive of every scene:
     *        flattens scene -> node -> mesh -> primitive, skipping empty levels.
     *        Pure cursor over the loaded data (no scratch storage), yields drawable_ref.
     */
    export class scene_iterator {
    public:
        using iterator_concept = std::input_iterator_tag;
        using iterator_category = std::input_iterator_tag;
        using value_type = drawable_ref;
        using difference_type = std::ptrdiff_t;
        using pointer = drawable_ref const*;
        using reference = drawable_ref const&;

        scene_iterator() = default; // end(): the default-constructed iterator is exhausted
        explicit scene_iterator(scenes const& owner);

        reference operator*() const noexcept;
        pointer operator->() const noexcept;
        scene_iterator& operator++();
        void operator++(int);

        friend bool operator==(scene_iterator const& a, scene_iterator const& b) noexcept {
            if (a.exhausted || b.exhausted) {
                return a.exhausted == b.exhausted;
            }
            return a.iterating_scene == b.iterating_scene && a.scene_i == b.scene_i && a.node_i == b.node_i && a.mesh_i == b.mesh_i && a.prim_i == b.prim_i;
        }

    private:
        void advance(); // move to the next primitive, or set exhausted

        scenes const* iterating_scene = nullptr;
        size_t scene_i = 0;
        size_t node_i = 0;
        size_t mesh_i = 0;
        size_t prim_i = 0;
        mutable drawable_ref current = {}; // operator* result cache (valid until ++)
        bool exhausted = true;             // default = end(); begin() clears it before advancing
    };

    /**
     * @ingroup gltf_loader
     * @brief single-pass input iterator over the node TREE of every scene: DFS pre-order
     *        over the retained hierarchy (same document order as the loader stored the pool),
     *        INCLUDING transform-only (mesh-less) nodes. Structural view for the runtime's
     *        scene_node_iterator concept: name + local transform + depth (for rebuilding
     *        parent/child edges with a stack) + how many drawables hang off this node.
     * @note pure CPU cursor over the loaded data (no scratch storage); the drawable
     *       geometry of a node is read through the existing drawable_iterator stream,
     *       which advances over the same pool in the same order (skipping mesh-less
     *       nodes), so the two streams stay aligned node-for-node.
     */
    export class scene_node_iterator {
    public:
        using iterator_concept = std::input_iterator_tag;
        using iterator_category = std::input_iterator_tag;
        using value_type = node const*;
        using difference_type = std::ptrdiff_t;
        using pointer = node const*;
        using reference = node const*;

        scene_node_iterator() = default; // end(): the default-constructed iterator is exhausted
        explicit scene_node_iterator(scenes const& owner);

        reference operator*() const noexcept;
        pointer operator->() const noexcept;
        scene_node_iterator& operator++();
        void operator++(int);

        // ---- structural getters (satisfy the runtime's scene_node_iterator concept) ----
        [[nodiscard]] std::string_view get_name() const noexcept;
        [[nodiscard]] glm::mat4 get_local_transform() const noexcept;
        [[nodiscard]] std::size_t get_depth() const noexcept;          // 0 = scene root
        [[nodiscard]] std::size_t get_drawable_count() const noexcept; // primitives hanging off this node
        [[nodiscard]] std::size_t get_source_index() const noexcept;   // asset node index (node::source_index)

        friend bool operator==(scene_node_iterator const& a, scene_node_iterator const& b) noexcept {
            if (a.exhausted || b.exhausted) {
                return a.exhausted == b.exhausted;
            }
            return a.iterating_scene == b.iterating_scene && a.scene_i == b.scene_i && a.stack == b.stack;
        }

    private:
        // DFS state: stack of {node index in the scene's pool, next child position to descend
        // into}; the top of the stack is the node currently being visited. root_i tracks which
        // root of the current scene has been pushed last (roots are visited in root_indices order).
        struct frame {
            std::size_t node_i = 0;
            std::size_t next_child = 0;
            friend bool operator==(frame const& a, frame const& b) noexcept {
                return a.node_i == b.node_i && a.next_child == b.next_child;
            }
        };
        void push_next_root(); // advance to the next scene root, or set exhausted
        void descend();        // move to the next node in DFS pre-order

        scenes const* iterating_scene = nullptr;
        std::size_t scene_i = 0;
        std::size_t root_i = 0; // next root of the current scene to visit (index into root_indices)
        std::vector<frame> stack = {};
        bool exhausted = true; // default = end(); begin() clears it before advancing
    };

    /**
     * @ingroup gltf_loader
     * @brief loaded result of a glTF file: textures, materials, animations and scenes
     * @note textures holds one entry per glTF texture (in texture order); material texture_indices
     *       values index into this array; primitive.material_index indexes into materials
     * @note animations holds the file's keyframe animations (glTF animation objects in order);
     *       see gltf::animation — channels target nodes by their asset node index, resolved
     *       against a scene's pool through gltf::node::source_index
     * @note scenes is a range: begin()/end() yield every drawable primitive with its node's
     *       world transform (see gltf::scene_iterator / gltf::drawable_ref), so callers can
     *       iterate the whole scene without manual scene -> node -> mesh -> primitive loops
     * @note nodes_begin()/nodes_end() yield the retained node tree (see gltf::scene_node_iterator)
     */
    export struct scenes {
        std::vector<texture_data> textures;
        std::vector<material> materials;
        std::vector<animation> animations = {};
        std::vector<skin> skins = {}; // file-scoped skins (glTF skin objects in order)
        std::vector<scene> scene;

        [[nodiscard]] scene_iterator begin() const;
        [[nodiscard]] scene_iterator end() const noexcept;
        [[nodiscard]] scene_node_iterator nodes_begin() const;
        [[nodiscard]] scene_node_iterator nodes_end() const noexcept;
    };

    /**
     * @ingroup gltf_loader
     * @brief load a glTF/GLB file into CPU-side scene data
     * @param file_name path to the .gltf or .glb file
     * @return scenes on success, error_code on failure (file_not_found/file_type_error/file_load_failed)
     */
    export std::expected<scenes, error_code> load_model(std::string_view file_name);

    /**
     * @ingroup gltf_loader
     * @brief async twin of load_model(): runs the parse + texture decode on a std::async
     *        thread; call get() on the returned future when the scene is actually needed
     *        (e.g. to overlap model loading with other startup work)
     */
    export std::future<std::expected<scenes, error_code>> load_model_async(std::string_view file_name);

    // ---- renderer-ready drawable iteration ------------------------------------------------
    // Pure CPU types. The runtime's vulkan::scene_drawable_iterator concept is STRUCTURAL over
    // the member shapes below, so this module never needs vulkan.model (or any Vulkan header);
    // the runtime template converts these values into its internal types itself.

    /** @brief interleaved vertex bytes of one drawable (spans into loader-owned storage) */
    export struct vertex_view {
        std::span<unsigned char const> data = {};
        uint32_t stride = 0;
        uint32_t count = 0;
    };

    /** @brief index bytes of one drawable; width is bytes per index (2 or 4) */
    export struct index_view {
        std::span<unsigned char const> data = {};
        unsigned char width = 2;
        uint32_t count = 0;
    };

    /**
     * @ingroup gltf_loader
     * @brief decoded RGBA8 texture bytes (mip-major) of one material slot
     * @note owner keeps the bytes alive while image_view is copied/moved around; data spans
     *       into *owner. valid == false means the slot is missing.
     */
    export struct image_view {
        std::span<unsigned char const> data = {};
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t mip_levels = 1;
        bool valid = false;
        std::shared_ptr<std::vector<unsigned char>> owner = {};
    };

    /** @brief PBR factors of one resolved material (mirrors the runtime's material_factors) */
    export struct resolved_factors {
        glm::vec4 base_color_factor = glm::vec4(1.0f);
        glm::vec4 emissive_factor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        float metallic_factor = 1.0f;
        float roughness_factor = 1.0f;
        float normal_scale = 1.0f;
    };

    /**
     * @ingroup gltf_loader
     * @brief one resolved material: the 5 texture slots + PBR factors, indexed by the glTF
     *        material index; a primitive without a material reads defaults instead
     */
    export struct resolved_material {
        std::array<image_view, 5> slots = {}; // albedo, metallic_roughness, normal, occlusion, emissive
        resolved_factors factors = {};
        bool double_sided = false; // glTF doubleSided: disable back-face culling + flip normals
    };

    /**
     * @ingroup gltf_loader
     * @brief resolve every scene material once into resolved_material: factors plus the 5
     *        texture slots, decoded to RGBA8 with full mip chains (CPU-side; shared glTF
     *        textures decode once). No Vulkan types involved.
     */
    export std::vector<resolved_material> resolve_materials(gltf::scenes const& scenes);

    /**
     * @ingroup gltf_loader
     * @brief async twin of resolve_materials(): runs the texture decode + mip generation on a
     *        std::async thread; @p scenes must stay alive until the future is consumed
     */
    export std::future<std::vector<resolved_material>> resolve_materials_async(gltf::scenes const& scenes);

    /**
     * @ingroup gltf_loader
     * @brief iterator over every drawable primitive of the scene that models
     *        vulkan::scene_drawable_iterator: ++ advances, then geometry/material are read
     *        through the getters (get_vertex / get_index / get_transform + one getter per
     *        material slot), all as pure CPU values. Interleaved geometry is built lazily per
     *        drawable and cached until the next increment. Feed it directly to
     *        runtime::import_scene(): the runtime drives the traversal and converts.
     * @note default-constructed instance == end() (exhausted inner gltf::scene_iterator)
     */
    export class drawable_iterator {
    public:
        drawable_iterator() = default; // end

        drawable_iterator(gltf::scenes const& scenes, std::span<resolved_material const> materials)
            : inner(scenes)
            , materials(materials) {
        }

        drawable_iterator& operator++();

        friend bool operator!=(drawable_iterator const& a, drawable_iterator const& b) {
            return a.inner != b.inner;
        }

        vertex_view get_vertex() const;
        index_view get_index() const;
        glm::mat4 get_transform() const;
        image_view get_albedo() const;
        image_view get_metallic_roughness() const;
        image_view get_normal() const;
        image_view get_occlusion() const;
        image_view get_emissive() const;
        resolved_factors get_factors() const;
        bool get_double_sided() const;

    private:
        void ensure_built() const; // build the current drawable's interleaved geometry lazily
        resolved_material const* current_material() const;
        image_view slot(int const i) const;

        gltf::scene_iterator inner;
        std::span<resolved_material const> materials = {};
        // scratch geometry of the current drawable (built lazily, valid until ++)
        mutable std::vector<unsigned char> vertex_bytes = {};
        mutable std::vector<unsigned char> index_bytes = {};
        mutable uint32_t vertex_stride = 0;
        mutable uint32_t vertex_count = 0;
        mutable unsigned char index_width = 2; // 2 or 4 bytes per index
        mutable uint32_t index_count = 0;
        mutable bool built = false;
    };

    /**
     * @ingroup gltf_loader
     * @brief world-space axis-aligned bounding box of a whole loaded model (see
     *        compute_scene_bounds): the minimal AABB enclosing every drawable primitive of
     *        every scene, with the glTF node world transforms applied
     */
    export struct scene_bounds {
        bool valid = false;              // false when the model has no position-carrying primitive
        glm::vec3 min = glm::vec3(0.0f); // AABB min corner
        glm::vec3 max = glm::vec3(0.0f); // AABB max corner
        std::size_t primitive_count = 0; // drawable primitives that contributed
    };

    /**
     * @ingroup gltf_loader
     * @brief world AABB of every drawable primitive of the model: each primitive's local AABB
     *        (from its raw POSITION attribute) is transformed by its node's world transform
     *        (TRS maps an AABB to an AABB, so transforming the 8 corners is exact) and united.
     * @note pure CPU over the retained scene data — the renderer uses this to frame the camera
     *       and center the scene before building anything (see main.cpp); primitives without
     *       POSITION are skipped
     */
    export scene_bounds compute_scene_bounds(gltf::scenes const& scenes);
} // namespace gltf
