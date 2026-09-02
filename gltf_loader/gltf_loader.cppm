module;

#include <cstdint>
#include <glm/glm.hpp>

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

    /**
     * @ingroup gltf_loader
     * @brief a scene node: meshes plus its world transform matrix
     * @note the scene hierarchy is expanded recursively, so every node reachable
     *       from a scene root (including intermediate transform-only nodes) appears here
     */
    export struct node {
        std::vector<mesh> meshes;
        glm::mat4 transform_matrix;
    };

    /**
     * @ingroup gltf_loader
     * @brief a named scene containing nodes
     */
    export struct scene {
        std::string name;
        std::vector<node> nodes;
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
            if (a.exhausted_ || b.exhausted_) {
                return a.exhausted_ == b.exhausted_;
            }
            return a.scenes_ == b.scenes_ && a.scene_i == b.scene_i && a.node_i == b.node_i && a.mesh_i == b.mesh_i && a.prim_i == b.prim_i;
        }

    private:
        void advance(); // move to the next primitive, or set exhausted_

        scenes const* scenes_ = nullptr;
        size_t scene_i = 0;
        size_t node_i = 0;
        size_t mesh_i = 0;
        size_t prim_i = 0;
        mutable drawable_ref current_ = {}; // operator* result cache (valid until ++)
        bool exhausted_ = true;             // default = end(); begin() clears it before advancing
    };

    /**
     * @ingroup gltf_loader
     * @brief loaded result of a glTF file: textures, materials and scenes
     * @note textures holds one entry per glTF texture (in texture order); material texture_indices
     *       values index into this array; primitive.material_index indexes into materials
     * @note scenes is a range: begin()/end() yield every drawable primitive with its node's
     *       world transform (see gltf::scene_iterator / gltf::drawable_ref), so callers can
     *       iterate the whole scene without manual scene -> node -> mesh -> primitive loops
     */
    export struct scenes {
        std::vector<texture_data> textures;
        std::vector<material> materials;
        std::vector<scene> scene;

        [[nodiscard]] scene_iterator begin() const;
        [[nodiscard]] scene_iterator end() const noexcept;
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
            : inner_(scenes)
            , materials_(materials) {
        }

        drawable_iterator& operator++();

        friend bool operator!=(drawable_iterator const& a, drawable_iterator const& b) {
            return a.inner_ != b.inner_;
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

        gltf::scene_iterator inner_;
        std::span<resolved_material const> materials_ = {};
        // scratch geometry of the current drawable (built lazily, valid until ++)
        mutable std::vector<unsigned char> vertex_bytes_ = {};
        mutable std::vector<unsigned char> index_bytes_ = {};
        mutable uint32_t vertex_stride_ = 0;
        mutable uint32_t vertex_count_ = 0;
        mutable unsigned char index_width_ = 2; // 2 or 4 bytes per index
        mutable uint32_t index_count_ = 0;
        mutable bool built_ = false;
    };
} // namespace gltf
