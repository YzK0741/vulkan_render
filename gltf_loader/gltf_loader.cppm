//
// Created by 小叶 on 2026/7/31.
//
module;

#include <string_view>
#include <expected>
#include <map>
#include <vector>
#include <glm/glm.hpp>
#include <tinygltf/tiny_gltf.h>

export module gltf_loader;

/**
 * @file gltf_loader.cppm
 * @defgroup gltf_loader glTF Loader
 * @brief load glTF/GLB files into pure CPU-side data structures, no Vulkan dependency
 * @note
 *      - built on tinygltf
 *      - load_model() returns std::expected, failures are reported via error_code
 */
namespace gltf {

    /**
     * @ingroup gltf_loader
     * @brief component type of an accessor element, mapped from tinygltf macros
     */
    export enum class component_type : int {
        byte_t = TINYGLTF_COMPONENT_TYPE_BYTE,
        unsigned_byte_t = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE,
        short_t = TINYGLTF_COMPONENT_TYPE_SHORT,
        unsigned_short_t = TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT,
        int_t = TINYGLTF_COMPONENT_TYPE_INT,
        unsigned_int_t = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT,
        float_t = TINYGLTF_COMPONENT_TYPE_FLOAT,
        double_t = TINYGLTF_COMPONENT_TYPE_DOUBLE,
        unknown
    };

    /**
     * @ingroup gltf_loader
     * @brief convert a tinygltf component type macro to component_type
     * @param tinygltf_type the tinygltf macro value
     * @return the mapped component_type, unknown for unrecognized values
     */
    export constexpr component_type to_component_type(const int tinygltf_type) {
        switch (tinygltf_type) {
            case TINYGLTF_COMPONENT_TYPE_BYTE: return component_type::byte_t;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: return component_type::unsigned_byte_t;
            case TINYGLTF_COMPONENT_TYPE_SHORT: return component_type::short_t;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: return component_type::unsigned_short_t;
            case TINYGLTF_COMPONENT_TYPE_INT: return component_type::int_t;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: return component_type::unsigned_int_t;
            case TINYGLTF_COMPONENT_TYPE_FLOAT: return component_type::float_t;
            case TINYGLTF_COMPONENT_TYPE_DOUBLE: return component_type::double_t;
            default: return component_type::unknown;
        }
    }

    /**
     * @ingroup gltf_loader
     * @brief byte size of a single component of the given component type
     * @param type the component type
     * @return byte size, 0 for unknown
     */
    export constexpr uint8_t get_component_size(const component_type type) {
        switch (type) {
            case component_type::byte_t: [[fallthrough]];
            case component_type::unsigned_byte_t: return 1;
            case component_type::short_t: [[fallthrough]];
            case component_type::unsigned_short_t: return 2;
            case component_type::int_t: [[fallthrough]];
            case component_type::unsigned_int_t: [[fallthrough]];
            case component_type::float_t: return 4;
            case component_type::double_t: return 8;
            case component_type::unknown: return 0;
        }
        std::unreachable();
    }

    /**
     * @ingroup gltf_loader
     * @brief convert component_type back to the tinygltf macro value
     * @param type the component type
     * @return the tinygltf macro value, -1 for unknown
     */
    export constexpr int to_gltf_macro_type(const component_type type) {
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
     * @brief convert a tinygltf type macro to element_type
     * @param type the tinygltf macro value
     * @return the mapped element_type
     */
    export constexpr element_type to_element_type(const int type) {
        switch (type) {
            case TINYGLTF_TYPE_SCALAR: return element_type::scale;
            case TINYGLTF_TYPE_VEC2: return element_type::vec2;
            case TINYGLTF_TYPE_VEC3: return element_type::vec3;
            case TINYGLTF_TYPE_VEC4: return element_type::vec4;
            case TINYGLTF_TYPE_MAT2: return element_type::mat2;
            case TINYGLTF_TYPE_MAT3: return element_type::mat3;
            case TINYGLTF_TYPE_MAT4: return element_type::mat4;
            default: break;
        }
        std::unreachable();
    }

    /**
     * @ingroup gltf_loader
     * @brief element count of the given element type
     * @param type the element type
     * @return element count, 0 for unknown
     */
    export constexpr uint8_t get_element_size(const element_type type) {
        switch (type) {
            case element_type::scale: return 1;
            case element_type::vec2: return 2;
            case element_type::vec3: return 3;
            case element_type::vec4: [[fallthrough]];
            case element_type::mat2: return 4;
            case element_type::mat3: return 9;
            case element_type::mat4: return 16;
            case element_type::unknown: return 0;
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
        file_load_failed
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
     * @brief a drawable primitive: vertex attributes, index data and texture indices
     */
    export struct primitive {
        std::map<std::string, vertex_portion> vertex;
        std::vector<unsigned char> index;
        component_type index_component_type;
        std::map<std::string, uint16_t> texture_indices;
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

    /**
     * @ingroup gltf_loader
     * @brief loaded result of a glTF file: textures and scenes
     */
    export struct scenes {
        std::vector<texture_data> textures;
        std::vector<scene> scene;
    };

    /**
     * @ingroup gltf_loader
     * @brief load a glTF/GLB file into CPU-side scene data
     * @param file_name path to the .gltf or .glb file
     * @return scenes on success, error_code on failure (file_not_found/file_type_error/file_load_failed)
     */
    export std::expected<scenes, error_code> load_model(std::string_view file_name);
}