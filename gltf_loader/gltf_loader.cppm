//
// Created by 小叶 on 2026/7/31.
//
module;

#include <string_view>
#include <expected>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>
#include <glm/glm.hpp>

export module gltf_loader;

/**
 * @file gltf_loader.cppm
 * @defgroup gltf_loader glTF Loader
 * @brief load glTF/GLB files into pure CPU-side data structures, no Vulkan dependency
 * @note
 *      - built on fastgltf
 *      - load_model() returns std::expected, failures are reported via error_code
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
        unknown = 0
    };

    /**
     * @ingroup gltf_loader
     * @brief convert a glTF component type constant to component_type
     * @param gltf_constant the OpenGL constant (5120..5130)
     * @return the mapped component_type, unknown for unrecognized values
     */
    export constexpr component_type to_component_type(const int gltf_constant) {
        switch (gltf_constant) {
            case 5120: return component_type::byte_t;
            case 5121: return component_type::unsigned_byte_t;
            case 5122: return component_type::short_t;
            case 5123: return component_type::unsigned_short_t;
            case 5124: return component_type::int_t;
            case 5125: return component_type::unsigned_int_t;
            case 5126: return component_type::float_t;
            case 5130: return component_type::double_t;
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
     * @brief convert component_type back to the glTF OpenGL constant
     * @param type the component type
     * @return the glTF OpenGL constant, -1 for unknown
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
     * @brief convert a glTF type constant to element_type
     * @param type the glTF type constant (0..6)
     * @return the mapped element_type
     */
    export constexpr element_type to_element_type(const int type) {
        switch (type) {
            case 0: return element_type::scale;
            case 1: return element_type::vec2;
            case 2: return element_type::vec3;
            case 3: return element_type::vec4;
            case 4: return element_type::mat2;
            case 5: return element_type::mat3;
            case 6: return element_type::mat4;
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

    /**
     * @ingroup gltf_loader
     * @brief loaded result of a glTF file: textures and scenes
     * @note textures holds one entry per glTF texture (in texture order); primitive
     *       texture_indices values index into this array
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
