//
// Created by 小叶 on 2026/8/2.
//

module;

#include <optional>
#include <vector>
#include <string_view>

export module vulkan.pipeline.spirv_parser;

namespace vulkan::pipeline {
    export enum class shader_variable_type {
        float_t, int_t, uint_t, bool_t,
        vec2, vec3, vec4,
        ivec2, ivec3, ivec4,
        uvec2, uvec3, uvec4,
        bvec2, bvec3, bvec4,
        mat2, mat3, mat4,
        mat2x2 ,mat2x3, mat3x2, mat2x4, mat4x2, mat3x4, mat4x3,
        sampler2D, samplerCube, sampler3D, sampler2DArray, sampler2DShadow,
        image2D, image3D, imageCube, image2DArray,
        structure
    };

    export struct shader_variable {
        shader_variable_type variable_type;
        std::vector<shader_variable> members;
        uint32_t array_size = 1; // ==1 as scale, >1 as array
    };

    export struct binding {
        std::vector<shader_variable> binding_elements;
    };

    export struct set {
        std::vector<binding> bindings;
    };

    export struct located_variable {
        uint32_t location;
        shader_variable_type type;
    };

    export struct shader_layout {
        std::vector<located_variable> in;
        std::vector<located_variable> out;
        std::vector<set> uniform;
        std::vector<shader_variable_type> push_constants;
    };

    export struct shader {
        shader_layout layout;
        std::vector<unsigned char> code;
    };

    export std::optional<shader> load_shader(std::string_view filename);
}