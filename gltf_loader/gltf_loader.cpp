//
// Created by 小叶 on 2026/7/31.
//
module;

#include <expected>
#include <filesystem>
#include <print>
#include <ranges>
#include <glm/glm.hpp>
#include <tinygltf/tiny_gltf.h>

module gltf_loader;

namespace {
    struct parsed_data {
        std::vector<unsigned char> data;
        gltf::component_type component_type = gltf::component_type::unknown;
        gltf::element_type element_type = gltf::element_type::unknown;
        uint64_t count = 0;
        uint64_t byte_size = 0;

        [[nodiscard]] [[maybe_unused]] bool is_legal() const noexcept {
            return gltf::get_element_size(this->element_type) * gltf::get_component_size(this->component_type) * this->count == this->byte_size;
        }
    };

    std::expected<tinygltf::Model, gltf::error_code> load_gltf(const std::string_view file) noexcept {
        if (const std::filesystem::path path(file); !std::filesystem::is_regular_file(path)) {
            return std::unexpected(gltf::error_code::file_not_found);
        }

        std::string file_name(file);

        tinygltf::TinyGLTF loader = {};
        tinygltf::Model model;
        std::string err;
        std::string warn;
        bool success = false;

        if (file_name.ends_with(".glb") || file_name.ends_with(".GLB")) {
            success = loader.LoadBinaryFromFile(&model, &err, &warn, file_name);
        } else if (file_name.ends_with(".gltf") || file_name.ends_with(".GLTF")) {
            success = loader.LoadASCIIFromFile(&model, &err, &warn, file_name);
        } else {
            std::println(stderr, "file type error");
            return std::unexpected(gltf::error_code::file_type_error);
        }

        if (!success && !err.empty()) {
            std::println(stderr, "gltf load err: {}", err.empty()? "unknown error" : err);
            return std::unexpected(gltf::error_code::file_load_failed);
        }
        if (!warn.empty()) {
            std::println(stderr, "gltf load warning: {}", warn);
        }
        return model;
    }

    parsed_data get_data_from_accessor(tinygltf::Accessor const& accessor, tinygltf::Model const& model) {
        std::vector<unsigned char> vertex_data;
        auto const& buffer_view = model.bufferViews[accessor.bufferView];
        auto const& buffer = model.buffers[buffer_view.buffer];
        const unsigned char* source = buffer.data.data() + buffer_view.byteOffset + accessor.byteOffset;
        const gltf::element_type element = gltf::to_element_type(accessor.type);
        const gltf::component_type component = gltf::to_component_type(accessor.componentType);
        const uint64_t element_size = gltf::get_element_size(element);
        const uint64_t component_size = gltf::get_component_size(component);
        const uint64_t byte_size = element_size * component_size * accessor.count;
        vertex_data.resize(byte_size);
        std::memcpy(vertex_data.data(), source, byte_size);

        return {.data = std::move(vertex_data), .component_type = component, .element_type = element, .count = accessor.count, .byte_size = byte_size};
    }

    std::map<std::string, uint16_t> get_texture_indices(tinygltf::Material const &material,
                                                        tinygltf::Model const &model) {
        std::map<std::string, uint16_t> texture_indices;

        if (const int texture_index = material.pbrMetallicRoughness.baseColorTexture.index; texture_index >=0) {
            texture_indices["albedo"] = texture_index;
        }
        if (const int texture_index = material.pbrMetallicRoughness.metallicRoughnessTexture.index; texture_index >=0) {
            texture_indices["metallic_roughness"] = texture_index;
        }
        if (const int texture_index = material.occlusionTexture.index; texture_index >=0) {
            texture_indices["occlusion"] = texture_index;
        }
        if (const int texture_index = material.normalTexture.index; texture_index >=0) {
            texture_indices["normal"] = texture_index;
        }
        if (const int texture_index = material.emissiveTexture.index; texture_index >=0) {
            texture_indices["emissive"] = texture_index;
        }
        return texture_indices;
    }


    gltf::primitive load_primitive(tinygltf::Primitive const& primitive, tinygltf::Model const& model) noexcept {
        std::map<std::string, gltf::vertex_portion> vertex;

        for (auto const& [name, index] : primitive.attributes) {
            auto data = get_data_from_accessor(model.accessors[index], model);
            vertex[name].component = data.component_type;
            vertex[name].data = std::move(data.data);
        }

        parsed_data index_data;

        if (primitive.indices >= 0) {
            auto const& accessor = model.accessors[primitive.indices];
            index_data = get_data_from_accessor(accessor, model);
        }

        auto& material = model.materials[primitive.material];

        const std::map<std::string, uint16_t> texture_indices = get_texture_indices(material, model);

        return {
            .vertex = std::move(vertex),
            .index = std::move(index_data.data),
            .index_component_type = index_data.component_type,
            .texture_indices = texture_indices
        };
    }

    glm::mat4 double_array_to_mat4(const std::vector<double>& src) {
        glm::mat4 result;
        for (int i = 0; i < 16; ++i) {
            result[i / 4][i % 4] = static_cast<float>(src[i]);
        }
        return result;
    }
}

namespace gltf {
    std::expected<scenes, error_code> load_model(std::string_view file_name) {
        auto model_exp = load_gltf(file_name);

        if (!model_exp) {
            return std::unexpected(model_exp.error());
        }

        const auto model = std::move(model_exp).value();

        scenes result;
        const auto scene_size = model.scenes.size();

        result.scene.reserve(scene_size);

        for (int current_scene_index = 0; current_scene_index < scene_size; current_scene_index++) {
            scene current_scene = {};
            current_scene.name = model.scenes[current_scene_index].name;
            auto node_size = model.scenes[current_scene_index].nodes.size();
            current_scene.nodes.reserve(node_size);
            for (auto node_index : model.scenes[current_scene_index].nodes) {

                if (node_index < 0) {
                    break;
                }

                node current_node = {};
                current_node.transform_matrix = double_array_to_mat4(model.nodes[node_index].matrix);
                if (model.nodes[node_index].mesh != -1) {
                    mesh current_mesh = {};
                    for (auto& primitive : model.meshes[model.nodes[node_index].mesh].primitives) {
                        current_mesh.primitives.push_back(load_primitive(primitive, model));
                    }

                    current_node.meshes.push_back(std::move(current_mesh));

                }

                current_scene.nodes.push_back(std::move(current_node));
            }

            result.scene.push_back(std::move(current_scene));
        }

        result.textures = model.images | std::views::transform(
                [](auto& a){
            return texture_data(std::move(a.image), a.width, a.height, a.component);
        }) | std::ranges::to<std::vector>();

        return result;
    }
}