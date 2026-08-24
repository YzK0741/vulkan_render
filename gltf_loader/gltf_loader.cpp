module;

#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <glm/glm.hpp>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

module gltf_loader;

namespace {
    using fastgltf::Asset;

    struct parsed_data {
        std::vector<unsigned char> data;
        gltf::component_type component_type = gltf::component_type::unknown;
        gltf::element_type element_type = gltf::element_type::unknown;
        uint64_t count = 0;
        uint64_t byte_size = 0;
    };

    gltf::component_type to_component_type(const fastgltf::ComponentType type) {
        switch (type) {
            case fastgltf::ComponentType::Byte: return gltf::component_type::byte_t;
            case fastgltf::ComponentType::UnsignedByte: return gltf::component_type::unsigned_byte_t;
            case fastgltf::ComponentType::Short: return gltf::component_type::short_t;
            case fastgltf::ComponentType::UnsignedShort: return gltf::component_type::unsigned_short_t;
            case fastgltf::ComponentType::Int: return gltf::component_type::int_t;
            case fastgltf::ComponentType::UnsignedInt: return gltf::component_type::unsigned_int_t;
            case fastgltf::ComponentType::Float: return gltf::component_type::float_t;
            case fastgltf::ComponentType::Double: return gltf::component_type::double_t;
            default: return gltf::component_type::unknown;
        }
    }

    gltf::element_type to_element_type(const fastgltf::AccessorType type) {
        switch (type) {
            case fastgltf::AccessorType::Scalar: return gltf::element_type::scale;
            case fastgltf::AccessorType::Vec2: return gltf::element_type::vec2;
            case fastgltf::AccessorType::Vec3: return gltf::element_type::vec3;
            case fastgltf::AccessorType::Vec4: return gltf::element_type::vec4;
            case fastgltf::AccessorType::Mat2: return gltf::element_type::mat2;
            case fastgltf::AccessorType::Mat3: return gltf::element_type::mat3;
            case fastgltf::AccessorType::Mat4: return gltf::element_type::mat4;
            default: return gltf::element_type::unknown;
        }
    }

    gltf::error_code to_error_code(const fastgltf::Error error) {
        switch (error) {
            case fastgltf::Error::InvalidFileData:
            case fastgltf::Error::InvalidGLB:
            case fastgltf::Error::InvalidJson:
            case fastgltf::Error::InvalidGltf:
            case fastgltf::Error::InvalidOrMissingAssetField:
            case fastgltf::Error::UnsupportedVersion:
                return gltf::error_code::file_type_error;
            default:
                return gltf::error_code::file_load_failed;
        }
    }

    // fastgltf does not define bitwise operators for Options, combine flags via the underlying type.
    constexpr fastgltf::Options load_options() {
        const auto flags = static_cast<std::uint64_t>(fastgltf::Options::LoadExternalBuffers)
                         | static_cast<std::uint64_t>(fastgltf::Options::LoadExternalImages);
        return static_cast<fastgltf::Options>(flags);
    }

    template <typename T>
    std::vector<unsigned char> copy_accessor(const Asset& asset, const fastgltf::Accessor& accessor) {
        std::vector<unsigned char> data(accessor.count * sizeof(T));
        fastgltf::copyFromAccessor<T>(asset, accessor, data.data());
        return data;
    }

    // copyFromAccessor handles byteStride de-interleaving, sparse accessors and
    // normalized component conversion for every (AccessorType, ComponentType) pair
    // that fastgltf's ElementTraits provides.
    parsed_data get_data_from_accessor(const Asset& asset, const fastgltf::Accessor& accessor) {
        using namespace fastgltf;
        using namespace fastgltf::math;

        parsed_data result;
        result.component_type = to_component_type(accessor.componentType);
        result.element_type = to_element_type(accessor.type);
        result.count = accessor.count;

        switch (accessor.type) {
            case AccessorType::Scalar:
                switch (accessor.componentType) {
                    case ComponentType::Byte: result.data = copy_accessor<std::int8_t>(asset, accessor); break;
                    case ComponentType::UnsignedByte: result.data = copy_accessor<std::uint8_t>(asset, accessor); break;
                    case ComponentType::Short: result.data = copy_accessor<std::int16_t>(asset, accessor); break;
                    case ComponentType::UnsignedShort: result.data = copy_accessor<std::uint16_t>(asset, accessor); break;
                    case ComponentType::Int: result.data = copy_accessor<std::int32_t>(asset, accessor); break;
                    case ComponentType::UnsignedInt: result.data = copy_accessor<std::uint32_t>(asset, accessor); break;
                    case ComponentType::Float: result.data = copy_accessor<float>(asset, accessor); break;
                    case ComponentType::Double: result.data = copy_accessor<double>(asset, accessor); break;
                    default: break;
                }
                break;
            case AccessorType::Vec2:
                switch (accessor.componentType) {
                    case ComponentType::Byte: result.data = copy_accessor<s8vec2>(asset, accessor); break;
                    case ComponentType::UnsignedByte: result.data = copy_accessor<u8vec2>(asset, accessor); break;
                    case ComponentType::Short: result.data = copy_accessor<s16vec2>(asset, accessor); break;
                    case ComponentType::UnsignedShort: result.data = copy_accessor<u16vec2>(asset, accessor); break;
                    case ComponentType::Int: result.data = copy_accessor<s32vec2>(asset, accessor); break;
                    case ComponentType::UnsignedInt: result.data = copy_accessor<u32vec2>(asset, accessor); break;
                    case ComponentType::Float: result.data = copy_accessor<fvec2>(asset, accessor); break;
                    case ComponentType::Double: result.data = copy_accessor<dvec2>(asset, accessor); break;
                    default: break;
                }
                break;
            case AccessorType::Vec3:
                switch (accessor.componentType) {
                    case ComponentType::Byte: result.data = copy_accessor<s8vec3>(asset, accessor); break;
                    case ComponentType::UnsignedByte: result.data = copy_accessor<u8vec3>(asset, accessor); break;
                    case ComponentType::Short: result.data = copy_accessor<s16vec3>(asset, accessor); break;
                    case ComponentType::UnsignedShort: result.data = copy_accessor<u16vec3>(asset, accessor); break;
                    case ComponentType::Int: result.data = copy_accessor<s32vec3>(asset, accessor); break;
                    case ComponentType::UnsignedInt: result.data = copy_accessor<u32vec3>(asset, accessor); break;
                    case ComponentType::Float: result.data = copy_accessor<fvec3>(asset, accessor); break;
                    case ComponentType::Double: result.data = copy_accessor<dvec3>(asset, accessor); break;
                    default: break;
                }
                break;
            case AccessorType::Vec4:
                switch (accessor.componentType) {
                    case ComponentType::Byte: result.data = copy_accessor<s8vec4>(asset, accessor); break;
                    case ComponentType::UnsignedByte: result.data = copy_accessor<u8vec4>(asset, accessor); break;
                    case ComponentType::Short: result.data = copy_accessor<s16vec4>(asset, accessor); break;
                    case ComponentType::UnsignedShort: result.data = copy_accessor<u16vec4>(asset, accessor); break;
                    case ComponentType::Int: result.data = copy_accessor<s32vec4>(asset, accessor); break;
                    case ComponentType::UnsignedInt: result.data = copy_accessor<u32vec4>(asset, accessor); break;
                    case ComponentType::Float: result.data = copy_accessor<fvec4>(asset, accessor); break;
                    case ComponentType::Double: result.data = copy_accessor<dvec4>(asset, accessor); break;
                    default: break;
                }
                break;
            case AccessorType::Mat2: result.data = copy_accessor<fmat2x2>(asset, accessor); break;
            case AccessorType::Mat3: result.data = copy_accessor<fmat3x3>(asset, accessor); break;
            case AccessorType::Mat4: result.data = copy_accessor<fmat4x4>(asset, accessor); break;
            default: break;
        }

        result.byte_size = result.data.size();
        return result;
    }

    // Extract raw bytes from a fastgltf data source (loaded buffers/images are sources::Array).
    std::span<const std::byte> get_source_bytes(const Asset& asset, const fastgltf::DataSource& source) {
        using namespace fastgltf;
        if (const auto* array = std::get_if<sources::Array>(&source)) {
            return {array->bytes.data(), array->bytes.size_bytes()};
        }
        if (const auto* vector = std::get_if<sources::Vector>(&source)) {
            return {vector->bytes.data(), vector->bytes.size()};
        }
        if (const auto* byte_view = std::get_if<sources::ByteView>(&source)) {
            return byte_view->bytes;
        }
        if (const auto* buffer_view = std::get_if<sources::BufferView>(&source)) {
            const auto& view = asset.bufferViews[buffer_view->bufferViewIndex];
            const auto& buffer = asset.buffers[view.bufferIndex];
            auto bytes = get_source_bytes(asset, buffer.data);
            return bytes.subspan(view.byteOffset, view.byteLength);
        }
        return {};
    }

    gltf::texture_data load_texture(const Asset& asset, const std::size_t texture_index) {
        gltf::texture_data out;
        if (texture_index >= asset.textures.size()) {
            return out;
        }
        const auto& texture = asset.textures[texture_index];
        if (!texture.imageIndex) {
            return out;
        }
        const auto& image = asset.images[*texture.imageIndex];
        const auto bytes = get_source_bytes(asset, image.data);
        if (bytes.empty()) {
            return out;
        }

        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char* pixels = stbi_load_from_memory(
                reinterpret_cast<const unsigned char*>(bytes.data()),
                static_cast<int>(bytes.size()),
                &width, &height, &channels, 0);
        if (pixels == nullptr) {
            return out;
        }

        out.data.assign(pixels, pixels + static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(channels));
        out.width = static_cast<uint32_t>(width);
        out.height = static_cast<uint32_t>(height);
        out.component = static_cast<uint8_t>(channels);
        stbi_image_free(pixels);
        return out;
    }

    std::map<std::string, uint16_t> get_texture_indices(const fastgltf::Material& material) {
        std::map<std::string, uint16_t> texture_indices;
        if (material.pbrData.baseColorTexture) {
            texture_indices["albedo"] = static_cast<uint16_t>(material.pbrData.baseColorTexture->textureIndex);
        }
        if (material.pbrData.metallicRoughnessTexture) {
            texture_indices["metallic_roughness"] = static_cast<uint16_t>(material.pbrData.metallicRoughnessTexture->textureIndex);
        }
        if (material.occlusionTexture) {
            texture_indices["occlusion"] = static_cast<uint16_t>(material.occlusionTexture->textureIndex);
        }
        if (material.normalTexture) {
            texture_indices["normal"] = static_cast<uint16_t>(material.normalTexture->textureIndex);
        }
        if (material.emissiveTexture) {
            texture_indices["emissive"] = static_cast<uint16_t>(material.emissiveTexture->textureIndex);
        }
        return texture_indices;
    }

    gltf::primitive load_primitive(const fastgltf::Primitive& primitive, const Asset& asset) {
        std::map<std::string, gltf::vertex_portion> vertex;
        for (const auto& attribute : primitive.attributes) {
            auto data = get_data_from_accessor(asset, asset.accessors[attribute.accessorIndex]);
            const std::string name(attribute.name);
            vertex[name].component = data.component_type;
            vertex[name].data = std::move(data.data);
        }

        parsed_data index_data;
        if (primitive.indicesAccessor) {
            index_data = get_data_from_accessor(asset, asset.accessors[*primitive.indicesAccessor]);
        }

        std::map<std::string, uint16_t> texture_indices;
        if (primitive.materialIndex) {
            texture_indices = get_texture_indices(asset.materials[*primitive.materialIndex]);
        }

        return {
            .vertex = std::move(vertex),
            .index = std::move(index_data.data),
            .index_component_type = index_data.component_type,
            .texture_indices = std::move(texture_indices)
        };
    }

    glm::mat4 to_glm_mat4(const fastgltf::math::fmat4x4& matrix) {
        glm::mat4 result;
        for (std::size_t column = 0; column < 4; ++column) {
            for (std::size_t row = 0; row < 4; ++row) {
                result[column][row] = matrix[column][row];
            }
        }
        return result;
    }

    gltf::scene load_scene(const Asset& asset, const std::size_t scene_index) {
        gltf::scene result;
        result.name = asset.scenes[scene_index].name;

        // Recursively walk the scene graph, computing world-space transforms.
        fastgltf::iterateSceneNodes(asset, scene_index, fastgltf::math::fmat4x4{},
                [&](const fastgltf::Node& node, const fastgltf::math::fmat4x4& matrix) {
                    gltf::node current_node{};
                    current_node.transform_matrix = to_glm_mat4(matrix);
                    if (node.meshIndex) {
                        gltf::mesh current_mesh{};
                        for (const auto& primitive : asset.meshes[*node.meshIndex].primitives) {
                            current_mesh.primitives.push_back(load_primitive(primitive, asset));
                        }
                        current_node.meshes.push_back(std::move(current_mesh));
                    }
                    result.nodes.push_back(std::move(current_node));
                });
        return result;
    }
}

namespace gltf {
    std::expected<scenes, error_code> load_model(std::string_view file_name) {
        const std::filesystem::path path(file_name);
        if (!std::filesystem::is_regular_file(path)) {
            return std::unexpected(error_code::file_not_found);
        }

        auto buffer_exp = fastgltf::GltfDataBuffer::FromPath(path);
        if (!buffer_exp) {
            std::println(stderr, "gltf load err: failed to read file: {}", fastgltf::getErrorMessage(buffer_exp.error()));
            return std::unexpected(error_code::file_load_failed);
        }

        fastgltf::Parser parser;
        auto asset_exp = parser.loadGltf(buffer_exp.get(), path.parent_path(), load_options());
        if (!asset_exp) {
            std::println(stderr, "gltf load err: {}", fastgltf::getErrorMessage(asset_exp.error()));
            return std::unexpected(to_error_code(asset_exp.error()));
        }

        Asset asset = std::move(asset_exp.get());

        scenes result;
        result.scene.reserve(asset.scenes.size());
        for (std::size_t i = 0; i < asset.scenes.size(); ++i) {
            result.scene.push_back(load_scene(asset, i));
        }

        result.textures.reserve(asset.textures.size());
        for (std::size_t i = 0; i < asset.textures.size(); ++i) {
            result.textures.push_back(load_texture(asset, i));
        }

        return result;
    }
}
