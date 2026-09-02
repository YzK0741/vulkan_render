module;

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <glm/glm.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

module gltf_loader;

import utility;

namespace {
    using fastgltf::Asset;

    struct parsed_data {
        std::vector<unsigned char> data;
        gltf::component_type component_type = gltf::component_type::unknown;
        gltf::element_type element_type = gltf::element_type::unknown;
        uint64_t count = 0;
        uint64_t byte_size = 0;
    };

    gltf::component_type to_component_type(fastgltf::ComponentType const type) {
        switch (type) {
        case fastgltf::ComponentType::Byte:
            return gltf::component_type::byte_t;
        case fastgltf::ComponentType::UnsignedByte:
            return gltf::component_type::unsigned_byte_t;
        case fastgltf::ComponentType::Short:
            return gltf::component_type::short_t;
        case fastgltf::ComponentType::UnsignedShort:
            return gltf::component_type::unsigned_short_t;
        case fastgltf::ComponentType::Int:
            return gltf::component_type::int_t;
        case fastgltf::ComponentType::UnsignedInt:
            return gltf::component_type::unsigned_int_t;
        case fastgltf::ComponentType::Float:
            return gltf::component_type::float_t;
        case fastgltf::ComponentType::Double:
            return gltf::component_type::double_t;
        default:
            return gltf::component_type::unknown;
        }
    }

    gltf::element_type to_element_type(fastgltf::AccessorType const type) {
        switch (type) {
        case fastgltf::AccessorType::Scalar:
            return gltf::element_type::scale;
        case fastgltf::AccessorType::Vec2:
            return gltf::element_type::vec2;
        case fastgltf::AccessorType::Vec3:
            return gltf::element_type::vec3;
        case fastgltf::AccessorType::Vec4:
            return gltf::element_type::vec4;
        case fastgltf::AccessorType::Mat2:
            return gltf::element_type::mat2;
        case fastgltf::AccessorType::Mat3:
            return gltf::element_type::mat3;
        case fastgltf::AccessorType::Mat4:
            return gltf::element_type::mat4;
        default:
            return gltf::element_type::unknown;
        }
    }

    gltf::error_code to_error_code(fastgltf::Error const error) {
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
        constexpr auto flags = static_cast<std::uint64_t>(fastgltf::Options::LoadExternalBuffers) | static_cast<std::uint64_t>(fastgltf::Options::LoadExternalImages);
        return static_cast<fastgltf::Options>(flags);
    }

    template <typename T>
    std::vector<unsigned char> copy_accessor(Asset const& asset, fastgltf::Accessor const& accessor) {
        std::vector<unsigned char> data(accessor.count * sizeof(T));
        fastgltf::copyFromAccessor<T>(asset, accessor, data.data());
        return data;
    }

    // copyFromAccessor handles byteStride de-interleaving, sparse accessors and
    // normalized component conversion for every (AccessorType, ComponentType) pair
    // that fastgltf's ElementTraits provides.
    parsed_data get_data_from_accessor(Asset const& asset, fastgltf::Accessor const& accessor) {
        using namespace fastgltf;
        using namespace fastgltf::math;

        parsed_data result;
        result.component_type = to_component_type(accessor.componentType);
        result.element_type = to_element_type(accessor.type);
        result.count = accessor.count;

        switch (accessor.type) {
        case AccessorType::Scalar:
            switch (accessor.componentType) {
            case ComponentType::Byte:
                result.data = copy_accessor<std::int8_t>(asset, accessor);
                break;
            case ComponentType::UnsignedByte:
                result.data = copy_accessor<std::uint8_t>(asset, accessor);
                break;
            case ComponentType::Short:
                result.data = copy_accessor<std::int16_t>(asset, accessor);
                break;
            case ComponentType::UnsignedShort:
                result.data = copy_accessor<std::uint16_t>(asset, accessor);
                break;
            case ComponentType::Int:
                result.data = copy_accessor<std::int32_t>(asset, accessor);
                break;
            case ComponentType::UnsignedInt:
                result.data = copy_accessor<std::uint32_t>(asset, accessor);
                break;
            case ComponentType::Float:
                result.data = copy_accessor<float>(asset, accessor);
                break;
            case ComponentType::Double:
                result.data = copy_accessor<double>(asset, accessor);
                break;
            default:
                break;
            }
            break;
        case AccessorType::Vec2:
            switch (accessor.componentType) {
            case ComponentType::Byte:
                result.data = copy_accessor<s8vec2>(asset, accessor);
                break;
            case ComponentType::UnsignedByte:
                result.data = copy_accessor<u8vec2>(asset, accessor);
                break;
            case ComponentType::Short:
                result.data = copy_accessor<s16vec2>(asset, accessor);
                break;
            case ComponentType::UnsignedShort:
                result.data = copy_accessor<u16vec2>(asset, accessor);
                break;
            case ComponentType::Int:
                result.data = copy_accessor<s32vec2>(asset, accessor);
                break;
            case ComponentType::UnsignedInt:
                result.data = copy_accessor<u32vec2>(asset, accessor);
                break;
            case ComponentType::Float:
                result.data = copy_accessor<fvec2>(asset, accessor);
                break;
            case ComponentType::Double:
                result.data = copy_accessor<dvec2>(asset, accessor);
                break;
            default:
                break;
            }
            break;
        case AccessorType::Vec3:
            switch (accessor.componentType) {
            case ComponentType::Byte:
                result.data = copy_accessor<s8vec3>(asset, accessor);
                break;
            case ComponentType::UnsignedByte:
                result.data = copy_accessor<u8vec3>(asset, accessor);
                break;
            case ComponentType::Short:
                result.data = copy_accessor<s16vec3>(asset, accessor);
                break;
            case ComponentType::UnsignedShort:
                result.data = copy_accessor<u16vec3>(asset, accessor);
                break;
            case ComponentType::Int:
                result.data = copy_accessor<s32vec3>(asset, accessor);
                break;
            case ComponentType::UnsignedInt:
                result.data = copy_accessor<u32vec3>(asset, accessor);
                break;
            case ComponentType::Float:
                result.data = copy_accessor<fvec3>(asset, accessor);
                break;
            case ComponentType::Double:
                result.data = copy_accessor<dvec3>(asset, accessor);
                break;
            default:
                break;
            }
            break;
        case AccessorType::Vec4:
            switch (accessor.componentType) {
            case ComponentType::Byte:
                result.data = copy_accessor<s8vec4>(asset, accessor);
                break;
            case ComponentType::UnsignedByte:
                result.data = copy_accessor<u8vec4>(asset, accessor);
                break;
            case ComponentType::Short:
                result.data = copy_accessor<s16vec4>(asset, accessor);
                break;
            case ComponentType::UnsignedShort:
                result.data = copy_accessor<u16vec4>(asset, accessor);
                break;
            case ComponentType::Int:
                result.data = copy_accessor<s32vec4>(asset, accessor);
                break;
            case ComponentType::UnsignedInt:
                result.data = copy_accessor<u32vec4>(asset, accessor);
                break;
            case ComponentType::Float:
                result.data = copy_accessor<fvec4>(asset, accessor);
                break;
            case ComponentType::Double:
                result.data = copy_accessor<dvec4>(asset, accessor);
                break;
            default:
                break;
            }
            break;
        case AccessorType::Mat2:
            result.data = copy_accessor<fmat2x2>(asset, accessor);
            break;
        case AccessorType::Mat3:
            result.data = copy_accessor<fmat3x3>(asset, accessor);
            break;
        case AccessorType::Mat4:
            result.data = copy_accessor<fmat4x4>(asset, accessor);
            break;
        default:
            break;
        }

        result.byte_size = result.data.size();
        return result;
    }

    // Extract raw bytes from a fastgltf data source (loaded buffers/images are sources::Array).
    std::span<std::byte const> get_source_bytes(Asset const& asset, fastgltf::DataSource const& source) {
        using namespace fastgltf;
        if (auto const* array = std::get_if<sources::Array>(&source)) {
            return {array->bytes.data(), array->bytes.size_bytes()};
        }
        if (auto const* vector = std::get_if<sources::Vector>(&source)) {
            return {vector->bytes.data(), vector->bytes.size()};
        }
        if (auto const* byte_view = std::get_if<sources::ByteView>(&source)) {
            return byte_view->bytes;
        }
        if (auto const* buffer_view = std::get_if<sources::BufferView>(&source)) {
            auto const& view = asset.bufferViews[buffer_view->bufferViewIndex];
            auto const& buffer = asset.buffers[view.bufferIndex];
            auto bytes = get_source_bytes(asset, buffer.data);
            return bytes.subspan(view.byteOffset, view.byteLength);
        }
        return {};
    }

    gltf::texture_data load_texture(Asset const& asset, std::size_t const texture_index) {
        gltf::texture_data out;
        if (texture_index >= asset.textures.size()) {
            return out;
        }
        auto const& texture = asset.textures[texture_index];
        if (!texture.imageIndex) {
            return out;
        }
        auto const& image = asset.images[*texture.imageIndex];
        auto const bytes = get_source_bytes(asset, image.data);
        if (bytes.empty()) {
            return out;
        }

        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char* pixels = stbi_load_from_memory(
            reinterpret_cast<unsigned char const*>(bytes.data()),
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

    std::map<std::string, uint16_t> get_texture_indices(fastgltf::Material const& material) {
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

    gltf::material load_material(fastgltf::Material const& material) {
        gltf::material result;
        result.factors.base_color_factor = glm::vec4(material.pbrData.baseColorFactor[0],
                                                     material.pbrData.baseColorFactor[1],
                                                     material.pbrData.baseColorFactor[2],
                                                     material.pbrData.baseColorFactor[3]);
        result.factors.metallic_factor = material.pbrData.metallicFactor;
        result.factors.roughness_factor = material.pbrData.roughnessFactor;
        result.factors.emissive_factor = glm::vec3(material.emissiveFactor[0],
                                                   material.emissiveFactor[1],
                                                   material.emissiveFactor[2]);
        result.factors.normal_scale = material.normalTexture ? material.normalTexture->scale : 1.0f;
        result.texture_indices = get_texture_indices(material);
        return result;
    }

    gltf::primitive load_primitive(fastgltf::Primitive const& primitive, Asset const& asset) {
        std::map<std::string, gltf::vertex_portion> vertex;
        for (auto const& attribute : primitive.attributes) {
            auto data = get_data_from_accessor(asset, asset.accessors[attribute.accessorIndex]);
            std::string const name(attribute.name);
            vertex[name].component = data.component_type;
            vertex[name].data = std::move(data.data);
        }

        parsed_data index_data;
        if (primitive.indicesAccessor) {
            index_data = get_data_from_accessor(asset, asset.accessors[*primitive.indicesAccessor]);
        }

        return {
            .vertex = std::move(vertex),
            .index = std::move(index_data.data),
            .index_component_type = index_data.component_type,
            .material_index = primitive.materialIndex
                                  ? static_cast<uint32_t>(*primitive.materialIndex)
                                  : std::numeric_limits<uint32_t>::max(),
        };
    }

    glm::mat4 to_glm_mat4(fastgltf::math::fmat4x4 const& matrix) {
        glm::mat4 result;
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                result[column][row] = matrix[column][row];
            }
        }
        return result;
    }

    gltf::scene load_scene(Asset const& asset, std::size_t const scene_index) {
        gltf::scene result;
        result.name = asset.scenes[scene_index].name;

        // Recursively walk the scene graph, computing world-space transforms.
        fastgltf::iterateSceneNodes(asset, scene_index, fastgltf::math::fmat4x4{},
                                    [&](fastgltf::Node const& node, fastgltf::math::fmat4x4 const& matrix) {
                                        gltf::node current_node = {};
                                        current_node.transform_matrix = to_glm_mat4(matrix);
                                        if (node.meshIndex) {
                                            gltf::mesh current_mesh = {};
                                            for (auto const& primitive : asset.meshes[*node.meshIndex].primitives) {
                                                current_mesh.primitives.push_back(load_primitive(primitive, asset));
                                            }
                                            current_node.meshes.push_back(std::move(current_mesh));
                                        }
                                        result.nodes.push_back(std::move(current_node));
                                    });
        return result;
    }
} // namespace

namespace gltf {
    std::expected<scenes, error_code> load_model(std::string_view file_name) {
        std::filesystem::path const path(file_name);
        if (!std::filesystem::is_regular_file(path)) {
            return std::unexpected(error_code::file_not_found);
        }

        auto buffer_exp = fastgltf::GltfDataBuffer::FromPath(path);
        if (!buffer_exp) {
            utility::error("gltf load err: failed to read file: {}", fastgltf::getErrorMessage(buffer_exp.error()));
            return std::unexpected(error_code::file_load_failed);
        }

        fastgltf::Parser parser;
        auto asset_exp = parser.loadGltf(buffer_exp.get(), path.parent_path(), load_options());
        if (!asset_exp) {
            utility::error("gltf load err: {}", fastgltf::getErrorMessage(asset_exp.error()));
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

        result.materials.reserve(asset.materials.size());
        for (std::size_t i = 0; i < asset.materials.size(); ++i) {
            result.materials.push_back(load_material(asset.materials[i]));
        }

        return result;
    }

    // ---- gltf::scenes iteration: flatten scene -> node -> mesh -> primitive ----

    scene_iterator::scene_iterator(scenes const& owner)
        : scenes_(&owner)
        , exhausted_(false) {
        this->advance();
    }

    scene_iterator::reference scene_iterator::operator*() const noexcept {
        auto const& owner = *this->scenes_;
        auto const& prim = owner.scene[this->scene_i].nodes[this->node_i].meshes[this->mesh_i].primitives[this->prim_i];
        this->current_.primitive = &prim;
        this->current_.transform_matrix = owner.scene[this->scene_i].nodes[this->node_i].transform_matrix;
        return this->current_;
    }

    scene_iterator::pointer scene_iterator::operator->() const noexcept {
        return &this->current_;
    }

    scene_iterator& scene_iterator::operator++() {
        ++this->prim_i; // step past the current primitive
        this->advance();
        return *this;
    }

    void scene_iterator::operator++(int) {
        ++*this;
    }

    void scene_iterator::advance() {
        if (this->scenes_ == nullptr || this->exhausted_) {
            this->exhausted_ = true;
            return;
        }
        auto const& owner = *this->scenes_;
        while (this->scene_i < owner.scene.size()) {
            auto const& s = owner.scene[this->scene_i];
            if (this->node_i >= s.nodes.size()) {
                ++this->scene_i;
                this->node_i = this->mesh_i = this->prim_i = 0;
                continue;
            }
            auto const& node = s.nodes[this->node_i];
            if (this->mesh_i >= node.meshes.size()) {
                ++this->node_i;
                this->mesh_i = this->prim_i = 0;
                continue;
            }
            auto const& mesh = node.meshes[this->mesh_i];
            if (this->prim_i >= mesh.primitives.size()) {
                ++this->mesh_i;
                this->prim_i = 0;
                continue;
            }
            return; // (scene_i, node_i, mesh_i, prim_i) is a valid drawable primitive
        }
        this->exhausted_ = true;
    }

    scene_iterator scenes::begin() const {
        return scene_iterator(*this);
    }

    scene_iterator scenes::end() const noexcept {
        return scene_iterator();
    }
} // namespace gltf
