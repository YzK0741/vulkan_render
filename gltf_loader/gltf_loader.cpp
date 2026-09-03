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
        result.double_sided = material.doubleSided;
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

        // DFS over the real node hierarchy (scene root -> node -> children), retaining the
        // parent->child structure. Each node keeps its LOCAL transform (relative to its parent)
        // plus the accumulated WORLD matrix; the DFS pre-order, the world computation and the
        // mesh building are identical to the previous iterateSceneNodes-based flattening, so
        // the drawable iterators (scenes::begin / drawable_iterator) yield exactly the same
        // primitives in the same order with the same world matrices as before.
        auto const build_node = [&result, &asset](auto&& self, std::size_t const node_index, fastgltf::math::fmat4x4 const& parent_world) -> void {
            fastgltf::Node const& fnode = asset.nodes[node_index];
            fastgltf::math::fmat4x4 const world = fastgltf::getTransformMatrix(fnode, parent_world);

            gltf::node current_node = {};
            current_node.name = fnode.name;
            current_node.local_transform = to_glm_mat4(fastgltf::getTransformMatrix(fnode)); // base = identity -> own transform
            current_node.transform_matrix = to_glm_mat4(world);
            if (fnode.meshIndex) {
                gltf::mesh current_mesh = {};
                for (auto const& primitive : asset.meshes[*fnode.meshIndex].primitives) {
                    current_mesh.primitives.push_back(load_primitive(primitive, asset));
                }
                current_node.meshes.push_back(std::move(current_mesh));
            }

            result.nodes.push_back(std::move(current_node));
            std::size_t const self_index = result.nodes.size() - 1;
            // children are appended right after their parent (DFS pre-order), so child indices
            // are always greater than self_index
            for (std::size_t const child : fnode.children) {
                self(self, child, world);
                result.nodes[self_index].children.push_back(result.nodes.size() - 1);
            }
        };

        for (std::size_t const root : asset.scenes[scene_index].nodeIndices) {
            result.root_indices.push_back(result.nodes.size());
            build_node(build_node, root, fastgltf::math::fmat4x4{});
        }
        return result;
    }

    // ---- CPU-side geometry building for drawable_iterator (interleaved pbr.vert layout) ----

    struct vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec3 tangent;
    };

    struct built_mesh {
        std::vector<vertex> vertices;
        std::vector<unsigned char> index_data;
        unsigned char index_width = 2; // bytes per index (2 or 4)
        uint32_t index_count = 0;
    };

    built_mesh build_mesh(gltf::primitive const& prim) {
        built_mesh result;
        auto const get_portion = [&prim](std::string_view const name) -> gltf::vertex_portion const* {
            auto const it = prim.vertex.find(std::string(name));
            return it == prim.vertex.end() ? nullptr : &it->second;
        };
        auto const* position_portion = get_portion("POSITION");
        auto const* normal_portion = get_portion("NORMAL");
        auto const* uv_portion = get_portion("TEXCOORD_0");
        auto const* tangent_portion = get_portion("TANGENT");
        if (position_portion == nullptr) {
            utility::panic("primitive has no POSITION attribute");
        }

        constexpr glm::vec3 default_normal(0.0f, 1.0f, 0.0f);
        constexpr glm::vec2 default_uv(0.0f, 0.0f);
        size_t const vertex_count = position_portion->data.size() / sizeof(glm::vec3);

        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> normals;
        std::vector<glm::vec2> uvs;
        positions.reserve(vertex_count);
        normals.reserve(vertex_count);
        uvs.reserve(vertex_count);
        for (size_t i = 0; i < vertex_count; ++i) {
            auto const* p = reinterpret_cast<glm::vec3 const*>(position_portion->data.data()) + i;
            auto const* n = normal_portion == nullptr ? &default_normal : reinterpret_cast<glm::vec3 const*>(normal_portion->data.data()) + i;
            auto const* uv = uv_portion == nullptr ? &default_uv : reinterpret_cast<glm::vec2 const*>(uv_portion->data.data()) + i;
            positions.push_back(*p);
            normals.push_back(*n);
            uvs.push_back(*uv);
        }

        // index data: 2 or 4 bytes per index
        if (prim.index.empty()) {
            utility::panic("primitive has no index data");
        }
        unsigned char index_width = 2;
        if (prim.index_component_type == gltf::component_type::unsigned_int_t) {
            index_width = 4;
        } else if (prim.index_component_type != gltf::component_type::unsigned_short_t) {
            utility::panic(std::source_location::current(), "unsupported index component type: {}", static_cast<int>(prim.index_component_type));
        }
        uint32_t const index_count = static_cast<uint32_t>(prim.index.size() / index_width);
        auto const read_index = [&prim, index_width](size_t const i) -> uint32_t {
            if (index_width == 4) {
                return reinterpret_cast<uint32_t const*>(prim.index.data())[i];
            }
            return reinterpret_cast<uint16_t const*>(prim.index.data())[i];
        };

        // tangents: use the model's TANGENT if present, otherwise compute per-triangle from position/uv
        // (classic approach: accumulate tangents per triangle, then Gram-Schmidt orthogonalize)
        std::vector<glm::vec3> tangents(vertex_count, glm::vec3(1.0f, 0.0f, 0.0f));
        if (tangent_portion != nullptr) {
            for (size_t i = 0; i < vertex_count; ++i) {
                auto const* t = reinterpret_cast<glm::vec4 const*>(tangent_portion->data.data()) + i;
                tangents[i] = glm::vec3(t->x, t->y, t->z);
            }
        } else {
            std::vector<glm::vec3> tangent_accumulator(vertex_count, glm::vec3(0.0f));
            for (uint32_t i = 0; i + 2 < index_count; i += 3) {
                uint32_t const i0 = read_index(i);
                uint32_t const i1 = read_index(i + 1);
                uint32_t const i2 = read_index(i + 2);
                glm::vec3 const e1 = positions[i1] - positions[i0];
                glm::vec3 const e2 = positions[i2] - positions[i0];
                glm::vec2 const duv1 = uvs[i1] - uvs[i0];
                glm::vec2 const duv2 = uvs[i2] - uvs[i0];
                float const denom = duv1.x * duv2.y - duv2.x * duv1.y;
                if (std::abs(denom) < 1e-8f) {
                    continue; // degenerate UV triangle
                }
                float const f = 1.0f / denom;
                glm::vec3 const tangent = f * duv2.y * e1 - f * duv1.y * e2;
                tangent_accumulator[i0] += tangent;
                tangent_accumulator[i1] += tangent;
                tangent_accumulator[i2] += tangent;
            }
            for (size_t i = 0; i < vertex_count; ++i) {
                glm::vec3 const t = tangent_accumulator[i] - normals[i] * glm::dot(normals[i], tangent_accumulator[i]);
                tangents[i] = glm::length(t) > 1e-8f ? glm::normalize(t) : glm::vec3(1.0f, 0.0f, 0.0f);
            }
        }

        // interleave into the single-binding layout the pbr pipeline expects (stride 44)
        result.vertices.reserve(vertex_count);
        for (size_t i = 0; i < vertex_count; ++i) {
            result.vertices.push_back(vertex{.position = positions[i], .normal = normals[i], .uv = uvs[i], .tangent = tangents[i]});
        }
        result.index_data = prim.index;
        result.index_width = index_width;
        result.index_count = index_count;
        return result;
    }

    // Convert stb-decoded texture data to RGBA (3 channels get alpha, 1 channel is gray-scaled)
    std::vector<unsigned char> to_rgba(gltf::texture_data const& texture) {
        size_t const pixel_count = static_cast<size_t>(texture.width) * texture.height;
        std::vector<unsigned char> rgba(pixel_count * 4, 255);
        switch (texture.component) {
        case 4:
            rgba = texture.data;
            break;
        case 3:
            for (size_t i = 0; i < pixel_count; ++i) {
                rgba[i * 4 + 0] = texture.data[i * 3 + 0];
                rgba[i * 4 + 1] = texture.data[i * 3 + 1];
                rgba[i * 4 + 2] = texture.data[i * 3 + 2];
            }
            break;
        case 1:
            for (size_t i = 0; i < pixel_count; ++i) {
                rgba[i * 4 + 0] = texture.data[i];
                rgba[i * 4 + 1] = texture.data[i];
                rgba[i * 4 + 2] = texture.data[i];
            }
            break;
        default:
            rgba.clear();
            break;
        }
        return rgba;
    }

    // 8-bit sRGB <-> linear conversion: color textures (slot 0) are averaged in linear space so
    // their mips keep correct brightness; the other (UNORM data) slots are averaged in byte space.
    float srgb_to_linear(unsigned char const c) {
        static std::array<float, 256> const table = [] {
            std::array<float, 256> t = {};
            for (int i = 0; i < 256; ++i) {
                float const v = static_cast<float>(i) / 255.0f;
                t[i] = v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
            }
            return t;
        }();
        return table[c];
    }

    unsigned char linear_to_srgb(float const v) {
        static std::array<unsigned char, 4096> const table = [] {
            std::array<unsigned char, 4096> t = {};
            for (int i = 0; i < 4096; ++i) {
                float const v = static_cast<float>(i) / 4095.0f;
                float const s = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
                t[i] = static_cast<unsigned char>(std::clamp(std::lround(s * 255.0f), 0L, 255L));
            }
            return t;
        }();
        int const idx = std::clamp(static_cast<int>(std::lround(v * 4095.0f)), 0, 4095);
        return table[idx];
    }

    // A full RGBA8 mip chain: mip0, mip1, ... laid out contiguously (mip-major). The level
    // count is floor(log2(min(width, height))) + 1.
    struct mip_chain {
        std::vector<unsigned char> data = {};
        uint32_t mip_levels = 0;
    };

    mip_chain generate_mip_chain(std::span<unsigned char const> const rgba, uint32_t const width, uint32_t const height, bool const srgb) {
        uint32_t const mip_count = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::min(width, height))))) + 1;
        mip_chain result;
        result.mip_levels = mip_count;
        result.data.reserve(static_cast<size_t>(width) * height * 4 * 4 / 3); // geometric series for power-of-two

        std::vector<unsigned char> a(rgba.begin(), rgba.end());
        std::vector<unsigned char> b;
        std::span<unsigned char const> cur = a;
        uint32_t w = width;
        uint32_t h = height;
        for (uint32_t mip = 0; mip < mip_count; ++mip) {
            result.data.insert(result.data.end(), cur.begin(), cur.end());
            if (w == 1 && h == 1) {
                break;
            }
            uint32_t const nw = std::max(1u, w / 2);
            uint32_t const nh = std::max(1u, h / 2);
            b.assign(static_cast<size_t>(nw) * nh * 4, 0);
            for (uint32_t y = 0; y < nh; ++y) {
                uint32_t const sy0 = std::min(y * 2, h - 1);
                uint32_t const sy1 = std::min(y * 2 + 1, h - 1);
                for (uint32_t x = 0; x < nw; ++x) {
                    uint32_t const sx0 = std::min(x * 2, w - 1);
                    uint32_t const sx1 = std::min(x * 2 + 1, w - 1);
                    size_t const p00 = (static_cast<size_t>(sy0) * w + sx0) * 4;
                    size_t const p10 = (static_cast<size_t>(sy0) * w + sx1) * 4;
                    size_t const p01 = (static_cast<size_t>(sy1) * w + sx0) * 4;
                    size_t const p11 = (static_cast<size_t>(sy1) * w + sx1) * 4;
                    size_t const dst = (static_cast<size_t>(y) * nw + x) * 4;
                    for (int c = 0; c < 4; ++c) {
                        if (srgb && c < 3) {
                            float const l = (srgb_to_linear(cur[p00 + c]) + srgb_to_linear(cur[p10 + c]) + srgb_to_linear(cur[p01 + c]) + srgb_to_linear(cur[p11 + c])) * 0.25f;
                            b[dst + c] = linear_to_srgb(l);
                        } else {
                            b[dst + c] = static_cast<unsigned char>((static_cast<unsigned>(cur[p00 + c]) + cur[p10 + c] + cur[p01 + c] + cur[p11 + c] + 2) / 4);
                        }
                    }
                }
            }
            std::swap(a, b);
            cur = a;
            w = nw;
            h = nh;
        }
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

    // ---- gltf::scenes node-tree iteration (structural view, includes transform-only nodes) ----

    void scene_node_iterator::push_next_root() {
        while (this->scene_i_ < this->owner_->scene.size()) {
            gltf::scene const& sc = this->owner_->scene[this->scene_i_];
            if (this->root_i_ < sc.root_indices.size()) {
                this->stack_.push_back({sc.root_indices[this->root_i_++], 0});
                return;
            }
            ++this->scene_i_;
            this->root_i_ = 0;
        }
        this->exhausted_ = true;
    }

    void scene_node_iterator::descend() {
        while (!this->exhausted_) {
            if (this->stack_.empty()) {
                this->push_next_root();
                if (this->exhausted_) {
                    return;
                }
                return; // visiting the fresh root now
            }
            frame& top = this->stack_.back();
            gltf::node const& node = this->owner_->scene[this->scene_i_].nodes[top.node_i];
            if (top.next_child < node.children.size()) {
                std::size_t const child = node.children[top.next_child++];
                this->stack_.push_back({child, 0});
                return; // visiting the child now
            }
            this->stack_.pop_back(); // this node's subtree done; retry its parent / next root
        }
    }

    scene_node_iterator::scene_node_iterator(scenes const& owner)
        : owner_(&owner)
        , exhausted_(false) {
        this->push_next_root(); // land on the first root of the first scene (or exhaust)
    }

    scene_node_iterator::reference scene_node_iterator::operator*() const noexcept {
        return &this->owner_->scene[this->scene_i_].nodes[this->stack_.back().node_i];
    }

    scene_node_iterator::pointer scene_node_iterator::operator->() const noexcept {
        return this->operator*();
    }

    scene_node_iterator& scene_node_iterator::operator++() {
        this->descend();
        return *this;
    }

    void scene_node_iterator::operator++(int) {
        ++*this;
    }

    std::string_view scene_node_iterator::get_name() const noexcept {
        return this->owner_->scene[this->scene_i_].nodes[this->stack_.back().node_i].name;
    }

    glm::mat4 scene_node_iterator::get_local_transform() const noexcept {
        return this->owner_->scene[this->scene_i_].nodes[this->stack_.back().node_i].local_transform;
    }

    std::size_t scene_node_iterator::get_depth() const noexcept {
        return this->stack_.size() - 1;
    }

    std::size_t scene_node_iterator::get_drawable_count() const noexcept {
        gltf::node const& node = this->owner_->scene[this->scene_i_].nodes[this->stack_.back().node_i];
        std::size_t count = 0;
        for (gltf::mesh const& mesh : node.meshes) {
            count += mesh.primitives.size();
        }
        return count;
    }

    scene_node_iterator scenes::nodes_begin() const {
        return scene_node_iterator(*this);
    }

    scene_node_iterator scenes::nodes_end() const noexcept {
        return scene_node_iterator();
    }

    // ---- resolved materials + renderer-ready drawable iteration (pure CPU) ----

    std::vector<resolved_material> resolve_materials(gltf::scenes const& scenes) {
        constexpr std::array<std::string_view, 5> slot_names = {"albedo", "metallic_roughness", "normal", "occlusion", "emissive"};

        // decoded texture cache: one entry per glTF texture index; shared textures decode once,
        // and the shared_ptr owners keep every image_view's span alive for the caller
        struct decoded_texture {
            std::shared_ptr<std::vector<unsigned char>> data = {};
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t mip_levels = 1;
        };
        std::vector<std::optional<decoded_texture>> cache(scenes.textures.size());

        auto const decode = [&scenes, &cache](uint16_t const texture_index, bool const srgb) -> decoded_texture {
            decoded_texture out = {};
            auto& entry = cache[texture_index];
            if (!entry) {
                entry = decoded_texture{};
                gltf::texture_data const& tex = scenes.textures[texture_index];
                std::vector<unsigned char> const rgba = to_rgba(tex);
                if (!rgba.empty() && tex.width > 0 && tex.height > 0) {
                    mip_chain const mips = generate_mip_chain(rgba, tex.width, tex.height, srgb);
                    entry->data = std::make_shared<std::vector<unsigned char>>(std::move(mips.data));
                    entry->width = tex.width;
                    entry->height = tex.height;
                    entry->mip_levels = mips.mip_levels;
                }
            }
            if (entry->data) {
                out = *entry;
            }
            return out;
        };

        std::vector<resolved_material> result;
        result.reserve(scenes.materials.size());
        for (auto const& mat : scenes.materials) {
            resolved_material out = {};
            out.factors.base_color_factor = mat.factors.base_color_factor;
            out.factors.emissive_factor = glm::vec4(mat.factors.emissive_factor, 1.0f);
            out.factors.metallic_factor = mat.factors.metallic_factor;
            out.factors.roughness_factor = mat.factors.roughness_factor;
            out.factors.normal_scale = mat.factors.normal_scale;
            out.double_sided = mat.double_sided;
            for (int i = 0; i < 5; ++i) {
                auto const it = mat.texture_indices.find(std::string(slot_names[i]));
                if (it == mat.texture_indices.end() || it->second >= scenes.textures.size()) {
                    continue;
                }
                // slot 0 (albedo) is sRGB color: average its mips in linear space
                decoded_texture const decoded = decode(it->second, i == 0);
                if (!decoded.data) {
                    continue;
                }
                image_view& slot = out.slots[i];
                slot.data = *decoded.data;
                slot.width = decoded.width;
                slot.height = decoded.height;
                slot.mip_levels = decoded.mip_levels;
                slot.owner = decoded.data;
                slot.valid = true;
            }
            result.push_back(std::move(out));
        }
        return result;
    }

    drawable_iterator& drawable_iterator::operator++() {
        ++this->inner_;
        this->built_ = false; // geometry of the next drawable is rebuilt lazily
        return *this;
    }

    void drawable_iterator::ensure_built() const {
        if (this->built_) {
            return;
        }
        built_mesh const mesh = build_mesh(*((*this->inner_).primitive));
        auto const* const first = reinterpret_cast<unsigned char const*>(mesh.vertices.data());
        this->vertex_bytes_.assign(first, first + mesh.vertices.size() * sizeof(vertex));
        this->vertex_stride_ = sizeof(vertex);
        this->vertex_count_ = static_cast<uint32_t>(mesh.vertices.size());
        this->index_bytes_ = mesh.index_data;
        this->index_width_ = mesh.index_width;
        this->index_count_ = mesh.index_count;
        this->built_ = true;
    }

    vertex_view drawable_iterator::get_vertex() const {
        this->ensure_built();
        return vertex_view{.data = this->vertex_bytes_, .stride = this->vertex_stride_, .count = this->vertex_count_};
    }

    index_view drawable_iterator::get_index() const {
        this->ensure_built();
        return index_view{.data = this->index_bytes_, .width = this->index_width_, .count = this->index_count_};
    }

    glm::mat4 drawable_iterator::get_transform() const {
        return (*this->inner_).transform_matrix;
    }

    resolved_material const* drawable_iterator::current_material() const {
        uint32_t const index = (*this->inner_).primitive->material_index;
        return index < this->materials_.size() ? &this->materials_[index] : nullptr;
    }

    image_view drawable_iterator::slot(int const i) const {
        resolved_material const* material = this->current_material();
        return material == nullptr ? image_view{} : material->slots[i];
    }

    image_view drawable_iterator::get_albedo() const {
        return this->slot(0);
    }

    image_view drawable_iterator::get_metallic_roughness() const {
        return this->slot(1);
    }

    image_view drawable_iterator::get_normal() const {
        return this->slot(2);
    }

    image_view drawable_iterator::get_occlusion() const {
        return this->slot(3);
    }

    image_view drawable_iterator::get_emissive() const {
        return this->slot(4);
    }

    resolved_factors drawable_iterator::get_factors() const {
        resolved_material const* material = this->current_material();
        return material == nullptr ? resolved_factors{} : material->factors;
    }

    bool drawable_iterator::get_double_sided() const {
        resolved_material const* material = this->current_material();
        return material != nullptr && material->double_sided;
    }

    // ---- async twins (see gltf_loader.cppm): delegate to the sync functions on a
    //      std::async thread; the caller consumes the future when the result is needed ----

    std::future<std::expected<scenes, error_code>> load_model_async(std::string_view const file_name) {
        // copy the path: the view must stay valid while the async task runs
        return std::async(std::launch::async, [path = std::string(file_name)] { return load_model(path); });
    }

    std::future<std::vector<resolved_material>> resolve_materials_async(gltf::scenes const& scenes) {
        return std::async(std::launch::async, [&scenes] { return resolve_materials(scenes); });
    }
} // namespace gltf
