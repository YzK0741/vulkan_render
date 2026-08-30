#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>
import std;
import gltf_loader;
import utility;
import vulkan.model;
import vulkan.runtime;

namespace {
    // Matches pbr.vert's vertex inputs: position(0) normal(1) uv(2) tangent(3), stride 44
    struct vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec3 tangent;
    };

    // Matches pbr.frag's PushConstants (48 bytes)
    struct pbr_push_constants {
        glm::vec4 base_color_factor = glm::vec4(1.0f);
        glm::vec4 emissive_factor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        float metallic_factor = 1.0f;
        float roughness_factor = 1.0f;
        float normal_scale = 1.0f;
        uint32_t flags = 0; // bit0: normal map, bit1: occlusion map, bit2: emissive map
    };
    static_assert(sizeof(pbr_push_constants) == 48);

    // Read a single shader SPIR-V file and print info; panic on failure
    void load_shader(const std::filesystem::path& dir, std::string_view file_name, std::vector<unsigned char>& out) {
        const std::filesystem::path path = dir / file_name;
        const std::optional<std::vector<unsigned char>> data = utility::read_binary_to_vector(path);
        if (!data) {
            utility::panic(std::source_location::current(), "cannot open shader file '{}'", path.string());
        }
        out = *data;
        utility::log("loaded shader: {} ({} bytes)", path.string(), out.size());
    }

    // Walk up from the working directory to find the shaders/ directory,
    // so it works when run from the project root or a cmake-build-* directory
    std::optional<std::filesystem::path> locate_shaders_dir() {
        std::filesystem::path current = std::filesystem::current_path();
        for (int depth = 0; depth < 4; ++depth) {
            std::filesystem::path candidate = current / "shaders";
            if (std::filesystem::is_directory(candidate)) {
                return candidate;
            }
            const std::filesystem::path parent = current.parent_path();
            if (parent == current) {
                break;
            }
            current = parent;
        }
        return std::nullopt;
    }

    // Walk up from the working directory to find the default model under gltf_model/
    std::optional<std::filesystem::path> locate_model_file() {
        std::filesystem::path current = std::filesystem::current_path();
        for (int depth = 0; depth < 4; ++depth) {
            std::filesystem::path candidate = current / "gltf_model" / "DamagedHelmet.gltf";
            if (std::filesystem::is_regular_file(candidate)) {
                return candidate;
            }
            const std::filesystem::path parent = current.parent_path();
            if (parent == current) {
                break;
            }
            current = parent;
        }
        return std::nullopt;
    }

    // Load a vertex/fragment SPIR-V pair and create the pipeline via runtime; panic on failure
    void load_and_create_pipeline(vulkan::runtime& runtime,
                                  const std::filesystem::path& shaders_dir,
                                  std::string_view pipeline_name,
                                  std::string_view vertex_file,
                                  std::string_view fragment_file) {
        std::vector<unsigned char> vertex_code;
        std::vector<unsigned char> fragment_code;
        load_shader(shaders_dir, vertex_file, vertex_code);
        load_shader(shaders_dir, fragment_file, fragment_code);

        const std::expected<void, std::string> result = runtime.make_pipeline(pipeline_name, vertex_code, fragment_code);
        if (!result) {
            utility::panic(std::source_location::current(), "failed to create pipeline '{}': {}", pipeline_name, result.error());
        }
        utility::log("SUCCESS: pipeline '{}' created and cached in the runtime", pipeline_name);
    }

    // Convert stb-decoded texture data to RGBA (3 channels get alpha, 1 channel is gray-scaled)
    std::vector<unsigned char> to_rgba(const gltf::texture_data& texture) {
        const size_t pixel_count = static_cast<size_t>(texture.width) * texture.height;
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

    // Upload the texture via VMA + create an image view
    struct texture_bundle {
        uint64_t image = 0;
        vulkan::vk_image_view view; // RAII; destroyed automatically on destruction
    };

    texture_bundle create_texture(vulkan::runtime& runtime, const gltf::texture_data& texture, const VkFormat format) {
        texture_bundle bundle;
        const std::vector<unsigned char> rgba = to_rgba(texture);
        if (rgba.empty()) {
            return bundle;
        }

        vulkan::image_create_info create_info = {};
        create_info.width = texture.width;
        create_info.height = texture.height;
        create_info.mip_levels = 1;
        create_info.array_layers = 1;
        create_info.format = format;
        bundle.image = runtime->vma.create_image(rgba.data(), rgba.size(), create_info, vulkan::image_type::texture_2d);
        const auto* detail = runtime->vma.get_image_detail(bundle.image);
        if (detail == nullptr) {
            return bundle;
        }

        bundle.view = runtime->make_image_view(detail->image, format, VK_IMAGE_VIEW_TYPE_2D);
        return bundle;
    }
} // namespace

int main(int argc, char** argv) {
    // 1. Locate the shaders directory (holds GLSL sources and compiled SPIR-V)
    const std::optional<std::filesystem::path> shaders_dir = locate_shaders_dir();
    if (!shaders_dir) {
        utility::panic("cannot find shaders/ directory. run the program from the project root or a cmake-build-* directory.");
    }

    // 2. Construct vulkan::runtime: the default constructor performs all window/instance/device/swapchain initialization
    vulkan::runtime runtime;

    // 3. Pipeline loading test: simple triangle + standard PBR
    load_and_create_pipeline(runtime, *shaders_dir, "triangle", "triangle.vert.spv", "triangle.frag.spv");
    load_and_create_pipeline(runtime, *shaders_dir, "pbr", "pbr.vert.spv", "pbr.frag.spv");

    // 4. Load a glTF model (defaults to DamagedHelmet under gltf_model/; other .gltf/.glb via command line)
    std::string model_path;
    if (argc > 1) {
        model_path = argv[1];
    } else if (const std::optional<std::filesystem::path> located = locate_model_file()) {
        model_path = located->string();
    } else {
        utility::panic("cannot find gltf_model/DamagedHelmet.gltf. run the program from the project root or pass a model path as argv[1]");
    }
    utility::log("loading model: {}", model_path);
    auto scenes = gltf::load_model(model_path);
    if (!scenes) {
        utility::panic(std::source_location::current(), "failed to load model '{}': error code {}", model_path, static_cast<int>(scenes.error()));
    }
    if (scenes->scene.empty() || scenes->scene[0].nodes.empty() ||
        scenes->scene[0].nodes[0].meshes.empty() || scenes->scene[0].nodes[0].meshes[0].primitives.empty()) {
        utility::panic(std::source_location::current(), "model '{}' has no drawable primitive", model_path);
    }
    const auto& prim = scenes->scene[0].nodes[0].meshes[0].primitives[0];
    utility::log("model loaded: {} textures, {} primitives", scenes->textures.size(), scenes->scene[0].nodes[0].meshes[0].primitives.size());

    // 5. Fetch vertex attributes (separate storage), defaulting missing ones
    const auto get_portion = [&prim](const std::string_view name) -> const gltf::vertex_portion* {
        const auto it = prim.vertex.find(std::string(name));
        return it == prim.vertex.end() ? nullptr : &it->second;
    };
    const auto* position_portion = get_portion("POSITION");
    const auto* normal_portion = get_portion("NORMAL");
    const auto* uv_portion = get_portion("TEXCOORD_0");
    const auto* tangent_portion = get_portion("TANGENT");
    if (position_portion == nullptr) {
        utility::panic("model has no POSITION attribute");
    }

    constexpr glm::vec3 default_normal(0.0f, 1.0f, 0.0f);
    constexpr glm::vec2 default_uv(0.0f, 0.0f);
    const size_t vertex_count = position_portion->data.size() / sizeof(glm::vec3);

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    positions.reserve(vertex_count);
    normals.reserve(vertex_count);
    uvs.reserve(vertex_count);
    for (size_t i = 0; i < vertex_count; ++i) {
        const auto* p = reinterpret_cast<const glm::vec3*>(position_portion->data.data()) + i;
        const auto* n = normal_portion == nullptr ? &default_normal : reinterpret_cast<const glm::vec3*>(normal_portion->data.data()) + i;
        const auto* uv = uv_portion == nullptr ? &default_uv : reinterpret_cast<const glm::vec2*>(uv_portion->data.data()) + i;
        positions.push_back(*p);
        normals.push_back(*n);
        uvs.push_back(*uv);
    }

    // 6. Index data and type
    if (prim.index.empty()) {
        utility::panic("model has no index data");
    }
    VkIndexType index_type = VK_INDEX_TYPE_UINT16;
    if (prim.index_component_type == gltf::component_type::unsigned_int_t) {
        index_type = VK_INDEX_TYPE_UINT32;
    } else if (prim.index_component_type != gltf::component_type::unsigned_short_t) {
        utility::panic(std::source_location::current(), "unsupported index component type: {}", static_cast<int>(prim.index_component_type));
    }
    const uint32_t index_count = static_cast<uint32_t>(prim.index.size() / (index_type == VK_INDEX_TYPE_UINT32 ? 4 : 2));
    const auto read_index = [&prim, index_type](const size_t i) -> uint32_t {
        if (index_type == VK_INDEX_TYPE_UINT32) {
            return reinterpret_cast<const uint32_t*>(prim.index.data())[i];
        }
        return reinterpret_cast<const uint16_t*>(prim.index.data())[i];
    };

    // 7. Tangents: use the model's TANGENT if present, otherwise compute per-triangle from position/uv
    //    (classic approach: accumulate tangents per triangle, then Gram-Schmidt orthogonalize)
    std::vector<glm::vec3> tangents(vertex_count, glm::vec3(1.0f, 0.0f, 0.0f));
    if (tangent_portion != nullptr) {
        for (size_t i = 0; i < vertex_count; ++i) {
            const auto* t = reinterpret_cast<const glm::vec4*>(tangent_portion->data.data()) + i;
            tangents[i] = glm::vec3(t->x, t->y, t->z);
        }
    } else {
        std::vector<glm::vec3> tangent_accumulator(vertex_count, glm::vec3(0.0f));
        for (uint32_t i = 0; i + 2 < index_count; i += 3) {
            const uint32_t i0 = read_index(i);
            const uint32_t i1 = read_index(i + 1);
            const uint32_t i2 = read_index(i + 2);
            const glm::vec3 e1 = positions[i1] - positions[i0];
            const glm::vec3 e2 = positions[i2] - positions[i0];
            const glm::vec2 duv1 = uvs[i1] - uvs[i0];
            const glm::vec2 duv2 = uvs[i2] - uvs[i0];
            const float denom = duv1.x * duv2.y - duv2.x * duv1.y;
            if (std::abs(denom) < 1e-8f) {
                continue; // degenerate UV triangle
            }
            const float f = 1.0f / denom;
            const glm::vec3 tangent = f * duv2.y * e1 - f * duv1.y * e2;
            tangent_accumulator[i0] += tangent;
            tangent_accumulator[i1] += tangent;
            tangent_accumulator[i2] += tangent;
        }
        for (size_t i = 0; i < vertex_count; ++i) {
            const glm::vec3 t = tangent_accumulator[i] - normals[i] * glm::dot(normals[i], tangent_accumulator[i]);
            tangents[i] = glm::length(t) > 1e-8f ? glm::normalize(t) : glm::vec3(1.0f, 0.0f, 0.0f);
        }
        utility::log("TANGENT not in model, computed from position/uv");
    }

    // 8. Interleave into the single-binding layout the pipeline expects (stride 44)
    std::vector<vertex> vertices;
    vertices.reserve(vertex_count);
    for (size_t i = 0; i < vertex_count; ++i) {
        vertices.push_back(vertex{.position = positions[i], .normal = normals[i], .uv = uvs[i], .tangent = tangents[i]});
    }

    // 9. Fit the camera from the bounding box
    auto b_min = glm::vec3(std::numeric_limits<float>::infinity());
    auto b_max = glm::vec3(-std::numeric_limits<float>::infinity());
    for (const auto& v : vertices) {
        b_min = glm::min(b_min, v.position);
        b_max = glm::max(b_max, v.position);
    }
    const glm::vec3 center = b_min * 0.5f + b_max * 0.5f;
    const glm::vec3 extent = b_max - b_min;
    const float max_extent = glm::max(extent.x, glm::max(extent.y, extent.z));
    const float fit_scale = max_extent > 0.0f ? 1.6f / max_extent : 1.0f;
    utility::log("mesh: {} vertices, {} indices, center ({:.3f}, {:.3f}, {:.3f}), extent {:.3f}", vertices.size(), index_count, center.x, center.y, center.z, max_extent);

    // 10. GPU buffers: vertices + indices
    const uint64_t vertex_buffer = runtime->vma.create_buffer(std::span(vertices), vulkan::buffer_type::vertex);
    const uint64_t index_buffer = runtime->vma.create_buffer(prim.index.data(), prim.index.size(), vulkan::buffer_type::index);
    const auto* vertex_detail = runtime->vma.get_buffer_detail(vertex_buffer);
    const auto* index_detail = runtime->vma.get_buffer_detail(index_buffer);
    if (vertex_detail == nullptr || index_detail == nullptr) {
        utility::panic("failed to create mesh buffers");
    }

    // 11. Fetch the PBR pipeline
    const auto* pipeline = runtime.get_pipeline("pbr");
    if (pipeline == nullptr) {
        utility::panic("pipeline 'pbr' not found in runtime");
    }

    // 12. Material textures: albedo as sRGB (PBR linear space), data maps as UNORM
    const auto& texture_indices = prim.texture_indices;
    const auto load_material_texture = [&runtime, &scenes, &texture_indices](const std::string_view semantic, const VkFormat format) -> texture_bundle {
        const auto it = texture_indices.find(std::string(semantic));
        if (it != texture_indices.end() && it->second < scenes->textures.size()) {
            return create_texture(runtime, scenes->textures[it->second], format);
        }
        return {};
    };
    const std::array<texture_bundle, 5> material_textures = {
        load_material_texture("albedo", VK_FORMAT_R8G8B8A8_SRGB),
        load_material_texture("metallic_roughness", VK_FORMAT_R8G8B8A8_UNORM),
        load_material_texture("normal", VK_FORMAT_R8G8B8A8_UNORM),
        load_material_texture("occlusion", VK_FORMAT_R8G8B8A8_UNORM),
        load_material_texture("emissive", VK_FORMAT_R8G8B8A8_UNORM),
    };

    // 1x1 white fallback for missing textures
    gltf::texture_data white_texture_data{};
    white_texture_data.data = {255, 255, 255, 255};
    white_texture_data.width = 1;
    white_texture_data.height = 1;
    white_texture_data.component = 4;
    const texture_bundle white_texture = create_texture(runtime, white_texture_data, VK_FORMAT_R8G8B8A8_UNORM);

    const vulkan::vk_sampler sampler = runtime->make_sampler(VK_SAMPLER_ADDRESS_MODE_REPEAT, 0.25f);

    // 12.1 Generate and upload IBL resources (split-sum: prefiltered env + irradiance + BRDF LUT)
    constexpr int env_size = 128;
    constexpr int env_mip_count = 5;
    utility::log("generating IBL environment (CPU)...");
    const auto ibl_start = std::chrono::steady_clock::now();
    const std::vector<float> env = vulkan::generate_environment_cubemap(env_size);
    const auto env_done = std::chrono::steady_clock::now();
    utility::log("  environment cubemap: {:.1f} ms", std::chrono::duration<double, std::milli>(env_done - ibl_start).count());
    const std::vector<float> prefiltered = vulkan::prefilter_environment(env, env_size, env_mip_count);
    const auto prefilter_done = std::chrono::steady_clock::now();
    utility::log("  prefilter (GGX importance sampling): {:.1f} ms", std::chrono::duration<double, std::milli>(prefilter_done - env_done).count());
    const std::vector<float> irradiance = vulkan::generate_irradiance_map(env, env_size, 32);
    const auto irradiance_done = std::chrono::steady_clock::now();
    utility::log("  irradiance map: {:.1f} ms", std::chrono::duration<double, std::milli>(irradiance_done - prefilter_done).count());
    const std::vector<float> brdf_lut = vulkan::generate_brdf_lut(256);
    const auto lut_done = std::chrono::steady_clock::now();
    utility::log("  BRDF LUT: {:.1f} ms", std::chrono::duration<double, std::milli>(lut_done - irradiance_done).count());
    utility::log("  IBL total: {:.1f} ms", std::chrono::duration<double, std::milli>(lut_done - ibl_start).count());

    const std::vector<unsigned char> env_bytes = vulkan::to_half_rgba(prefiltered);
    const std::vector<unsigned char> irr_bytes = vulkan::to_half_rgba(irradiance);
    const std::vector<unsigned char> lut_bytes = vulkan::to_half_rg(brdf_lut);

    vulkan::image_create_info env_create_info{};
    env_create_info.width = env_size;
    env_create_info.height = env_size;
    env_create_info.mip_levels = env_mip_count;
    env_create_info.array_layers = 6;
    env_create_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    const uint64_t env_image = runtime->vma.create_image(env_bytes.data(), env_bytes.size(), env_create_info, vulkan::image_type::texture_cubemap);

    vulkan::image_create_info irr_create_info{};
    irr_create_info.width = 32;
    irr_create_info.height = 32;
    irr_create_info.mip_levels = 1;
    irr_create_info.array_layers = 6;
    irr_create_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    const uint64_t irr_image = runtime->vma.create_image(irr_bytes.data(), irr_bytes.size(), irr_create_info, vulkan::image_type::texture_cubemap);

    vulkan::image_create_info lut_create_info{};
    lut_create_info.width = 256;
    lut_create_info.height = 256;
    lut_create_info.mip_levels = 1;
    lut_create_info.array_layers = 1;
    lut_create_info.format = VK_FORMAT_R16G16_SFLOAT;
    const uint64_t lut_image = runtime->vma.create_image(lut_bytes.data(), lut_bytes.size(), lut_create_info, vulkan::image_type::texture_2d);

    const auto* env_detail = runtime->vma.get_image_detail(env_image);
    const auto* irr_detail = runtime->vma.get_image_detail(irr_image);
    const auto* lut_detail = runtime->vma.get_image_detail(lut_image);
    if (env_detail == nullptr || irr_detail == nullptr || lut_detail == nullptr) {
        utility::panic("failed to create IBL images");
    }
    const vulkan::vk_image_view env_view = runtime->make_image_view(env_detail->image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_CUBE);
    const vulkan::vk_image_view irr_view = runtime->make_image_view(irr_detail->image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_CUBE);
    const vulkan::vk_image_view lut_view = runtime->make_image_view(lut_detail->image, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_VIEW_TYPE_2D);
    const vulkan::vk_sampler env_sampler = runtime->make_sampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, static_cast<float>(env_mip_count - 1));
    utility::log("IBL ready: {} mips, irradiance 32x32, LUT 256x256", env_mip_count);

    // set 1: 5 textures (binding 0-4) + 3 IBL resources (binding 5-7)
    auto texture_set = runtime->make_descriptor_set(pipeline->get_descriptor_set_layouts()[1]);
    std::array<VkDescriptorImageInfo, 8> image_infos{};
    std::array<VkWriteDescriptorSet, 8> texture_writes{};
    for (int binding = 0; binding < 5; ++binding) {
        const auto& [image, view] = material_textures[binding].view.get() != VK_NULL_HANDLE ? material_textures[binding] : white_texture;
        image_infos[binding] = {.sampler = *sampler, .imageView = *view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        texture_writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        texture_writes[binding].dstSet = *texture_set;
        texture_writes[binding].dstBinding = static_cast<uint32_t>(binding);
        texture_writes[binding].descriptorCount = 1;
        texture_writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        texture_writes[binding].pImageInfo = &image_infos[binding];
    }
    image_infos[5] = {.sampler = *env_sampler, .imageView = *env_view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    image_infos[6] = {.sampler = *env_sampler, .imageView = *irr_view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    image_infos[7] = {.sampler = *env_sampler, .imageView = *lut_view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    for (int binding = 5; binding < 8; ++binding) {
        texture_writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        texture_writes[binding].dstSet = *texture_set;
        texture_writes[binding].dstBinding = static_cast<uint32_t>(binding);
        texture_writes[binding].descriptorCount = 1;
        texture_writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        texture_writes[binding].pImageInfo = &image_infos[binding];
    }
    vkUpdateDescriptorSets(runtime->device, static_cast<uint32_t>(texture_writes.size()), texture_writes.data(), 0, nullptr);

    // 13. One CameraUBO per frame slot + a set-0 descriptor set
    constexpr size_t ubo_size = sizeof(vulkan::camera_ubo);
    std::vector<uint64_t> ubo_buffers;
    std::vector<vulkan::vk_descriptor_set> ubo_sets;
    ubo_buffers.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
    ubo_sets.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
    for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
        vulkan::camera_ubo initial{};
        ubo_buffers.push_back(runtime->vma.create_buffer(std::span(&initial, 1), vulkan::buffer_type::uniform_coherent));
        const auto* ubo_detail = runtime->vma.get_buffer_detail(ubo_buffers.back());
        if (ubo_detail == nullptr) {
            utility::panic("failed to create ubo buffer");
        }

        auto ubo_set = runtime->make_descriptor_set(pipeline->get_descriptor_set_layouts()[0]);
        const VkDescriptorBufferInfo ubo_info{ubo_detail->buffer, 0, ubo_size};
        VkWriteDescriptorSet write_info{};
        write_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_info.dstSet = *ubo_set;
        write_info.dstBinding = 0;
        write_info.descriptorCount = 1;
        write_info.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write_info.pBufferInfo = &ubo_info;
        vkUpdateDescriptorSets(runtime->device, 1, &write_info, 0, nullptr);
        ubo_sets.push_back(std::move(ubo_set));
    }

    // 14. One command buffer per frame slot
    std::vector<vulkan::vk_command_buffer> command_buffers;
    command_buffers.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
    for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
        command_buffers.push_back(runtime->make_command_buffer());
    }

    // 15. Material push constants
    pbr_push_constants push = {};
    push.flags = 0;
    if (texture_indices.contains("normal")) {
        push.flags |= 1u;
    }
    if (texture_indices.contains("occlusion")) {
        push.flags |= 2u;
    }
    if (texture_indices.contains("emissive")) {
        push.flags |= 4u;
    }

    // 16. Camera: orbit view (left-drag to rotate, wheel to zoom).
    //    Camera state lives in runtime (mouse callbacks registered at construction); matrices are built in vulkan::model
    const float aspect = static_cast<float>(runtime->swap_chain_extent.width) / static_cast<float>(runtime->swap_chain_extent.height);

    // Model matrix (fit scale + centered at origin)
    const glm::mat4 model_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(fit_scale)) * glm::translate(glm::mat4(1.0f), -center);

    // 17. Main render loop: until the window closes or ESC is pressed
    utility::log("rendering '{}' with PBR... left-drag to orbit, wheel to zoom, ESC to exit", model_path);

    // FPS statistics: accumulate frame times, report once per second (log + window title)
    std::chrono::steady_clock::time_point last_frame_time = std::chrono::steady_clock::now();
    double fps_elapsed = 0.0;
    uint32_t fps_frame_count = 0;

    while (!glfwWindowShouldClose(runtime->window)) {
        glfwPollEvents();
        if (glfwGetKey(runtime->window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(runtime->window, GLFW_TRUE);
        }

        const size_t frame_slot = runtime->current_frame;

        vkWaitForFences(runtime->device, 1, &runtime->in_flight_fences[frame_slot], VK_TRUE, UINT64_MAX);
        vkResetFences(runtime->device, 1, &runtime->in_flight_fences[frame_slot]);

        // Update this slot's UBO (orbit camera)
        const vulkan::camera_ubo ubo = vulkan::make_orbit_camera_ubo(
            runtime.camera.yaw, runtime.camera.pitch, runtime.camera.distance, model_matrix, aspect);
        const auto* ubo_detail = runtime->vma.get_buffer_detail(ubo_buffers[frame_slot]);
        std::memcpy(ubo_detail->allocation_info.pMappedData, &ubo, ubo_size);

        uint32_t image_index = 0;
        if (vkAcquireNextImageKHR(runtime->device,
                                  runtime->swap_chain,
                                  UINT64_MAX,
                                  runtime->image_available_semaphores[frame_slot],
                                  VK_NULL_HANDLE,
                                  &image_index) != VK_SUCCESS) {
            break;
        }

        auto& command_buffer = command_buffers[frame_slot];
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(*command_buffer, &begin_info) != VK_SUCCESS) {
            break;
        }

        // Begin the render pass (clear + framebuffer/render area, wrapped in runtime)
        runtime.begin_render_pass(*command_buffer, image_index);

        // Bind pipeline + set viewport/scissor (wrapped in vk_pipeline)
        pipeline->begin_pipeline(*command_buffer);

        const VkBuffer vertex_handle = vertex_detail->buffer;
        constexpr VkDeviceSize vertex_offset = 0;
        vkCmdBindVertexBuffers(*command_buffer, 0, 1, &vertex_handle, &vertex_offset);
        vkCmdBindIndexBuffer(*command_buffer, index_detail->buffer, 0, index_type);

        const std::array<VkDescriptorSet, 2> bound_sets = {*ubo_sets[frame_slot], *texture_set};
        vkCmdBindDescriptorSets(*command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline->get_pipeline_layout(),
                                0,
                                bound_sets.size(),
                                bound_sets.data(),
                                0,
                                nullptr);
        vkCmdPushConstants(*command_buffer,
                           pipeline->get_pipeline_layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(pbr_push_constants),
                           &push);
        vkCmdDrawIndexed(*command_buffer, index_count, 1, 0, 0, 0);

        vkCmdEndRenderPass(*command_buffer);
        if (vkEndCommandBuffer(*command_buffer) != VK_SUCCESS) {
            break;
        }

        // Submit + present (wrapped in core: semaphores/fences/queue selection)
        if (runtime->submit(*command_buffer, image_index) != VK_SUCCESS) {
            break;
        }
        if (runtime->present(image_index) != VK_SUCCESS) {
            break;
        }

        runtime->to_next_frame();

        // FPS statistics: frame time = wall time since the previous frame
        const auto now = std::chrono::steady_clock::now();
        fps_elapsed += std::chrono::duration<double>(now - last_frame_time).count();
        last_frame_time = now;
        ++fps_frame_count;
        if (fps_elapsed >= 1.0) {
            const double fps = fps_frame_count / fps_elapsed;
            utility::log("fps: {:.1f} ({:.2f} ms/frame)", fps, 1000.0 * fps_elapsed / fps_frame_count);
            glfwSetWindowTitle(runtime->window, std::format("vulkan_render - {:.1f} fps", fps).c_str());
            fps_elapsed = 0.0;
            fps_frame_count = 0;
        }
    }

    // 18. Wait for the GPU to finish before releasing resources (image views / samplers freed by RAII handles)
    vkDeviceWaitIdle(runtime->device);
    runtime->vma.free_image(lut_image);
    runtime->vma.free_image(irr_image);
    runtime->vma.free_image(env_image);
    for (const auto& bundle : material_textures) {
        if (bundle.image != 0) {
            runtime->vma.free_image(bundle.image);
        }
    }
    if (white_texture.image != 0) {
        runtime->vma.free_image(white_texture.image);
    }
    for (const uint64_t ubo : ubo_buffers) {
        runtime->vma.free_buffer(ubo);
    }
    runtime->vma.free_buffer(index_buffer);
    runtime->vma.free_buffer(vertex_buffer);
    utility::log("render loop finished");
    return 0;
}
