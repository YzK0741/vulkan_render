#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>
import std;
import gltf_loader;
import utility;
import vulkan.math;
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

    // Read a single shader SPIR-V file and print info; panic on failure
    void load_shader(std::filesystem::path const& dir, std::string_view file_name, std::vector<unsigned char>& out) {
        std::filesystem::path const path = dir / file_name;
        std::optional<std::vector<unsigned char>> const data = utility::read_binary_to_vector(path);
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
            std::filesystem::path const parent = current.parent_path();
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
            std::filesystem::path const parent = current.parent_path();
            if (parent == current) {
                break;
            }
            current = parent;
        }
        return std::nullopt;
    }

    // Load a vertex/fragment SPIR-V pair and create the pipeline via runtime; panic on failure
    void load_and_create_pipeline(vulkan::runtime& runtime,
                                  std::filesystem::path const& shaders_dir,
                                  std::string_view pipeline_name,
                                  std::string_view vertex_file,
                                  std::string_view fragment_file) {
        std::vector<unsigned char> vertex_code;
        std::vector<unsigned char> fragment_code;
        load_shader(shaders_dir, vertex_file, vertex_code);
        load_shader(shaders_dir, fragment_file, fragment_code);

        std::expected<void, std::string> const result = runtime.make_pipeline(pipeline_name, vertex_code, fragment_code);
        if (!result) {
            utility::panic(std::source_location::current(), "failed to create pipeline '{}': {}", pipeline_name, result.error());
        }
        utility::log("SUCCESS: pipeline '{}' created and cached in the runtime", pipeline_name);
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

} // namespace

int main(int argc, char** argv) {
    // 1. Locate the shaders directory (holds GLSL sources and compiled SPIR-V)
    std::optional<std::filesystem::path> const shaders_dir = locate_shaders_dir();
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
    } else if (std::optional<std::filesystem::path> const located = locate_model_file()) {
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
    auto const& prim = scenes->scene[0].nodes[0].meshes[0].primitives[0];
    utility::log("model loaded: {} textures, {} primitives", scenes->textures.size(), scenes->scene[0].nodes[0].meshes[0].primitives.size());

    // 5. Fetch vertex attributes (separate storage), defaulting missing ones
    auto const get_portion = [&prim](std::string_view const name) -> gltf::vertex_portion const* {
        auto const it = prim.vertex.find(std::string(name));
        return it == prim.vertex.end() ? nullptr : &it->second;
    };
    auto const* position_portion = get_portion("POSITION");
    auto const* normal_portion = get_portion("NORMAL");
    auto const* uv_portion = get_portion("TEXCOORD_0");
    auto const* tangent_portion = get_portion("TANGENT");
    if (position_portion == nullptr) {
        utility::panic("model has no POSITION attribute");
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
    uint32_t const index_count = static_cast<uint32_t>(prim.index.size() / (index_type == VK_INDEX_TYPE_UINT32 ? 4 : 2));
    auto const read_index = [&prim, index_type](size_t const i) -> uint32_t {
        if (index_type == VK_INDEX_TYPE_UINT32) {
            return reinterpret_cast<uint32_t const*>(prim.index.data())[i];
        }
        return reinterpret_cast<uint16_t const*>(prim.index.data())[i];
    };

    // 7. Tangents: use the model's TANGENT if present, otherwise compute per-triangle from position/uv
    //    (classic approach: accumulate tangents per triangle, then Gram-Schmidt orthogonalize)
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
    for (auto const& v : vertices) {
        b_min = glm::min(b_min, v.position);
        b_max = glm::max(b_max, v.position);
    }
    glm::vec3 const center = b_min * 0.5f + b_max * 0.5f;
    glm::vec3 const extent = b_max - b_min;
    float const max_extent = glm::max(extent.x, glm::max(extent.y, extent.z));
    float const fit_scale = max_extent > 0.0f ? 1.6f / max_extent : 1.0f;
    utility::log("mesh: {} vertices, {} indices, center ({:.3f}, {:.3f}, {:.3f}), extent {:.3f}", vertices.size(), index_count, center.x, center.y, center.z, max_extent);

    // 12. Material textures: decode glTF textures to RGBA and hand them to the model (missing -> white fallback)
    auto const& texture_indices = prim.texture_indices;
    constexpr std::array<std::pair<std::string_view, VkFormat>, 5> texture_slots = {
        std::pair{"albedo", VK_FORMAT_R8G8B8A8_SRGB},
        std::pair{"metallic_roughness", VK_FORMAT_R8G8B8A8_UNORM},
        std::pair{"normal", VK_FORMAT_R8G8B8A8_UNORM},
        std::pair{"occlusion", VK_FORMAT_R8G8B8A8_UNORM},
        std::pair{"emissive", VK_FORMAT_R8G8B8A8_UNORM},
    };
    std::array<std::vector<unsigned char>, 5> texture_rgba;
    std::array<vulkan::texture_input, 5> texture_inputs;
    for (int i = 0; i < 5; ++i) {
        auto const it = texture_indices.find(std::string(texture_slots[i].first));
        if (it != texture_indices.end() && it->second < scenes->textures.size()) {
            gltf::texture_data const& source = scenes->textures[it->second];
            texture_rgba[i] = to_rgba(source);
            if (!texture_rgba[i].empty()) {
                texture_inputs[i].data = texture_rgba[i];
                texture_inputs[i].width = source.width;
                texture_inputs[i].height = source.height;
                texture_inputs[i].format = texture_slots[i].second;
                texture_inputs[i].valid = true;
            }
        }
    }

    // 12.1 Generate IBL resources on the CPU (split-sum: prefiltered env + irradiance + BRDF LUT)
    constexpr int env_size = 128;
    constexpr int env_mip_count = 5;
    utility::log("generating IBL environment (CPU)...");
    auto const ibl_start = std::chrono::steady_clock::now();
    std::vector<float> const env = vulkan::generate_environment_cubemap(env_size);
    auto const env_done = std::chrono::steady_clock::now();
    utility::log("  environment cubemap: {:.1f} ms", std::chrono::duration<double, std::milli>(env_done - ibl_start).count());
    std::vector<float> const prefiltered = vulkan::prefilter_environment(env, env_size, env_mip_count);
    auto const prefilter_done = std::chrono::steady_clock::now();
    utility::log("  prefilter (GGX importance sampling): {:.1f} ms", std::chrono::duration<double, std::milli>(prefilter_done - env_done).count());
    std::vector<float> const irradiance = vulkan::generate_irradiance_map(env, env_size, 32);
    auto const irradiance_done = std::chrono::steady_clock::now();
    utility::log("  irradiance map: {:.1f} ms", std::chrono::duration<double, std::milli>(irradiance_done - prefilter_done).count());
    std::vector<float> const brdf_lut = vulkan::generate_brdf_lut(256);
    auto const lut_done = std::chrono::steady_clock::now();
    utility::log("  BRDF LUT: {:.1f} ms", std::chrono::duration<double, std::milli>(lut_done - irradiance_done).count());
    utility::log("  IBL total: {:.1f} ms", std::chrono::duration<double, std::milli>(lut_done - ibl_start).count());

    std::vector<unsigned char> const env_bytes = vulkan::to_half_rgba(prefiltered);
    std::vector<unsigned char> const irr_bytes = vulkan::to_half_rgba(irradiance);
    std::vector<unsigned char> const lut_bytes = vulkan::to_half_rg(brdf_lut);

    // 12.2 Upload the scene-wide IBL once: shared by every model (bindings 2-4 of the scene set)
    runtime.set_ibl(vulkan::ibl_input{.prefiltered_env = env_bytes, .irradiance = irr_bytes, .brdf_lut = lut_bytes, .env_size = env_size, .env_mip_count = env_mip_count, .irr_size = 32, .lut_size = 256});

    // 13. Build the model: geometry + material textures; IBL and the camera UBO are scene-wide
    //     and owned by the runtime, the model only registers its textures into the shared array
    vulkan::model_create_info model_info = {};
    model_info.vertex_data = std::span(reinterpret_cast<unsigned char const*>(vertices.data()), vertices.size() * sizeof(vertex));
    model_info.vertex_stride = sizeof(vertex);
    model_info.vertex_count = static_cast<uint32_t>(vertices.size());
    model_info.index_data = std::span<unsigned char const>(prim.index);
    model_info.index_type = index_type;
    model_info.index_count = index_count;
    model_info.albedo = texture_inputs[0];
    model_info.metallic_roughness = texture_inputs[1];
    model_info.normal = texture_inputs[2];
    model_info.occlusion = texture_inputs[3];
    model_info.emissive = texture_inputs[4];

    // Model matrix (fit scale + centered at origin), pushed per model (shared UBO carries none)
    model_info.model_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(fit_scale)) * glm::translate(glm::mat4(1.0f), -center);

    auto* model = runtime.make_model("pbr", model_info);
    if (model == nullptr) {
        utility::panic("failed to create model (pipeline 'pbr' missing)");
    }

    // 14. Main render loop: until the window closes or ESC is pressed.
    //     Every Vulkan frame step (fences, acquire, command buffers, render pass, submit, present)
    //     lives inside runtime::render_frame()
    utility::log("rendering '{}' with PBR... left-drag to orbit, wheel to zoom, ESC to exit", model_path);

    // FPS statistics: accumulate frame times, report once per second (log + window title)
    std::chrono::steady_clock::time_point last_frame_time = std::chrono::steady_clock::now();
    double fps_elapsed = 0.0;
    uint32_t fps_frame_count = 0;

    // All per-frame decisions (event polling, ESC/close response, minimize skip, swapchain
    // recreation on restore/resize) live inside runtime::render_frame(); main only reacts to
    // the returned frame_result.
    while (true) {
        vulkan::frame_result const result = runtime.render_frame();
        if (result == vulkan::frame_result::closed || result == vulkan::frame_result::failed) {
            break;
        }
        if (result == vulkan::frame_result::skipped) {
            // Minimized or swapchain recreated: skip this frame; keep the FPS timer fresh so the
            // pause is not counted as one huge rendered frame.
            last_frame_time = std::chrono::steady_clock::now();
            std::this_thread::yield();
            continue;
        }

        // render_success: frame time = wall time since the previous rendered frame
        auto const now = std::chrono::steady_clock::now();
        fps_elapsed += std::chrono::duration<double>(now - last_frame_time).count();
        last_frame_time = now;
        ++fps_frame_count;
        if (fps_elapsed >= 1.0) {
            double const fps = fps_frame_count / fps_elapsed;
            utility::log("fps: {:.1f} ({:.2f} ms/frame)", fps, 1000.0 * fps_elapsed / fps_frame_count);
            glfwSetWindowTitle(runtime->get_window(), std::format("vulkan_render - {:.1f} fps", fps).c_str());
            fps_elapsed = 0.0;
            fps_frame_count = 0;
        }
    }

    // 18. Wait for the GPU to finish; models and pipelines are released by the runtime destructor
    vkDeviceWaitIdle(runtime->get_device());
    utility::log("render loop finished");
    return 0;
}
