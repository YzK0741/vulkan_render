#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
import std;
import gltf_loader;
import utility;
import vulkan.math;
import vulkan.runtime.scene_tree; // scene storage + GPU primitives (was vulkan.model)
import vulkan.runtime;

namespace {
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

} // namespace

int main(int argc, char** argv) {
    // 1. Locate the shaders directory (holds GLSL sources and compiled SPIR-V)
    std::optional<std::filesystem::path> const shaders_dir = locate_shaders_dir();
    if (!shaders_dir) {
        utility::panic("cannot find shaders/ directory. run the program from the project root or a cmake-build-* directory.");
    }

    // 2. Pick the model file: independent of the runtime, needed before kicking off the async
    //    load below (defaults to DamagedHelmet under gltf_model/; other .gltf/.glb via argv)
    std::string model_path;
    if (argc > 1) {
        model_path = argv[1];
    } else if (std::optional<std::filesystem::path> const located = locate_model_file()) {
        model_path = located->string();
    } else {
        utility::panic("cannot find gltf_model/DamagedHelmet.gltf. run the program from the project root or pass a model path as argv[1]");
    }

    // 3. Kick off the runtime-independent heavy CPU stages BEFORE constructing the (heavy)
    //    Vulkan runtime, so window/instance/device/swapchain init overlaps the model parse +
    //    texture decode and the base environment cubemap generation.
    constexpr int env_size = 256;
    constexpr int env_mip_count = 5;
    auto const startup_start = std::chrono::steady_clock::now();
    auto env_future = vulkan::generate_environment_cubemap_async(env_size);
    auto load_future = gltf::load_model_async(model_path);

    // 4. Construct vulkan::runtime: the default constructor performs all window/instance/device/swapchain initialization
    vulkan::runtime runtime;
    auto const runtime_ready = std::chrono::steady_clock::now();
    utility::log("vulkan runtime initialized: {:.1f} ms (async model load + env generation running in background)", std::chrono::duration<double, std::milli>(runtime_ready - startup_start).count());

    // 5. Pipelines: triangle + standard PBR + skybox background (fullscreen environment pass)
    load_and_create_pipeline(runtime, *shaders_dir, "triangle", "triangle.vert.spv", "triangle.frag.spv");
    load_and_create_pipeline(runtime, *shaders_dir, "pbr", "pbr.vert.spv", "pbr.frag.spv");
    {
        std::vector<unsigned char> vertex_code;
        std::vector<unsigned char> fragment_code;
        load_shader(*shaders_dir, "skybox.vert.spv", vertex_code);
        load_shader(*shaders_dir, "skybox.frag.spv", fragment_code);
        auto const skybox_result = runtime.make_skybox_pipeline(vertex_code, fragment_code);
        if (!skybox_result) {
            utility::panic(std::source_location::current(), "failed to create skybox pipeline: {}", skybox_result.error());
        }
        utility::log("SUCCESS: skybox pipeline created (fullscreen environment background)");
    }
    {
        // Shadow pass pipeline (depth-only): renders the scene from the light into the shadow
        // map. Created once; enable_shadows() below activates the pass after the scene import.
        std::vector<unsigned char> vertex_code;
        std::vector<unsigned char> fragment_code;
        load_shader(*shaders_dir, "shadow.vert.spv", vertex_code);
        load_shader(*shaders_dir, "shadow.frag.spv", fragment_code);
        auto const shadow_result = runtime.make_shadow_pipeline(vertex_code, fragment_code);
        if (!shadow_result) {
            utility::log("shadow pipeline disabled: {}", shadow_result.error());
        } else {
            utility::log("SUCCESS: shadow pipeline created (directional shadow map pass)");
        }
    }

    // 6. Collect the async startup results
    auto scenes = load_future.get();
    if (!scenes) {
        utility::panic(std::source_location::current(), "failed to load model '{}': error code {}", model_path, static_cast<int>(scenes.error()));
    }
    std::vector<float> const env = env_future.get();
    auto const startup_done = std::chrono::steady_clock::now();
    utility::log("model loaded + environment cubemap (startup window incl. runtime init): {:.1f} ms", std::chrono::duration<double, std::milli>(startup_done - startup_start).count());

    // 7. Scene bounds from the raw POSITION attributes (no geometry building): the world AABB
    //    of every drawable primitive's local AABB transformed by its node's world transform.
    //    gltf::scenes is iterable (begin()/end() flatten scene -> node -> mesh -> primitive).
    auto const local_bounds = [](gltf::primitive const& prim) -> std::pair<glm::vec3, glm::vec3> {
        glm::vec3 min(std::numeric_limits<float>::infinity());
        glm::vec3 max(-std::numeric_limits<float>::infinity());
        auto const it = prim.vertex.find("POSITION");
        if (it != prim.vertex.end()) {
            size_t const count = it->second.data.size() / sizeof(glm::vec3);
            for (size_t i = 0; i < count; ++i) {
                glm::vec3 const p = reinterpret_cast<glm::vec3 const*>(it->second.data.data())[i];
                min = glm::min(min, p);
                max = glm::max(max, p);
            }
        }
        return {min, max};
    };
    glm::vec3 scene_min(std::numeric_limits<float>::infinity());
    glm::vec3 scene_max(-std::numeric_limits<float>::infinity());
    size_t primitive_count = 0;
    for (gltf::drawable_ref const& drawable : *scenes) {
        ++primitive_count;
        auto const [lmin, lmax] = local_bounds(*drawable.primitive);
        // TRS transforms map an AABB to an AABB, so transforming the 8 corners is exact
        for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
                for (int z = 0; z < 2; ++z) {
                    glm::vec3 const corner(x ? lmax.x : lmin.x, y ? lmax.y : lmin.y, z ? lmax.z : lmin.z);
                    glm::vec4 const world = drawable.transform_matrix * glm::vec4(corner, 1.0f);
                    scene_min = glm::min(scene_min, glm::vec3(world));
                    scene_max = glm::max(scene_max, glm::vec3(world));
                }
            }
        }
    }
    if (primitive_count == 0) {
        utility::panic("model has no drawable primitives");
    }
    glm::vec3 const scene_center = scene_min * 0.5f + scene_max * 0.5f;
    float const scene_radius = glm::length(scene_max - scene_min) * 0.5f;
    utility::log("scene loaded: {} textures, {} materials, {} primitives", scenes->textures.size(), scenes->materials.size(), primitive_count);
    utility::log("scene bounds: center ({:.3f}, {:.3f}, {:.3f}), radius {:.3f}", scene_center.x, scene_center.y, scene_center.z, scene_radius);

    // Scene hierarchy summary: the loader retains the node tree (children + local transforms),
    // so report the tree shape (roots / total nodes / max depth / mesh-bearing nodes) for
    // diagnostics. Walks the retained tree through gltf::scene_node_iterator (DFS pre-order,
    // transform-only nodes included) — the same iterator the runtime's import consumes.
    {
        size_t total_nodes = 0;
        size_t mesh_nodes = 0;
        size_t max_depth = 0;
        std::vector<std::string> tree_lines;
        for (gltf::scene_node_iterator it = scenes->nodes_begin(); it != scenes->nodes_end(); ++it) {
            ++total_nodes;
            size_t const depth = it.get_depth();
            max_depth = std::max(max_depth, depth);
            bool const has_mesh = it.get_drawable_count() > 0;
            if (has_mesh) {
                ++mesh_nodes;
            }
            std::string_view const name = it.get_name();
            tree_lines.push_back(std::format("{}{}{}", std::string(depth * 2, ' '),
                                             name.empty() ? std::string("<unnamed>") : std::string(name),
                                             has_mesh ? " [mesh]" : ""));
        }
        utility::log("scene hierarchy: {} roots, {} nodes total ({} with meshes), max depth {}",
                     scenes->scene.size() > 0 ? scenes->scene.front().root_indices.size() : 0,
                     total_nodes, mesh_nodes, max_depth);
        for (std::string const& line : tree_lines) {
            utility::log("  {}", line);
        }
    }

    // Sink the model so it sits near the world horizon (y = 0) and move the camera target with it:
    // the camera then orbits/looks at the model's position instead of the scene origin.
    glm::vec3 const scene_sink(0.0f, -scene_radius, 0.0f);
    runtime.camera.target = scene_sink;

    // 8. IBL stage 2 + material resolve run concurrently via their _async wrappers: the
    //    prefilter (GGX importance sampling), irradiance map, BRDF LUT and the per-material
    //    texture decode + mip chains only depend on what we already have (env, scenes). The
    //    per-stage times are not reported individually: get() orders the waits, so only the
    //    wall-clock of the parallel stage is meaningful (the other tasks hide under the
    //    slowest one).
    utility::log("generating IBL (prefilter/irradiance/BRDF LUT) + resolving materials...");
    auto const stage2_start = std::chrono::steady_clock::now();
    auto prefilter_future = vulkan::prefilter_environment_async(env, env_size, env_mip_count);
    auto irradiance_future = vulkan::generate_irradiance_map_async(env, env_size, 32);
    auto lut_future = vulkan::generate_brdf_lut_async(256);
    auto resolve_future = gltf::resolve_materials_async(*scenes);

    std::vector<float> const prefiltered = prefilter_future.get();
    std::vector<float> const irradiance = irradiance_future.get();
    std::vector<float> const brdf_lut = lut_future.get();
    std::vector<gltf::resolved_material> const materials = resolve_future.get();
    auto const stage2_done = std::chrono::steady_clock::now();
    utility::log("  IBL (prefilter/irradiance/BRDF LUT) + material resolve, parallel wall: {:.1f} ms", std::chrono::duration<double, std::milli>(stage2_done - stage2_start).count());

    std::vector<unsigned char> const env_bytes = vulkan::to_half_rgba(prefiltered);
    std::vector<unsigned char> const irr_bytes = vulkan::to_half_rgba(irradiance);
    std::vector<unsigned char> const lut_bytes = vulkan::to_half_rg(brdf_lut);

    // 10. Upload the scene-wide IBL once: shared by every primitive (bindings 2-4 of the scene set)
    runtime.set_ibl(vulkan::ibl_input{.prefiltered_env = env_bytes, .irradiance = irr_bytes, .brdf_lut = lut_bytes, .env_size = env_size, .env_mip_count = env_mip_count, .irr_size = 32, .lut_size = 256});

    // 11. Batch-import: the runtime drives the traversal itself through two aligned loader
    //     streams — the retained node hierarchy (gltf::scene_node_iterator: DFS pre-order,
    //     transform-only nodes included, name + local transform per node) and the drawables
    //     of those nodes (gltf::drawable_iterator: geometry/material getters, node-aligned).
    //     The runtime rebuilds the scene tree (node per loader node) and attaches each
    //     drawable as a leaf primitive under its node, so whole-group transforms work on the
    //     imported hierarchy. The orbit camera looks at the origin, so center the scene and
    //     pull it back to fit its radius (same framing as the old single-model fit).
    runtime.camera.distance = scene_radius * 2.75f;
    gltf::scene_node_iterator const node_first = scenes->nodes_begin();
    gltf::scene_node_iterator const node_last;
    gltf::drawable_iterator const scene_first(*scenes, materials);
    gltf::drawable_iterator const scene_last;
    vulkan::scene_import_result const imported = runtime.import_scene(node_first, node_last, scene_first, scene_last, -scene_center + scene_sink);
    utility::log("imported {} primitives ({} new materials)", imported.primitive_count, imported.material_count);
    runtime.log_scene_tree();

    // 11b. Enable directional shadow mapping over the imported scene: the shadow frustum frames
    //      the sphere around where the primitives actually sit (they were translated by the import
    //      offset above, so their world-space center is scene_sink) with their original radius
    runtime.enable_shadows(scene_sink, scene_radius);

    // 12. Optional instancing stress: `vulkan_render <model> <grid_side>` draws the first
    //     imported primitive as a grid_side x grid_side grid in ONE instanced draw call
    //     (an instanced_draw_primitive appended to the scene tree — the frame loop is untouched)
    if (argc > 2) {
        int const side = std::atoi(argv[2]);
        if (side > 1) {
            std::vector<vulkan::primitive const*> const pbr_primitives = runtime.get_primitives("pbr");
            if (!pbr_primitives.empty()) {
                vulkan::primitive const& source = *pbr_primitives[0];
                std::vector<glm::mat4> transforms;
                transforms.reserve(static_cast<size_t>(side) * side);
                float const spacing = 2.5f * scene_radius; // keep instances apart: measure draw scaling, not overdraw
                for (int i = 0; i < side; ++i) {
                    for (int j = 0; j < side; ++j) {
                        float const dx = (static_cast<float>(i) - (side - 1) * 0.5f) * spacing;
                        float const dz = (static_cast<float>(j) - (side - 1) * 0.5f) * spacing;
                        transforms.push_back(glm::translate(glm::mat4(1.0f), glm::vec3(dx, 0.0f, dz)) * source.push.model);
                    }
                }
                runtime.make_instanced_primitive(source, transforms);
                utility::log("instancing stress: {} x {} grid ({} instances, 1 draw call)", side, side, transforms.size());
            }
        }
    }

    // 14. Main render loop: until the window closes or ESC is pressed.
    //     Every Vulkan frame step (fences, acquire, command buffers, render pass, submit, present)
    //     lives inside runtime::render_frame()
    utility::log("rendering '{}' with PBR... left-drag to orbit, wheel to zoom, ESC to exit", model_path);

    // Optional demo transforms (argv[2] when it is not a grid-side number, or argv[3]):
    //   "spin"          — whole-scene rotation via runtime::set_scene_transform (extra world
    //                     matrix on top of every root; the whole tree moves together).
    //   "spin-subtree"  — per-node local transform: rotate ONE primitive leaf node around its
    //                     own position in parent space (the scene tree's per-node local
    //                     transforms make this possible). On the hierarchy asset this spins a
    //                     single helmet in place while its sibling stays still; on the default
    //                     single-node asset the primitive spins about the scene sink.
    //   "nocull"        — verification: disable frustum culling (force every leaf visible); run
    //                     the same camera path with and without it and compare the cull log + fps.
    bool spin_scene = false;
    bool spin_subtree = false;
    bool no_cull = false;
    bool closeup = false;
    char const* const demo = argc > 3 ? argv[3] : (argc > 2 ? argv[2] : nullptr);
    if (demo != nullptr) {
        std::string_view const demo_view(demo);
        spin_scene = demo_view == "spin";
        spin_subtree = demo_view == "spin-subtree";
        no_cull = demo_view == "nocull";
        closeup = demo_view == "closeup";
    }
    if (no_cull) {
        runtime.set_frustum_culling(false);
        utility::log("nocull: frustum culling disabled (all leaves drawn every frame)");
    }
    if (closeup) {
        // pull the camera close so only part of the scene fits the frustum -> partial culling
        runtime.camera.distance *= 0.22f;
        utility::log("closeup: camera pulled in (partial frustum culling expected)");
    }
    double spin_angle = 0.0;
    if (spin_scene) {
        utility::log("spin: rotating the whole scene about the scene sink");
    }
    // target node + its initial local transform for the subtree demo (found once, before the loop)
    vulkan::scene_tree::scene_node* subtree_node = nullptr;
    glm::mat4 subtree_local0 = glm::mat4(1.0f);
    glm::vec3 subtree_pivot = glm::vec3(0.0f);
    if (spin_subtree) {
        // find the first node carrying a primitive leaf (DFS pre-order over all roots)
        std::vector<vulkan::scene_tree::scene_node*> stack;
        for (vulkan::scene_tree::scene_node& root : runtime.scene().roots) {
            stack.push_back(&root);
        }
        while (!stack.empty() && subtree_node == nullptr) {
            vulkan::scene_tree::scene_node* const node = stack.back();
            stack.pop_back();
            if (node->primitive_leaf != nullptr) {
                subtree_node = node;
                subtree_local0 = node->local;
                // pivot = where this node sits in parent space (translation column of its local)
                subtree_pivot = glm::vec3(subtree_local0[3]);
                break;
            }
            for (vulkan::scene_tree::scene_node& child : node->children) {
                stack.push_back(&child);
            }
        }
        if (subtree_node == nullptr) {
            utility::log("spin-subtree: scene has no primitive leaf node to rotate");
        } else {
            utility::log("spin-subtree: rotating node '{}' about its own position", subtree_node->name);
        }
    }

    // FPS statistics: accumulate frame times, report once per second (log + window title)
    std::chrono::steady_clock::time_point last_frame_time = std::chrono::steady_clock::now();
    double fps_elapsed = 0.0;
    uint32_t fps_frame_count = 0;

    // All per-frame decisions (event polling, ESC/close response, minimize skip, swapchain
    // recreation on restore/resize) live inside runtime::render_frame(); main only reacts to
    // the returned frame_result.
    while (true) {
        if (spin_scene) {
            // rotate the whole scene around scene_sink (its own center): shadows stay valid
            spin_angle += 0.6 * std::chrono::duration<double>(std::chrono::steady_clock::now() - last_frame_time).count();
            glm::mat4 const center = glm::translate(glm::mat4(1.0f), scene_sink);
            runtime.set_scene_transform(center * glm::rotate(glm::mat4(1.0f), static_cast<float>(spin_angle), glm::vec3(0.0f, 1.0f, 0.0f)) * glm::inverse(center));
        }
        if (spin_subtree && subtree_node != nullptr) {
            // rotate ONE node's local transform about its own position (pivot in parent space):
            // the leaf primitive under it spins in place while sibling nodes stay put — the scene
            // tree's per-node locals make whole-group AND per-primitive transforms possible.
            spin_angle += 0.6 * std::chrono::duration<double>(std::chrono::steady_clock::now() - last_frame_time).count();
            glm::mat4 const pivot = glm::translate(glm::mat4(1.0f), subtree_pivot);
            subtree_node->local = pivot * glm::rotate(glm::mat4(1.0f), static_cast<float>(spin_angle), glm::vec3(0.0f, 1.0f, 0.0f)) * glm::inverse(pivot) * subtree_local0;
        }
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
            runtime->set_window_title(std::format("vulkan_render - {:.1f} fps", fps));
            fps_elapsed = 0.0;
            fps_frame_count = 0;
        }
    }

    // 18. Wait for the GPU to finish; primitives and pipelines are released by the runtime destructor
    runtime->wait_idle();
    utility::log("render loop finished");
    return 0;
}
