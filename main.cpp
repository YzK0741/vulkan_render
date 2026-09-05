#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
import std;
import app_config;
import gltf_loader;
import utility;
import vulkan.math;
import vulkan.runtime.scene_tree; // scene storage + GPU primitives (was vulkan.model)
import vulkan.runtime;

// Route std::pmr allocations through mimalloc (utility.better_pmr) before main(): this
// file-scope reference's dynamic initialization runs at startup, so every runtime/scene
// object built below already allocates its std::pmr vectors from mimalloc. Idempotent —
// other TUs (vulkan/runtime.cpp) keep their own copy of the same singleton.
[[maybe_unused]] static auto& pmr = utility::init_pmr(); // NOLINT(keep-alive)

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
    // 1. Resolve startup settings first: config file (config.toml by default, --config <path>
    //    to override) merged with positional argv overrides. argv[1] = model, argv[2] = grid
    //    side (numeric) or demo, argv[3] = demo.
    app_config::app_settings const settings = app_config::resolve_from_argv(argc, argv);
    if (!settings.config_file.empty()) {
        utility::log("app_config: loaded startup settings from '{}'", settings.config_file);
    }

    // 2. Shaders directory (holds GLSL sources and compiled SPIR-V): explicit config path when
    //    given, otherwise walk up from the working directory to find shaders/.
    std::filesystem::path shaders_dir;
    if (!settings.paths.shaders_dir.empty()) {
        shaders_dir = settings.paths.shaders_dir;
        if (!std::filesystem::is_directory(shaders_dir)) {
            utility::panic(std::source_location::current(), "cannot find configured shaders_dir '{}'.", settings.paths.shaders_dir);
        }
    } else if (std::optional<std::filesystem::path> const located = locate_shaders_dir()) {
        shaders_dir = *located;
    } else {
        utility::panic("cannot find shaders/ directory. run the program from the project root, pass shaders_dir in config.toml, or use a cmake-build-* directory.");
    }

    // 3. Pick the model file: settings.model when configured/argv-given; else the default model
    //    under settings.paths.model_dir (or the auto-located gltf_model/).
    std::string model_path;
    if (!settings.model.empty()) {
        model_path = settings.model;
    } else if (!settings.paths.model_dir.empty()) {
        model_path = (std::filesystem::path(settings.paths.model_dir) / "DamagedHelmet.gltf").string();
        if (!std::filesystem::is_regular_file(model_path)) {
            utility::panic(std::source_location::current(), "cannot find model '{}' under configured model_dir '{}'.", "DamagedHelmet.gltf", settings.paths.model_dir);
        }
    } else if (std::optional<std::filesystem::path> const located = locate_model_file()) {
        model_path = located->string();
    } else {
        utility::panic("cannot find gltf_model/DamagedHelmet.gltf. run the program from the project root or pass a model path as argv[1]");
    }

    // 4. Kick off the runtime-independent heavy CPU stages BEFORE constructing the (heavy)
    //    Vulkan runtime, so window/instance/device/swapchain init overlaps the model parse +
    //    texture decode and the base environment cubemap generation. IBL resolutions come from
    //    the [lighting] config (smaller = faster startup, larger = higher quality).
    auto const env_size = settings.lighting.env_size;
    auto const env_mip_count = settings.lighting.env_mip_count;
    auto const irr_size = settings.lighting.irr_size;
    auto const lut_size = settings.lighting.lut_size;
    auto const startup_start = std::chrono::steady_clock::now();
    auto env_future = vulkan::generate_environment_cubemap_async(env_size);
    auto load_future = gltf::load_model_async(model_path);

    // 5. Construct vulkan::runtime from the startup render settings (window size / title /
    //    vsync / MSAA; the defaults in render_settings mirror the historic hardcoded values)
    vulkan::core_create_info core_options = {};
    core_options.window_width = settings.render.window_width;
    core_options.window_height = settings.render.window_height;
    core_options.window_title = settings.render.window_title;
    core_options.vsync = settings.render.vsync;
    core_options.msaa_samples = settings.render.msaa;
    vulkan::runtime runtime{core_options};
    runtime.clear_color = glm::vec3(settings.render.clear_color[0], settings.render.clear_color[1], settings.render.clear_color[2]);
    // render-stage toggles from config: skybox applies immediately (only affects recording);
    // shadow is applied after enable_shadows() below (it needs the shadow maps to exist)
    runtime.set_skybox_enabled(settings.render.skybox);
    auto const runtime_ready = std::chrono::steady_clock::now();
    utility::log("vulkan runtime initialized: {:.1f} ms (async model load + env generation running in background)", std::chrono::duration<double, std::milli>(runtime_ready - startup_start).count());

    // 5. Pipelines: triangle + standard PBR + skybox background (fullscreen environment pass)
    load_and_create_pipeline(runtime, shaders_dir, "triangle", "triangle.vert.spv", "triangle.frag.spv");
    load_and_create_pipeline(runtime, shaders_dir, "pbr", "pbr.vert.spv", "pbr.frag.spv");
    {
        std::vector<unsigned char> vertex_code;
        std::vector<unsigned char> fragment_code;
        load_shader(shaders_dir, "skybox.vert.spv", vertex_code);
        load_shader(shaders_dir, "skybox.frag.spv", fragment_code);
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
        load_shader(shaders_dir, "shadow.vert.spv", vertex_code);
        load_shader(shaders_dir, "shadow.frag.spv", fragment_code);
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

    // Animation summary (diagnostics): gltf::scenes::animations holds the file's decoded
    // keyframe animations (channels -> samplers, see docs/gltf_loader_usage.md). Playback is
    // not implemented yet; this logs what the loader exported.
    if (!scenes->animations.empty()) {
        utility::log("animations: {}", scenes->animations.size());
        for (gltf::animation const& anim : scenes->animations) {
            std::string_view const anim_name = anim.name.empty() ? std::string_view("<unnamed>") : std::string_view(anim.name);
            // a typical animation shares one keyframe count across its samplers; report the first
            size_t const keys = anim.samplers.empty() ? 0 : anim.samplers.front().times.size();
            utility::log("  animation '{}': {} channels, {} samplers, {} keyframes", anim_name, anim.channels.size(), anim.samplers.size(), keys);
        }
    }

    // Skin summary (diagnostics): scenes::skins holds the file's skins (joint asset-node
    // indices + inverse bind matrices); playback of the joint transforms is a later stage.
    if (!scenes->skins.empty()) {
        utility::log("skins: {}", scenes->skins.size());
        for (gltf::skin const& skin : scenes->skins) {
            std::string_view const skin_name = skin.name.empty() ? std::string_view("<unnamed>") : std::string_view(skin.name);
            utility::log("  skin '{}': {} joints", skin_name, skin.joints.size());
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
    auto irradiance_future = vulkan::generate_irradiance_map_async(env, env_size, irr_size);
    auto lut_future = vulkan::generate_brdf_lut_async(lut_size);
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
    runtime.set_ibl(vulkan::ibl_input{.prefiltered_env = env_bytes, .irradiance = irr_bytes, .brdf_lut = lut_bytes, .env_size = static_cast<uint32_t>(env_size), .env_mip_count = static_cast<uint32_t>(env_mip_count), .irr_size = static_cast<uint32_t>(irr_size), .lut_size = static_cast<uint32_t>(lut_size)});

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
    glm::vec3 const scene_import_shift = -scene_center + scene_sink;
    vulkan::scene_import_result const imported = runtime.import_scene(node_first, node_last, scene_first, scene_last, scene_import_shift);
    utility::log("imported {} primitives ({} new materials)", imported.primitive_count, imported.material_count);
    runtime.log_scene_tree();

    // 11b. Enable directional shadow mapping over the imported scene: the shadow frustum frames
    //      the sphere around where the primitives actually sit (they were translated by the import
    //      offset above, so their world-space center is scene_sink) with their original radius
    runtime.enable_shadows(scene_sink, scene_radius);
    // apply the config shadow toggle now that the shadow maps exist (turning it off clears them)
    if (!settings.render.shadow) {
        runtime.set_shadow_enabled(false);
    }

    // 12. Optional instancing stress: grid_side > 1 (config or argv) draws the first imported
    //     primitive as a grid_side x grid_side grid in ONE instanced draw call
    //     (an instanced_draw_primitive appended to the scene tree — the frame loop is untouched)
    if (settings.grid_side > 1) {
        int const side = settings.grid_side;
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

    // Optional demo transforms (settings.demo: config.toml demo = ... or argv):
    //   "spin"          — whole-scene rotation via runtime::set_scene_transform (extra world
    //                     matrix on top of every root; the whole tree moves together).
    //   "spin-subtree"  — per-node local transform: rotate ONE primitive leaf node around its
    //                     own position in parent space (the scene tree's per-node local
    //                     transforms make this possible). On the hierarchy asset this spins a
    //                     single helmet in place while its sibling stays still; on the default
    //                     single-node asset the primitive spins about the scene sink.
    //   "nocull"        — verification: disable frustum culling (force every leaf visible); run
    //                     the same camera path with and without it and compare the cull log + fps.
    //   "gui"           — enable the Dear ImGui debug overlay (fps text + culling checkbox)
    bool spin_scene = false;
    bool spin_subtree = false;
    bool no_cull = false;
    bool closeup = false;
    bool use_gui = false;
    if (!settings.demo.empty()) {
        std::string_view const demo_view(settings.demo);
        spin_scene = demo_view == "spin";
        spin_subtree = demo_view == "spin-subtree";
        no_cull = demo_view == "nocull";
        closeup = demo_view == "closeup";
        use_gui = demo_view == "gui";
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
        for (vulkan::scene_tree::scene_node& root : runtime.get_scene().roots) {
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

    // ---- keyframe animation playback (the loader exports keyframes; see docs/gltf_loader_usage.md) ----
    // The first animation with channels plays on loop against the imported tree. Runtime nodes
    // record the loader's asset node index (scene_node::source_index, set by import_scene), so
    // every channel target maps onto the live tree; node locals are re-evaluated each frame from
    // the node's TRS base pose (kept in the retained loader tree) plus the animation channels.
    struct anim_playback {
        gltf::animation const* animation = nullptr; // the animation being played (null = none)
        double time = 0.0;                          // playback clock, seconds
        float duration = 1.0f;                      // loop length = max sampler time
        bool playing = true;
        std::chrono::steady_clock::time_point last_tick = {};               // real-time anchor of the clock
        std::size_t debug_source = std::numeric_limits<std::size_t>::max(); // first animated source present (reporting)
        glm::vec3 debug_translation = glm::vec3(0.0f);
    } anim;
    anim.last_tick = std::chrono::steady_clock::now();
    for (gltf::animation const& candidate : scenes->animations) {
        if (!candidate.channels.empty()) {
            anim.animation = &candidate;
            break;
        }
    }
    // live-tree lookup: asset node index -> runtime scene nodes (a node may be reachable from
    // several roots) + whether each occurrence sits at a scene root (import_scene applied the
    // import shift to root locals only, so animated roots must re-apply it)
    struct anim_target {
        vulkan::scene_tree::scene_node* node = nullptr;
        bool scene_root = false;
    };
    std::unordered_map<std::size_t, std::vector<anim_target>> source_nodes;
    auto const collect = [&source_nodes](auto&& self, vulkan::scene_tree::scene_node& node, bool const scene_root) -> void {
        source_nodes[node.source_index].push_back(anim_target{&node, scene_root});
        for (vulkan::scene_tree::scene_node& child : node.children) {
            self(self, child, false);
        }
    };
    for (vulkan::scene_tree::scene_node& root : runtime.get_scene().roots) {
        collect(collect, root, true);
    }
    // TRS base pose per asset node index, from the retained loader tree (matrix nodes cannot be
    // animated per the glTF spec and keep identity here)
    std::unordered_map<std::size_t, gltf::node_pose> anim_base_poses;
    // loader nodes by asset node index (base-pose / skin lookups; the loader tree stays alive)
    std::unordered_map<std::size_t, gltf::node const*> loader_nodes;
    for (gltf::scene const& loader_scene : scenes->scene) {
        for (gltf::node const& loader_node : loader_scene.nodes) {
            anim_base_poses.try_emplace(loader_node.source_index,
                                        gltf::node_pose{.translation = loader_node.translation,
                                                        .rotation = loader_node.rotation,
                                                        .scale = loader_node.scale});
            loader_nodes.try_emplace(loader_node.source_index, &loader_node);
        }
    }
    // the node reported per second: prefer a translation channel target (its value is visible
    // in the log), fall back to the first channel target present in the tree
    auto const pick_debug_source = [&source_nodes](gltf::animation const& animation) {
        std::size_t fallback = std::numeric_limits<std::size_t>::max();
        for (gltf::animation_channel const& channel : animation.channels) {
            if (!source_nodes.contains(channel.target_node)) {
                continue;
            }
            if (channel.path == gltf::animation_path::translation) {
                return channel.target_node;
            }
            if (fallback == std::numeric_limits<std::size_t>::max()) {
                fallback = channel.target_node;
            }
        }
        return fallback;
    };
    if (anim.animation != nullptr) {
        for (gltf::animation_sampler const& sampler : anim.animation->samplers) {
            if (!sampler.times.empty()) {
                anim.duration = std::max(anim.duration, sampler.times.back());
            }
        }
        anim.debug_source = pick_debug_source(*anim.animation);
        std::string_view const anim_name = anim.animation->name.empty() ? std::string_view("<unnamed>") : std::string_view(anim.animation->name);
        utility::log("animation: playing '{}' ({} channels, {:.2f}s loop)", anim_name, anim.animation->channels.size(), anim.duration);
    }

    // Playback controls (gui): the animation combo lists every channel-bearing animation, the
    // time slider needs a float mirror of the clock (slider_widget binds an external float)
    // whose range covers all playable animations.
    std::vector<gltf::animation const*> playable_animations;
    for (gltf::animation const& candidate : scenes->animations) {
        if (!candidate.channels.empty()) {
            playable_animations.push_back(&candidate);
        }
    }
    auto const animation_duration = [](gltf::animation const& animation) {
        float duration = 1.0f; // avoid a zero-length loop
        for (gltf::animation_sampler const& sampler : animation.samplers) {
            if (!sampler.times.empty()) {
                duration = std::max(duration, sampler.times.back());
            }
        }
        return duration;
    };
    float gui_anim_max = 1.0f;
    for (gltf::animation const* playable : playable_animations) {
        gui_anim_max = std::max(gui_anim_max, animation_duration(*playable));
    }
    int gui_anim_index = 0;     // selected item of the animation combo (0 = the auto-played one)
    float gui_anim_time = 0.0f; // float mirror of the playback clock (time-slider target)

    // ---- glTF skinning ----
    // Every exported skin that drives an imported mesh becomes a "rig": its joints resolve onto
    // live scene nodes (they follow the animation playback above), each skinned primitive points
    // at a block of the scene skin buffer (indices 0-3 are the identity block for unskinned
    // draws), and per frame the skin matrices skinMat_j = inv(W_mesh) * W_joint_j * IBM_j are
    // rebuilt and uploaded. Assumptions typical of glTF assets: one occurrence per skinned node
    // and a static skinned node (its inverse is taken per frame).
    struct skin_rig {
        gltf::skin const* skin = nullptr; // loader skin: joints (asset node indices) + IBM
        std::size_t mesh_source = 0;      // asset node index of the skinned mesh node
        uint32_t block_base = 0;          // block start in the skin buffer (after the identity block)
    };
    std::vector<skin_rig> skin_rigs;
    if (!scenes->skins.empty()) {
        uint32_t next_block = 4; // identity block occupies indices 0-3
        for (std::size_t skin_id = 0; skin_id < scenes->skins.size(); ++skin_id) {
            gltf::skin const& loader_skin = scenes->skins[skin_id];
            // the first loader node referencing this skin that is present in the imported tree
            std::size_t mesh_source = std::numeric_limits<std::size_t>::max();
            for (auto const& [source, loader_node] : loader_nodes) {
                if (loader_node->skin_index && *loader_node->skin_index == skin_id && source_nodes.contains(source)) {
                    mesh_source = source;
                    break;
                }
            }
            if (mesh_source == std::numeric_limits<std::size_t>::max()) {
                continue; // the skin is not used by the imported scene
            }
            bool const all_joints_present = std::all_of(loader_skin.joints.begin(), loader_skin.joints.end(), [&source_nodes](std::size_t const joint) { return source_nodes.contains(joint); });
            if (!all_joints_present) {
                std::string_view const sname = loader_skin.name.empty() ? std::string_view("<unnamed>") : std::string_view(loader_skin.name);
                utility::log("skinning: skin '{}' skipped (joint(s) missing from the imported scene)", sname);
                continue;
            }
            uint32_t const block_base = next_block;
            next_block += static_cast<uint32_t>(loader_skin.joints.size());
            // point every primitive leaf of the skinned node at the block: the node's own leaf
            // plus extra-primitive child leaves (import adds them under the node with the
            // default source_index 0); real child nodes (other source indices) keep skin_base 0
            vulkan::scene_tree::scene_node* const mesh_node = source_nodes.at(mesh_source).front().node;
            auto const assign_block = [block_base, mesh_source](auto&& self, vulkan::scene_tree::scene_node& node) -> void {
                if (node.primitive_leaf != nullptr && (node.source_index == 0 || node.source_index == mesh_source)) {
                    static_cast<vulkan::primitive*>(node.primitive_leaf.get())->push.skin_base = block_base;
                }
                for (vulkan::scene_tree::scene_node& child : node.children) {
                    self(self, child);
                }
            };
            assign_block(assign_block, *mesh_node);
            skin_rigs.push_back(skin_rig{&loader_skin, mesh_source, block_base});
        }
        if (!skin_rigs.empty()) {
            utility::log("skinning: {} skin rig(s) active ({} joint matrix block(s) + identity block)", skin_rigs.size(), next_block - 4);
        }
    }
    // identity block for unskinned draws: upload once (the skin buffer starts zeroed)
    {
        std::array<glm::mat4, 4> const identity_block = {glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f)};
        runtime.set_skin_matrices(identity_block);
    }
    // per-second skin report (first rig's joint 0 world position, follows the animation)
    glm::vec3 skin_debug_translation = glm::vec3(0.0f);
    bool skin_debug_valid = false;

    // Optional Dear ImGui debug overlay: the runtime drives new_frame/record inside its frame
    // steps; main only enables it and manages its content through the panel/widget API (fps
    // text widget bound to a live lambda + a frustum-culling checkbox that forwards to the
    // runtime). The checkbox is a slider-free toggle bound to an external bool.
    double gui_fps = 0.0;
    bool gui_cull_enabled = true;
    bool gui_skybox_enabled = settings.render.skybox;
    bool gui_shadow_enabled = settings.render.shadow;
    if (use_gui) {
        runtime.enable_debug_gui();
        vulkan::gui::debug_panel& panel = runtime.debug_gui().add_panel("vulkan_render debug");
        panel.set_default_size(settings.gui.panel_width, settings.gui.panel_height);
        panel.push_back(std::make_unique<vulkan::gui::label_widget>([&gui_fps] { return std::format("fps: {:.1f}", gui_fps); }));
        panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>(
            "frustum culling",
            &gui_cull_enabled,
            [&runtime](bool const enabled) { runtime.set_frustum_culling(enabled); }));
        panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>(
            "skybox",
            &gui_skybox_enabled,
            [&runtime](bool const enabled) { runtime.set_skybox_enabled(enabled); }));
        panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>(
            "shadow",
            &gui_shadow_enabled,
            [&runtime](bool const enabled) { runtime.set_shadow_enabled(enabled); }));
        // camera orbit target: dragging it moves what the camera looks at / orbits around
        // (camera.target is a glm::vec3, i.e. three contiguous floats; the runtime rebuilds the
        // camera UBO from it every frame, so no on_change callback is needed)
        panel.push_back(std::make_unique<vulkan::gui::vec3_widget>("camera target", &runtime.camera.target.x, 0.05f));
        // playback controls (only when the model carries animations): play/pause toggle bound
        // to the playback state, a time scrubber (pauses on drag so the clock cannot fight the
        // scrub; the play checkbox resumes), and — for multi-animation assets — a dropdown to
        // pick which animation plays. All state lives in main's anim struct above.
        if (anim.animation != nullptr) {
            panel.push_back(std::make_unique<vulkan::gui::label_widget>([&anim] {
                std::string_view const name = anim.animation->name.empty() ? std::string_view("<unnamed>") : std::string_view(anim.animation->name);
                return std::format("animation '{}' ({} channels)", name, anim.animation->channels.size());
            }));
            panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>("play", &anim.playing));
            panel.push_back(std::make_unique<vulkan::gui::slider_widget>(
                "time",
                &gui_anim_time,
                0.0f,
                gui_anim_max,
                [&anim](float const value) {
                    anim.time = value;
                    anim.playing = false; // scrubbing pauses so the clock does not fight the drag
                }));
            if (playable_animations.size() > 1) {
                std::vector<std::string> names;
                names.reserve(playable_animations.size());
                for (gltf::animation const* playable : playable_animations) {
                    names.push_back(playable->name.empty() ? "<unnamed>" : playable->name);
                }
                panel.push_back(std::make_unique<vulkan::gui::combo_widget>(
                    "animation",
                    std::move(names),
                    &gui_anim_index,
                    [&anim, &gui_anim_time, &playable_animations, &animation_duration, &pick_debug_source](int const index) {
                        gltf::animation const* const next = playable_animations[static_cast<std::size_t>(index)];
                        anim.animation = next;
                        anim.time = 0.0;
                        gui_anim_time = 0.0f;
                        anim.duration = animation_duration(*next);
                        anim.debug_source = pick_debug_source(*next);
                    }));
            }
            utility::log("gui: playback controls added ({} animation(s))", playable_animations.size());
        }
        utility::log("gui: Dear ImGui debug overlay enabled");
    }

    // All per-frame decisions (event polling, ESC/close response, minimize skip, swapchain
    // recreation on restore/resize) live inside runtime::render_frame(); main only reacts to
    // the returned frame_status.
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
            runtime.scene_changed(); // edited node.local directly -> culling BVH must track it
        }

        // advance the animation clock by real time and re-evaluate the animated node locals
        // (same "edit node.local, then mark the scene changed" contract the demos above use)
        if (anim.animation != nullptr) {
            auto const anim_now = std::chrono::steady_clock::now();
            if (anim.playing) {
                anim.time += std::chrono::duration<double>(anim_now - anim.last_tick).count();
                if (anim.time >= anim.duration) {
                    anim.time = std::fmod(anim.time, anim.duration);
                }
            }
            anim.last_tick = anim_now;
            bool changed = false;
            for (auto const& [source, targets] : source_nodes) {
                auto const base_it = anim_base_poses.find(source);
                gltf::node_pose const base = base_it == anim_base_poses.end() ? gltf::node_pose{} : base_it->second;
                gltf::node_pose const pose = gltf::sample_node(*anim.animation, source, base, static_cast<float>(anim.time));
                if (!pose.any_channel) {
                    continue;
                }
                // compose T * R * S; scene-root occurrences also keep the import shift that
                // placed the model (import_scene applied it to each root's local transform)
                glm::mat4 const trs = glm::translate(glm::mat4(1.0f), pose.translation) * glm::mat4_cast(pose.rotation) * glm::scale(glm::mat4(1.0f), pose.scale);
                for (anim_target const& target : targets) {
                    target.node->local = target.scene_root ? glm::translate(glm::mat4(1.0f), scene_import_shift) * trs : trs;
                }
                changed = true;
                if (source == anim.debug_source) {
                    anim.debug_translation = pose.translation;
                }
            }
            if (changed) {
                runtime.scene_changed(); // node.locals edited directly -> culling BVH must track them
            }
            gui_anim_time = static_cast<float>(anim.time); // keep the gui time slider in sync
        }

        // skin matrices: the joint worlds follow the animation applied above, so rebuild every
        // frame [identity block | per-rig joint blocks] and upload to the scene skin buffer
        if (!skin_rigs.empty()) {
            // collect the world matrices of the mesh + joint nodes with one DFS (same
            // accumulation rule as the runtime's update_world: world = parent_world * local)
            std::unordered_map<std::size_t, glm::mat4> skin_worlds;
            auto const collect_worlds = [&skin_worlds, &skin_rigs](auto&& self, vulkan::scene_tree::scene_node& node, glm::mat4 const& parent_world) -> void {
                glm::mat4 const world = parent_world * node.local;
                for (skin_rig const& rig : skin_rigs) {
                    if (node.source_index == rig.mesh_source) {
                        skin_worlds.try_emplace(node.source_index, world);
                    }
                    for (std::size_t const joint : rig.skin->joints) {
                        if (node.source_index == joint) {
                            skin_worlds.try_emplace(node.source_index, world);
                        }
                    }
                }
                for (vulkan::scene_tree::scene_node& child : node.children) {
                    self(self, child, world);
                }
            };
            for (vulkan::scene_tree::scene_node& root : runtime.get_scene().roots) {
                collect_worlds(collect_worlds, root, glm::mat4(1.0f));
            }
            // skinMat_j = inv(W_mesh) * W_joint_j * IBM_j; joints whose world is missing fall
            // back to identity so vertex joint ids stay aligned with their block
            std::vector<glm::mat4> matrices;
            matrices.reserve(4 + (skin_rigs.size() * 8));
            matrices.insert(matrices.end(), {glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f)});
            for (skin_rig const& rig : skin_rigs) {
                auto const mesh_it = skin_worlds.find(rig.mesh_source);
                glm::mat4 const mesh_world_inv = glm::inverse(mesh_it == skin_worlds.end() ? glm::mat4(1.0f) : mesh_it->second);
                for (std::size_t j = 0; j < rig.skin->joints.size(); ++j) {
                    auto const joint_it = skin_worlds.find(rig.skin->joints[j]);
                    glm::mat4 const joint_world = joint_it == skin_worlds.end() ? glm::mat4(1.0f) : joint_it->second;
                    matrices.push_back(mesh_world_inv * joint_world * rig.skin->inverse_bind_matrices[j]);
                }
            }
            runtime.set_skin_matrices(matrices);
            // per-second report data: first rig's LAST joint world x-axis (rotations change it,
            // unlike the joint's position which stays fixed under rotation-only animations)
            skin_debug_valid = false;
            if (!skin_rigs.empty() && !skin_rigs.front().skin->joints.empty()) {
                std::size_t const last_joint = skin_rigs.front().skin->joints.back();
                auto const joint_it = skin_worlds.find(last_joint);
                if (joint_it != skin_worlds.end()) {
                    skin_debug_valid = true;
                    skin_debug_translation = glm::vec3(joint_it->second[0]); // world x axis
                }
            }
        }
        vulkan::frame_status const result = runtime.render_frame();
        if (result == vulkan::frame_status::closed || vulkan::is_failure(result)) {
            break;
        }
        if (result == vulkan::frame_status::skipped) {
            // Minimized or swapchain recreated: skip this frame; keep the FPS timer fresh so the
            // pause is not counted as one huge rendered frame.
            last_frame_time = std::chrono::steady_clock::now();
            std::this_thread::yield();
            continue;
        }

        // proceed: frame time = wall time since the previous rendered frame
        auto const now = std::chrono::steady_clock::now();
        fps_elapsed += std::chrono::duration<double>(now - last_frame_time).count();
        last_frame_time = now;
        ++fps_frame_count;
        if (use_gui) {
            gui_fps = fps_frame_count / fps_elapsed; // smooth per-second value for the overlay
        }
        if (fps_elapsed >= 1.0) {
            double const fps = fps_frame_count / fps_elapsed;
            // fps is shown inside the ImGui overlay (when enabled); the log line stays for
            // headless / non-gui runs
            utility::log("fps: {:.1f} ({:.2f} ms/frame)", fps, 1000.0 * fps_elapsed / fps_frame_count);
            if (anim.animation != nullptr) {
                // report the playback clock + the first animated node's evaluated translation
                // (proves the keyframes are actually moving the tree)
                std::string_view const anim_name = anim.animation->name.empty() ? std::string_view("<unnamed>") : std::string_view(anim.animation->name);
                std::string_view const node_name = anim.debug_source == std::numeric_limits<std::size_t>::max()
                                                       ? std::string_view("<no target in scene>")
                                                       : std::string_view(source_nodes.at(anim.debug_source).front().node->name);
                utility::log("  anim '{}': t={:.3f}s/{:.2f}s, '{}' at ({:.3f}, {:.3f}, {:.3f})",
                             anim_name, anim.time, anim.duration, node_name,
                             anim.debug_translation.x, anim.debug_translation.y, anim.debug_translation.z);
            }
            if (skin_debug_valid) {
                std::string_view const skin_name = skin_rigs.front().skin->name.empty() ? std::string_view("<unnamed>") : std::string_view(skin_rigs.front().skin->name);
                utility::log("  skin '{}': last joint world x-axis ({:.3f}, {:.3f}, {:.3f})", skin_name,
                             skin_debug_translation.x, skin_debug_translation.y, skin_debug_translation.z);
            }
            fps_elapsed = 0.0;
            fps_frame_count = 0;
        }
    }

    // 18. Wait for the GPU to finish; primitives and pipelines are released by the runtime destructor
    runtime->wait_idle();
    utility::log("render loop finished");
    return 0;
}
