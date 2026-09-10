#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
import vstd;
import app_config;
import chores; // demo bootstrap helpers (shader loading / dir locating / pipelines)
import gltf_loader;
import utility;          // re-exports utility.frame_clock / frame_stats / bvh / better_pmr / thread_pool / data_block
import vulkan.animation; // animation::controller: glTF playback / skinning / morphs on the runtime tree
import vulkan.math;
import vulkan.scene_tree; // scene storage + GPU primitives (was vulkan.model)
import vulkan.runtime;

// Route std::pmr allocations through mimalloc (utility.better_pmr) before main(): this
// file-scope reference's dynamic initialization runs at startup, so every runtime/scene
// object built below already allocates its std::pmr vectors from mimalloc. Idempotent —
// other TUs (vulkan/runtime.cpp) keep their own copy of the same singleton.
[[maybe_unused]] static auto& pmr = utility::init_pmr(); // NOLINT(keep-alive)

namespace {

    // ---- scripted capture (dev tool) ----
    // Verifying anything visual used to need a human at the keyboard (F12). Two flags remove
    // that: `--capture-frames <n>` renders n frames, saves a screenshot through EXACTLY the F12
    // path (same read-back, same PNG writer, same logged line) and quits; `--capture-camera
    // <yaw,pitch,distance>` overrides the orbit camera at startup (degrees / scene units) so a
    // specific view - e.g. looking down-sun, where a shadow leak shows - reproduces on demand.
    // Both are stripped from argv before app_config sees them, so the positional model /
    // grid-side slots keep their meaning.
    struct capture_options {
        int frames = 0;                                            // 0 = normal interactive run
        std::optional<std::array<float, 3>> camera = std::nullopt; // yaw(deg), pitch(deg), distance
        std::optional<glm::vec3> target = std::nullopt;            // orbit target override (optional)
    };

    // strtof with a full-string check (no exceptions: std::stof would abort under -fno-exceptions)
    std::optional<float> parse_number(std::string_view const text) {
        if (text.empty()) {
            return std::nullopt;
        }
        std::string const copy(text); // strtof needs a null-terminated buffer
        char* end = nullptr;
        float const value = std::strtof(copy.c_str(), &end);
        if (end == copy.c_str() || *end != '\0') {
            return std::nullopt;
        }
        return value;
    }

    capture_options parse_capture_options(int const argc, char** argv, std::vector<char*>& filtered) {
        capture_options options = {};
        filtered.push_back(argv[0]);
        // "--flag value" or "--flag=value"; returns the value and advances i past it
        auto const take_value = [&](int& i, std::string_view const arg, std::string_view const name) -> std::optional<std::string_view> {
            if (arg == name) {
                return i + 1 < argc ? std::optional<std::string_view>(argv[++i]) : std::nullopt;
            }
            if (arg.size() > name.size() && arg[name.size()] == '=' && arg.starts_with(name)) {
                return arg.substr(name.size() + 1);
            }
            return std::nullopt;
        };
        for (int i = 1; i < argc; ++i) {
            std::string_view const arg(argv[i]);
            if (std::optional<std::string_view> const value = take_value(i, arg, "--capture-frames")) {
                if (std::optional<float> const frames = parse_number(*value)) {
                    options.frames = static_cast<int>(std::max(0.0f, *frames));
                } else {
                    utility::log("capture: ignoring '--capture-frames {}' (expected a frame count)", *value);
                }
                continue;
            }
            if (std::optional<std::string_view> const value = take_value(i, arg, "--capture-camera")) {
                // yaw,pitch,distance[,target.x,target.y,target.z] - comma-separated numbers
                std::array<float, 6> parsed = {};
                std::size_t cursor = 0;
                std::size_t count = 0;
                bool valid = true;
                while (cursor <= value->size()) {
                    std::size_t const comma = value->find(',', cursor);
                    std::string_view const piece = value->substr(cursor, comma == std::string_view::npos ? std::string_view::npos : comma - cursor);
                    std::optional<float> const number = parse_number(piece);
                    if (!number || count == parsed.size()) {
                        valid = false;
                        break;
                    }
                    parsed[count++] = *number;
                    if (comma == std::string_view::npos) {
                        break;
                    }
                    cursor = comma + 1;
                }
                if (valid && (count == 3 || count == 6)) {
                    options.camera = std::array<float, 3>{parsed[0], parsed[1], parsed[2]};
                    if (count == 6) {
                        options.target = glm::vec3(parsed[3], parsed[4], parsed[5]);
                    }
                } else {
                    utility::log("capture: ignoring '--capture-camera {}' (expected yaw,pitch,distance[,target.x,target.y,target.z])", *value);
                }
                continue;
            }
            filtered.push_back(argv[i]);
        }
        if (options.frames > 0) {
            utility::log("capture mode: {} frames, then screenshot + quit", options.frames);
        }
        return options;
    }

} // namespace

int main(int argc, char** argv) {
    // --version: print the version (single source: project(VERSION) in CMakeLists.txt, injected
    // as VULKAN_RENDER_VERSION_*) and exit before any config / Vulkan init.
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--version") {
            std::print("vulkan_render {}.{}.{}\n", VULKAN_RENDER_VERSION_MAJOR, VULKAN_RENDER_VERSION_MINOR, VULKAN_RENDER_VERSION_PATCH);
            return 0;
        }
    }

    // 1-3. Resolve the startup config in one step (chores): merge the config file (config.toml
    // by default, --config <path> to override) with positional argv overrides (argv[1] = model,
    // argv[2] = grid side (numeric)), then locate the shaders/ dir and pick the model file.
    // Panics on any missing configured/located resource. The dev-tool capture flags are removed
    // from argv first (see parse_capture_options) so they cannot land in the positional slots.
    std::vector<char*> filtered_argv;
    capture_options const capture = parse_capture_options(argc, argv, filtered_argv);
    chores::startup_config const config = chores::analyse_config(static_cast<int>(filtered_argv.size()), filtered_argv.data());
    app_config::app_settings const& settings = config.settings;
    std::filesystem::path const& shaders_dir = config.shaders_dir;
    std::string const& model_path = config.model_path;

    // startup banner: version (single source: project(VERSION) in CMakeLists.txt)
    utility::log("vulkan_render {}.{}.{}", VULKAN_RENDER_VERSION_MAJOR, VULKAN_RENDER_VERSION_MINOR, VULKAN_RENDER_VERSION_PATCH);

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
    core_options.validation_layers = settings.render.validation_layers;
    vulkan::runtime runtime{core_options};
    runtime.clear_color = glm::vec3(settings.render.clear_color[0], settings.render.clear_color[1], settings.render.clear_color[2]);
    // render-stage toggles from config: skybox applies immediately (only affects recording);
    // shadow is applied after enable_shadows() below (it needs the shadow maps to exist)
    runtime.set_skybox_enabled(settings.render.skybox);
    // per-pass GPU timings (timestamp queries): on by default, reported in the log + overlay
    runtime.set_gpu_timings(settings.render.gpu_timings);
    // Shadow cascades: applied here (BEFORE the scene import) because the shadow map's layered image
    // is created when the first scene set binds it - see runtime::set_shadow_cascades.
    runtime.set_shadow_cascades(static_cast<uint32_t>(settings.render.shadow_cascades));
    runtime.set_shadow_cascade_blend(settings.render.shadow_cascade_blend);
    // shadow map edge length: same startup-only rule as the cascade count (the layered image and its
    // views are created when the first scene set binds them, so this must precede the scene import)
    runtime.set_shadow_map_size(static_cast<uint32_t>(settings.render.shadow_map_size));
    auto const runtime_ready = std::chrono::steady_clock::now();
    utility::log("vulkan runtime initialized: {:.1f} ms (async model load + env generation running in background)", std::chrono::duration<double, std::milli>(runtime_ready - startup_start).count());

    // 6. Pipelines up front (chores::setup_pipeline): the standard PBR pipeline (used by the
    //    imported scene) plus the skybox background and the directional shadow pass. The legacy
    //    triangle demo pipeline is no longer created - nothing draws it.
    chores::setup_pipeline(runtime, shaders_dir);

    // 7. Collect the async startup results
    auto scenes = load_future.get();
    if (!scenes) {
        utility::panic(std::source_location::current(), "failed to load model '{}': error code {}", model_path, static_cast<int>(scenes.error()));
    }
    std::vector<float> const env = env_future.get();
    auto const startup_done = std::chrono::steady_clock::now();
    utility::log("model loaded + environment cubemap (startup window incl. runtime init): {:.1f} ms", std::chrono::duration<double, std::milli>(startup_done - startup_start).count());

    // 8. Whole-model world AABB + loader diagnostics (gltf_loader, pure CPU over the retained
    //    scene data): logs the scene summary (contents, hierarchy, animations/skins/morphs/
    //    cameras/lights) and returns the world bounds that frame the orbit camera and center
    //    the scene before import. Panics when the model has no drawable primitives.
    gltf::scene_bounds const bounds = gltf::log_scene_diagnostics(*scenes);
    glm::vec3 const scene_center = bounds.min * 0.5f + bounds.max * 0.5f;
    float const scene_radius = glm::length(bounds.max - bounds.min) * 0.5f;

    // Sink the model so it sits near the world horizon (y = 0) and move the camera target with it:
    // the camera then orbits/looks at the model's position instead of the scene origin.
    glm::vec3 const scene_sink(0.0f, -scene_radius, 0.0f);
    runtime.camera.target = scene_sink;

    // 9. IBL stage 2 + material resolve run concurrently via their _async wrappers: the
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
    //
    //     The scene tree is CALLER-OWNED: main declares it (AFTER the runtime, so C++ reverse
    //     declaration order destroys it BEFORE the runtime — the leaves' GPU buffers release
    //     through the runtime's vma allocator while it is still alive) and binds it with
    //     set_scene() before any import.
    vulkan::scene_tree::scene scene;
    runtime.set_scene(scene);
    runtime.camera.distance = scene_radius * 2.75f;
    gltf::scene_node_iterator const node_first = scenes->nodes_begin();
    gltf::scene_node_iterator const node_last;
    gltf::drawable_iterator const scene_first(*scenes, materials);
    gltf::drawable_iterator const scene_last;
    glm::vec3 const scene_import_shift = -scene_center + scene_sink;
    vulkan::scene_import_result const imported = runtime.import_scene(node_first, node_last, scene_first, scene_last, scene_import_shift);
    utility::log("imported {} primitives ({} new materials)", imported.primitive_count, imported.material_count);
    runtime.log_scene_tree();

    // 12. Enable directional shadow mapping over the imported scene: the shadow frustum frames
    //      the sphere around where the primitives actually sit (they were translated by the import
    //      offset above, so their world-space center is scene_sink) with their original radius
    runtime.enable_shadows(scene_sink, scene_radius);
    // apply the config shadow toggle now that the shadow maps exist (turning it off clears them)
    if (!settings.render.shadow) {
        runtime.set_shadow_enabled(false);
    }

    // 13. Optional instancing stress (chores::add_instancing_grid): grid_side > 1 (config or
    //     argv) draws the first imported primitive as a grid_side x grid_side grid in ONE
    //     instanced draw call (the frame loop is untouched); no-op otherwise.
    chores::add_instancing_grid(runtime, settings.grid_side, scene_radius);

    // 13b. Clustered-light stress ([lighting] demo_lights): spawn N procedural punctual lights on a
    //      helix around the scene bounds, pushed every frame together with the overlay's slots. The
    //      clustered path's whole point is a light count the brute-force loop could not afford, and
    //      the overlay's four slots cannot show that - this is what makes the difference measurable
    //      (and what the [render] clustered_lights A/B is compared against).
    std::vector<vulkan::punctual_light> demo_lights;
    if (settings.lighting.demo_lights > 0) {
        int const total = std::min(settings.lighting.demo_lights, static_cast<int>(app_config::max_demo_lights));
        demo_lights.reserve(static_cast<std::size_t>(total));
        for (int i = 0; i < total; ++i) {
            float const t = static_cast<float>(i) / static_cast<float>(total);
            float const angle = t * 6.2831853f * 3.0f; // three turns around the scene
            float const radius = scene_radius * 0.85f;
            vulkan::punctual_light light = {};
            light.position = scene_sink + glm::vec3(std::cos(angle) * radius, scene_radius * (t - 0.5f), std::sin(angle) * radius);
            // hue cycle: a warm/cool strip of colors makes the per-cluster lists visible as color
            light.color = glm::vec3(0.5f + 0.5f * std::cos(angle), 0.5f + 0.5f * std::cos(angle + 2.094f), 0.5f + 0.5f * std::cos(angle + 4.188f));
            light.intensity = 12.0f;
            light.range = scene_radius * 0.55f; // finite range: what the cluster sphere test culls on
            demo_lights.push_back(light);
        }
        utility::log("demo lights: {} procedural punctual lights around the scene (clustered light stress)", total);
    }

    // 14. Main render loop: until the window closes or ESC is pressed.
    //     Every Vulkan frame step (fences, acquire, command buffers, render pass, submit, present)
    //     lives inside runtime::render_frame()
    runtime.log_feature_status(); // one line naming every optional feature that could not be created
    utility::log("rendering '{}' with PBR... left-drag to orbit, wheel to zoom, ESC to exit", model_path);

    // Dear ImGui debug overlay on by default ([gui] show)
    bool const use_gui = settings.gui.show;

    // FPS statistics (utility.frame_stats): a rolling one-second window of frame gaps.
    // tick() once per presented frame, on_skipped() on minimized/recreate iterations, and
    // the once-per-second report (log + the overlay's smoothed value) keys off window_rolled().
    utility::frame_stats frame_stats;

    // ---- keyframe animation playback + skinning + morph targets (vulkan.animation) ----
    // The controller owns playback (sampling + writing node locals), the skin rigs (per-frame
    // joint matrices) and the morph rigs (static deltas + per-frame weights) against the
    // runtime scene tree; initialize it before the first frame (it bakes morph deltas and the
    // identity skin block into every frame slot's buffers). A float mirror of the playback
    // clock feeds the gui time slider (slider_widget binds an external float).
    vulkan::animation::controller animation;
    // the controller drives the runtime through an injected surface (chores wires the scene,
    // per-slot buffers and task pool), so it never depends on vulkan::runtime itself
    animation.init(*scenes, chores::make_animation_backend(runtime), scene_import_shift);
    // Live gui widget state (chores::gui_bindings) is declared after the authored-camera
    // seeding below, right before chores::setup_gui() builds the overlay.

    // ---- authored (glTF) camera selection ----
    // A glTF camera is used as a VIEWPOINT SEED for the orbit camera: picking one places the
    // orbit (target = scene center, distance/yaw/pitch derived from the camera node's world),
    // and from there the mouse keeps working normally (drag orbits, wheel zooms). The gui
    // combo below switches among "orbit" and the scene's cameras; the runtime's external-camera
    // override is not used by the demo (it stays available for exact/animated authored cameras).
    // usable cameras: those whose owning node exists in the imported tree, in scenes.cameras order
    struct authored_camera {
        gltf::camera const* camera = nullptr;
        std::size_t source = 0; // owning loader node (asset node index)
    };
    std::vector<authored_camera> authored_cameras;
    for (gltf::camera const& cam : scenes->cameras) {
        // find a node referencing this camera that is present in the imported tree (the loader's
        // asset-level node table + the controller's tree membership test)
        for (auto const& [source, loader_node] : scenes->node_by_source) {
            if (loader_node->camera_index && *loader_node->camera_index == static_cast<std::size_t>(&cam - scenes->cameras.data()) && animation.has_runtime_node(source)) {
                authored_cameras.push_back(authored_camera{&cam, source});
                break;
            }
        }
    }
    // current selection: 0 = orbit, 1..N = authored_cameras[i - 1]; default = the first
    // authored camera when the scene has any (same initial view as before, but now movable)
    int current_camera = authored_cameras.empty() ? 0 : 1;
    // point the orbit camera at the authored pose: target = scene center (frame like the
    // default view), yaw/pitch/distance solved from the camera node's world transform
    auto const seed_orbit_from_camera = [&](int const index) {
        if (index <= 0 || index > static_cast<int>(authored_cameras.size())) {
            return; // "orbit": keep the current free orbit
        }
        authored_camera const& ac = authored_cameras[static_cast<std::size_t>(index - 1)];
        glm::mat4 camera_world = glm::mat4(1.0f);
        bool found = false;
        auto const find_world = [&](auto&& self, vulkan::scene_tree::scene_node& node, glm::mat4 const& parent_world) -> bool {
            glm::mat4 const world = parent_world * node.local;
            if (node.source_index == ac.source) {
                camera_world = world;
                return true;
            }
            for (vulkan::scene_tree::scene_node& child : node.children) {
                if (self(self, child, world)) {
                    return true;
                }
            }
            return false;
        };
        for (vulkan::scene_tree::scene_node& root : runtime.get_scene().roots) {
            if (find_world(find_world, root, glm::mat4(1.0f))) {
                found = true;
                break;
            }
        }
        if (!found) {
            return;
        }
        glm::vec3 const eye = glm::vec3(camera_world[3]);
        glm::vec3 const center = scene_center;
        glm::vec3 const dir = center - eye;
        float const dist = glm::length(dir);
        if (dist < 1e-4f) {
            return;
        }
        glm::vec3 const d = dir / dist;
        runtime.camera.target = center;
        runtime.camera.distance = dist;
        runtime.camera.yaw = std::atan2(d.x, d.z);
        runtime.camera.pitch = std::asin(std::clamp(d.y, -1.0f, 1.0f));
        std::string_view const cam_name = ac.camera->name.empty() ? std::string_view("<unnamed>") : std::string_view(ac.camera->name);
        utility::log("camera: starting pose from glTF camera '{}' ({}) - you can still orbit/zoom", cam_name,
                     ac.camera->type == gltf::camera_type::perspective ? "perspective" : "orthographic");
    };
    if (!authored_cameras.empty()) {
        seed_orbit_from_camera(current_camera);
    }

    // ---- authored (glTF) punctual lights: KHR_lights_punctual lights up automatically ----
    // They load straight into the editable gui light slots below (see the gui_bindings setup),
    // so imported lights are adjustable in the overlay like the demo ones.

    // Optional Dear ImGui debug overlay: chores::setup_gui enables it on the runtime (when
    // use_gui) and assembles the whole panel - fps label, frustum-culling / skybox / shadow
    // toggles, the camera-target drag, animation playback controls, the camera selector and
    // the shadow-bias sliders. The widgets bind to the live gui_bindings below (checkbox and
    // slider mirrors + the animation mirrors, which the frame loop keeps in sync each frame);
    // authored-camera names and the orbit-seeding callback are passed in, so chores never
    // touches glTF types.
    chores::gui_bindings gui;
    gui.skybox_enabled = settings.render.skybox; // checkbox initial states mirror the config
    gui.shadow_enabled = settings.render.shadow;
    gui.fxaa_enabled = settings.render.fxaa;
    gui.gbuffer_debug = settings.render.gbuffer_debug; // gbuffer debug view initial state (M1)
    gui.gbuffer_channel = settings.render.gbuffer_channel;
    gui.deferred_enabled = settings.render.deferred;                 // deferred lighting render mode (M2)
    gui.render_mode = settings.render.unlit ? 1 : 0;                 // render-mode combo (0 = pbr, 1 = unlit)
    gui.taa_enabled = settings.render.taa;                           // temporal anti-aliasing (M3)
    gui.shadow_cascades = settings.render.shadow_cascades - 1;       // cascade combo index (0 = single map)
    gui.shadow_cascade_blend = settings.render.shadow_cascade_blend; // cascaded shadow maps (M4)
    gui.clustered_lights = settings.render.clustered_lights;         // clustered light culling (M5)
    gui.ssao_enabled = settings.render.ssao;                         // screen-space AO (M6)
    gui.ssao_radius = settings.render.ssao_radius;
    gui.ssao_intensity = settings.render.ssao_intensity;
    gui.ssao_samples = static_cast<float>(settings.render.ssao_samples);
    gui.anim_playing = animation.is_playing(); // play checkbox initial state
    gui.current_camera = current_camera;       // combo selection (the pose seeded above)

    // ---- authored (glTF) punctual lights -> the editable gui light slots ----
    // KHR_lights_punctual lights load straight into the gui slots (up to
    // vulkan::max_punctual_lights): main pushes the enabled set through
    // chores::apply_point_lights() every frame, so imported lights are adjustable in the overlay
    // like the demo ones. Position comes from the owning node's loader-space world matrix,
    // shifted by the same import offset the geometry got. KHR directional lights are NOT mapped:
    // the engine sun is the shadow-casting analytic light configured by enable_shadows() above -
    // the log keeps that miss visible instead of silent. The lights are fixed to the base pose
    // (an animated light node would need per-frame resolution - not wired yet).
    std::size_t imported_lights = 0;
    std::size_t imported_directional = 0;
    std::size_t imported_truncated = 0;
    for (auto const& [source, loader_node] : scenes->node_by_source) {
        if (!loader_node->light_index.has_value() || !animation.has_runtime_node(source)) {
            continue; // no light, or the node is absent from the imported tree
        }
        gltf::light const& src = scenes->lights[*loader_node->light_index];
        if (src.type == gltf::light_type::directional) {
            ++imported_directional;
            continue;
        }
        if (imported_lights >= vulkan::max_punctual_lights) {
            ++imported_truncated;
            continue;
        }
        chores::gui_bindings::light_slot& slot = gui.point_lights[imported_lights++];
        glm::vec3 const position = glm::vec3(loader_node->transform_matrix[3]) + scene_import_shift;
        slot.enabled = true;
        slot.position[0] = position.x;
        slot.position[1] = position.y;
        slot.position[2] = position.z;
        slot.color[0] = src.color.x;
        slot.color[1] = src.color.y;
        slot.color[2] = src.color.z;
        slot.intensity = src.intensity;
        slot.range = src.range.value_or(0.0f); // 0 = infinite falloff (UBO semantics)
        if (src.type == gltf::light_type::spot) {
            slot.spot = true;
            glm::vec3 const dir = glm::mat3(loader_node->transform_matrix) * glm::vec3(0.0f, 0.0f, -1.0f); // glTF spot axis
            glm::vec3 const axis = glm::dot(dir, dir) > 1e-8f ? glm::normalize(dir) : glm::vec3(0.0f, -1.0f, 0.0f);
            slot.direction[0] = axis.x;
            slot.direction[1] = axis.y;
            slot.direction[2] = axis.z;
            float const outer = src.spot_outer_cone.value_or(glm::radians(45.0f)); // KHR default cone
            slot.outer_cone_deg = glm::degrees(outer);
            // inner cone: the KHR angle when authored, else the legacy soft-inner derived in
            // cosine space (mix(outerCos, 1, 0.6)) converted back to degrees for the slider
            float const inner_cos = src.spot_inner_cone.has_value()
                                        ? std::cos(*src.spot_inner_cone)
                                        : 0.6f + 0.4f * std::cos(outer);
            slot.inner_cone_deg = glm::degrees(std::acos(std::clamp(inner_cos, -1.0f, 1.0f)));
        }
    }
    if (imported_lights > 0) {
        utility::log("KHR_lights_punctual: {} point/spot light(s) loaded into the editable gui light slots (base pose; adjustable in the overlay)",
                     imported_lights);
    }
    if (imported_directional > 0) {
        utility::log("KHR_lights_punctual: {} directional light(s) ignored - the engine sun is enable_shadows()'s analytic light",
                     imported_directional);
    }
    if (imported_truncated > 0) {
        utility::log("KHR_lights_punctual: {} additional light(s) dropped (GPU punctual-light cap = {})", imported_truncated, vulkan::max_punctual_lights);
    }
    std::vector<std::string> gui_camera_names; // selector items: authored names (orbit added inside)
    gui_camera_names.reserve(authored_cameras.size());
    for (authored_camera const& ac : authored_cameras) {
        gui_camera_names.push_back(ac.camera->name.empty() ? "<unnamed>" : std::string(ac.camera->name));
    }
    chores::setup_gui(runtime, use_gui, settings, gui, animation, gui_camera_names, seed_orbit_from_camera);

    // Per-frame cheap clock: stamp() once per presented frame on this (the frame owner) thread,
    // so any other thread can read the current frame time as a plain atomic load. Animation /
    // future parallel workers should prefer frame_clock.last_ns()/delta_ns() over now().
    utility::frame_clock frame_clock;

    // All per-frame decisions (event polling, ESC/close response, minimize skip, swapchain
    // recreation on restore/resize) live inside the runtime's frame phases, which main calls at
    // fine granularity so it can write per-frame data (scene node locals -> culling, skin
    // matrices, morph weights) between pacing and recording.
    int last_render_mode = 0; // gui render-mode combo (0 = pbr); applied between frames below
    // scripted capture: apply the camera override LAST, so nothing in the setup above (the orbit
    // framing of the imported scene, an authored glTF camera) can win over the requested view
    if (capture.camera) {
        runtime.camera.yaw = glm::radians((*capture.camera)[0]);
        runtime.camera.pitch = glm::radians((*capture.camera)[1]);
        runtime.camera.distance = (*capture.camera)[2];
        if (capture.target) {
            runtime.camera.target = *capture.target;
        }
        utility::log("capture camera: yaw {:.1f} deg, pitch {:.1f} deg, distance {:.2f}, target ({:.2f}, {:.2f}, {:.2f})",
                     (*capture.camera)[0], (*capture.camera)[1], (*capture.camera)[2],
                     runtime.camera.target.x, runtime.camera.target.y, runtime.camera.target.z);
    }
    int captured_frames = 0; // presented frames so far (scripted capture; see --capture-frames)
    while (true) {
        // Phase 1: poll window events (ESC / native close -> closed, minimized -> skipped)
        vulkan::frame_status const polled = runtime.poll_events();
        if (polled == vulkan::frame_status::closed || vulkan::is_failure(polled)) {
            break;
        }
        if (polled == vulkan::frame_status::skipped) {
            // Minimized: skip this frame's CPU work too; refresh the fps baseline so the pause
            // is not counted as one huge rendered frame.
            frame_stats.on_skipped();
            std::this_thread::yield();
            continue;
        }
        runtime.recreate_if_minimized();

        // Phase 2: pace + acquire the next frame slot. After pace_and_acquire() returns
        // proceed, this slot's previous submission has completed, so the per-frame host writes
        // below (scene node locals -> culling, skin matrices, morph weights) cannot race an
        // in-flight frame.
        vulkan::frame_status const paced = runtime.pace_and_acquire();
        if (paced == vulkan::frame_status::closed || vulkan::is_failure(paced)) {
            break;
        }
        if (paced == vulkan::frame_status::skipped) {
            // zero-sized swapchain (not sized yet / restored minimized): no attachments to render
            // into - skip this frame's CPU work too, like the minimized case above
            frame_stats.on_skipped();
            std::this_thread::yield();
            continue;
        }
        if (paced == vulkan::frame_status::skipped) {
            // Swapchain recreated during acquire: retry next iteration
            frame_stats.on_skipped();
            std::this_thread::yield();
            continue;
        }

        // drive the animation controller: sample the active animation into node locals (T/R/S +
        // morph weights) and rebuild the skin matrices, into the frame slot pace_and_acquire()
        // just paced. dt comes from frame_clock (stamped after the previous presented frame);
        // clamp it so a pause (minimized / swapchain-recreate gaps that never stamped) does not
        // fast-forward the animation by the whole gap - playback resumes where it paused.
        float const dt = static_cast<float>(std::min(frame_clock.delta_seconds(), 0.25));
        animation.update(dt);
        gui.anim_time = animation.current_time();  // keep the gui time slider in sync
        gui.anim_playing = animation.is_playing(); // reflect controller-side pauses (scrub / select)
        gui.anim_index = static_cast<int>(animation.current());

        // Phase 3: record + submit + present the paced frame
        vulkan::frame_status const rec = runtime.begin_recording();
        if (rec == vulkan::frame_status::closed || vulkan::is_failure(rec)) {
            break;
        }
        runtime.record_main_drawcalls();
        vulkan::frame_status const ended = runtime.end_recording();
        if (ended == vulkan::frame_status::closed || vulkan::is_failure(ended)) {
            break;
        }
        vulkan::frame_status const result = runtime.submit_and_present();
        if (result == vulkan::frame_status::closed || vulkan::is_failure(result)) {
            break;
        }
        if (result == vulkan::frame_status::skipped) {
            // present reported the swapchain out of date / recreated it: retry next iteration
            frame_stats.on_skipped();
            std::this_thread::yield();
            continue;
        }

        // A frame was presented: publish its stamp for cheap readers (frame_clock)
        frame_clock.stamp();

        // scripted capture: once the requested number of frames has been PRESENTED, request the
        // screenshot from the runtime - the F12 block below consumes the request in this same
        // iteration, so the capture happens on a fully warmed-up frame
        if (capture.frames > 0 && ++captured_frames >= capture.frames) {
            runtime.request_screenshot();
        }

        // gui "render mode": switch the runtime's default pipeline BETWEEN frames (the pipeline
        // registry must not be mutated while a frame records; this point is after submit, before
        // the next frame's recording). Default-semantics leaves re-shade on the next frame.
        if (gui.render_mode != last_render_mode) {
            last_render_mode = gui.render_mode;
            std::string_view const mode_name = gui.render_mode == 0 ? "pbr" : "unlit";
            runtime.set_default_pipeline(mode_name);
            // the deferred path cannot switch pipelines per fragment, so tell its lighting stage that the
            // default pipeline is the flat one - both paths then mean the same thing by "unlit"
            runtime.set_unlit(gui.render_mode == 1);
            utility::log("render mode: {} ({})", mode_name, gui.render_mode == 0 ? "lit" : "unlit / flat");
        }

        // fps statistics: accumulate the frame gap into the rolling window
        frame_stats.tick();
        // punctual lights: push the gui slot set every frame, INDEPENDENT of the overlay being
        // visible - imported model lights were loaded into those slots, so they must stay lit in
        // headless-overlay runs too (the demo slots stay off unless the user enabled them)
        chores::apply_point_lights(runtime, gui, demo_lights);
        runtime.set_exposure(gui.exposure);                                                     // gui exposure slider -> linear scale (post-process pass)
        runtime.set_bloom(gui.bloom_enabled ? gui.bloom_intensity : 0.0f, gui.bloom_threshold); // bloom checkbox + knobs -> post pass
        // FXAA: mirrored every frame like the other post-process values (the runtime clamps them and
        // ignores the flag when no fxaa pipeline was created)
        runtime.set_fxaa(gui.fxaa_enabled, gui.fxaa_subpixel, gui.fxaa_edge_threshold);
        // G-buffer debug view (the deferred path's data): mirrored every frame like the FXAA state,
        // so the config, the overlay checkbox and the channel combo all take effect immediately
        runtime.set_gbuffer_debug(gui.gbuffer_debug);
        runtime.set_gbuffer_channel(gui.gbuffer_channel);
        // deferred lighting (the deferred path's render mode): same mirror rule. It needs the same
        // pipelines as the debug view, so enabling it without them leaves the forward path running.
        runtime.set_deferred(gui.deferred_enabled);
        // TAA (the deferred path's answer to MSAA): mirrored like the other render toggles. The
        // jitter follows automatically - it is applied to the projection when TAA is active.
        runtime.set_taa(gui.taa_enabled, gui.taa_blend_static, gui.taa_blend_min);

        // Order matters for the M5/M6 mirrors: their availability checks read the state the lines
        // above just set (SSAO and TAA only apply to the deferred path, clustered lighting only when
        // the cluster pipeline exists). Mirroring them before set_deferred() would make the first
        // frame of every run report 'SSAO does nothing' from a stale deferred flag.
        runtime.set_clustered_lights(gui.clustered_lights);
        runtime.set_ssao(gui.ssao_enabled, gui.ssao_radius, gui.ssao_intensity, static_cast<uint32_t>(std::max(gui.ssao_samples, 0.0f) + 0.5f));
        // cel shading: the combo picks a discrete band count (index 0 = off); every entry is a
        // visibly different look, unlike a continuous strength that had dead zones between bands
        constexpr std::array<float, 7> toon_band_counts = {0.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 8.0f};
        auto const toon_index = static_cast<std::size_t>(std::clamp(gui.toon_bands_index, 0, static_cast<int>(toon_band_counts.size()) - 1));
        runtime.set_toon_shading(toon_band_counts[toon_index], gui.toon_softness);
        // F12 screenshot: the runtime reports the request (edge-triggered in poll_events), main
        // captures the presented swapchain image and writes it as a PNG (dependency-free encoder)
        if (runtime.consume_screenshot_request()) {
            auto const image = runtime.acquire_current_frame_image();
            if (!image) {
                utility::log("screenshot failed: {}", image.error());
            } else {
                // base directory from [paths] screenshot_dir (empty = the current working
                // directory); created on demand so a fresh checkout can capture immediately
                std::filesystem::path directory(settings.paths.screenshot_dir);
                if (!directory.empty()) {
                    std::error_code ec;
                    std::filesystem::create_directories(directory, ec);
                    if (ec) {
                        utility::log("screenshot: cannot create '{}' - saving to the working directory", directory.string());
                        directory.clear();
                    }
                }
                std::filesystem::path const path = directory / std::format("screenshot_{:%Y%m%d_%H%M%S}.png", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
                auto const written = utility::write_png(path, image->width, image->height, image->rgba);
                if (written) {
                    utility::log("screenshot saved: {} ({}x{})", path.string(), image->width, image->height);
                } else {
                    utility::log("screenshot save failed: {}", written.error());
                }
            }
            // scripted capture: the frame is captured (saved or not - do not spin forever on a
            // failing writer), so leave the render loop and shut down cleanly
            if (capture.frames > 0) {
                break;
            }
        }
        // overlay fps mirror: updated unconditionally - the overlay can be hidden with F1 and
        // shown again at runtime, so its data must stay fresh even while it is not drawn
        gui.fps = frame_stats.smoothed_fps();
        if (frame_stats.window_rolled()) {
            // once per second: the fps log line stays for headless / non-gui runs; the overlay
            // shows the same number via smoothed_fps()
            utility::log("fps: {:.1f} ({:.2f} ms/frame)", frame_stats.window_fps(), frame_stats.window_frame_ms());
            if (animation.has_active()) {
                // report the playback clock + the first animated node's evaluated translation
                // (proves the keyframes are actually moving the tree)
                std::string_view const node_name = animation.get_debug_node_name().empty()
                                                       ? std::string_view("<no target in scene>")
                                                       : animation.get_debug_node_name();
                utility::log("  anim '{}': t={:.3f}s/{:.2f}s, '{}' at ({:.3f}, {:.3f}, {:.3f})",
                             animation.active_name(), animation.current_time(), animation.loop_duration(), node_name,
                             animation.get_debug_translation().x, animation.get_debug_translation().y, animation.get_debug_translation().z);
            }
            if (animation.is_skin_debug_valid()) {
                utility::log("  skin '{}': last joint world x-axis ({:.3f}, {:.3f}, {:.3f})", animation.get_skin_debug_name(),
                             animation.get_skin_debug_translation().x, animation.get_skin_debug_translation().y, animation.get_skin_debug_translation().z);
            }
        }
    }

    // 15. Wait for the GPU to finish; primitives and pipelines are released by the runtime destructor
    runtime->wait_idle();
    utility::log("render loop finished");
    return 0;
}
