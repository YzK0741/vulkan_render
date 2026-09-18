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
import vulkan.render_start_demo; // the example's pass wiring: this app's chain, from outside the renderer

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
        // Degrees of YAW added per presented frame (`--capture-sweep`). 0 - the default - is a fixed
        // camera, which is what every other capture in this repository is.
        //
        // WHY IT EXISTS: with the camera still, every reprojection path in the renderer is exercised only
        // in its trivial case - a motion vector of zero, a history fetched from the pixel it came from.
        // TAA's resolve, the GI temporal accumulation, the reflection's history and the velocity target
        // itself are all "correct" under a static camera no matter how they are written, so a break in
        // any of them passed the capture harness. This makes a capture move the camera by a fixed amount
        // per FRAME INDEX (not per wall-clock second), so a sweep capture is as reproducible as a still
        // one - the gate compares two runs of it like any other scenario.
        float sweep_yaw_deg_per_frame = 0.0f;
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
            if (std::optional<std::string_view> const value = take_value(i, arg, "--capture-sweep")) {
                // degrees of yaw per presented frame - see capture_options::sweep_yaw_deg_per_frame
                if (std::optional<float> const number = parse_number(*value)) {
                    options.sweep_yaw_deg_per_frame = *number;
                } else {
                    utility::log("capture: ignoring '--capture-sweep {}' (expected degrees of yaw per frame)", *value);
                }
                continue;
            }
            filtered.push_back(argv[i]);
        }
        if (options.frames > 0) {
            utility::log("capture mode: {} frames, then screenshot + quit", options.frames);
            if (options.sweep_yaw_deg_per_frame != 0.0f) {
                utility::log("capture camera sweep: {:.3f} deg of yaw per frame, from whatever pose the scene settled on", options.sweep_yaw_deg_per_frame);
            }
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
    //    vsync; the defaults in render_settings mirror the historic hardcoded values)
    vulkan::core_create_info core_options = {};
    core_options.window_width = settings.render.window_width;
    core_options.window_height = settings.render.window_height;
    core_options.window_title = settings.render.window_title;
    core_options.vsync = settings.render.vsync;
    core_options.validation_layers = settings.render.validation_layers;
    vulkan::runtime runtime{core_options};
    runtime.clear_color = glm::vec3(settings.render.clear_color[0], settings.render.clear_color[1], settings.render.clear_color[2]);
    // shadow is applied after enable_shadows() below (it needs the shadow maps to exist)
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
    //    imported scene) and the directional shadow pass. The legacy
    //    triangle demo pipeline is no longer created - nothing draws it.
    chores::setup_pipeline(runtime, shaders_dir);
    // THE EXAMPLE'S OWN WIRING: this application's passes are fed by `vulkan.render_start_demo`, which finds them
    // in the chain the runtime owns and answers the frame's per-stage questions (see the module's header). The
    // runtime holds none of these references itself any more, which is what lets a second application hand it a
    // different chain - and the demo object outlives the frame loop because it lives here, in the app's own scope.
    vulkan::render_start_demo start_demo;
    static_cast<void>(start_demo.attach(runtime)); // builds this app's chain and hands it over
    // ... and the CREATE step runs over that chain (the shaders above are registered by now): every pass builds what
    // it owns, and the renderer's two jobs - which are not passes - are created with them.
    runtime.create_passes();
    // Stochastic punctual lighting (docs/megalights.md): the switch and the sample count, with the two bias
    // terms left at the pass's own defaults (they are self-intersection guards rather than look knobs, and the
    // pass clamps them). OFF by default, so a stock config is the unshadowed path it always was.
    // The two bias radii are UE's pair scaled by the config's dial (see app_config's note): the floor at
    // normal incidence and the larger offset at grazing incidence, where the ray leaves nearly parallel to the
    // surface and a small offset would let it re-hit the surface it started from.
    float const ml_bias_floor = 0.01f * settings.render.megalights_bias;
    float const ml_bias_grazing = 0.1f * settings.render.megalights_bias;
    start_demo.set_megalights(settings.render.megalights, static_cast<uint32_t>(settings.render.megalights_samples), 0.001f, ml_bias_floor, ml_bias_grazing);
    // ... and the chain's policy: UE's relative depth tolerance (0.03) and frame-count cap (12) for the temporal
    // running mean, plus this chain's own spatial pre-filter width, which is the config's because it is the dial
    // between grain and detail (see app_config's note and docs/megalights.md's measurement).
    start_demo.set_megalights_accumulation(settings.render.megalights_history_tolerance, 12.0f, settings.render.megalights_spatial_sigma);
    // Same bargain as the ray-traced shadows: a request the runtime grants only on a device with ray
    // queries and a built top level structure - otherwise the GI rays keep marching the depth buffer.
    // The furnace verification mode: an analytic reference rather than another estimator of ours.
    runtime.set_furnace(settings.render.furnace);
    runtime.set_rt_shadows(settings.render.rt_shadows);
    // ... and the alphaMode MASK bake, which is what keeps a masked surface from being SOLID to those rays:
    // a compute pass collapses the triangles the material's alpha cuts out, before the structures are built.
    runtime.set_rt_mask_bake(settings.render.rt_mask_bake);
    // ... and the per-frame skinning pass, which is what keeps an ANIMATED caster's traced shadow where the
    // caster actually is: the structures are built from the bind pose, so without it the ray sees the mesh
    // at rest.
    runtime.set_rt_skin_bake(settings.render.rt_skin_bake);

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

    // 10. Upload the scene-wide IBL once: shared by every primitive (bindings 2-4 of the scene block)
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
    // Initial camera framing ([render] camera_fit). "exterior" is the historic fit: 2.75 scene radii
    // frames a compact object, and for that it stays the default. It is also useless for a building -
    // the camera ends up well outside its own walls, so the frame is a facade with no interior to
    // bounce light in, which is why GI work needs the other mode. "interior" stands inside and looks
    // along the LONGEST HORIZONTAL AXIS, because that is the axis a hall, nave or corridor runs down
    // (for Sponza: 29.8 x 18.3 in the ground plane, so the look direction is +X). The distance is a
    // fraction of the SMALLER horizontal half-extent, which is what keeps the eye inside even a
    // narrow hall while still leaving enough parallax for the view to read as a space.
    {
        glm::vec3 const half_extent = (bounds.max - bounds.min) * 0.5f;
        bool const interior = settings.render.camera_fit == "interior";
        if (interior) {
            runtime.camera.yaw = half_extent.x >= half_extent.z ? glm::radians(90.0f) : 0.0f;
            runtime.camera.pitch = 0.0f;
            runtime.camera.distance = std::min(half_extent.x, half_extent.z) * 0.7f;
        } else {
            runtime.camera.distance = scene_radius * 2.75f;
        }
        utility::log("initial camera: fit={} yaw {:.1f} deg, pitch {:.1f} deg, distance {:.2f} (scene radius {:.2f}, ground half-extent {:.2f} x {:.2f})",
                     settings.render.camera_fit,
                     glm::degrees(runtime.camera.yaw),
                     glm::degrees(runtime.camera.pitch),
                     runtime.camera.distance,
                     scene_radius,
                     half_extent.x,
                     half_extent.z);
    }
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
            float const radius = scene_radius * settings.lighting.demo_light_radius;
            vulkan::punctual_light light = {};
            light.position = scene_sink + glm::vec3(std::cos(angle) * radius, scene_radius * (t - 0.5f), std::sin(angle) * radius);
            // hue cycle: a warm/cool strip of colors makes the per-cluster lists visible as color
            light.color = glm::vec3(0.5f + 0.5f * std::cos(angle), 0.5f + 0.5f * std::cos(angle + 2.094f), 0.5f + 0.5f * std::cos(angle + 4.188f));
            light.intensity = 12.0f;
            light.range = scene_radius * settings.lighting.demo_light_range; // finite range: what the cluster sphere test culls on
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
    // [render] animation_time >= 0 PINS the pose: playback is wall-clock driven, so two captures of an
    // animated scene differ unless the time is fixed - and this is also what makes such a scene usable in
    // a measurement or a regression scenario at all. scrub() is the overlay's time slider, so the pose is a
    // function of the value alone; a negative value (the default) plays as always.
    if (settings.render.animation_time >= 0.0f) {
        animation.set_time(settings.render.animation_time);
        utility::log("animation: pinned at {:.2f}s by [render] animation_time (playback is wall-clock driven, so captures of an animated scene are only reproducible this way)", settings.render.animation_time);
    }
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
    // use_gui) and assembles the whole panel - fps label, frustum-culling / shadow
    // toggles, the camera-target drag, animation playback controls, the camera selector and
    // the shadow-bias sliders. The widgets bind to the live gui_bindings below (checkbox and
    // slider mirrors + the animation mirrors, which the frame loop keeps in sync each frame);
    // authored-camera names and the orbit-seeding callback are passed in, so chores never
    // touches glTF types.
    chores::gui_bindings gui;
    gui.shadow_enabled = settings.render.shadow; // checkbox initial states mirror the config
    gui.fxaa_enabled = settings.render.fxaa;
    gui.gbuffer_debug = settings.render.gbuffer_debug; // gbuffer debug view initial state (M1)
    gui.gbuffer_channel = settings.render.gbuffer_channel;
    gui.render_mode = settings.render.unlit ? 1 : 0; // render-mode combo (0 = pbr, 1 = unlit)
    gui.taa_enabled = settings.render.taa;           // temporal anti-aliasing (M3)
    // ... and its TWO BLEND WEIGHTS, which the frame loop mirrors into the runtime every frame
    // (start_demo.set_taa below). Without these two lines the config's values never reached the
    // renderer: chores' gui_bindings defaults (0.9 / 0.5) are what set_taa received on every frame,
    // so editing `[render] taa_blend_static` in the file changed NOTHING - measured, the flicker at a
    // pinned close-up was byte-identical at 0.90, 0.95 and 0.98 (38.22% of pixels changing per frame
    // in all three). `taa_enabled` alone happened to look wired because it IS copied here.
    gui.taa_blend_static = settings.render.taa_blend_static;
    gui.taa_blend_min = settings.render.taa_blend_min;
    gui.megalights_enabled = settings.render.megalights;
    gui.megalights_samples = static_cast<float>(settings.render.megalights_samples);
    gui.megalights_spatial_sigma = settings.render.megalights_spatial_sigma;
    gui.megalights_history_tolerance = settings.render.megalights_history_tolerance;
    gui.megalights_bias = settings.render.megalights_bias;
    gui.megalights_light_angle = settings.render.megalights_light_angle;
    start_demo.set_megalights_light_angle(settings.render.megalights_light_angle);
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
    // The sweep's BASE pose is whatever the camera ended up as - the scene's own `camera_fit`, an authored
    // glTF camera, or the pinned `--capture-camera` above - so a sweep composes with all three instead of
    // demanding a pinned pose it would have to be told twice.
    float const sweep_base_yaw = runtime.camera.yaw;
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

        // Scripted capture: the camera sweep, advanced by the number of PRESENTED frames - a frame index,
        // not a clock reading, so two runs of one sweep are byte-identical (the harness's determinism run
        // is what verifies it). It has to happen before pace_and_acquire(), which is the phase that writes
        // the camera UBO.
        if (capture.sweep_yaw_deg_per_frame != 0.0f) {
            runtime.camera.yaw = sweep_base_yaw + glm::radians(capture.sweep_yaw_deg_per_frame) * static_cast<float>(captured_frames);
        }

        // Phase 2: pace + acquire the next frame slot. After pace_and_acquire() returns
        // proceed, this slot's previous submission has completed, so the per-frame host writes
        // below (scene node locals -> culling, skin matrices, morph weights) cannot race an
        // in-flight frame.
        vulkan::frame_status const paced = runtime.pace_and_acquire();
        if (paced == vulkan::frame_status::closed || vulkan::is_failure(paced)) {
            break;
        }
        if (paced == vulkan::frame_status::skipped) {
            // The swapchain is not usable this iteration - zero-sized (not sized yet / restored
            // minimized) so there are no attachments to render into, or it was recreated during the
            // acquire. Either way: skip this frame's CPU work too, like the minimized case above.
            // (There used to be a SECOND identical check here with a "recreated during acquire"
            // comment: unreachable, since the first one already covered it - both return the same
            // status. The reasons differ, the handling does not.)
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
            // the lighting stage cannot switch pipelines per fragment, so tell it that the default
            // pipeline is the flat one - it then writes the stored albedo instead of shading, so
            // "unlit" means the same thing for the opaque scene and for the transparent pass
            start_demo.set_unlit(gui.render_mode == 1);
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
        runtime.set_max_fps(config.settings.render.max_fps);                                    // 0 = uncapped (see config.example.toml)
        // FXAA: mirrored every frame like the other post-process values (the runtime clamps them and
        // ignores the flag when no fxaa pipeline was created)
        runtime.set_fxaa(gui.fxaa_enabled, gui.fxaa_subpixel, gui.fxaa_edge_threshold);
        // G-buffer debug view (the G-buffer's stored data): mirrored every frame like the FXAA state,
        // so the config, the overlay checkbox and the channel combo all take effect immediately
        runtime.set_gbuffer_debug(gui.gbuffer_debug);
        start_demo.set_gbuffer_channel(gui.gbuffer_channel);
        // TAA (the engine's anti-aliasing): mirrored like the other render toggles. The jitter
        // follows automatically - it is applied to the projection when TAA is active.
        start_demo.set_taa(gui.taa_enabled, gui.taa_blend_static, gui.taa_blend_min);
        // Stochastic punctual lighting: the overlay's switch and sample count, mirrored like the GI's - the two
        // bias terms are the pass's constants and are passed through at their shipped values.
        start_demo.set_megalights(gui.megalights_enabled, static_cast<uint32_t>(std::max(gui.megalights_samples, 1.0f) + 0.5f), 0.001f, 0.01f * gui.megalights_bias,
                                  0.1f * gui.megalights_bias);
        start_demo.set_megalights_light_angle(gui.megalights_light_angle);
        // ... and the chain's policy, so the overlay's own slider moves the spatial pre-filter live (0 = the
        // temporal-only chain, which is also the A/B the measurement uses).
        start_demo.set_megalights_accumulation(gui.megalights_history_tolerance, gui.megalights_frames, gui.megalights_spatial_sigma);

        // Order matters for the M5/M6 mirrors: their availability checks read the state the lines
        // above just set (the debug view replaces the lighting stage, clustered lighting only exists
        // when the cluster pipeline does), so mirroring them earlier would report a stale answer for
        // the first frame of every run.
        runtime.set_clustered_lights(gui.clustered_lights);
        start_demo.set_ssao(gui.ssao_enabled, gui.ssao_radius, gui.ssao_intensity, static_cast<uint32_t>(std::max(gui.ssao_samples, 0.0f) + 0.5f));
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
