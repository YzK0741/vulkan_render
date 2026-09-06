#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
import std;
import app_config;
import chores; // demo bootstrap helpers (shader loading / dir locating / pipelines)
import gltf_loader;
import utility;
import utility.frame_clock; // per-frame stamp: cheap time reads for (future) parallel workers / animation
import vulkan.animation;    // animation_controller: glTF playback / skinning / morphs on the runtime tree
import vulkan.math;
import vulkan.runtime.scene_tree; // scene storage + GPU primitives (was vulkan.model)
import vulkan.runtime;

// Route std::pmr allocations through mimalloc (utility.better_pmr) before main(): this
// file-scope reference's dynamic initialization runs at startup, so every runtime/scene
// object built below already allocates its std::pmr vectors from mimalloc. Idempotent —
// other TUs (vulkan/runtime.cpp) keep their own copy of the same singleton.
[[maybe_unused]] static auto& pmr = utility::init_pmr(); // NOLINT(keep-alive)

int main(int argc, char** argv) {
    // 1-3. Resolve the startup config in one step (chores): merge the config file (config.toml
    // by default, --config <path> to override) with positional argv overrides (argv[1] = model,
    // argv[2] = grid side (numeric) or demo, argv[3] = demo), then locate the shaders/ dir and
    // pick the model file. Panics on any missing configured/located resource.
    chores::startup_config const config = chores::analyse_config(argc, argv);
    app_config::app_settings const& settings = config.settings;
    std::filesystem::path const& shaders_dir = config.shaders_dir;
    std::string const& model_path = config.model_path;

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
    //   "gui"           — force-enable the Dear ImGui debug overlay (also the default: the
    //                     overlay shows unless config sets [gui] show = false)
    bool spin_scene = false;
    bool spin_subtree = false;
    bool no_cull = false;
    bool closeup = false;
    bool use_gui = settings.gui.show; // overlay defaults on ([gui] show); demo "gui" forces it
    if (!settings.demo.empty()) {
        std::string_view const demo_view(settings.demo);
        spin_scene = demo_view == "spin";
        spin_subtree = demo_view == "spin-subtree";
        no_cull = demo_view == "nocull";
        closeup = demo_view == "closeup";
        if (demo_view == "gui") {
            use_gui = true;
        }
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
        if (subtree_node == nullptr) { // NOLINT(bugprone-branch-clone): CLion FP - the branches log different messages
            utility::log("spin-subtree: scene has no primitive leaf node to rotate");
        } else {
            utility::log("spin-subtree: rotating node '{}' about its own position", subtree_node->name);
        }
    }

    // FPS statistics: accumulate frame times, report once per second (log + window title)
    std::chrono::steady_clock::time_point last_frame_time = std::chrono::steady_clock::now();
    double fps_elapsed = 0.0;
    uint32_t fps_frame_count = 0;

    // ---- keyframe animation playback + skinning + morph targets (vulkan.animation) ----
    // The controller owns playback (sampling + writing node locals), the skin rigs (per-frame
    // joint matrices) and the morph rigs (static deltas + per-frame weights) against the
    // runtime scene tree; initialize it before the first frame (it bakes morph deltas and the
    // identity skin block into every frame slot's buffers). A float mirror of the playback
    // clock feeds the gui time slider (slider_widget binds an external float).
    vulkan::animation_controller animation;
    animation.init(*scenes, runtime, scene_import_shift);
    float gui_anim_time = 0.0f; // float mirror of the playback clock (time-slider target)
    bool gui_anim_playing = animation.is_playing();
    int gui_anim_index = 0; // selected item of the animation combo (0 = the auto-played one)

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
        // find a node referencing this camera that is present in the imported tree
        for (auto const& [source, loader_node] : animation.get_loader_nodes()) {
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

    // Optional Dear ImGui debug overlay: the runtime drives new_frame/record inside its frame
    // steps; main only enables it and manages its content through the panel/widget API (fps
    // text widget bound to a live lambda + a frustum-culling checkbox that forwards to the
    // runtime). The checkbox is a slider-free toggle bound to an external bool.
    double gui_fps = 0.0;
    bool gui_cull_enabled = true;
    bool gui_skybox_enabled = settings.render.skybox;
    bool gui_shadow_enabled = settings.render.shadow;
    // live shadow depth-bias mirrors: sliders write both the mirror (drag feedback) and the
    // runtime's per-frame vkCmdSetDepthBias values
    float gui_shadow_bias_constant = 0.0f;
    float gui_shadow_bias_slope = 1.5f;
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
        // shadow depth bias (dynamic state, applied every frame): slope factor removes acne on
        // angled surfaces, the constant adds a fixed push - tune per model when shadows show
        // acne (raise slope) or peter-panning (lower / raise constant)
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>(
            "shadow bias slope",
            &gui_shadow_bias_slope,
            0.0f,
            10.0f,
            [&runtime, &gui_shadow_bias_constant, &gui_shadow_bias_slope](float const value) {
                gui_shadow_bias_slope = value;
                runtime.set_shadow_depth_bias(gui_shadow_bias_constant, gui_shadow_bias_slope, 0.0f);
            }));
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>(
            "shadow bias constant",
            &gui_shadow_bias_constant,
            0.0f,
            10.0f,
            [&runtime, &gui_shadow_bias_constant, &gui_shadow_bias_slope](float const value) {
                gui_shadow_bias_constant = value;
                runtime.set_shadow_depth_bias(gui_shadow_bias_constant, gui_shadow_bias_slope, 0.0f);
            }));
        // camera orbit target: dragging it moves what the camera looks at / orbits around
        // (camera.target is a glm::vec3, i.e. three contiguous floats; the runtime rebuilds the
        // camera UBO from it every frame, so no on_change callback is needed)
        panel.push_back(std::make_unique<vulkan::gui::vec3_widget>("camera target", &runtime.camera.target.x, 0.05f));
        // playback controls (only when the model carries animations): play/pause toggle bound
        // to the playback state, a time scrubber (pauses on drag so the clock cannot fight the
        // scrub; the play checkbox resumes), and — for multi-animation assets — a dropdown to
        // pick which animation plays. All playback state lives in the animation_controller.
        if (animation.has_active()) {
            panel.push_back(std::make_unique<vulkan::gui::label_widget>([&animation] {
                return std::format("animation '{}' ({}s)", animation.active_name(), animation.loop_duration());
            }));
            panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>(
                "play",
                &gui_anim_playing,
                [&animation](bool const enabled) { animation.set_playing(enabled); }));
            panel.push_back(std::make_unique<vulkan::gui::slider_widget>(
                "time",
                &gui_anim_time,
                0.0f,
                animation.playable_max_duration(),
                [&animation](float const value) {
                    animation.set_time(value); // scrubbing pauses so the clock does not fight the drag
                }));
            if (animation.playable_count() > 1) {
                std::vector<std::string> names;
                names.reserve(animation.playable_count());
                for (std::size_t i = 0; i < animation.playable_count(); ++i) {
                    names.push_back(std::string(animation.playable_name(i)));
                }
                panel.push_back(std::make_unique<vulkan::gui::combo_widget>(
                    "animation",
                    std::move(names),
                    &gui_anim_index,
                    [&animation](int const index) { animation.select(static_cast<std::size_t>(index)); }));
            }
            utility::log("gui: playback controls added ({} animation(s))", animation.playable_count());
        }
        // camera selector: "orbit" (free) or any scene camera (its pose seeds the orbit camera,
        // so the mouse keeps working after switching)
        if (!authored_cameras.empty()) {
            std::vector<std::string> camera_names;
            camera_names.reserve(authored_cameras.size() + 1);
            camera_names.push_back("orbit");
            for (authored_camera const& ac : authored_cameras) {
                camera_names.push_back(ac.camera->name.empty() ? "<unnamed>" : ac.camera->name);
            }
            panel.push_back(std::make_unique<vulkan::gui::combo_widget>(
                "camera",
                std::move(camera_names),
                &current_camera,
                [&seed_orbit_from_camera](int const index) { seed_orbit_from_camera(index); }));
            utility::log("gui: camera selector added ({} camera(s))", authored_cameras.size());
        }
        utility::log("gui: Dear ImGui debug overlay enabled");
    }

    // Per-frame cheap clock: stamp() once per presented frame on this (the frame owner) thread,
    // so any other thread can read the current frame time as a plain atomic load. Animation /
    // future parallel workers should prefer frame_clock.last_ns()/delta_ns() over now().
    utility::frame_clock frame_clock;

    // All per-frame decisions (event polling, ESC/close response, minimize skip, swapchain
    // recreation on restore/resize) live inside the runtime's frame phases, which main calls at
    // fine granularity so it can write per-frame data (scene node locals -> culling, skin
    // matrices, morph weights) between pacing and recording.
    while (true) {
        // Phase 1: poll window events (ESC / native close -> closed, minimized -> skipped)
        vulkan::frame_status const polled = runtime.poll_events();
        if (polled == vulkan::frame_status::closed || vulkan::is_failure(polled)) {
            break;
        }
        if (polled == vulkan::frame_status::skipped) {
            // Minimized: skip this frame's CPU work too; keep the FPS timer fresh so the pause
            // is not counted as one huge rendered frame.
            last_frame_time = std::chrono::steady_clock::now();
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
            // Swapchain recreated during acquire: retry next iteration
            last_frame_time = std::chrono::steady_clock::now();
            std::this_thread::yield();
            continue;
        }
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

        // drive the animation controller: sample the active animation into node locals (T/R/S +
        // morph weights) and rebuild the skin matrices, into the frame slot pace_and_acquire()
        // just paced. dt comes from frame_clock (stamped after the previous presented frame).
        animation.update(static_cast<float>(frame_clock.delta_seconds()));
        gui_anim_time = animation.current_time();  // keep the gui time slider in sync
        gui_anim_playing = animation.is_playing(); // reflect controller-side pauses (scrub / select)
        gui_anim_index = static_cast<int>(animation.current());

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
            last_frame_time = std::chrono::steady_clock::now();
            std::this_thread::yield();
            continue;
        }

        // A frame was presented: publish its stamp for cheap readers (frame_clock)
        frame_clock.stamp();

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
            fps_elapsed = 0.0;
            fps_frame_count = 0;
        }
    }

    // 15. Wait for the GPU to finish; primitives and pipelines are released by the runtime destructor
    runtime->wait_idle();
    utility::log("render loop finished");
    return 0;
}
