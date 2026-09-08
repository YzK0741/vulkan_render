module;

// The global-module-fragment include below is load-bearing, not stylistic: with -fno-exceptions
// and the vendored std module, a chores implementation unit that instantiates std::vector (this
// file does, in load_shader) sees TWO 'operator new(size_t, align_val_t)' declarations - module
// std's and the textual libc++ copy baked into utility.data_block.pcm (data_block is the one
// module that never imports std; it includes libc++ headers in its own GMF). The result is
// 'call to operator new is ambiguous' at allocate.h. Textually including glm here (the same
// trick vulkan/animation/animation_controller.cpp uses) makes clang merge the two copies, so
// the allocator instantiations resolve. Do not remove this include to "clean up".
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

module chores;

namespace chores {
    // Resolve the startup config in one step: merge config file + argv into the app settings,
    // then locate the shaders/ dir and pick the model file. Panics when a resource is missing.
    startup_config analyse_config(int argc, char** argv) {
        // 1. Resolve startup settings first: config file (config.toml by default, --config <path>
        //    to override) merged with positional argv overrides. argv[1] = model, argv[2] = grid
        //    side (numeric) or demo, argv[3] = demo.
        app_config::app_settings const settings = app_config::resolve_from_argv(argc, argv);
        if (!settings.config_file.empty()) {
            utility::log("app_config: loaded startup settings from '{}'", settings.config_file);
        }

        startup_config config = {settings, {}, {}};

        // 2. Shaders directory (holds GLSL sources and compiled SPIR-V): explicit config path when
        //    given, otherwise walk up from the working directory to find shaders/.
        if (!settings.paths.shaders_dir.empty()) {
            config.shaders_dir = settings.paths.shaders_dir;
            if (!std::filesystem::is_directory(config.shaders_dir)) {
                utility::panic(std::source_location::current(), "cannot find configured shaders_dir '{}'.", settings.paths.shaders_dir);
            }
        } else if (std::optional<std::filesystem::path> const located = locate_shaders_dir()) {
            config.shaders_dir = *located;
        } else {
            utility::panic("cannot find shaders/ directory. run the program from the project root, pass shaders_dir in config.toml, or use a cmake-build-* directory.");
        }

        // 3. Pick the model file: settings.model when configured/argv-given; else the default model
        //    under settings.paths.model_dir (or the auto-located gltf_model/).
        if (!settings.model.empty()) {
            config.model_path = settings.model;
        } else if (!settings.paths.model_dir.empty()) {
            config.model_path = (std::filesystem::path(settings.paths.model_dir) / "DamagedHelmet.gltf").string();
            if (!std::filesystem::is_regular_file(config.model_path)) {
                utility::panic(std::source_location::current(), "cannot find model '{}' under configured model_dir '{}'.", "DamagedHelmet.gltf", settings.paths.model_dir);
            }
        } else if (std::optional<std::filesystem::path> const located = locate_model_file()) {
            config.model_path = located->string();
        } else {
            utility::panic("cannot find gltf_model/DamagedHelmet.gltf. run the program from the project root or pass a model path as argv[1]");
        }

        return config;
    }

    // Read a single shader SPIR-V file and print info; panic on failure
    void load_shader(std::filesystem::path const& dir, std::string_view const file_name, std::vector<unsigned char>& out) {
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
                                  std::string_view const pipeline_name,
                                  std::string_view const vertex_file,
                                  std::string_view const fragment_file) {
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

    // Create the demo pipelines up front: the standard PBR pipeline (used by the imported scene)
    // plus the skybox background (fullscreen environment pass) and the directional shadow pass
    // (depth-only). The legacy triangle demo pipeline is deliberately not created here - nothing
    // draws it anymore.
    void setup_pipeline(vulkan::runtime& runtime, std::filesystem::path const& shaders_dir) {
        // Standard PBR pipeline: the imported scene's primitives bind to it
        load_and_create_pipeline(runtime, shaders_dir, "pbr", "pbr.vert.spv", "pbr.frag.spv");

        {
            // Skybox background pipeline (fullscreen environment pass): drawn first every frame,
            // behind the scene. Panic on failure - the frame needs a background to clear to.
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
            // map. Created once; enable_shadows() activates the pass after the scene import.
            // Failure is not fatal - the scene simply renders without shadows.
            std::vector<unsigned char> vertex_code;
            std::vector<unsigned char> fragment_code;
            load_shader(shaders_dir, "shadow.vert.spv", vertex_code);
            load_shader(shaders_dir, "shadow.frag.spv", fragment_code);
            auto const shadow_result = runtime.make_shadow_pipeline(vertex_code, fragment_code);
            if (!shadow_result) { // NOLINT(bugprone-branch-clone): CLion FP - the branches log different messages
                utility::log("shadow pipeline disabled: {}", shadow_result.error());
            } else {
                utility::log("SUCCESS: shadow pipeline created (directional shadow map pass)");
            }
        }
    }

    // Optional instancing stress: grid_side > 1 (config or argv) draws the first imported
    // primitive as a grid_side x grid_side grid in ONE instanced draw call (an
    // instanced_draw_primitive appended to the scene tree - the frame loop is untouched)
    void add_instancing_grid(vulkan::runtime& runtime, int const grid_side, float const scene_radius) {
        if (grid_side <= 1) {
            return;
        }
        std::vector<vulkan::primitive const*> const pbr_primitives = runtime.get_primitives("pbr");
        if (pbr_primitives.empty()) {
            return;
        }
        vulkan::primitive const& source = *pbr_primitives[0];
        std::vector<glm::mat4> transforms;
        transforms.reserve(static_cast<size_t>(grid_side) * grid_side);
        float const spacing = 2.5f * scene_radius; // keep instances apart: measure draw scaling, not overdraw
        for (int i = 0; i < grid_side; ++i) {
            for (int j = 0; j < grid_side; ++j) {
                float const dx = (static_cast<float>(i) - static_cast<float>(grid_side - 1) * 0.5f) * spacing;
                float const dz = (static_cast<float>(j) - static_cast<float>(grid_side - 1) * 0.5f) * spacing;
                transforms.push_back(glm::translate(glm::mat4(1.0f), glm::vec3(dx, 0.0f, dz)) * source.push.model);
            }
        }
        runtime.make_instanced_primitive(source, transforms);
        utility::log("instancing stress: {} x {} grid ({} instances, 1 draw call)", grid_side, grid_side, transforms.size());
    }

    // Build the demo's Dear ImGui debug overlay (when use_gui): enable it on the runtime and
    // assemble the "vulkan_render debug" panel. The widgets bind to @p bindings (fps text,
    // toggles, sliders, animation mirrors, camera selection) - the frame loop keeps the fps
    // and animation mirrors in sync. The overlay's glTF-side content (authored camera names,
    // orbit-camera seeding) arrives as display names + a selection callback, so this helper
    // never touches glTF types.
    void setup_gui(vulkan::runtime& runtime,
                   bool const use_gui,
                   app_config::app_settings const& settings,
                   gui_bindings& bindings,
                   vulkan::animation::controller& animation,
                   std::vector<std::string> const& camera_names,
                   std::function<void(int)> const& on_camera_selected) {
        if (!use_gui) {
            return;
        }
        runtime.enable_debug_gui();
        vulkan::gui::debug_panel& panel = runtime.debug_gui().add_panel("vulkan_render debug");
        panel.set_default_size(settings.gui.panel_width, settings.gui.panel_height);
        panel.push_back(std::make_unique<vulkan::gui::label_widget>([&bindings] { return std::format("fps: {:.1f}", bindings.fps); }));
        panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>(
            "frustum culling",
            &bindings.cull_enabled,
            [&runtime](bool const enabled) { runtime.set_frustum_culling(enabled); }));
        panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>(
            "skybox",
            &bindings.skybox_enabled,
            [&runtime](bool const enabled) { runtime.set_skybox_enabled(enabled); }));
        panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>(
            "shadow",
            &bindings.shadow_enabled,
            [&runtime](bool const enabled) { runtime.set_shadow_enabled(enabled); }));
        // camera orbit target: dragging it moves what the camera looks at / orbits around
        // (camera.target is a glm::vec3, i.e. three contiguous floats; the runtime rebuilds the
        // camera UBO from it every frame, so no on_change callback is needed)
        panel.push_back(std::make_unique<vulkan::gui::vec3_widget>("camera target", &runtime.camera.target.x, 0.05f));
        // playback controls (only when the model carries animations): play/pause toggle bound
        // to the playback state, a time scrubber (pauses on drag so the clock cannot fight the
        // scrub; the play checkbox resumes), and - for multi-animation assets - a dropdown to
        // pick which animation plays. All playback state lives in the animation::controller.
        if (animation.has_active()) {
            panel.push_back(std::make_unique<vulkan::gui::label_widget>([&animation] {
                return std::format("animation '{}' ({}s)", animation.active_name(), animation.loop_duration());
            }));
            panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>(
                "play",
                &bindings.anim_playing,
                [&animation](bool const enabled) { animation.set_playing(enabled); }));
            panel.push_back(std::make_unique<vulkan::gui::slider_widget>(
                "time",
                &bindings.anim_time,
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
                    &bindings.anim_index,
                    [&animation](int const index) { animation.select(static_cast<std::size_t>(index)); }));
            }
            utility::log("gui: playback controls added ({} animation(s))", animation.playable_count());
        }
        // camera selector: "orbit" (free) or any scene camera (its pose seeds the orbit camera,
        // so the mouse keeps working after switching)
        if (!camera_names.empty()) {
            std::vector<std::string> items;
            items.reserve(camera_names.size() + 1);
            items.push_back("orbit");
            items.insert(items.end(), camera_names.begin(), camera_names.end());
            panel.push_back(std::make_unique<vulkan::gui::combo_widget>(
                "camera",
                std::move(items),
                &bindings.current_camera,
                on_camera_selected));
            utility::log("gui: camera selector added ({} camera(s))", camera_names.size());
        }
        // Shadow depth bias (bottom of the panel - a rarely-used tuning aid): the pass's bias
        // is dynamic state applied every frame; the slope factor removes acne on angled
        // surfaces, the constant adds a fixed push. Note it cannot fix geometry that is simply
        // too coarse (e.g. RecursiveSkeletons' sides are large flat triangles - the depth
        // gradient across them is what it is), it only tunes the bias offset.
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>(
            "shadow bias slope",
            &bindings.shadow_bias_slope,
            0.0f,
            10.0f,
            [&runtime, &bindings](float const value) {
                bindings.shadow_bias_slope = value;
                runtime.set_shadow_depth_bias(bindings.shadow_bias_constant, bindings.shadow_bias_slope, 0.0f);
            }));
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>(
            "shadow bias constant",
            &bindings.shadow_bias_constant,
            0.0f,
            10.0f,
            [&runtime, &bindings](float const value) {
                bindings.shadow_bias_constant = value;
                runtime.set_shadow_depth_bias(bindings.shadow_bias_constant, bindings.shadow_bias_slope, 0.0f);
            }));
        utility::log("gui: Dear ImGui debug overlay enabled");
    }

    // Wire an animation backend to the runtime: the scene tree it drives, its per-frame-slot
    // morph/skin buffers (active slot for per-frame writes, explicit slot for setup bakes) and
    // its shared task pool. The controller sees only this surface, never vulkan::runtime.
    vulkan::animation::backend make_animation_backend(vulkan::runtime& runtime) {
        vulkan::animation::backend backend;
        backend.scene = &runtime.get_scene();
        backend.morph_scratch_active = [&runtime]() -> float* {
            return static_cast<float*>(runtime.morph_scratch());
        };
        backend.morph_scratch_slot = [&runtime](uint32_t const slot) -> float* {
            return static_cast<float*>(runtime.morph_scratch(slot));
        };
        backend.set_skin_matrices_active = [&runtime](std::span<glm::mat4 const> matrices) {
            runtime.set_skin_matrices(matrices);
        };
        backend.set_skin_matrices_slot = [&runtime](std::span<glm::mat4 const> matrices, uint32_t const slot) {
            runtime.set_skin_matrices(matrices, slot);
        };
        backend.scene_changed = [&runtime]() {
            runtime.scene_changed();
        };
        backend.run_tasks = [&runtime](std::span<std::function<void()>> tasks) {
            runtime.run_tasks(tasks, vulkan::task_priority::animation);
        };
        backend.task_worker_count = [&runtime]() -> int {
            return runtime.task_pool_threads();
        };
        return backend;
    }
} // namespace chores
