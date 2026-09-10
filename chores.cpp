module;

// The global-module-fragment include below is load-bearing, not stylistic: with -fno-exceptions
// and the vendored std module, a chores implementation unit that instantiates std::vector (this
// file does, in load_shader) sees TWO 'operator new(size_t, align_val_t)' declarations - module
// std's and the textual libc++ copy baked into utility.data_block.pcm (data_block is the one
// module that never imports std; it includes libc++ headers in its own GMF). The result is
// 'call to operator new is ambiguous' at allocate.h. Textually including glm here (the same
// trick vulkan/animation/controller.cpp uses) makes clang merge the two copies, so
// the allocator instantiations resolve. Do not remove this include to "clean up".
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <span>

module chores;

namespace chores {
    // Resolve the startup config in one step: merge config file + argv into the app settings,
    // then locate the shaders/ dir and pick the model file. Panics when a resource is missing.
    startup_config analyse_config(int argc, char** argv) {
        // 1. Resolve startup settings first: config file (config.toml by default, --config <path>
        //    to override) merged with positional argv overrides. argv[1] = model, argv[2] = grid
        //    side (numeric).
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
        // Standard PBR pipeline: the imported scene's primitives bind to it (the FIRST pipeline
        // created becomes the runtime's implicit default)
        load_and_create_pipeline(runtime, shaders_dir, "pbr", "pbr.vert.spv", "pbr.frag.spv");
        // Non-PBR "unlit" pipeline: flat base color, no lighting/shadows/IBL (see unlit.frag).
        // Registered as a SECOND named pipeline - the scene tree's default-semantics leaves draw
        // with whatever the runtime default is, so switching set_default_pipeline() between
        // "pbr" and "unlit" (gui "render mode") re-shades the whole scene without re-baking.
        load_and_create_pipeline(runtime, shaders_dir, "unlit", "pbr.vert.spv", "unlit.frag.spv");

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
            // Post-process pipeline (HDR scene target -> exposure + ACES + gamma -> swapchain):
            // the forward passes render into an HDR offscreen target, so this pass is required to
            // present anything meaningful. Panic on failure like the skybox.
            std::vector<unsigned char> vertex_code;
            std::vector<unsigned char> fragment_code;
            load_shader(shaders_dir, "post.vert.spv", vertex_code);
            load_shader(shaders_dir, "post.frag.spv", fragment_code);
            auto const post_result = runtime.make_post_pipeline(vertex_code, fragment_code);
            if (!post_result) {
                utility::panic(std::source_location::current(), "failed to create post-process pipeline: {}", post_result.error());
            }
            utility::log("SUCCESS: post-process pipeline created (HDR -> exposure/tonemap -> swapchain)");
        }
        {
            // FXAA pipeline (gamma-encoded LDR image -> anti-aliased swapchain). Optional: it
            // reuses post.vert and the post set layout, so it must come after the post pipeline.
            // Failure is not fatal - FXAA simply stays unavailable and runtime::set_fxaa() logs it.
            std::vector<unsigned char> vertex_code;
            std::vector<unsigned char> fragment_code;
            load_shader(shaders_dir, "post.vert.spv", vertex_code);
            load_shader(shaders_dir, "fxaa.frag.spv", fragment_code);
            auto const fxaa_result = runtime.make_fxaa_pipeline(vertex_code, fragment_code);
            if (!fxaa_result) {
                utility::log("fxaa pipeline disabled: {}", fxaa_result.error());
            } else {
                utility::log("SUCCESS: fxaa pipeline created (LDR -> anti-aliased swapchain)");
            }
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

        {
            // Clustered light culling compute pass (M5): one dispatch per frame sorts the punctual
            // lights into the screen-tile x depth-slice grid the shading stage then reads. Optional -
            // without it (or with [render] clustered_lights = false) shade_surface() loops every
            // active light, which is the brute-force reference the clustered path is verified on.
            std::vector<unsigned char> compute_code;
            load_shader(shaders_dir, "light_cluster.comp.spv", compute_code);
            auto const cluster_result = runtime.make_cluster_pipeline(compute_code);
            if (!cluster_result) {
                utility::log("clustered light culling disabled: {}", cluster_result.error());
            } else {
                utility::log("SUCCESS: cluster compute pipeline created (clustered light culling)");
            }
        }

        {
            // G-buffer pair (the deferred path's first half): the surface-writing pipeline the
            // opaque pass binds when it writes the G-buffer, and the fullscreen debug view that
            // turns one stored channel into a visible image. Both optional - without them
            // runtime::set_gbuffer_debug() has no effect and the forward path keeps running.
            std::vector<unsigned char> vertex_code;
            std::vector<unsigned char> fragment_code;
            load_shader(shaders_dir, "pbr.vert.spv", vertex_code); // the vertex stage is the forward one
            load_shader(shaders_dir, "gbuffer.frag.spv", fragment_code);
            auto const gbuffer_result = runtime.make_gbuffer_pipeline(vertex_code, fragment_code);
            if (!gbuffer_result) {
                utility::log("gbuffer pipeline disabled: {}", gbuffer_result.error());
            } else {
                // the debug view is a FULLSCREEN pass: it needs post.vert's synthetic triangle, not
                // the scene vertex stage (which declares vertex inputs, instancing/skin/morph
                // descriptors and the scene push block - all of which the debug pass has no use for)
                load_shader(shaders_dir, "post.vert.spv", vertex_code);
                load_shader(shaders_dir, "gbuffer_debug.frag.spv", fragment_code);
                auto const debug_result = runtime.make_gbuffer_debug_pipeline(vertex_code, fragment_code);
                if (!debug_result) {
                    utility::log("gbuffer debug view disabled: {}", debug_result.error());
                } else {
                    // the deferred lighting stage reads the G-buffer through the debug view's set
                    // layout, so it must be created after it
                    load_shader(shaders_dir, "deferred.frag.spv", fragment_code);
                    auto const deferred_result = runtime.make_deferred_pipeline(vertex_code, fragment_code);
                    if (!deferred_result) {
                        utility::log("deferred lighting disabled: {}", deferred_result.error());
                    } else {
                        // TAA resolve (deferred-only): its own set layout (scene color, history,
                        // motion vectors, depth) and a fullscreen pipeline writing the HDR target
                        load_shader(shaders_dir, "taa.frag.spv", fragment_code);
                        auto const taa_result = runtime.make_taa_pipeline(vertex_code, fragment_code);
                        if (!taa_result) {
                            utility::log("taa disabled: {}", taa_result.error());
                        } else {
                            utility::log("SUCCESS: gbuffer + deferred pipelines created (surface write, debug view, deferred lighting, taa)");
                        }
                    }
                }
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
        // per-pass GPU milliseconds (runtime::gpu_timing_summary): the timing that steers the
        // renderer's performance work, so it sits with the fps line at the top of the panel
        panel.push_back(std::make_unique<vulkan::gui::label_widget>([&runtime] { return runtime.gpu_timing_summary(); }));
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
        // clustered light culling (M5): off = every active light is evaluated per pixel (the
        // brute-force reference), on = only the pixel's cluster list. Mirroring it every frame in
        // main() keeps the config and the checkbox in agreement.
        panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>("clustered lights", &bindings.clustered_lights));
        // screen-space ambient occlusion (M6): the deferred lighting stage traces the G-buffer.
        // The sliders edit the radius (world units), the applied intensity and the sample count.
        panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>("ssao", &bindings.ssao_enabled));
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>("ssao radius", &bindings.ssao_radius, 0.05f, 3.0f));
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>("ssao intensity", &bindings.ssao_intensity, 0.0f, 1.0f));
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>("ssao samples", &bindings.ssao_samples, 1.0f, 16.0f));
        // render mode: pbr (lit) vs unlit (flat base color, no shading). Default-semantics leaves
        // draw with the runtime's default pipeline, so this only records a combo selection here;
        // main() applies it BETWEEN frames via runtime.set_default_pipeline (the registry may
        // not be mutated while a frame records).
        panel.push_back(std::make_unique<vulkan::gui::combo_widget>(
            "render mode",
            std::vector<std::string>{"pbr (lit)", "unlit (flat)"},
            &bindings.render_mode));
        // selectable BRDF theory models (pbr.frag): preset 0 is the default GGX + joint-Smith;
        // each other preset differs by exactly one piece (NDF or visibility), so the gui is a
        // live A/B comparison. CPU-side write-through (safe mid-run, see runtime::set_brdf_model).
        panel.push_back(std::make_unique<vulkan::gui::combo_widget>(
            "brdf model",
            std::vector<std::string>{"GGX + joint Smith", "GGX + height-corr. Smith", "Beckmann + Smith", "Blinn-Phong + Smith"},
            &bindings.brdf_model,
            [&runtime](int const index) { runtime.set_brdf_model(index); }));
        panel.push_back(std::make_unique<vulkan::gui::combo_widget>(
            "diffuse model",
            std::vector<std::string>{"Lambert", "Oren-Nayar"},
            &bindings.diffuse_model,
            [&runtime](int const index) { runtime.set_diffuse_model(index); }));
        // linear exposure applied before tonemapping (pbr.frag + skybox.frag); main pushes it
        // into the runtime every frame like the light slots
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>("exposure", &bindings.exposure, 0.1f, 5.0f));
        // bloom (bright-pass threshold + blend weight); 0 intensity disables it
        // the useful ranges: a threshold above ~0.75 leaves almost no pixel over it (so nothing
        // glows), and the intensity needed for a visible glow grows with the threshold - keeping
        // the threshold low is what makes the whole intensity slider effective
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>("bloom intensity", &bindings.bloom_intensity, 0.0f, 3.0f));
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>("bloom threshold", &bindings.bloom_threshold, 0.0f, 0.75f));
        // FXAA: a checkbox plus its two shader knobs (main mirrors all three into the runtime every
        // frame). The knobs are genuine effects, not strength padding - "subpixel" trades edge
        // smoothing for the single-pixel sparkle FXAA leaves on near-axis-aligned edges, and the
        // threshold decides how much contrast counts as an edge (lower = softer whole image).
        panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>("fxaa", &bindings.fxaa_enabled));
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>("fxaa subpixel", &bindings.fxaa_subpixel, 0.0f, 1.0f));
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>("fxaa edge threshold", &bindings.fxaa_edge_threshold, 0.05f, 0.5f));
        // G-buffer debug view: what the deferred path stores - the one part of the renderer whose
        // contents cannot be judged from a shaded screenshot, so it gets a channel selector rather
        // than a strength knob. main() mirrors both fields into the runtime every frame.
        panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>("gbuffer debug", &bindings.gbuffer_debug));
        panel.push_back(std::make_unique<vulkan::gui::combo_widget>(
            "gbuffer channel",
            std::vector<std::string>{"albedo", "normal", "roughness", "metallic", "ao", "material id", "depth", "flags"},
            &bindings.gbuffer_channel));
        // deferred lighting: the render-mode switch for the deferred path (the debug view above wins
        // when both are on, which is why it is not part of the same combo as pbr/unlit)
        panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>("deferred lighting", &bindings.deferred_enabled));
        // TAA: the deferred path's anti-aliasing, with the two history-weight knobs. The static weight
        // decides how smooth a still image gets (higher = smoother, slower to react to lighting
        // changes); the minimum is what a fast-moving pixel falls back to (lower = trusts the current
        // frame more, which trades smoothing for less ghosting).
        panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>("taa", &bindings.taa_enabled));
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>("taa history (static)", &bindings.taa_blend_static, 0.0f, 0.98f));
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>("taa history (min)", &bindings.taa_blend_min, 0.0f, 0.98f));
        // cel/toon shading: quantize the diffuse falloff (and harden shadows/highlights);
        // 0 steps leaves plain PBR, softness shrinks toward hard comic edges
        // cel/toon shading is discrete: every listed band count gives a visibly different look
        // (more bands converge back to smooth PBR, so a continuous slider had dead zones).
        // Softness stays small - a wide band edge erases the steps entirely.
        panel.push_back(std::make_unique<vulkan::gui::combo_widget>(
            "toon shading",
            std::vector<std::string>{"off (plain pbr)", "2 bands (hardest)", "3 bands", "4 bands", "5 bands", "6 bands", "8 bands (softest)"},
            &bindings.toon_bands_index));
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>("toon softness", &bindings.toon_softness, 0.01f, 0.25f));
        // ---- punctual lights (demo lights; see apply_point_lights): the widgets edit
        //      bindings.point_lights live and main() pushes the enabled set once per frame.
        //      Each slot is a point light or - with `spot` checked - a cone light -------
        for (std::size_t i = 0; i < std::size(bindings.point_lights); ++i) {
            gui_bindings::light_slot& slot = bindings.point_lights[i];
            panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>(
                std::format("point light {}", i + 1), &slot.enabled));
            panel.push_back(std::make_unique<vulkan::gui::vec3_widget>(
                std::format("  position {}", i + 1), slot.position, 0.1f));
            panel.push_back(std::make_unique<vulkan::gui::vec3_widget>(
                std::format("  color {}", i + 1), slot.color, 0.02f));
            panel.push_back(std::make_unique<vulkan::gui::slider_widget>(
                std::format("  intensity {}", i + 1), &slot.intensity, 0.0f, 50.0f));
            panel.push_back(std::make_unique<vulkan::gui::slider_widget>(
                std::format("  range {}", i + 1), &slot.range, 0.1f, 100.0f));
            // spot cone editing (ignored while the slot stays a point light)
            panel.push_back(std::make_unique<vulkan::gui::checkbox_widget>(
                std::format("  spot {}", i + 1), &slot.spot));
            panel.push_back(std::make_unique<vulkan::gui::vec3_widget>(
                std::format("  direction {}", i + 1), slot.direction, 0.1f));
            panel.push_back(std::make_unique<vulkan::gui::slider_widget>(
                std::format("  inner cone deg {}", i + 1), &slot.inner_cone_deg, 0.0f, 89.0f));
            panel.push_back(std::make_unique<vulkan::gui::slider_widget>(
                std::format("  outer cone deg {}", i + 1), &slot.outer_cone_deg, 1.0f, 89.0f));
        }
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
        // Cascaded shadow maps: how many cascades the sun's shadow pass fills (1 = the historic
        // single map) and how much of a cascade's range fades into the next one. Both are
        // write-through: the runtime refits the cascade boxes on the next frame, and the blend is a
        // shader constant in the light UBO - no pipeline or image rebuild, so they are live.
        panel.push_back(std::make_unique<vulkan::gui::combo_widget>(
            "shadow cascades",
            std::vector<std::string>{"1 (single map)", "2", "3", "4"},
            &bindings.shadow_cascades,
            [&runtime](int const index) { runtime.set_shadow_cascades(static_cast<uint32_t>(index) + 1u); }));
        panel.push_back(std::make_unique<vulkan::gui::slider_widget>(
            "shadow cascade blend",
            &bindings.shadow_cascade_blend,
            0.0f,
            0.5f,
            [&runtime](float const value) { runtime.set_shadow_cascade_blend(value); }));
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

    void apply_point_lights(vulkan::runtime& runtime, gui_bindings const& bindings, std::span<vulkan::punctual_light const> const extra) {
        // build the enabled demo lights into a fixed stack array (limit = the light UBO's
        // array size) and push it; the span form keeps set_point_lights cheap to call per frame.
        // `extra` is the [lighting] demo_lights stress set (M5), appended after the overlay's slots
        // so the overlay keeps working - both share the UBO's light array, so the overlay's slots
        // win when the two together would overflow it.
        std::array<vulkan::punctual_light, vulkan::max_punctual_lights> active = {};
        uint32_t count = 0;
        for (gui_bindings::light_slot const& slot : bindings.point_lights) {
            if (!slot.enabled || count >= vulkan::max_punctual_lights) {
                continue;
            }
            vulkan::punctual_light& light = active[count++];
            light.position = glm::vec3(slot.position[0], slot.position[1], slot.position[2]);
            light.color = glm::vec3(slot.color[0], slot.color[1], slot.color[2]);
            light.intensity = slot.intensity;
            light.range = slot.range;
            light.spot = slot.spot;
            if (slot.spot) {
                light.spot_direction = glm::vec3(slot.direction[0], slot.direction[1], slot.direction[2]);
                // clamp the cone (inner <= outer, both < 90 deg) and hand the shader the cosines
                float const outer_deg = std::clamp(slot.outer_cone_deg, 1.0f, 89.0f);
                float const inner_deg = std::clamp(slot.inner_cone_deg, 0.0f, outer_deg);
                light.spot_outer_cos = std::cos(glm::radians(outer_deg));
                light.spot_inner_cos = std::cos(glm::radians(inner_deg));
            }
        }
        for (vulkan::punctual_light const& light : extra) {
            if (count >= vulkan::max_punctual_lights) {
                break;
            }
            active[count++] = light;
        }
        runtime.set_point_lights(std::span(active.data(), count));
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
