#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
import vstd;
import application_configuration;
import chores; // demo bootstrap helpers (shader loading / dir locating / pipelines)
import gltf_loader;
import toon_material_sidecar; // the .toon.tsv a character model carries: its toon maps, and the _Use flags
import utility;               // re-exports utility:frame_clock / frame_stats / bvh / better_pmr / thread_pool / data_block
import vulkan.animation;      // animation::controller: glTF playback / skinning / morphs on the runtime tree
import vulkan.math;
import vulkan.scene_tree; // scene storage + GPU primitives (was vulkan.model)
import vulkan.runtime;
import vulkan.render_start_demo; // the example's pass wiring: this app's chain, from outside the renderer

// Route std::pmr allocations through mimalloc (utility:better_pmr) before main(): this
// file-scope reference's dynamic initialization runs at startup, so every runtime/scene
// object built below already allocates its std::pmr vectors from mimalloc. Idempotent —
// other TUs (vulkan/runtime/runtime.cpp) keep their own copy of the same singleton.
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
        // Seconds of ANIMATION time added per PRESENTED frame (`--capture-animation-sweep`). 0 - the
        // default - leaves playback on the wall clock, which is what an interactive run wants.
        //
        // WHY IT EXISTS, and it is the camera sweep's argument one member up, for the other half of the
        // frame: a capture of a DEFORMING mesh is only reproducible if the pose is a function of the frame
        // index. [render] animation_time makes such a capture reproducible and USELESS for this purpose - a
        // pinned pose uploads the same skin matrices every frame, so the previous-frame deformation equals
        // the current one and the deformation term of the motion vector is exactly zero, which is why a
        // pinned-pose screenshot cannot tell a deformation-aware renderer from the one that ignores
        // deformation. The wall clock (frame_clock::delta_seconds()) is reproducible in neither direction.
        // So playback gets the frame-indexed clock the camera already has: the pose is a function of the
        // frames PRESENTED, and two runs of one animation sweep are byte-identical.
        float animation_seconds_per_frame = 0.0f;
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
            if (std::optional<std::string_view> const value = take_value(i, arg, "--capture-animation-sweep")) {
                // seconds of animation time per presented frame - see capture_options::animation_seconds_per_frame
                if (std::optional<float> const number = parse_number(*value)) {
                    options.animation_seconds_per_frame = *number;
                } else {
                    utility::log("capture: ignoring '--capture-animation-sweep {}' (expected seconds per frame)", *value);
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
            if (options.animation_seconds_per_frame != 0.0f) {
                utility::log("capture animation sweep: {:.4f} s of animation per frame, from the clip's own start", options.animation_seconds_per_frame);
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
    runtime.set_shadow_depth_bias(settings.render.shadow_bias_constant, settings.render.shadow_bias_slope, 0.0f);
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

    // 8b. THE TOON MATERIAL SIDECAR, RESOLVED AGAINST THIS MODEL'S OWN IMAGES.
    //
    //     A toon character's materials read a SECOND set of texture slots that glTF has no concept of - a
    //     diffuse ramp, a shadow LUT, a specular ramp, a matcap, a face SDF - and the `.toon.tsv` beside the
    //     model names them. Nothing connects that file to the model except IMAGE NAMES, which is why
    //     `gltf_loader` now carries them (see `texture_data::name` and `scenes::texture_index_by_name`).
    //
    //     THIS BLOCK RESOLVES THE JOIN AND REPORTS IT, which is deliberately all it does so far: it is the one
    //     place the resolution can be SEEN on a real character rather than on a test fixture, and what it
    //     prints is exactly the data the shading stage needs next - each material's family, and for every toon
    //     slot the sidecar declares, whether the artist switched it on and which of the model's images it
    //     names. The three outcomes are each reported distinctly because they are three different situations:
    //     no sidecar at all (the normal case for a model that is not a character), a slot whose flag is off,
    //     and a slot that is ON while the model has no such image - which is the case a consumer must handle by
    //     leaving the feature off rather than by substituting something.
    // THE ASSET PIPELINE'S VOCABULARY FOR THE FOUR TOON LANES, and the only place it appears: one entry per
    // `vulkan::toon_slot`, in lane order, carrying BOTH names the pipeline uses for it.
    //
    // TWO NAMES RATHER THAN ONE, and the second is not derivable from the first: the ramp and LUT lanes are
    // switched on by `_Use<Slot>`, but the MATCAP lane `_MatcapTex` is switched on by `_UseMatcap` - the slot's
    // `Tex` suffix is not in the flag, which is a fact about the pipeline rather than a rule. Deriving the flag
    // from the slot answered "off" for every matcap in the file, silently, which is the failure the sidecar
    // module exists to prevent and precisely the one its own convention could not see: it is right for three
    // lanes and wrong for the fourth.
    //
    // IT SITS HERE, ABOVE THE DIAGNOSTIC, because the diagnostic prints every slot in the file and has to answer
    // the same question the lookup does. It did not, once: the lookup was taught the real flag names while the log
    // went on calling the convention, and the result was a log line reading `| off` about a lane the renderer was
    // reading - a diagnostic that contradicts the renderer is worse than no diagnostic, so both now ask
    // `toon_flag_for`.
    struct toon_lane_names {
        std::string_view slot;
        std::string_view flag;
    };
    static constexpr std::array<toon_lane_names, static_cast<std::size_t>(vulkan::toon_slot::count)> toon_lane = {{
        {"_DiffRampMap", "_UseDiffRampMap"},
        {"_ShadowLutTex", "_UseShadowLutTex"},
        {"_SpecRampMap", "_UseSpecRampMap"},
        {"_MatcapTex", "_UseMatcap"},
        {"_SDFLightmap", "_UseSDFLightmap"},
    }};
    // The declared flag for a toon lane; the `_Use<Slot>` convention for every OTHER slot, which the diagnostic
    // needs because it walks the whole file (`_BaseMap`, `_BumpMap`, the outline and SDF masks and the rest).
    auto const toon_flag_for = [](std::string_view const slot_name) -> std::string {
        for (toon_lane_names const& lane : toon_lane) {
            if (lane.slot == slot_name) {
                return std::string(lane.flag);
            }
        }
        std::string flag{toon::enable_flag_prefix};
        flag.append(slot_name.starts_with('_') ? slot_name.substr(1) : slot_name);
        return flag;
    };

    std::optional<toon::sidecar> toon_sidecar = {}; // kept in scope: the import below is what consumes it
    {
        auto const sidecar = toon::load_sidecar(model_path);
        if (!sidecar.has_value()) {
            utility::log("toon sidecar: NOT READ - {}", sidecar.error());
        } else if (sidecar->empty()) {
            utility::log("toon sidecar: none beside '{}' (the normal case for a model that is not a character)", model_path);
        } else {
            toon_sidecar = *sidecar;
            utility::log("toon sidecar: {} material(s) described ({} line(s) skipped)", sidecar->materials.size(), sidecar->skipped_lines);
            for (toon::material_sidecar const& material : sidecar->materials) {
                // THE FAMILY COMES FROM THE LOADER'S CLASSIFIER over the SAME name, so the sidecar (which
                // supplies the parameters) and the renderer (which selects them) cannot disagree about which
                // family a material is: there is one classifier and both sides ask it.
                utility::log("  '{}' -> family {} | {} slot(s), {} scalar(s)", material.name, static_cast<uint32_t>(gltf::toon_family_of(material.name)), material.slots.size(), material.scalars.size());
                for (auto const& [slot_name, texture_name] : material.slots) {
                    std::optional<uint16_t> const index = scenes->texture_index_by_name(texture_name);
                    utility::log("      {} = '{}' -> {} | {}", slot_name, texture_name, index.has_value() ? std::format("texture #{}", *index) : std::string("ABSENT from this model"), material.enabled_by_flag(toon_flag_for(slot_name)) ? "ON" : "off");
                }
            }
        }
    }

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
        if (settings.render.camera_pose_set) {
            // A pinned pose from the config: the same six numbers `--capture-camera` takes, so a view a user
            // reports (printed by the F12 block below) can be pasted straight back in here.
            runtime.camera.yaw = glm::radians(settings.render.camera_pose[0]);
            runtime.camera.pitch = glm::radians(settings.render.camera_pose[1]);
            runtime.camera.distance = settings.render.camera_pose[2];
            runtime.camera.target = glm::vec3(settings.render.camera_pose[3], settings.render.camera_pose[4], settings.render.camera_pose[5]);
        }
        bool const interior = settings.render.camera_fit == "interior";
        if (settings.render.camera_pose_set) {
            // the pinned pose above already decided everything; the fit's numbers would overwrite it
        } else if (interior) {
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
    // ONE place that turns the live pose into the six numbers the config and the command line take: the F12
    // screenshot prints it (so a bug report carries its own camera) and so does the exit below.
    auto const log_camera_pose = [&runtime](char const* const why) {
        utility::log("camera pose ({}): {:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}   -> [render] camera_pose = [...] or --capture-camera",
                     why,
                     glm::degrees(runtime.camera.yaw),
                     glm::degrees(runtime.camera.pitch),
                     runtime.camera.distance,
                     runtime.camera.target.x,
                     runtime.camera.target.y,
                     runtime.camera.target.z);
    };

    gltf::scene_node_iterator const node_first = scenes->nodes_begin();
    gltf::scene_node_iterator const node_last;
    gltf::drawable_iterator const scene_first(*scenes, materials);
    gltf::drawable_iterator const scene_last;
    glm::vec3 const scene_import_shift = -scene_center + scene_sink;

    // ---- THE TOON LOOKUP: the one place the sidecar reader and the renderer meet ----
    // `vulkancorekit` deliberately does not depend on `gltf_loader` (see that target's note: the engine is
    // loader-agnostic), so the runtime cannot hold a sidecar and must not learn what one is. What it CAN ask for
    // is "the texture input for this material and this lane", and this application is the layer that links both
    // - so this is where they are joined.
    struct toon_lookup_state {
        toon::sidecar const* sidecar = nullptr;
        gltf::scenes const* scenes = nullptr;
        /// the PROCEDURALLY BAKED neutral unit ramp (see the baker below), kept alive here because the texture
        /// input the lookup returns is a SPAN INTO IT and `register_material` reads it during the import. ONE
        /// buffer serves BOTH ramp lanes: the diffuse ramp and the specular ramp are the same neutral step, and
        /// what tells them apart is which family numbers the shader moves it with
        std::vector<unsigned char> baked_unit_ramp = {};
        uint32_t baked_ramp_width = 0;
        uint32_t baked_ramp_height = 0;
        /// the SHADOW LUT cube (see the baker below) - a different shape from the ramp and so a different
        /// buffer, kept alive for the same reason: the lookup hands out a SPAN INTO IT
        std::vector<unsigned char> baked_shadow_lut = {};
        uint32_t baked_lut_width = 0;
        uint32_t baked_lut_height = 0;
        /// the MATCAP ball (see the baker below) - a third shape again, and a black one, because this stage had
        /// no matcap term to reproduce
        std::vector<unsigned char> baked_matcap = {};
        uint32_t baked_matcap_size = 0;
    };

    // ---- THE PROCEDURAL RAMP, which is what replaces the game's own ramps on the read path ----
    //
    // WHY IT IS BAKED RATHER THAN SAMPLED FROM THE MODEL: the ramps a character ships with are the GAME's
    // textures, and `ASSET_LICENSE_BOUNDARY_CN.md` excludes those from redistribution - so a read path that
    // depends on them is a read path this repository cannot carry. This bakes an equivalent SHAPE instead: a flat
    // shadow side, a step, a flat lit side, which is what a toon ramp IS (see character_forward.slang's note on
    // why a ramp replaces the procedural threshold rather than layering with it).
    //
    // ONE ASSET SERVES BOTH LANES, the diffuse ramp and the specular ramp. The reference keeps them in two V
    // bands of one atlas for exactly this reason - they are the same step read with two different coordinates,
    // one from the shadow-gated half-Lambert and one from the half-vector angle - and a neutral step is the
    // shape both of them need. Two files would be two things to keep in step for no gain.
    //
    // THE BAKE IS A NEUTRAL UNIT STEP, NOT ANY ONE FAMILY'S RAMP, and that is what lets a SINGLE redistributable
    // asset serve every family without copying the family table into this file. The texture holds the SHAPE -
    // zero through the shadow side, one through the lit side, a step at x = 0.5 - and the SHADER maps the
    // family's own threshold and edge width onto that step when it builds the coordinate it reads at (see
    // `character_ramp_half_width` in character_forward.slang). The family's numbers therefore stay in ONE place,
    // in the shader, where the procedural branch and the rim stage already read them; what this file supplies is
    // the shape, and what the shader supplies is where along it this material's terminator sits.
    //
    // THAT IS NOT A WORKAROUND FOR THE TABLE BEING IN THE SHADER, it is the structure the reference uses: its
    // ramp atlas is shared across materials and each material's own `_ShadowCenter` / `_ShadowSmoothness` decide
    // the UV it is read at. A per-family BAKE would be the copy - and it would be a worse one, because the
    // family's edge width would be frozen into pixels at load time instead of staying a number the shader can
    // evaluate the procedural branch with.
    //
    // `baked_ramp_half_width` IS HALF OF A SHARED CONTRACT; the other half is the shader's
    // `character_ramp_half_width`. The shader's remap inverts this bake exactly when the two agree, which is
    // what makes the texture branch and the procedural branch produce the same tint for the same family rather
    // than merely similar ones. `tests/test_toon_material_sidecar.cpp` reads both files and fails on drift -
    // and it is there rather than in a test of its own because the contract is a sidecar-lane contract.
    constexpr uint32_t baked_ramp_width = 256;
    constexpr uint32_t baked_ramp_height = 8;
    constexpr float baked_ramp_half_width = 0.035f;
    auto const bake_unit_ramp = []() {
        // `smoothstep(0.5 - w, 0.5 + w, x)`: the same Hermite step the shader's procedural branch builds, with
        // the step moved to the ramp's centre and its width normalised so the remap can undo it.
        auto const srgb_encode = [](float const linear) {
            return linear <= 0.0031308f ? linear * 12.92f : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
        };
        auto const smoothstep = [](float const edge0, float const edge1, float const x) {
            float const t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
        };
        std::vector<unsigned char> pixels(static_cast<std::size_t>(baked_ramp_width) * baked_ramp_height * 4u, 255u);
        for (uint32_t v = 0; v < baked_ramp_height; ++v) {
            for (uint32_t u = 0; u < baked_ramp_width; ++u) {
                float const x = static_cast<float>(u) / static_cast<float>(baked_ramp_width - 1u);
                float const step = smoothstep(0.5f - baked_ramp_half_width, 0.5f + baked_ramp_half_width, x);
                unsigned char* const texel = pixels.data() + (static_cast<std::size_t>(v) * baked_ramp_width + u) * 4u;
                // GREY, and the SHADER READS `.r`: the ramp carries the step and nothing else, so the family's
                // own `shadow_tint` is what colours the dark side rather than a bake that could only ever hold
                // one family's tint. Writing the same value into R, G and B keeps the asset readable as a ramp
                // by eye.
                //
                // THE UPLOAD IS SRGB (see register_material's slot table), so the texel holds the ENCODED value:
                // what the sampler hands the shader is then the LINEAR step, which is the number the procedural
                // branch's `smoothstep` produced. Encoding is what makes the two branches agree on the VALUE and
                // not just on the shape.
                unsigned char const encoded = static_cast<unsigned char>(std::clamp(srgb_encode(step), 0.0f, 1.0f) * 255.0f + 0.5f);
                texel[0] = encoded;
                texel[1] = encoded;
                texel[2] = encoded;
                texel[3] = 255u;
            }
        }
        return pixels;
    };

    // ---- THE SHADOW LUT, WHICH IS A CUBE RATHER THAN A STEP ----
    //
    // IT IS THE ONE LANE THAT CANNOT REUSE THE UNIT RAMP, and the shape is the reason: a ramp answers "how deep
    // is the shadow here", a shadow LUT answers "what is THIS MATERIAL's colour in shadow" and is indexed by the
    // material's own albedo. That is not a variation on the ramp - it is a function of a different variable -
    // and it is why the reference ships skin and cloth a `1024x32` cube at all rather than another band.
    //
    // THE LAYOUT IS THE REFERENCE'S, TO THE TILE COUNT: 32 horizontal `32x32` tiles holding a `32^3` cube, the
    // tile chosen by the X channel and a bilinear hop between adjacent tiles because X is continuous. The
    // shader's `toon_shadow_lut` inverts this exactly, and `baked_lut_tiles` is HALF OF THAT CONTRACT - the other
    // half is `character_shadow_lut_tiles` - because a bake laid out for a different tile count reads back as a
    // different colour with no other symptom. `tests/test_toon_material_sidecar.cpp` compares the two.
    //
    // THE CONTENT IS THE IDENTITY CUBE, which is the same choice the ramps make and for the same reason: a
    // neutral bake reproduces the look the procedural branch already had (`albedo` in, `albedo` out), so the
    // lane is EXERCISED without the frame being changed by a guess at what the artist meant. What the lane buys
    // is that an AUTHORED cube would be honoured - a real skin palette makes the shadow a function of the skin
    // tone, which no per-family constant can be - and that is not something a neutral bake can substitute for.
    //
    // Note the axes, because they are the ones that invert: X picks the TILE, the in-tile X is the GREEN channel,
    // the in-tile Y is the BLUE channel FLIPPED, and each channel's stored value is `index / (tiles - 1)`. The
    // half-texel offsets are in the SHADER rather than here - the bake writes texel centres, and a reader that
    // sampled corners would be off by half a texel in every direction.
    constexpr uint32_t baked_lut_tiles = 32;
    constexpr uint32_t baked_lut_width = baked_lut_tiles * baked_lut_tiles;
    constexpr uint32_t baked_lut_height = baked_lut_tiles;
    auto const bake_shadow_lut = []() {
        auto const srgb_encode = [](float const linear) {
            return linear <= 0.0031308f ? linear * 12.92f : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
        };
        std::vector<unsigned char> pixels(static_cast<std::size_t>(baked_lut_width) * baked_lut_height * 4u, 255u);
        for (uint32_t row = 0; row < baked_lut_height; ++row) {
            for (uint32_t column = 0; column < baked_lut_width; ++column) {
                // The identity cube at this texel's coordinate: the X channel from which tile it is in, the Y
                // channel from where it sits inside that tile, the Z channel from the row - flipped, because the
                // shader reads the cube's Z from the BOTTOM of the strip upward.
                float const x = static_cast<float>(column / baked_lut_tiles) / static_cast<float>(baked_lut_tiles - 1u);
                float const y = static_cast<float>(column % baked_lut_tiles) / static_cast<float>(baked_lut_tiles - 1u);
                float const z = static_cast<float>(baked_lut_tiles - 1u - row) / static_cast<float>(baked_lut_tiles - 1u);
                float const cube[3] = {x, y, z};
                unsigned char* const texel = pixels.data() + (static_cast<std::size_t>(row) * baked_lut_width + column) * 4u;
                for (int c = 0; c < 3; ++c) {
                    // SRGB-ENCODED for the same reason the ramp is: the lane is uploaded as an sRGB texture, so
                    // the sampler hands the shader the LINEAR value the cube is written to mean, and the identity
                    // survives the round trip instead of being gamma-shifted by it.
                    texel[c] = static_cast<unsigned char>(std::clamp(srgb_encode(cube[c]), 0.0f, 1.0f) * 255.0f + 0.5f);
                }
                texel[3] = 255u;
            }
        }
        return pixels;
    };

    // ---- THE MATCAP, WHICH IS BLACK, AND THAT IS THE WHOLE OF ITS NEUTRALITY ----
    //
    // THE THIRD SHAPE AND THE THIRD VARIABLE: a ramp is indexed by the shading term, the shadow LUT by the
    // material's albedo, and a matcap by the VIEW-SPACE NORMAL - `normalVS.xy * 0.5 + 0.5`, which is a sphere
    // map's own UV and the reason a matcap image is a picture of a ball.
    //
    // BLACK IS NOT A PLACEHOLDER, IT IS THE ONLY CONTENT THAT IS NEUTRAL HERE, and the difference from the other
    // two lanes is worth stating because it looks like a weaker bake and is not. The ramps and the cube have a
    // procedural branch to reproduce, so their neutral bake is a SHAPE the shader already computed (`smoothstep`,
    // the identity). This stage had NO matcap term at all before this lane existed, so "the look without the
    // artist's matcap" is the reference's own expression with the lookup at zero - `1 + 0 * strength` - and any
    // other content would be this repository inventing art direction and calling it a substitute. The lane is
    // real, it is read, and what it is FOR is honouring an authored ball.
    constexpr uint32_t baked_matcap_size = 256;
    auto const bake_matcap = []() {
        // A matcap is an sRGB reference image (the reference's `EfClothSampleMatcap` says so explicitly), so the
        // lane's upload format applies here as it does to the ramps: zero is zero in both encodings, which is the
        // one value where the encode step cannot disagree with itself.
        return std::vector<unsigned char>(static_cast<std::size_t>(baked_matcap_size) * baked_matcap_size * 4u, 0u);
    };

    // THE LOOKUP, whose lane vocabulary lives with the diagnostic above so that the two cannot disagree about
    // which flag switches a lane on.
    auto const toon_texture = [](void* const owner, std::string_view const material_name, vulkan::toon_slot const lane) -> vulkan::texture_input {
        toon_lookup_state const& state = *static_cast<toon_lookup_state*>(owner);
        vulkan::texture_input out = {};
        if (state.sidecar == nullptr || state.scenes == nullptr) {
            return out;
        }
        toon::material_sidecar const* const material = state.sidecar->find(material_name);
        if (material == nullptr) {
            return out;
        }
        toon_lane_names const& names = toon_lane[static_cast<std::size_t>(lane)];
        std::string_view const slot_name = names.slot;
        // THE ARTIST'S SWITCH DECIDES, and an off flag produces an INVALID input, which the record turns into the
        // white fallback - i.e. "do not read" (see material_record::toon_indices). A map that exists while its
        // flag is off must NOT be read: that is the first rule the sidecar module exists to keep. Asked BY NAME
        // rather than by slot, because that is the only spelling that is true for all four lanes.
        if (!material->enabled_by_flag(names.flag)) {
            return out;
        }
        // THE DIFFUSE AND SPECULAR RAMP LANES BOTH GET THE BAKED UNIT STEP, NOT THE MODEL'S IMAGE - see
        // bake_unit_ramp for why a path that read the game's own ramps is a path this repository cannot carry,
        // and note that the artist's switch above is still what decides WHETHER there is a ramp at all. The
        // other lanes keep the model's images for now: nothing reads them yet, and baking them is the same
        // question one lane at a time.
        if ((lane == vulkan::toon_slot::diffuse_ramp || lane == vulkan::toon_slot::specular_ramp) && !state.baked_unit_ramp.empty()) {
            out.data = std::span<unsigned char const>(state.baked_unit_ramp.data(), state.baked_unit_ramp.size());
            out.width = state.baked_ramp_width;
            out.height = state.baked_ramp_height;
            out.mip_levels = 1;
            out.valid = true;
            return out;
        }
        // THE SHADOW LUT LANE GETS THE BAKED CUBE, NOT THE MODEL'S PALETTE - the same trade as the ramps, and
        // for a stronger version of the same reason: `T_actor_common_femaleskincolor01_lut_D` is not the game's
        // only skin palette but one of several per-character variants, so a read path carrying it would be
        // carrying a specific character's skin tone. The artist's switch above still decides whether there is a
        // LUT at all, which is why hair - whose `_UseShadowLutTex` is off - is unaffected by any of this.
        if (lane == vulkan::toon_slot::shadow_lut && !state.baked_shadow_lut.empty()) {
            out.data = std::span<unsigned char const>(state.baked_shadow_lut.data(), state.baked_shadow_lut.size());
            out.width = state.baked_lut_width;
            out.height = state.baked_lut_height;
            out.mip_levels = 1;
            out.valid = true;
            return out;
        }
        // THE MATCAP LANE GETS THE BAKED BALL, NOT THE MODEL'S - `T_actor_common_matcap_10_D` is a game image
        // like the rest. THIS IS ALSO THE LANE THE FLAG TABLE ABOVE WAS FIXED FOR: until the flag was asked by
        // name, `_MatcapTex` resolved to "off" for the iris that ships it on, so this branch was unreachable and
        // the game's matcap was neither read nor replaced - the lane was simply absent, which is the quietest
        // possible version of getting it wrong.
        if (lane == vulkan::toon_slot::matcap && !state.baked_matcap.empty()) {
            out.data = std::span<unsigned char const>(state.baked_matcap.data(), state.baked_matcap.size());
            out.width = state.baked_matcap_size;
            out.height = state.baked_matcap_size;
            out.mip_levels = 1;
            out.valid = true;
            return out;
        }
        std::string_view const texture_name = material->slot(slot_name);
        if (texture_name.empty()) {
            return out;
        }
        std::optional<uint16_t> const index = state.scenes->texture_index_by_name(texture_name);
        if (!index.has_value()) {
            return out; // the sidecar names a map this model does not have
        }
        gltf::texture_data const& tex = state.scenes->textures[*index];
        if (tex.data.empty() || tex.width == 0 || tex.height == 0) {
            return out; // present but unusable: still "do not read" rather than a guess
        }
        out.data = std::span<unsigned char const>(tex.data.data(), tex.data.size());
        out.width = tex.width;
        out.height = tex.height;
        out.mip_levels = 1;
        out.valid = true;
        return out;
    };
    toon_lookup_state toon_state{.sidecar = toon_sidecar.has_value() ? &*toon_sidecar : nullptr, .scenes = &*scenes};
    toon_state.baked_unit_ramp = bake_unit_ramp();
    toon_state.baked_ramp_width = baked_ramp_width;
    toon_state.baked_ramp_height = baked_ramp_height;
    toon_state.baked_shadow_lut = bake_shadow_lut();
    toon_state.baked_lut_width = baked_lut_width;
    toon_state.baked_lut_height = baked_lut_height;
    toon_state.baked_matcap = bake_matcap();
    toon_state.baked_matcap_size = baked_matcap_size;
    runtime.set_toon_lookup(vulkan::runtime::toon_lookup{.owner = &toon_state, .texture = toon_texture});

    // ---- THE HEAD FRAME the face SDF shades against, resolved once here and published every frame ----
    //
    // THE LOADER IS ASKED FIRST, and the answer is a decision this code makes rather than a constant it never
    // questioned - which matters, because for the models in this repository the two answers happen to coincide.
    //
    // THE FALLBACK IS THE REFERENCE'S OWN FRAME (`EfFaceGetHeadBasis`'s `valid < 0.5` branch), so a model with
    // no usable skeleton is shaded by the answer the reference itself gives for that case rather than by a guess
    // made here. This model is exactly that case: `zhuangfy_scalar.glb` is seven nodes and zero skins, a static
    // pose split by body part, and a model that cannot turn its head has one head frame whether it is read from
    // a bone or from these three constants.
    // WHICH RIG AND WHICH JOINT, kept for the frame loop below: both are needed there to read the bone's world
    // matrix, and the joint index is the SAME number `head_joint_of` returns (an index into `skin::joints`, not
    // an asset node index) because that is what picks a matrix out of the rig's joint block.
    std::size_t head_rig = 0;
    std::optional<std::size_t> head_joint = std::nullopt;
    {
        gltf::head_basis head = gltf::head_basis_fallback();
        for (std::size_t scene_index = 0; scene_index < scenes->scene.size() && !head_joint.has_value(); ++scene_index) {
            for (std::size_t skin_index = 0; skin_index < scenes->skins.size(); ++skin_index) {
                if (std::optional<std::size_t> const joint = gltf::head_joint_of(*scenes, scene_index, skin_index); joint.has_value()) {
                    head_rig = skin_index;
                    head_joint = joint;
                    break;
                }
            }
        }
        if (head_joint.has_value()) {
            utility::log("head frame: '{}' HAS a head bone at rig {} joint {} - the frame is read from it every frame", model_path, head_rig, *head_joint);
        } else {
            utility::log("head frame: no head bone in '{}' ({} skin(s)) - shading from the reference's fallback frame", model_path, scenes->skins.size());
        }
        runtime.set_head_basis(vulkan::head_ubo{.front = glm::vec4(head.front, 0.0f), .right = glm::vec4(head.right, 0.0f), .up = glm::vec4(head.up, 0.0f)});
    }

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

    // FPS statistics (utility:frame_stats): a rolling one-second window of frame gaps.
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
        if (capture.animation_seconds_per_frame != 0.0f) {
            utility::log("capture: --capture-animation-sweep is IGNORED - [render] animation_time pins the pose, so the clock never advances (set animation_time = -1 to play)");
        }
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
    // THE TOON CHARACTER STAGE: copied here like every other toggle, and that copy is load-bearing rather than
    // tidiness - the frame loop mirrors `gui.character_forward` into the runtime every frame, so a config value
    // that never reached the gui (the trap the two lines above record) would be overwritten by the binding's
    // default on the very first frame.
    gui.character_forward = settings.render.character_forward;
    gui.megalights_enabled = settings.render.megalights;
    gui.megalights_samples = static_cast<float>(settings.render.megalights_samples);
    gui.megalights_spatial_sigma = settings.render.megalights_spatial_sigma;
    gui.megalights_history_tolerance = settings.render.megalights_history_tolerance;
    gui.sun_intensity = settings.render.sun_intensity; // a scale on the sun (see [render] sun_intensity)
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
        // just paced. dt is the WALL CLOCK - clamped, so a pause (minimized / swapchain-recreate
        // gaps that never stamped) does not fast-forward the animation by the whole gap, and
        // playback resumes where it paused - UNLESS --capture-animation-sweep asked for a
        // frame-indexed clock instead, which is the only form a capture of a deforming mesh can
        // be gated on (see capture_options::animation_seconds_per_frame).
        float const dt = capture.animation_seconds_per_frame != 0.0f
                             ? capture.animation_seconds_per_frame
                             : static_cast<float>(std::min(frame_clock.delta_seconds(), 0.25));
        animation.update(dt);
        // ---- THE HEAD FRAME FOLLOWS THE BONE, published here because this is where the pose exists ----
        //
        // IT LANDS ONE FRAME LATE, and that is a consequence of the per-frame-slot arrangement rather than an
        // oversight: the runtime copies `head_state` into the paced slot INSIDE pace_and_acquire(), which has
        // already run by the time this frame's pose is produced. So a pose is shaded with the head frame
        // computed from the previous pose. For a pose that is standing still - which is what a capture gates on,
        // and what this model's rest pose is - the difference is exactly zero, and for a turning head it is one
        // frame of lag: the same trade every other per-frame write in this loop makes.
        //
        // THE EXTRACTION IS THE REFERENCE'S (`EfFaceGetHeadBasis`): `-row3` of the bone's world matrix is the
        // head's forward and `-row1` its right, where "row" is HLSL's and therefore a COLUMN of the
        // column-major matrix this engine stores. `head_basis_from_axes` negates, re-orthogonalises and falls
        // back when the bone is degenerate, so a zero matrix here cannot put NaNs in the sigmoid.
        if (head_joint.has_value()) {
            if (std::optional<glm::mat4> const bone = animation.joint_world(head_rig, *head_joint); bone.has_value()) {
                glm::mat4 const& joint = *bone;
                glm::vec3 const forward_row(joint[0][2], joint[1][2], joint[2][2]); // HLSL `_31_32_33`
                glm::vec3 const right_row(joint[0][0], joint[1][0], joint[2][0]);   // HLSL `_11_12_13`
                gltf::head_basis const basis = gltf::head_basis_from_axes(forward_row, right_row);
                runtime.set_head_basis(vulkan::head_ubo{.front = glm::vec4(basis.front, 0.0f), .right = glm::vec4(basis.right, 0.0f), .up = glm::vec4(basis.up, 0.0f)});
                static bool logged_head_probe = false;
                if (!logged_head_probe) {
                    logged_head_probe = true;
                    utility::log("head probe: bone forward row ({:.3f} {:.3f} {:.3f}) right row ({:.3f} {:.3f} {:.3f}) -> basis front ({:.3f} {:.3f} {:.3f}) from_skeleton {}", forward_row.x, forward_row.y, forward_row.z, right_row.x, right_row.y, right_row.z, basis.front.x, basis.front.y, basis.front.z, basis.from_skeleton);
                }
            } else {
                static bool logged_head_miss = false;
                if (!logged_head_miss) {
                    logged_head_miss = true;
                    utility::log("head probe: joint_world({}, {}) returned NOTHING", head_rig, *head_joint);
                }
            }
        }
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
        // The toon character stage: mirrored like the other render toggles. Its own gate composes this knob
        // with the frame's opaque leaf list (feature_facts::character_forward_pending), so turning it on in a
        // frame with no opaque geometry records nothing rather than an empty instance.
        runtime.set_character_forward(gui.character_forward);
        start_demo.set_ssao(gui.ssao_enabled, gui.ssao_radius, gui.ssao_intensity, static_cast<uint32_t>(std::max(gui.ssao_samples, 0.0f) + 0.5f));
        // cel shading: the combo picks a discrete band count (index 0 = off); every entry is a
        // visibly different look, unlike a continuous strength that had dead zones between bands
        constexpr std::array<float, 7> toon_band_counts = {0.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 8.0f};
        auto const toon_index = static_cast<std::size_t>(std::clamp(gui.toon_bands_index, 0, static_cast<int>(toon_band_counts.size()) - 1));
        runtime.set_toon_shading(toon_band_counts[toon_index], gui.toon_softness);
        runtime.set_sun_intensity(gui.sun_intensity);
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
                // ... AND THE POSE THAT FRAME WAS RENDERED WITH, in the form both the config and the
                // command line take. A screenshot without its camera is a picture nobody can reproduce,
                // which is how a user's "the face goes grey at some angle" cost a day of guessing.
                log_camera_pose("F12 screenshot");
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

    // The pose the session ENDED on, so a view reached by orbiting can be written down without
    // re-deriving it from the fit. Printed whether the loop ended by closing the window or by a
    // scripted capture finishing.
    log_camera_pose("exit");
    // 15. Wait for the GPU to finish; primitives and pipelines are released by the runtime destructor
    runtime->wait_idle();
    utility::log("render loop finished");
    return 0;
}
