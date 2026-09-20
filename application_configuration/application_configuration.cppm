// ============================================================================
// module: app_config
// module version: 0.40.0  (independent of the app version in CMakeLists project(VERSION))
//
// Startup configuration: TOML file (config.toml / --config) merged with argv.
// Pure CPU, no Vulkan dependency.
//
// evolve: bump MAJOR on breaking interface changes, MINOR on additive features,
//         PATCH on internal fixes - independently of the rest of the project.
// ============================================================================
module;

// toml++ is header-only and auto-detects -fno-exceptions (TOML_EXCEPTIONS=0),
// so including it in the global module fragment works under the project flags.
#include <toml++/toml.hpp>

export module application_configuration;
export import vstd;
import utility;

/**
 * @file application_configuration.cppm
 * @defgroup app_config Application Startup Config
 * @brief load vulkan_render startup settings from a TOML file, merged with command-line
 *        arguments (--config <path> overrides the default file; explicit argv values for the
 *        model / grid override the file). Pure CPU, no Vulkan dependency.
 *
 * Example config.toml:
 * @code
 * # top level: model to load
 * model = "gltf_model/DamagedHelmet.gltf"
 * grid_side = 0     # > 1 enables the instancing stress grid (0 = off)
 *
 * [paths]
 * shaders_dir = ""  # shader SPIR-V dir (empty = auto-locate "shaders/" upward)
 * model_dir   = ""  # default model dir used when model is empty (auto-locate gltf_model/)
 * screenshot_dir = ""  # base directory for F12 screenshots (empty = current working directory)
 *
 * [render]
 * window_width  = 1080
 * window_height = 960
 * window_title  = "vulkan_render"
 * vsync = true     # true = FIFO_LATEST_READY (vsync), false = mailbox (uncapped)
 * max_fps = 0      # 0 = uncapped (what a throughput measurement needs), else a frame rate cap
 * clear_color = [0.02, 0.02, 0.03]  # background clear color, RGB in 0..1
 * camera_fit = "exterior"  # "exterior" fits the whole model from outside; "interior" stands inside
 *                          # and looks down the longest horizontal axis (a hall / nave / corridor)
 * shadow = true    # record the directional shadow pass each frame
 * fxaa   = false   # anti-alias the final image (adds one fullscreen pass; needs fxaa.frag.spv)
 * gpu_timings = true  # measure + report per-pass GPU milliseconds (timestamp queries)
 * gbuffer_debug = false  # draw the G-buffer + one of its channels instead of the shaded scene
 * taa = false            # temporal anti-aliasing (jitter + resolved history)
 * rt_shadows = false    # ray-traced sun shadows (needs a device with ray queries; else ignored)
 * rt_mask_bake = false  # bake alphaMode MASK into the acceleration structures (off: the rule measured worse)
 * furnace = false       # verification mode: sun off, environment a constant, so the answer is analytic
 * animation_time = -1.0 # pin a keyframe animation at N seconds (-1 = play it; playback is wall-clock)
 * gbuffer_channel = 1    # which channel: 0 albedo, 1 normal, 2 roughness, 3 metallic, 4 ao, 5 id,
 *                        # 6 depth, 7 flags, 8 motion (the motion vector, amplified - see the shader)
 * validation_layers = true  # Vulkan validation layers + debug messenger (Debug builds default on, Release off)
 *
 * [gui]
 * show         = true    # show the Dear ImGui debug overlay by default
 * panel_width  = 380     # default debug-panel width (0 = auto-size)
 * panel_height = 140     # default debug-panel height (0 = auto-size)
 *
 * [lighting]
 * env_size     = 256   # environment cubemap size
 * env_mip_count = 5    # prefiltered env mip chain length
 * irr_size     = 32    # irradiance cubemap size
 * lut_size     = 256   # BRDF LUT size
 * @endcode
 */
namespace app_config {
#ifdef NDEBUG
    // Release builds default validation layers OFF (historic behavior); enable them when needed
    // via [render] validation_layers = true (e.g. debugging in a Release build).
    inline constexpr bool default_validation_layers = false;
#else
    // Debug builds keep the validation layers + debug messenger ON by default (historic
    // behavior); the config can turn them off for raw performance.
    inline constexpr bool default_validation_layers = true;
#endif

    export struct path_settings {
        std::string shaders_dir = {};    // shader SPIR-V dir (empty = auto-locate "shaders/" upward)
        std::string model_dir = {};      // default model dir used when model is empty (auto-locate gltf_model/ if empty)
        std::string screenshot_dir = {}; // F12 screenshot base directory (empty = current working directory)
    };

    /**
     * @ingroup app_config
     * @brief Vulkan/render preferences consumed by the runtime/core (applied via a create_info
     *        that the runtime/core layers add); parsed here but not interpreted by app_config.
     */
    export struct render_settings {
        int window_width = 1080;
        int window_height = 960;
        std::string window_title = "vulkan_render";               // GLFW window title
        bool vsync = true;                                        // true = FIFO_LATEST_READY (FIFO fallback), false = mailbox (uncapped)
        double max_fps = 0.0;                                     // 0 = uncapped; a positive value caps the render loop
        std::array<float, 3> clear_color = {0.02f, 0.02f, 0.03f}; // background clear color (RGB, 0..1)
        // Initial camera framing ([render] camera_fit): how main() aims the orbit camera at the
        // imported scene. "exterior" (default) fits the WHOLE model from outside - the right answer
        // for a compact object, and what every earlier version did. "interior" stands inside the
        // scene and looks along its longest horizontal axis, because that is the axis a hall, a nave
        // or a corridor runs down; a building framed from outside is a facade and nothing else, which
        // makes it useless as a global-illumination reference (there is no interior to bounce in).
        // See the log's "initial camera" line for the numbers this resolves to, and use
        // --capture-camera to pin an exact pose.
        std::string camera_fit = "exterior";
        // A PINNED initial camera pose: yaw (deg), pitch (deg), distance, target x, y, z - the same six
        // numbers `--capture-camera` takes and the same six the log prints (`camera pose: ...`), so a view
        // can be reproduced from a config, pasted between the two, or reported by a user. ABSENT by
        // default, which leaves `camera_fit` in charge; when present it wins over the fit and is itself
        // overridden by `--capture-camera`.
        std::array<float, 6> camera_pose = {};
        bool camera_pose_set = false;
        bool shadow = true; // record the directional shadow pass each frame
        // Cascaded shadow maps ([render] shadow_cascades / shadow_cascade_blend): how many cascades the
        // shadow pass fits, renders and samples (1 = one box over the whole visible range, the historic
        // single-map behavior) and the fraction of a cascade's range over which the shader blends into
        // the next one. 3 by default: the cheapest point where the near range stops paying for the far
        // range's texel size. Applied BEFORE the scene import - see runtime::set_shadow_cascades.
        int shadow_cascades = 3;
        float shadow_cascade_blend = 0.1f;
        // Shadow depth bias ([render] shadow_bias_constant / shadow_bias_slope): the rasterization bias the
        // shadow pass pushes a caster's depth by, which is what keeps a lit surface from shadowing itself
        // (acne). The defaults are the runtime's own (runtime::shadow_depth_bias_constant/_slope), so a config
        // that omits them behaves exactly as before they existed. `slope` scales with the surface's depth
        // gradient and is the one that usually matters; `constant` is a flat offset. The GUI panel exposes the
        // same two values as sliders.
        float shadow_bias_constant = 0.0f;
        float shadow_bias_slope = 1.5f;
        // ---- ZZZ-style NPR ([render] toon_steps / toon_softness / toon_shadow_tint / toon_rim): the
        // cel/toon path plus the two knobs that XIYAG's ZZZ shader adds to it - a diffuse WARP towards a
        // tinted shadow colour, and a view-space rim (see docs/zzz_shading.md for the credit and for
        // what this engine's version leaves out). EVERY DEFAULT HERE IS THE NEUTRAL VALUE: 0 steps, a
        // (1,1,1) tint, a 0 rim - so a config that omits all four shades exactly as the renderer did
        // before these keys existed, which is what the gate's references rely on. `toon_steps` is
        // matched into the band counts the overlay offers (main.cpp), so a value the combo does not
        // offer lands on the nearest one.
        int toon_steps = 0;
        float toon_softness = 0.15f;
        std::array<float, 3> toon_shadow_tint = {1.0f, 1.0f, 1.0f};
        // The linear exposure scale applied before the tonemapper (README's "Exposure"), which used to
        // be reachable only through the GUI slider. It is a config key now because matching a reference
        // render needs it reproducible: it decides how far into the tonemapper's flat, desaturating
        // region the lit side sits, which is measurable (see docs/zzz_shading.md).
        float exposure = 1.0f;
        // A scale on the sun's radiance (the shading path's constant 7.5). The reference render's lit side is
        // cooler than ours in the tonemapper's sense - ours sits deep in the flat, desaturating region - and
        // this is the knob for that, independent of the exposure, which scales the whole frame.
        float sun_intensity = 1.0f;
        // THE NON-PBR BRIGHTNESS COEFFICIENT, and the strength of the nose mark. `unlit_gain` scales every
        // surface this engine draws from its albedo rather than from the lighting stack - today that is the
        // painted face (see docs/zzz_shading.md), and the name says so rather than calling it a face knob:
        // the sun scale deliberately cannot reach these surfaces, so this is their brightness.
        float unlit_gain = 1.3f;
        float face_nose_strength = 1.0f;
        // The LIT face's brightness multiplier ([render] face_gain): a face material's lit result is scaled
        // by it, so a frame can bring the face down without touching the sun every surface shares. 1.0 is
        // neutral; the gui exposes it as "face gain".
        float face_gain = 1.0f;
        // The environment light's brightness and colour ([render] ambient_gain / ambient_tint). The ambient
        // is the sky, so it is bright and blue; these are what a frame uses to take it down and warm it.
        float ambient_gain = 1.0f;
        std::array<float, 3> ambient_tint = {1.0f, 1.0f, 1.0f};
        float toon_rim = 0.0f;
        // The reference's own two shading parameters, which a ZZZ model carries in its ILM light map and a
        // PMX does not: the band factor its five-colour shadow cascade is walked with (0 = its deepest
        // shadow colour, 1 = its lit end) and the mask on its stepped highlight term (0 = no highlight).
        float toon_shadow_band = 0.3f;
        float toon_specular = 0.0f;
        // How much of each surface's own albedo luminance is added to `toon_shadow_band`, i.e. the
        // per-texel half of the reference's light-map input (it reads its band from an ILM texture's R
        // channel; a PMX carries none). 0 = the frame-wide constant, the neutral default.
        float toon_shadow_band_gain = 0.0f;
        // ---- the OUTLINE ([render] outline_color / outline_width; see docs/zzz_shading.md): an inverted
        // hull of the scene, drawn into the G-buffer in `outline_color`, expanded by `outline_width` WORLD
        // units. Width 0 is the default and means the hull is not recorded at all, so the default frame is
        // the frame recorded before this existed. The width is absolute rather than relative to the scene,
        // which is the honest form for a per-model look: a model ten times larger wants a width ten times
        // larger, and the value that suits a given asset is measured rather than guessed (the reference
        // takes it per-vertex from the model's own edge scale instead - see the doc's "not here yet").
        std::array<float, 3> outline_color = {0.0f, 0.0f, 0.0f};
        float outline_width = 0.0f;
        // Clustered light culling ([render] clustered_lights, M5): the punctual lights are sorted
        // into a screen-tile x depth-slice grid once per frame and the shading stage loops only its
        // own cluster's list. false = the brute-force loop over every active light - the reference
        // path the clustered one is verified against (and what every pre-M5 frame did).
        bool clustered_lights = true;
        // Screen-space ambient occlusion (M6): the lighting stage traces a hemisphere of samples
        // against the G-buffer depth and scales the IBL ambient by the result. `ssao_radius` is in
        // world units (a fraction of the scene scale), `ssao_samples` is clamped to the shader
        // maximum of 16.
        // Shadow map edge length in texels ([render] shadow_map_size): 1024/2048/4096 are the usual
        // choices - resolution against the pass cost and memory (the layered map is
        // shadow_map_size^2 x 4 layers x 4 bytes per cascade set, per frame slot). Applied before the
        // scene import; the runtime clamps it to 256..8192 and rounds to a power of two.
        int shadow_map_size = 2048;
        // Stochastic PUNCTUAL lighting (docs/megalights.md): sample a few of each pixel's clustered lights, trace
        // one visibility ray per sample and add the shadowed estimate where the raster loop would have added an
        // unshadowed one. OFF by default, and deliberately: the estimator is the first stage of a chain whose
        // denoiser is not built yet, so a stock config must not inherit its raw noise.
        bool megalights = false;
        // How many samples per HALF-RESOLUTION pixel (1..4, the shader's compile-time bound). This is the knob the
        // cost and the noise both scale with, and it is the one the overlay exposes next to the switch.
        int megalights_samples = 4;
        // The chain's SPATIAL pre-filter width in half-resolution texels (0 = off, which is the temporal-only
        // chain). It is the "detail versus grain" dial: a filter that removes signal and noise at the same rate is
        // worse than none, so this one is configurable and its own measurement is in docs/megalights.md.
        float megalights_spatial_sigma = 1.5f;
        // The temporal resolve's history depth tolerance, RELATIVE to the pixel's view depth: the reject
        // threshold for "the history I reprojected belongs to this surface". UE's value is 0.03, widened at
        // grazing angles by 1/lerp(0.1, 1, N dot V) - see docs/reference/megalights_stochastic_lighting.md.
        // It is a knob because it is the balance between a stale history (ghosting) and a lost one (flicker at
        // every depth discontinuity TAA's jitter lands on the wrong side of), and the flicker measurement needs
        // to move it to attribute one to the other.
        float megalights_history_tolerance = 0.03f;
        // A scale on the ray origin's NORMAL OFFSET, which is the self-intersection guard for the visibility
        // rays (UE's mix(0.1, 0.01, N dot L) shape, in world units). 1 is the shipped pair; raising it is the
        // experiment that says whether a flickering terminator is the ray re-hitting its own surface.
        float megalights_bias = 1.0f;
        // The emitter's ANGULAR radius in radians: the soft-shadow knob, and this feature's cure for a flickering
        // hard shadow edge (see shaders/megalights_trace.comp's soft-shadow block). A point light's visibility is
        // binary, so a shadow boundary crossing the pixel flips it; an emitter with size makes the answer the
        // fraction of the emitter the pixel sees, which moves gradually. DEFAULT 0 - hard shadows, the behaviour
        // this feature shipped with - because the soft look is a choice, not a fix: measured, it does NOT reduce
        // the edge flicker this session chased (that flicker is the renderer's, and it survives every shadow
        // algorithm being off), so it stays an opt-in quality dial rather than a new default.
        float megalights_light_angle = 0.0f;
        // Ray-traced sun shadows ([render] rt_shadows): one ray per pixel against the scene's
        // acceleration structures instead of a sample of the cascaded shadow maps. Off by default, and
        // granted only on a device with ray queries - a device without them keeps running the cascaded
        // maps, which is what makes the key safe to leave in a shared config file.
        bool rt_shadows = false;
        // Bake alphaMode MASK into the acceleration structures ([render] rt_mask_bake): a compute pass
        // collapses the triangles a material's alpha covers nowhere. This is the SUBSTITUTE for an any-hit
        // stage and the shipped path no longer needs it: the shadow runs on a ray-tracing pipeline whose
        // any-hit shader cuts the mask per hit (shaders/rt_shadow.rahit). It stays because a knob is the only
        // way to measure the arms against each other. OFF BY DEFAULT, because the measurement says the
        // per-triangle rule is not good enough to be on: on a MASK-heavy sample asset it removes triangles
        // the raster path's own filtered sampling keeps, and the frame comes out 1.29 of mean brightness
        // BRIGHTER than the raster shadow it should match.
        bool rt_mask_bake = false;
        // Re-skin animated casters and refit their acceleration structures every frame ([render]
        // rt_skin_bake). The structures are built from the bind pose, so without this a ray-traced shadow of
        // an animated mesh is cast by the mesh where it is not. OFF BY DEFAULT: it is the A/B whose effect
        // the L2.2b measurement is about, and leaving it off keeps the traced shadow path byte-identical to
        // the frames every earlier measurement was taken with.
        bool rt_skin_bake = false;
        // The furnace verification mode ([render] furnace): the sun is turned off and the environment becomes
        // a constant level, so the correct frame is computable by hand - a diffuse surface's outgoing
        // radiance is exactly albedo * L, so anything added on top of it is double counting.
        // It is the one reference in this project that no estimator of its own can flatter, because it is not
        // an estimator. Off by default, and off is byte-exact.
        bool furnace = false;
        // Pin the keyframe animation at a time in seconds ([render] animation_time), or -1 to play it: a
        // NEGATIVE value is the default and plays as always. Playback is driven by the wall clock
        // (frame_clock::delta_seconds), so a capture of an animated scene is NOT reproducible - two runs of
        // one config differ, measured - and anything that has to compare two captures of one pose (the
        // skinned-mesh ray-tracing work is the first) needs this. It scrubs and pauses, which is what the
        // debug overlay's time slider does, so the pose is a function of the value alone.
        float animation_time = -1.0f;
        bool ssao = true;
        float ssao_radius = 0.5f;
        float ssao_intensity = 1.0f;
        int ssao_samples = 8;
        // Render mode ([render] unlit): the "pbr (lit)" / "unlit (flat)" combo of the debug overlay
        // as a startup setting - the flat base-color reference, useful as a shading-free view. It
        // selects the runtime's default pipeline for the transparent pass, and the lighting stage is
        // told explicitly (set_unlit) because it binds its own pipeline and cannot follow a per-leaf
        // switch: it then outputs the stored albedo instead of shading it.
        bool unlit = false;
        bool fxaa = false; // FXAA the final image (one extra fullscreen pass)
        // measure per-pass GPU time with timestamp queries: one vkCmdWriteTimestamp per pass
        // boundary, read back after the frame slot completed, averaged over a 60-frame window
        // (logged + shown in the debug overlay). A no-op on devices that cannot timestamp.
        bool gpu_timings = true;
        // G-buffer debug view ([render] gbuffer_debug / gbuffer_channel): draws the opaque scene
        // into the three G-buffer targets and shows the selected channel through the ordinary post
        // chain. A development view of the deferred path's data - the deferred lighting pass (M2)
        // takes over the display role and this stays as the inspection tool.
        bool gbuffer_debug = false;
        int gbuffer_channel = 1; // 0 albedo, 1 normal, 2 roughness, 3 metallic, 4 ao, 5 material id, 6 depth, 7 flags, 8 motion
        // Temporal anti-aliasing ([render] taa / taa_blend_static / taa_blend_min): the deferred path's
        // anti-aliasing (a G-buffer cannot be multisampled, so there is no MSAA to fall back on). The projection is jittered every frame and a resolve pass blends the
        // reprojected, neighborhood-clamped history in - see runtime::set_taa. There is no per-object
        // motion yet: the G-buffer motion vectors are camera-only at this milestone.
        bool taa = false;
        float taa_blend_static = 0.9f;                      // history weight for a pixel that did not move
        float taa_blend_min = 0.5f;                         // history weight floor under motion (lower = less ghosting)
        bool validation_layers = default_validation_layers; // Vulkan validation layers + debug messenger ([render])
    };

    /**
     * @ingroup app_config
     * @brief image-based-lighting precompute resolutions ([lighting] in the config)
     */
    export struct lighting_settings {
        int env_size = 256;    // base environment cubemap size
        int env_mip_count = 5; // prefiltered-environment mip chain length
        int irr_size = 32;     // irradiance cubemap size
        int lut_size = 256;    // BRDF LUT size
        // demo_lights ([lighting] demo_lights): spawn this many procedural punctual lights around
        // the scene (a helix at the scene bounds, cycling colors). This is the clustered-light stress
        // mode: with the debug overlay's four light slots the cluster lists and the brute-force loop
        // visit the same handful of lights, so the win is invisible. 0 (default) = overlay lights
        // only; the generated ones are pushed every frame together with the overlay's slots.
        // Capped at max_demo_lights: the light UBO holds max_punctual_lights lights in total, and the
        // overlay's own slots share that array.
        int demo_lights = 0;
        /**
         * WHERE the generated lights sit and how far they reach, both as FRACTIONS of the scene radius.
         *
         * They default to the helix the clustered-light stress mode has always used (0.85 of the scene radius
         * out, 0.55 of it as the range), so a config that does not set them renders exactly what it did
         * before. They are configurable because the two USES of this set pull in opposite directions: the
         * cluster stress wants lights AROUND the scene, where a light count the brute-force loop could not
         * afford is what shows, while measuring what a punctual light does to a frame (its shadows, and the
         * noise of a stochastic estimate of it - see docs/megalights.md) needs lights INSIDE the view, close
         * enough to matter. A radius near 0 puts them at the scene centre with a range that reaches the
         * camera's neighbourhood.
         */
        float demo_light_radius = 0.85f; // helix radius, as a fraction of the scene radius
        float demo_light_range = 0.55f;  // the lights' range, as the same fraction (0 = no cutoff)
    };

    /** @brief upper bound for [lighting] demo_lights (the UBO's light array is vulkan::max_punctual_lights) */
    export constexpr uint32_t max_demo_lights = 64;

    /**
     * @ingroup app_config
     * @brief debug-overlay panel settings ([gui] in the config)
     */
    export struct gui_settings {
        bool show = true;            // show the Dear ImGui debug overlay by default
        float panel_width = 380.0f;  // default debug-panel width (0 = ImGui auto-size)
        float panel_height = 140.0f; // default debug-panel height (0 = ImGui auto-size)
    };

    /**
     * @ingroup app_config
     * @brief the resolved startup settings after config-file + argv merging.
     * @note empty string / zero fields mean "not specified": the caller falls back to its
     *       built-in defaults, mirroring the pre-config argv behavior.
     */
    export struct app_settings {
        // model file (empty = locate default via paths.model_dir; "ask" opens the platform's own file
        // dialog at startup - see wants_model_dialog, which is the only thing that may interpret it)
        std::string model = {};
        int grid_side = 0; // > 1 enables the instancing stress grid
        path_settings paths = {};
        render_settings render = {};
        lighting_settings lighting = {};
        gui_settings gui = {};
        std::string config_file = {}; // path actually read (empty = no config file found / used)
    };

    /**
     * @ingroup app_config
     * @brief the `model` value that means "ask me with the platform's file dialog at startup"
     */
    export inline constexpr std::string_view model_ask = "ask";

    /**
     * @ingroup app_config
     * @brief whether @p settings asks for the model dialog instead of naming a file
     * @return true when `model` is exactly the @ref model_ask sentinel
     *
     * A function rather than a field, so that exactly ONE place decides what the sentinel is: a caller
     * must never write `settings.model == "ask"` itself, because the day the sentinel changes, that copy
     * opens a file called "ask" instead of asking. The comparison is exact and case-sensitive on purpose -
     * a model may legitimately be named ASK, and a near miss should be treated as the path it looks like
     * rather than as an instruction.
     */
    export bool wants_model_dialog(app_settings const& settings);

    /**
     * @ingroup app_config
     * @brief parse @p path as TOML into the settings it specifies; missing keys keep defaults.
     * @return app_settings with config_file = @p path on success; a partially filled structure
     *         with empty config_file on a read/parse error (the error is logged)
     */
    export app_settings load_settings(std::string const& path);

    /**
     * @ingroup app_config
     * @brief resolve the effective startup settings from argv: a --config <path> argument picks
     *        the config file (default: "config.toml" in the working directory if present), then
     *        positional argv values (with --config <path> consumed as an option) override the
     *        file: positional[0] = model path, positional[1] = grid side (numeric)
     * @return the merged settings (see app_settings notes for the "not specified" semantics)
     */
    export app_settings resolve_from_argv(int argc, char const* const* argv);

    /**
     * @ingroup app_config
     * @brief like resolve_from_argv() but with an explicit default config path when no --config
     *        argument is present (used when the caller does not want the cwd-relative default)
     */
    export app_settings resolve_from_argv(int argc, char const* const* argv, std::string const& default_config_path);
} // namespace app_config