// ============================================================================
// module: app_config
// module version: 0.22.0  (independent of the app version in CMakeLists project(VERSION))
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

export module app_config;
export import vstd;
import utility;

/**
 * @file app_config.cppm
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
 * ssgi = false          # trace one bounce of screen-space diffuse indirect (adds to the IBL probe)
 * ssgi_intensity = 0.7  # weight on the traced indirect (it overlaps the probe; see the note below)
 * ssgi_radius = 0.12    # ray length as a fraction of the scene radius
 * ssgi_rays = 2         # rays per pixel per frame (1..16)
 * ssgi_steps = 6        # depth samples per ray (1..64)
 * ssgi_spatial_sigma = 2.0  # GI spatial filter width in GI texels; 0 = off (a pass-through)
 * ssgi_upsample = true  # joint-bilateral upsample of the half-res GI in the composite; false = bilinear
 * rt_shadows = false    # ray-traced sun shadows (needs a device with ray queries; else ignored)
 * rt_mask_bake = false  # bake alphaMode MASK into the acceleration structures (off: the rule measured worse)
 * ssgi_ray_tracing = false # trace the GI rays instead of marching the depth buffer (same conditions)
 * ssgi_bounce = 0.0     # re-emit this fraction of the previous frame's indirect at a hit (multi-bounce)
 * ssgi_probes = false   # world-space probe cache: answers for hits the screen cannot resolve
 * ssgi_hit_shading = false # shade the surface a ray hit from its geometry, not from the screen
 * furnace = false       # verification mode: sun off, environment a constant, so the answer is analytic
 * ssgi_probe_rate = 0.08 # how much of a cell one frame's observation replaces
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
        bool shadow = true; // record the directional shadow pass each frame
        // Cascaded shadow maps ([render] shadow_cascades / shadow_cascade_blend): how many cascades the
        // shadow pass fits, renders and samples (1 = one box over the whole visible range, the historic
        // single-map behavior) and the fraction of a cascade's range over which the shader blends into
        // the next one. 3 by default: the cheapest point where the near range stops paying for the far
        // range's texel size. Applied BEFORE the scene import - see runtime::set_shadow_cascades.
        int shadow_cascades = 3;
        float shadow_cascade_blend = 0.1f;
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
        // Screen-space global illumination ([render] ssgi*): one bounce of diffuse indirect, traced
        // against the depth buffer at half resolution and added by the post composite. It can only see
        // what is on screen, so it is an ADDITION to the IBL probe rather than a replacement for it -
        // which is why it has an intensity: the probe already claims some of this light, and the two
        // together over-brighten unless the screen-space part is dialled back. `ssgi_radius` is a
        // fraction of the scene radius, `ssgi_rays` x `ssgi_steps` is the cost per half-res pixel.
        // `ssgi_spatial_sigma` is the last pass's filter width: the temporal resolve averages frames,
        // this removes the spatially-fixed grain it cannot, and 0 turns it off (a pass-through), which
        // is what its effect is measured against.
        bool ssgi = false;
        float ssgi_intensity = 0.7f;
        float ssgi_radius = 0.12f;
        int ssgi_rays = 2;
        int ssgi_steps = 6;
        float ssgi_spatial_sigma = 2.0f;
        // Whether the composite upsamples that half-resolution result with a joint-bilateral gather
        // (true) or a plain bilinear fetch (false). Not a quality knob: bilinear is what the chain did
        // before the upsample existed, and keeping it reachable is what makes its effect measurable.
        bool ssgi_upsample = true;
        // Ray-traced sun shadows ([render] rt_shadows): one ray per pixel against the scene's
        // acceleration structures instead of a sample of the cascaded shadow maps. Off by default, and
        // granted only on a device with ray queries - a device without them keeps running the cascaded
        // maps, which is what makes the key safe to leave in a shared config file.
        bool rt_shadows = false;
        // Bake alphaMode MASK into the acceleration structures ([render] rt_mask_bake): a compute pass
        // collapses the triangles a material's alpha covers nowhere, so a ray-traced shadow can agree with
        // the raster one instead of treating the surface as solid. OFF BY DEFAULT, because the measurement
        // says the per-triangle rule is not good enough to be on: on a MASK-heavy sample asset it removes
        // triangles the raster path's own filtered sampling keeps, and the frame comes out 1.29 of mean
        // brightness BRIGHTER than the raster shadow it should match (docs/gi_hit_shading.md's L2.2
        // section has the numbers and the verified plumbing). It stays as an instrument: a knob is the only
        // way to measure the next attempt at the same mechanism.
        bool rt_mask_bake = false;
        // Trace the screen-space GI rays against the acceleration structures instead of marching the depth
        // buffer ([render] ssgi_ray_tracing). Same estimator, better hit oracle; ignored unless the device
        // has ray queries and ssgi itself is on.
        bool ssgi_ray_tracing = false;
        // Re-emit a fraction of the previous frame's accumulated indirect at every GI hit ([render]
        // ssgi_bounce): the multi-bounce approximation, so that a ray also carries the light that
        // already bounced once at the surface it hit. 0 - the default - is the single-bounce estimator
        // every earlier measurement was taken with. The image being fed back already carries
        // ssgi_intensity, so the loop's effective gain is this value times that one.
        float ssgi_bounce = 0.0f;
        // The world-space probe cache ([render] ssgi_probes): a persistent grid of SH-2 cells over the
        // scene's bounds - four coefficients per channel, so a cell answers for a DIRECTION - each filled by
        // tracing its OWN rays and sampled by the tracer for a hit the screen cannot resolve. The pass
        // declares no camera at all, which is what makes a cell a fact about the scene rather than about the
        // frame. Off by default, and a no-op when off - the tracer's fallback for those hits is
        // then exactly what it was. `ssgi_probe_rate` is how much of a cell one frame's observation
        // replaces (its own loop gain), `ssgi_probe_rounds` how far a frame spreads what it deposited
        // (0 = injection only, which is how the propagation is measured), and `ssgi_probe_gain` how much
        // of the cache's answer is added on top of the environment probe (0 = the cache runs, and is
        // still not sampled: that is the A/B that measures what it adds). The gain's SIGN is a second A/B:
        // negative means the same gain with the cell looked up along the opposite direction of the ray,
        // which differs from the positive one only through the cache - so the two captures were identical
        // until a cell carried a direction (measured: same SHA256), and differ on 22.7% of pixels now.
        bool ssgi_probes = false;
        // Shade the surface a GI ray hit from the geometry it landed on, instead of sampling the screen's
        // direct-radiance image there ([render] ssgi_hit_shading). Off by default; it needs the traced GI
        // path (the marched one never leaves the frame) and the acceleration structures, and it costs the
        // vertex/index/texture fetches a shaded hit makes.
        bool ssgi_hit_shading = false;
        // The furnace verification mode ([render] furnace): the sun is turned off and the environment becomes
        // a constant level, so the correct frame is computable by hand - a diffuse surface's outgoing
        // radiance is exactly albedo * L, and a GI chain that adds anything on top of it is double counting.
        // It is the one reference in this project that no estimator of its own can flatter, because it is not
        // an estimator. Off by default, and off is byte-exact.
        bool furnace = false;
        float ssgi_probe_rate = 0.08f;
        int ssgi_probe_rounds = 2;
        float ssgi_probe_gain = 1.0f;
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
        // deferred lighting ([render] deferred): the opaque scene is stored in the G-buffer and shaded
        // in screen space afterwards, through the same lighting code the forward path runs per
        // fragment; alpha-blended
        // geometry is not drawn in this mode yet (see runtime::set_deferred).
        bool deferred = false;
        // Temporal anti-aliasing ([render] taa / taa_blend_static / taa_blend_min): the deferred path's
        // anti-aliasing (a G-buffer cannot be multisampled, so there is no MSAA to fall back on). The projection is jittered every frame and a resolve pass blends the
        // reprojected, neighborhood-clamped history in - see runtime::set_taa. The forward path keeps
        // object motion yet: the G-buffer motion vectors are camera-only for now.
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
        std::string model = {}; // model file (empty = locate default via paths.model_dir)
        int grid_side = 0;      // > 1 enables the instancing stress grid
        path_settings paths = {};
        render_settings render = {};
        lighting_settings lighting = {};
        gui_settings gui = {};
        std::string config_file = {}; // path actually read (empty = no config file found / used)
    };

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
