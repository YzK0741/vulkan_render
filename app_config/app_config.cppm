// ============================================================================
// module: app_config
// module version: 0.10.0  (independent of the app version in CMakeLists project(VERSION))
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
 * vsync = false    # false = mailbox (current default), true = FIFO
 * msaa  = 0        # 0 = auto (device max), else a fixed sample count (4/8/...)
 * clear_color = [0.02, 0.02, 0.03]  # background clear color, RGB in 0..1
 * skybox = true    # draw the environment skybox pass each frame
 * shadow = true    # record the directional shadow pass each frame
 * fxaa   = false   # anti-alias the final image (adds one fullscreen pass; needs fxaa.frag.spv)
 * gpu_timings = true  # measure + report per-pass GPU milliseconds (timestamp queries)
 * gbuffer_debug = false  # draw the G-buffer + one of its channels instead of the shaded scene
deferred = false       # shade the opaque scene from the G-buffer (deferred lighting) instead of forward
taa = false            # temporal anti-aliasing on the deferred path (jitter + resolved history)
 * gbuffer_channel = 1    # which channel: 0 albedo, 1 normal, 2 roughness, 3 metallic, 4 ao, 5 id, 6 depth, 7 flags
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
        bool vsync = false;                                       // false = mailbox present mode, true = FIFO
        int msaa = 0;                                             // 0 = auto (device max usable), otherwise a fixed sample count
        std::array<float, 3> clear_color = {0.02f, 0.02f, 0.03f}; // background clear color (RGB, 0..1)
        bool skybox = true;                                       // draw the environment skybox pass each frame
        bool shadow = true;                                       // record the directional shadow pass each frame
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
        // Screen-space ambient occlusion (M6): the deferred lighting stage traces a hemisphere of
        // samples against the G-buffer depth and scales the IBL ambient by the result. `ssao_radius`
        // is in world units (a fraction of the scene scale), `ssao_samples` is clamped to the
        // shader maximum of 16. Deferred-only - the forward path stores no depth/normals to trace.
        // Shadow map edge length in texels ([render] shadow_map_size): 1024/2048/4096 are the usual
        // choices - resolution against the pass cost and memory (the layered map is
        // shadow_map_size^2 x 4 layers x 4 bytes per cascade set, per frame slot). Applied before the
        // scene import; the runtime clamps it to 256..8192 and rounds to a power of two.
        int shadow_map_size = 2048;
        bool ssao = true;
        float ssao_radius = 0.5f;
        float ssao_intensity = 1.0f;
        int ssao_samples = 8;
        // Render mode ([render] unlit): the "pbr (lit)" / "unlit (flat)" combo of the debug overlay
        // as a startup setting - the flat base-color pass, useful as a shading-free reference. It
        // selects the runtime's default pipeline, so it applies to forward geometry AND to the
        // deferred path (the lighting stage then outputs the stored albedo instead of shading it).
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
        int gbuffer_channel = 1; // 0 albedo, 1 normal, 2 roughness, 3 metallic, 4 ao, 5 material id, 6 depth, 7 flags
        // deferred lighting ([render] deferred): the opaque scene is stored in the G-buffer and shaded
        // in screen space afterwards, through the same lighting code the forward path runs per
        // fragment. The G-buffer pass is 1x whatever MSAA the forward path uses; alpha-blended
        // geometry is not drawn in this mode yet (see runtime::set_deferred).
        bool deferred = false;
        // Temporal anti-aliasing ([render] taa / taa_blend_static / taa_blend_min): the deferred path's
        // answer to MSAA. The projection is jittered every frame and a resolve pass blends the
        // reprojected, neighborhood-clamped history in - see runtime::set_taa. The forward path keeps
        // MSAA and writes no motion vectors, so TAA is deferred-only for now.
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
