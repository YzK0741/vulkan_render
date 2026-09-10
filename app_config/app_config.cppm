// ============================================================================
// module: app_config
// module version: 0.1.1  (independent of the app version in CMakeLists project(VERSION))
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
        bool validation_layers = default_validation_layers;       // Vulkan validation layers + debug messenger ([render])
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
    };

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
