module;

// toml++ is header-only and auto-detects -fno-exceptions (TOML_EXCEPTIONS=0),
// so including it in the global module fragment works under the project flags.
#include <toml++/toml.hpp>

export module app_config;
export import std;
import utility;

/**
 * @file app_config.cppm
 * @defgroup app_config Application Startup Config
 * @brief load vulkan_render startup settings from a TOML file, merged with command-line
 *        arguments (--config <path> overrides the default file; explicit argv values for the
 *        model / demo / grid override the file). Pure CPU, no Vulkan dependency.
 *
 * Example config.toml:
 * @code
 * # 顶层：加载的模型与演示模式
 * model = "gltf_model/DamagedHelmet.gltf"
 * demo  = ""        # spin | spin-subtree | nocull | closeup | gui (empty = none)
 * grid_side = 0     # > 1 enables the instancing stress grid (0 = off)
 *
 * [paths]
 * shaders_dir = ""  # shader SPIR-V dir (empty = auto-locate "shaders/" upward)
 * model_dir   = ""  # default model dir used when model is empty (auto-locate gltf_model/)
 *
 * [render]
 * window_width  = 1080
 * window_height = 960
 * window_title  = "vulkan_render"
 * vsync = false    # false = mailbox (current default), true = FIFO
 * msaa  = 0        # 0 = auto (device max), else a fixed sample count (4/8/...)
 * clear_color = [0.02, 0.02, 0.03]  # background clear color, RGB in 0..1
 *
 * [lighting]
 * env_size     = 256   # environment cubemap size
 * env_mip_count = 5    # prefiltered env mip chain length
 * irr_size     = 32    # irradiance cubemap size
 * lut_size     = 256   # BRDF LUT size
 * @endcode
 */
namespace app_config {
    export struct path_settings {
        std::string shaders_dir = {}; // shader SPIR-V dir (empty = auto-locate "shaders/" upward)
        std::string model_dir = {};   // default model dir used when model is empty (auto-locate gltf_model/ if empty)
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
        std::string demo = {};  // spin | spin-subtree | nocull | closeup | gui (empty = none)
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
     *        file: positional[0] = model path, positional[1] = grid side (numeric) or demo,
     *        positional[2] = demo
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
