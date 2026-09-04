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
 * model = "gltf_model/DamagedHelmet.gltf"
 * demo  = ""        # spin | spin-subtree | nocull | closeup | gui (empty = none)
 * grid_side = 0     # > 1 enables the instancing stress grid (0 = off)
 * shaders_dir = ""  # shader SPIR-V directory (empty = auto-locate "shaders/" upward)
 * model_dir = ""    # default model dir used when model is empty (empty = auto-locate gltf_model/)
 *
 * [render]
 * window_width  = 1080
 * window_height = 960
 * window_title  = "vulkan"
 * vsync = false    # false = mailbox (current default), true = FIFO
 * msaa  = 0        # 0 = auto (device max), else a fixed sample count (4/8/...)
 * @endcode
 */
namespace app_config {
    /**
     * @ingroup app_config
     * @brief Vulkan/render preferences consumed by the runtime/core (applied via a create_info
     *        that the runtime/core layers add); parsed here but not interpreted by app_config.
     */
    export struct render_settings {
        int window_width = 1080;
        int window_height = 960;
        std::string window_title = "vulkan_render"; // GLFW window title
        bool vsync = false;                         // false = mailbox present mode, true = FIFO
        int msaa = 0;                               // 0 = auto (device max usable), otherwise a fixed sample count
    };

    /**
     * @ingroup app_config
     * @brief the resolved startup settings after config-file + argv merging.
     * @note empty string / zero fields mean "not specified": the caller falls back to its
     *       built-in defaults, mirroring the pre-config argv behavior.
     */
    export struct app_settings {
        std::string model = {};       // model file (empty = locate default, e.g. gltf_model/DamagedHelmet.gltf)
        std::string demo = {};        // spin | spin-subtree | nocull | closeup | gui (empty = none)
        int grid_side = 0;            // > 1 enables the instancing stress grid
        std::string shaders_dir = {}; // shader directory (empty = auto-locate "shaders/" upward from cwd)
        std::string model_dir = {};   // default model directory (used when model is empty; auto-locate gltf_model/ if empty)
        render_settings render = {};
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
