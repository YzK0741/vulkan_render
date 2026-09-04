module;

#include <toml++/toml.hpp>

module app_config;

import utility;

namespace app_config {
    app_settings load_settings(std::string const& path) {
        app_settings settings;

        // toml++ parses to a parse_result when exceptions are disabled; operator bool reports
        // success and error() carries the message
        toml::parse_result const parsed = toml::parse_file(path);
        if (!parsed) {
            utility::log("app_config: cannot load '{}': {}", path, parsed.error().description());
            return settings; // config_file stays empty -> caller falls back to defaults
        }
        toml::table const& table = parsed.table();

        settings.config_file = path;
        // read keys only when present: absent keys keep the struct defaults (an empty model /
        // demo / grid_side = 0 also mean "not specified" to the caller)
        if (toml::node const* node = table.get("model")) {
            if (std::optional<std::string> const value = node->value<std::string>()) {
                settings.model = *value;
            }
        }
        if (toml::node const* node = table.get("demo")) {
            if (std::optional<std::string> const value = node->value<std::string>()) {
                settings.demo = *value;
            }
        }
        if (toml::node const* node = table.get("grid_side")) {
            if (std::optional<int64_t> const value = node->value<int64_t>()) {
                settings.grid_side = static_cast<int>(*value);
            }
        }

        if (toml::table const* paths = table.get_as<toml::table>("paths")) {
            if (toml::node const* node = paths->get("shaders_dir")) {
                if (std::optional<std::string> const value = node->value<std::string>()) {
                    settings.paths.shaders_dir = *value;
                }
            }
            if (toml::node const* node = paths->get("model_dir")) {
                if (std::optional<std::string> const value = node->value<std::string>()) {
                    settings.paths.model_dir = *value;
                }
            }
        }

        if (toml::table const* render = table.get_as<toml::table>("render")) {
            if (toml::node const* node = render->get("window_width")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.render.window_width = static_cast<int>(*value);
                }
            }
            if (toml::node const* node = render->get("window_height")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.render.window_height = static_cast<int>(*value);
                }
            }
            if (toml::node const* node = render->get("window_title")) {
                if (std::optional<std::string> const value = node->value<std::string>()) {
                    settings.render.window_title = *value;
                }
            }
            if (toml::node const* node = render->get("vsync")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.vsync = *value;
                }
            }
            if (toml::node const* node = render->get("msaa")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.render.msaa = static_cast<int>(*value);
                }
            }
            if (toml::node const* node = render->get("clear_color")) {
                if (toml::array const* color = node->as_array()) {
                    std::size_t i = 0;
                    for (toml::node const& element : *color) {
                        if (i >= settings.render.clear_color.size()) {
                            break;
                        }
                        if (std::optional<double> const channel = element.value<double>()) {
                            settings.render.clear_color[i++] = static_cast<float>(*channel);
                        }
                    }
                }
            }
            if (toml::node const* node = render->get("skybox")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.skybox = *value;
                }
            }
            if (toml::node const* node = render->get("shadow")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.shadow = *value;
                }
            }
        }

        if (toml::table const* lighting = table.get_as<toml::table>("lighting")) {
            if (toml::node const* node = lighting->get("env_size")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.lighting.env_size = static_cast<int>(*value);
                }
            }
            if (toml::node const* node = lighting->get("env_mip_count")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.lighting.env_mip_count = static_cast<int>(*value);
                }
            }
            if (toml::node const* node = lighting->get("irr_size")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.lighting.irr_size = static_cast<int>(*value);
                }
            }
            if (toml::node const* node = lighting->get("lut_size")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.lighting.lut_size = static_cast<int>(*value);
                }
            }
        }

        if (toml::table const* gui = table.get_as<toml::table>("gui")) {
            if (toml::node const* node = gui->get("panel_width")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.gui.panel_width = static_cast<float>(*value);
                }
            }
            if (toml::node const* node = gui->get("panel_height")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.gui.panel_height = static_cast<float>(*value);
                }
            }
        }
        return settings;
    }

    app_settings resolve_from_argv(int const argc, char const* const* const argv, std::string const& default_config_path) {
        // 1. Collect the non-option positional arguments (--config <path> / --config=<path> is
        //    consumed as an option, not a positional), so model/grid/demo positions stay stable
        //    regardless of where --config appears.
        std::string config_path = default_config_path;
        std::vector<std::string_view> positional;
        for (int i = 1; i < argc; ++i) {
            std::string_view const arg(argv[i]);
            if (arg == "--config") {
                if (i + 1 < argc) {
                    config_path = argv[i + 1];
                }
                ++i; // skip the option's value
                continue;
            }
            if (arg.starts_with("--config=")) {
                config_path = std::string(arg.substr(9));
                continue;
            }
            positional.push_back(arg);
        }

        app_settings settings;
        std::error_code ec;
        if (!config_path.empty() && std::filesystem::is_regular_file(config_path, ec)) {
            settings = load_settings(config_path);
        } else if (!config_path.empty()) {
            utility::log("app_config: config file '{}' not found, using defaults", config_path);
        }

        // 2. Positional argv overrides the file: [0] = model, [1] = grid side (numeric) or demo,
        //    [2] = demo.
        if (positional.size() > 0 && !positional[0].empty()) {
            settings.model = std::string(positional[0]);
        }
        if (positional.size() > 1 && !positional[1].empty()) {
            std::string const arg(positional[1]);
            char* end = nullptr;
            long const side = std::strtol(arg.c_str(), &end, 10);
            if (end != arg.c_str() && *end == '\0') {
                settings.grid_side = static_cast<int>(side); // positional[1] is a number
            } else {
                settings.demo = arg; // positional[1] is a demo name
            }
        }
        if (positional.size() > 2 && !positional[2].empty()) {
            settings.demo = std::string(positional[2]); // positional[2] is always the demo name
        }
        return settings;
    }

    app_settings resolve_from_argv(int const argc, char const* const* const argv) {
        // no explicit default path: fall back to "config.toml" in the working directory
        return resolve_from_argv(argc, argv, "config.toml");
    }
} // namespace app_config
