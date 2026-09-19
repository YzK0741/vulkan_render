module;

#include <sstream>
#include <toml++/toml.hpp>

module application_configuration;

import utility;

namespace app_config {
    app_settings load_settings(std::string const& path) {
        app_settings settings = {};

        // toml++ parses to a parse_result when exceptions are disabled; operator bool reports
        // success and error() carries the message
        toml::parse_result const parsed = toml::parse_file(path);
        if (!parsed) {
            utility::log("app_config: cannot load '{}': {}", path, parsed.error().description());
            return settings; // config_file stays empty -> caller falls back to defaults
        }
        toml::table const& table = parsed.table();

        // Dump every key the file actually carried, so "which config was loaded and what did it say" is
        // answerable from the log alone. A key that is missing from this dump fell back to the compiled-in
        // default - which is precisely what made a frame-rate cap look like a no-op: the file that was
        // loaded (the one next to the executable, not the one being edited) simply had no max_fps key.
        {
            std::ostringstream dumped;
            dumped << table;
            utility::log("app_config: '{}' contents:\n{}", path, dumped.str());
        }

        settings.config_file = path;
        // read keys only when present: absent keys keep the struct defaults (an empty model /
        // grid_side = 0 also mean "not specified" to the caller)
        if (toml::node const* node = table.get("model")) {
            if (std::optional<std::string> const value = node->value<std::string>()) {
                settings.model = *value;
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

            if (toml::node const* node = paths->get("screenshot_dir")) {
                if (std::optional<std::string> const value = node->value<std::string>()) {
                    settings.paths.screenshot_dir = *value;
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
            if (toml::node const* node = render->get("max_fps")) {
                // TOML integers are not doubles: max_fps = 60 must work as written, so try both.
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.max_fps = *value;
                } else if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.render.max_fps = static_cast<double>(*value);
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
            if (toml::node const* node = render->get("megalights")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.megalights = *value;
                }
            }
            if (toml::node const* node = render->get("megalights_samples")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.render.megalights_samples = static_cast<int>(*value);
                }
            }
            if (toml::node const* node = render->get("megalights_spatial_sigma")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.megalights_spatial_sigma = static_cast<float>(*value);
                }
            }
            if (toml::node const* node = render->get("megalights_history_tolerance")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.megalights_history_tolerance = static_cast<float>(*value);
                }
            }
            if (toml::node const* node = render->get("megalights_light_angle")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.megalights_light_angle = static_cast<float>(*value);
                }
            }
            if (toml::node const* node = render->get("megalights_bias")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.megalights_bias = static_cast<float>(*value);
                }
            }
            if (toml::node const* node = render->get("furnace")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.furnace = *value;
                }
            }
            if (toml::node const* node = render->get("rt_shadows")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.rt_shadows = *value;
                }
            }
            if (toml::node const* node = render->get("rt_mask_bake")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.rt_mask_bake = *value;
                }
            }
            if (toml::node const* node = render->get("rt_skin_bake")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.rt_skin_bake = *value;
                }
            }
            if (toml::node const* node = render->get("animation_time")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.animation_time = static_cast<float>(*value);
                }
            }
            if (toml::node const* node = render->get("camera_fit")) {
                if (std::optional<std::string> const value = node->value<std::string>()) {
                    settings.render.camera_fit = *value;
                }
            }
            if (toml::node const* node = render->get("shadow")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.shadow = *value;
                }
            }
            if (toml::node const* node = render->get("shadow_cascades")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.render.shadow_cascades = static_cast<int>(*value);
                }
            }
            if (toml::node const* node = render->get("shadow_cascade_blend")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.shadow_cascade_blend = static_cast<float>(*value);
                }
            }
            if (toml::node const* node = render->get("shadow_bias_constant")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.shadow_bias_constant = static_cast<float>(*value);
                }
            }
            if (toml::node const* node = render->get("shadow_bias_slope")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.shadow_bias_slope = static_cast<float>(*value);
                }
            }
            // ---- ZZZ-style NPR (see docs/zzz_shading.md). The bounds are the runtime setters' own
            //      (set_toon_shading / set_toon_warp), so a config cannot ask for a value the renderer
            //      would clamp anyway; the NEUTRAL values (0 steps, (1,1,1) tint, 0 rim) are what leave
            //      the shading plain PBR.
            if (toml::node const* node = render->get("toon_steps")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.render.toon_steps = static_cast<int>(std::clamp<int64_t>(*value, 0, 8));
                }
            }
            if (toml::node const* node = render->get("toon_softness")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.toon_softness = static_cast<float>(std::clamp(*value, 0.01, 0.5));
                }
            }
            if (toml::node const* node = render->get("toon_shadow_tint")) {
                if (toml::array const* tint = node->as_array()) {
                    std::size_t i = 0;
                    for (toml::node const& element : *tint) {
                        if (i >= settings.render.toon_shadow_tint.size()) {
                            break;
                        }
                        if (std::optional<double> const channel = element.value<double>()) {
                            settings.render.toon_shadow_tint[i++] = static_cast<float>(std::clamp(*channel, 0.0, 2.0));
                        }
                    }
                }
            }
            if (toml::node const* node = render->get("toon_rim")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.toon_rim = static_cast<float>(std::clamp(*value, 0.0, 2.0));
                }
            }
            // the reference's own two shading parameters (see docs/zzz_shading.md): its light-map band and
            // its highlight mask, which this engine substitutes frame-wide because a PMX carries neither
            if (toml::node const* node = render->get("exposure")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.exposure = static_cast<float>(std::clamp(*value, 0.05, 4.0));
                }
            }
            if (toml::node const* node = render->get("toon_shadow_band")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.toon_shadow_band = static_cast<float>(std::clamp(*value, 0.0, 1.0));
                }
            }
            if (toml::node const* node = render->get("toon_shadow_band_gain")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.toon_shadow_band_gain = static_cast<float>(std::clamp(*value, 0.0, 1.0));
                }
            }
            if (toml::node const* node = render->get("toon_specular")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.toon_specular = static_cast<float>(std::clamp(*value, 0.0, 2.0));
                }
            }
            // ---- the outline (see docs/zzz_shading.md): the hull's colour and its width in world units,
            //      clamped to the bounds runtime::set_outline applies (0 width = no hull at all).
            if (toml::node const* node = render->get("outline_color")) {
                if (toml::array const* color = node->as_array()) {
                    std::size_t i = 0;
                    for (toml::node const& element : *color) {
                        if (i >= settings.render.outline_color.size()) {
                            break;
                        }
                        if (std::optional<double> const channel = element.value<double>()) {
                            settings.render.outline_color[i++] = static_cast<float>(std::clamp(*channel, 0.0, 2.0));
                        }
                    }
                }
            }
            if (toml::node const* node = render->get("outline_width")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.outline_width = static_cast<float>(std::clamp(*value, 0.0, 2.0));
                }
            }
            if (toml::node const* node = render->get("clustered_lights")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.clustered_lights = *value;
                }
            }
            if (toml::node const* node = render->get("shadow_map_size")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.render.shadow_map_size = static_cast<int>(*value);
                }
            }
            if (toml::node const* node = render->get("ssao")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.ssao = *value;
                }
            }
            if (toml::node const* node = render->get("ssao_radius")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.ssao_radius = static_cast<float>(*value);
                }
            }
            if (toml::node const* node = render->get("ssao_intensity")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.ssao_intensity = static_cast<float>(*value);
                }
            }
            if (toml::node const* node = render->get("ssao_samples")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.render.ssao_samples = static_cast<int>(*value);
                }
            }
            if (toml::node const* node = render->get("unlit")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.unlit = *value;
                }
            }
            if (toml::node const* node = render->get("fxaa")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.fxaa = *value;
                }
            }
            if (toml::node const* node = render->get("taa")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.taa = *value;
                }
            }
            if (toml::node const* node = render->get("taa_blend_static")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.taa_blend_static = static_cast<float>(*value);
                }
            }
            if (toml::node const* node = render->get("taa_blend_min")) {
                if (std::optional<double> const value = node->value<double>()) {
                    settings.render.taa_blend_min = static_cast<float>(*value);
                }
            }
            if (toml::node const* node = render->get("gbuffer_debug")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.gbuffer_debug = *value;
                }
            }
            if (toml::node const* node = render->get("gbuffer_channel")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.render.gbuffer_channel = static_cast<int>(*value);
                }
            }
            if (toml::node const* node = render->get("gpu_timings")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.gpu_timings = *value;
                }
            }
            if (toml::node const* node = render->get("validation_layers")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.render.validation_layers = *value;
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
            if (toml::node const* node = lighting->get("demo_lights")) {
                if (std::optional<int64_t> const value = node->value<int64_t>()) {
                    settings.lighting.demo_lights = static_cast<int>(*value);
                }
            }
            // The generated lights' geometry (see the struct's own note): a radius and a range, both as
            // fractions of the scene radius, whose defaults are the helix this mode has always used.
            for (auto const& [key, target] : {std::pair{"demo_light_radius", &settings.lighting.demo_light_radius},
                                              std::pair{"demo_light_range", &settings.lighting.demo_light_range}}) {
                if (toml::node const* node = lighting->get(key)) {
                    if (std::optional<double> const value = node->value<double>()) {
                        *target = static_cast<float>(*value);
                    }
                }
            }
        }

        if (toml::table const* gui = table.get_as<toml::table>("gui")) {
            if (toml::node const* node = gui->get("show")) {
                if (std::optional<bool> const value = node->value<bool>()) {
                    settings.gui.show = *value;
                }
            }
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

        // Sanity-clamp numeric settings: negative/absurd values would break window/swapchain
        // creation or reserve huge buffers. Non-positive window sizes fall back to the defaults;
        // grid_side is capped (the instancing stress reserves side*side transforms).
        if (settings.render.window_width <= 0) {
            settings.render.window_width = 1080;
        }
        if (settings.render.window_height <= 0) {
            settings.render.window_height = 960;
        }
        settings.gui.panel_width = std::max(settings.gui.panel_width, 0.0f); // 0 = ImGui auto-size
        settings.gui.panel_height = std::max(settings.gui.panel_height, 0.0f);
        settings.grid_side = std::clamp(settings.grid_side, 0, 90);
        if (settings.render.shadow_cascades < 1 || settings.render.shadow_cascades > 4) {
            utility::log("app_config: invalid shadow_cascades {} (use 1..4), falling back to 3", settings.render.shadow_cascades);
            settings.render.shadow_cascades = 3;
        }
        if (settings.render.shadow_map_size < 256 || settings.render.shadow_map_size > 8192) {
            utility::log("app_config: invalid shadow_map_size {} (use 256..8192), falling back to 2048", settings.render.shadow_map_size);
            settings.render.shadow_map_size = 2048;
        }
        if (settings.render.ssao_samples < 1 || settings.render.ssao_samples > 16) {
            utility::log("app_config: invalid ssao_samples {} (use 1..16), falling back to 8", settings.render.ssao_samples);
            settings.render.ssao_samples = 8;
        }
        settings.render.ssao_radius = std::clamp(settings.render.ssao_radius, 0.0f, 100.0f);
        settings.render.ssao_intensity = std::clamp(settings.render.ssao_intensity, 0.0f, 1.0f);
        // IBL resolutions drive the CPU precompute that runs BEFORE the runtime validates anything
        // (main.cpp bakes the cubemap right after analyse_config), so an out-of-range value is not a
        // quality knob but a crash: env_size = 0 indexes an empty source buffer inside
        // build_environment_pyramid (its source_size floors at 1, so every fetch runs off the end),
        // a negative one turns "6 * size * size * 4" into a huge size_t (length_error -> terminate
        // with exceptions disabled), and env_mip_count < 2 makes the sampler's
        // "pyramid.size() - 1" wrap around. Clamp to the range the precompute actually supports.
        if (settings.lighting.env_size < 16 || settings.lighting.env_size > 4096) {
            utility::log("app_config: invalid env_size {} (use 16..4096), falling back to 256", settings.lighting.env_size);
            settings.lighting.env_size = 256;
        }
        if (settings.lighting.env_mip_count < 2 || settings.lighting.env_mip_count > 12) {
            utility::log("app_config: invalid env_mip_count {} (use 2..12), falling back to 5", settings.lighting.env_mip_count);
            settings.lighting.env_mip_count = 5;
        }
        if (settings.lighting.irr_size < 1 || settings.lighting.irr_size > 1024) {
            utility::log("app_config: invalid irr_size {} (use 1..1024), falling back to 32", settings.lighting.irr_size);
            settings.lighting.irr_size = 32;
        }
        if (settings.lighting.lut_size < 1 || settings.lighting.lut_size > 1024) {
            utility::log("app_config: invalid lut_size {} (use 1..1024), falling back to 256", settings.lighting.lut_size);
            settings.lighting.lut_size = 256;
        }
        if (settings.lighting.demo_lights < 0 || settings.lighting.demo_lights > static_cast<int>(max_demo_lights)) {
            utility::log("app_config: invalid demo_lights {} (use 0..{}), clamping", settings.lighting.demo_lights, max_demo_lights);
            settings.lighting.demo_lights = std::clamp(settings.lighting.demo_lights, 0, static_cast<int>(max_demo_lights));
        }
        // The stochastic punctual lighting's sample count, clamped to the shader's own compile-time bound (see
        // vulkan.pass.megalights_trace::max_samples): the shader's loop is bounded by that constant, so a larger
        // value here would silently do nothing rather than cost more.
        settings.render.megalights_samples = std::clamp(settings.render.megalights_samples, 1, 4);
        // 0..4: the same bound the pass clamps to, and for the same reason (past a few texels it is wider than
        // the neighbourhood it can read).
        settings.render.megalights_spatial_sigma = std::clamp(settings.render.megalights_spatial_sigma, 0.0f, 4.0f);
        settings.render.megalights_history_tolerance = std::clamp(settings.render.megalights_history_tolerance, 0.0f, 1.0f);
        settings.render.megalights_bias = std::clamp(settings.render.megalights_bias, 0.0f, 32.0f);
        settings.render.megalights_light_angle = std::clamp(settings.render.megalights_light_angle, 0.0f, 0.1f);
        if (settings.render.camera_fit != "exterior" && settings.render.camera_fit != "interior") {
            utility::log("app_config: invalid camera_fit '{}' (use exterior/interior), falling back to exterior", settings.render.camera_fit);
            settings.render.camera_fit = "exterior";
        }
        return settings;
    }

    bool wants_model_dialog(app_settings const& settings) {
        // the sentinel is compared in exactly one function on purpose (see the declaration's note)
        return settings.model == model_ask;
    }

    app_settings resolve_from_argv(int const argc, char const* const* const argv, std::string const& default_config_path) {
        // 1. Collect the non-option positional arguments (--config <path> / --config=<path> is
        //    consumed as an option, not a positional), so model/grid positions stay stable
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

        app_settings settings = {};
        std::error_code ec;
        if (!config_path.empty() && std::filesystem::is_regular_file(config_path, ec)) {
            settings = load_settings(config_path);
        } else if (!config_path.empty()) {
            utility::log("app_config: config file '{}' not found, using defaults", config_path);
        }

        // 2. Positional argv overrides the file: [0] = model, [1] = grid side (numeric).
        if (!positional.empty() && !positional[0].empty()) {
            settings.model = std::string(positional[0]);
        }
        if (positional.size() > 1 && !positional[1].empty()) {
            std::string const arg(positional[1]);
            char* end = nullptr;
            long const side = std::strtol(arg.c_str(), &end, 10);
            if (end != arg.c_str() && *end == '\0') {
                // grid side: clamp instead of trusting the raw value - a huge argv number would
                // otherwise make the instancing stress reserve enormous buffers (and strtol
                // overflow saturates to LONG_MAX, which the clamp also absorbs)
                settings.grid_side = static_cast<int>(std::clamp(side, 0L, 90L));
            } else {
                utility::log("app_config: ignoring unrecognized positional argument '{}' (expected a numeric grid side)", arg);
            }
        }
        return settings;
    }

    app_settings resolve_from_argv(int const argc, char const* const* const argv) {
        // no explicit default path: fall back to "config.toml" in the working directory
        return resolve_from_argv(argc, argv, "config.toml");
    }
} // namespace app_config