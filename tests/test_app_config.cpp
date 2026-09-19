// Headless unit tests: app_config (pure CPU) ===================================
// The test executable runs in the build's per-test scratch directory (see the CMakeLists.txt
// VR_BUILD_TESTS block), so fixtures are addressed through the absolute VR_TEST_SOURCE_DIR the
// build injects rather than relative to the working directory.
#include "vk_test.h"

#include <string_view>

import application_configuration;

namespace {
    void test_load_settings_applies_toml() {
        app_config::app_settings const settings = app_config::load_settings(VR_TEST_SOURCE_DIR "/tests/fixtures/config_full.toml");
        CHECK(settings.model == "Models/tri.gltf");
        CHECK(settings.render.vsync);
        CHECK(!settings.render.shadow);
        CHECK(settings.render.shadow_cascades == 2);
        CHECK(settings.render.shadow_cascade_blend > 0.24f && settings.render.shadow_cascade_blend < 0.26f);
        CHECK(settings.render.shadow_map_size == 1024);
        CHECK(!settings.render.clustered_lights); // fixture turns the M5 cluster pass off
        CHECK(settings.lighting.demo_lights == 3);
        CHECK(!settings.render.ssao); // fixture turns the M6 screen-space AO off
        CHECK(settings.render.ssao_radius > 1.49f && settings.render.ssao_radius < 1.51f);
        CHECK(settings.render.ssao_intensity > 0.49f && settings.render.ssao_intensity < 0.51f);
        CHECK(settings.render.ssao_samples == 4);
        // ZZZ-style NPR: this fixture is the one that proves the four keys are READ (three of them are
        // neutral in the generated defaults, so a parser that ignored them would read the same values).
        CHECK(settings.render.exposure > 0.749f && settings.render.exposure < 0.751f);
        CHECK(settings.render.sun_intensity > 0.799f && settings.render.sun_intensity < 0.801f);
        CHECK(settings.render.toon_steps == 5);
        CHECK(settings.render.toon_softness > 0.049f && settings.render.toon_softness < 0.051f);
        CHECK(settings.render.toon_shadow_tint[0] > 0.549f && settings.render.toon_shadow_tint[0] < 0.551f);
        CHECK(settings.render.toon_shadow_tint[1] > 0.499f && settings.render.toon_shadow_tint[1] < 0.501f);
        CHECK(settings.render.toon_shadow_tint[2] > 0.749f && settings.render.toon_shadow_tint[2] < 0.751f);
        CHECK(settings.render.toon_rim > 0.799f && settings.render.toon_rim < 0.801f);
        CHECK(settings.render.toon_shadow_band > 0.149f && settings.render.toon_shadow_band < 0.151f);
        CHECK(settings.render.toon_specular > 0.899f && settings.render.toon_specular < 0.901f);
        CHECK(settings.render.toon_shadow_band_gain > 0.399f && settings.render.toon_shadow_band_gain < 0.401f);
        // the outline: both keys at non-default values (the width is the one that matters - 0 is the
        // compiled default AND the generated default, so only this fixture proves it is read at all)
        CHECK(settings.render.outline_color[0] > 0.099f && settings.render.outline_color[0] < 0.101f);
        CHECK(settings.render.outline_color[1] > 0.049f && settings.render.outline_color[1] < 0.051f);
        CHECK(settings.render.outline_color[2] > 0.199f && settings.render.outline_color[2] < 0.201f);
        CHECK(settings.render.outline_width > 0.059f && settings.render.outline_width < 0.061f);
        CHECK(settings.render.rt_shadows);                                                       // fixture: the ray-traced sun shadows
        CHECK(!settings.render.rt_mask_bake);                                                    // fixture: the bake off (the A/B)
        CHECK(settings.render.rt_skin_bake);                                                     // fixture: the per-frame skin refit on
        CHECK(settings.render.animation_time > 0.74f && settings.render.animation_time < 0.76f); // fixture: a pinned pose
        CHECK(settings.render.furnace);                                                          // fixture: the analytic verification mode
        CHECK(settings.render.unlit);                                                            // fixture: the flat render mode
        CHECK(settings.render.fxaa);
        CHECK(!settings.render.gpu_timings);
        CHECK(settings.render.gbuffer_debug);
        CHECK(settings.render.gbuffer_channel == 5);
        CHECK(settings.render.taa);
        CHECK(settings.render.taa_blend_static > 0.79f && settings.render.taa_blend_static < 0.81f);
        CHECK(settings.render.taa_blend_min > 0.19f && settings.render.taa_blend_min < 0.21f);
        CHECK(!settings.gui.show);
        CHECK(settings.lighting.irr_size == 64);
        CHECK(settings.paths.shaders_dir == "shaders");
        CHECK(settings.paths.screenshot_dir == "captures");
    }

    void test_load_settings_missing_file_keeps_defaults() {
        app_config::app_settings const settings = app_config::load_settings(VR_TEST_SOURCE_DIR "/tests/fixtures/does_not_exist.toml");
        CHECK(settings.model.empty());
        CHECK(settings.render.shadow);
        CHECK(!settings.render.fxaa);
        CHECK(settings.render.gpu_timings); // default: pass timings are collected
        CHECK(!settings.render.gbuffer_debug);
        CHECK(settings.render.gbuffer_channel == 1);
        CHECK(!settings.render.taa);
        CHECK(settings.render.shadow_cascades == 3); // default: three cascades
        CHECK(settings.render.shadow_cascade_blend > 0.09f && settings.render.shadow_cascade_blend < 0.11f);
        CHECK(settings.render.clustered_lights); // default: the cluster pass runs
        CHECK(settings.lighting.demo_lights == 0);
        CHECK(settings.render.ssao); // default: screen-space AO runs
        CHECK(settings.render.ssao_samples == 8);
        CHECK(!settings.render.rt_shadows);             // default: the cascaded shadow maps, not traced rays
        CHECK(!settings.render.rt_mask_bake);           // default: OFF - the per-triangle rule measured worse than the raster path
        CHECK(!settings.render.rt_skin_bake);           // default: OFF - the bind-pose shadow is what the L2.2b baseline measures
        CHECK(settings.render.animation_time < 0.0f);   // default: animations play (no pose pinned)
        CHECK(settings.render.shadow_map_size == 2048); // default: 2048^2 per cascade layer
        CHECK(settings.gui.show);
    }

    // The documented example is what users copy: parsing it must succeed and must produce the
    // values its comments claim, or the docs and the parser have drifted apart (M7 collation).
    void test_example_config_matches_documentation() {
        app_config::app_settings const settings = app_config::load_settings(VR_TEST_SOURCE_DIR "/config.example.toml");
        CHECK(!settings.config_file.empty()); // parsed, not rejected
        CHECK(!settings.model.empty());
        CHECK(settings.render.window_width == 1080);
        CHECK(settings.render.shadow);
        CHECK(settings.render.shadow_cascades == 3);
        CHECK(settings.render.shadow_map_size == 2048);
        CHECK(!settings.render.unlit); // default: the lit PBR path
        CHECK(settings.render.clustered_lights);
        CHECK(settings.render.ssao);
        CHECK(settings.render.ssao_samples == 8);
        CHECK(!settings.render.taa);
        CHECK(settings.lighting.demo_lights == 0);
        CHECK(settings.lighting.env_size == 256);
        CHECK(settings.gui.show);
    }

    // The [lighting] resolutions feed the CPU IBL precompute, which runs BEFORE the runtime sees the
    // settings - so an unclamped out-of-range value is a crash rather than a bad-looking frame:
    // env_size = 0 reads past an empty pyramid source, a negative size becomes a huge allocation
    // request (length_error -> terminate, exceptions are off), and env_mip_count < 2 wraps the
    // sampler's "pyramid.size() - 1". Every one of these must land on its documented default.
    void test_lighting_sizes_are_clamped() {
        app_config::app_settings const settings = app_config::load_settings(VR_TEST_SOURCE_DIR "/tests/fixtures/config_bad_lighting.toml");
        CHECK(settings.lighting.env_size == 256);
        CHECK(settings.lighting.env_mip_count == 5);
        CHECK(settings.lighting.irr_size == 32);
        CHECK(settings.lighting.lut_size == 256);
    }

    // scripts/make_config.py WRITES config.toml, so every value it can emit has to be a value
    // app_config can read back - including the ones whose bounds the loader clamps. This fixture is
    // that generator's full default output (one entry per key it writes); it exists because the two
    // drifted apart once already: the generator silently omitted eleven [render] keys the parser
    // understood, and its env_mip_count default range (1..10) overlapped a value the loader rejects.
    // A key added to the generator without a parser (or a clamp added without the generator) shows
    // up here as a wrong value rather than as a user's surprise.
    void test_generated_config_parses() {
        app_config::app_settings const settings = app_config::load_settings(VR_TEST_SOURCE_DIR "/tests/fixtures/config_generated_defaults.toml");
        CHECK(!settings.config_file.empty()); // parsed, not rejected
        CHECK(settings.model == "gltf_model/DamagedHelmet.gltf");
        CHECK(settings.grid_side == 0);
        // [render] presentation
        CHECK(settings.render.window_width == 1080);
        CHECK(settings.render.window_height == 960);
        CHECK(settings.render.window_title == "vulkan_render");
        CHECK(!settings.render.vsync);       // generator default: Mailbox (uncapped)
        CHECK(settings.render.max_fps == 0); // generator default: 0 = uncapped, and the compiled default too
        CHECK(settings.render.camera_fit == "exterior");
        CHECK(!settings.render.unlit);
        // [render] shadow mapping
        CHECK(settings.render.shadow);
        CHECK(settings.render.shadow_cascades == 3);
        CHECK(settings.render.shadow_map_size == 2048);
        CHECK(settings.render.shadow_cascade_blend > 0.09f && settings.render.shadow_cascade_blend < 0.11f);
        // [render] shading + post-processing: the eleven keys the generator used to omit. TWO OF THESE
        // ARE BOOLEANS WHOSE GENERATOR DEFAULT IS false, which is the compiled default as well - so
        // `CHECK(!taa)` proves the key is ACCEPTED, not that the parser read it (a parser that ignored
        // `taa` altogether would read false too). `config_full.toml` is the fixture that proves the
        // read-back, with every one of these keys at a non-default value.
        CHECK(!settings.render.taa);
        CHECK(settings.render.taa_blend_static > 0.89f && settings.render.taa_blend_static < 0.91f);
        CHECK(settings.render.taa_blend_min > 0.49f && settings.render.taa_blend_min < 0.51f);
        CHECK(!settings.render.fxaa);
        CHECK(!settings.render.gbuffer_debug);
        CHECK(settings.render.gbuffer_channel == 1);
        CHECK(settings.render.gpu_timings);
        // [render] lights + SSAO
        CHECK(settings.render.clustered_lights);
        CHECK(settings.render.ssao);
        CHECK(settings.render.ssao_radius > 0.49f && settings.render.ssao_radius < 0.51f);
        CHECK(settings.render.ssao_intensity > 0.99f && settings.render.ssao_intensity < 1.01f);
        CHECK(settings.render.ssao_samples == 8);
        // [render] cel/toon + ZZZ-style NPR: the generator writes all four, and every one of its defaults
        // is the NEUTRAL value - which is the property that matters here, because a neutral value is what
        // leaves the shading plain PBR (see docs/zzz_shading.md and the light UBO's npr_ lanes).
        CHECK(settings.render.exposure == 1.0f);
        CHECK(settings.render.sun_intensity == 1.0f);
        CHECK(settings.render.toon_steps == 0);
        CHECK(settings.render.toon_softness > 0.14f && settings.render.toon_softness < 0.16f);
        CHECK(settings.render.toon_shadow_tint[0] == 1.0f && settings.render.toon_shadow_tint[1] == 1.0f && settings.render.toon_shadow_tint[2] == 1.0f);
        CHECK(settings.render.toon_rim == 0.0f);
        CHECK(settings.render.toon_shadow_band > 0.29f && settings.render.toon_shadow_band < 0.31f);
        CHECK(settings.render.toon_specular == 0.0f);
        CHECK(settings.render.toon_shadow_band_gain == 0.0f);
        // the outline: the generator writes both, and its defaults are the NEUTRAL pair (black, width 0),
        // which is what leaves the hull unrecorded in a frame that does not ask for one
        CHECK(settings.render.outline_color[0] == 0.0f && settings.render.outline_color[1] == 0.0f && settings.render.outline_color[2] == 0.0f);
        CHECK(settings.render.outline_width == 0.0f);
        // [render] ray tracing: written by the generator like every other switch, so it round-trips
        CHECK(!settings.render.rt_shadows);
        CHECK(!settings.render.rt_mask_bake);         // default: the mask bake is off (see the generated-defaults fixture)
        CHECK(!settings.render.rt_skin_bake);         // default: the per-frame skin refit is off (same fixture)
        CHECK(settings.render.animation_time < 0.0f); // default: -1, i.e. play (the generator writes it out)
        CHECK(!settings.render.furnace);              // default: a normal frame, not the verification mode
        // [render] validation
        CHECK(settings.render.validation_layers);
        // [gui]
        CHECK(settings.gui.show);
        CHECK(settings.gui.panel_width > 379.0f && settings.gui.panel_width < 381.0f);
        CHECK(settings.gui.panel_height > 139.0f && settings.gui.panel_height < 141.0f);
        // [lighting]: the loader's clamps must leave every generator default untouched
        CHECK(settings.lighting.env_size == 256);
        CHECK(settings.lighting.env_mip_count == 5);
        CHECK(settings.lighting.irr_size == 32);
        CHECK(settings.lighting.lut_size == 256);
    }

    void test_model_ask_sentinel_is_recognized_and_never_a_path() {
        // `model = "ask"` is the config's way of asking for the startup file dialog, and the ONLY thing that
        // may read it that way is app_config::wants_model_dialog - every other caller has to see the string
        // as the path it looks like. The near misses below are why that comparison is exact and
        // case-sensitive: a build that opened a file named "ask" instead of asking is the bug this guards.
        app_config::app_settings const asking = app_config::load_settings(VR_TEST_SOURCE_DIR "/tests/fixtures/config_ask_model.toml");
        CHECK(asking.model == "ask"); // the parser keeps the sentinel verbatim
        CHECK(app_config::wants_model_dialog(asking));

        app_config::app_settings const named = app_config::load_settings(VR_TEST_SOURCE_DIR "/tests/fixtures/config_full.toml");
        CHECK(!app_config::wants_model_dialog(named)); // a real path is a real path

        app_config::app_settings near = {};
        near.model = "ASK"; // a model may legitimately be called this
        CHECK(!app_config::wants_model_dialog(near));
        near.model = "ask.glb";
        CHECK(!app_config::wants_model_dialog(near));
        near.model = " ask";
        CHECK(!app_config::wants_model_dialog(near));
        near.model = {}; // empty means "locate the default", not "ask"
        CHECK(!app_config::wants_model_dialog(near));
    }

    void test_resolve_from_argv_merges_config_and_positional() {
        char const* argv[] = {"vk_test", "Models/tri.gltf", "3"};
        app_config::app_settings const settings =
            app_config::resolve_from_argv(3, argv, VR_TEST_SOURCE_DIR "/tests/fixtures/config_full.toml");
        CHECK(settings.model == "Models/tri.gltf");
        CHECK(settings.grid_side == 3);
    }
} // namespace

int main() {
    test_load_settings_applies_toml();
    test_example_config_matches_documentation();
    test_load_settings_missing_file_keeps_defaults();
    test_lighting_sizes_are_clamped();
    test_generated_config_parses();
    test_model_ask_sentinel_is_recognized_and_never_a_path();
    test_resolve_from_argv_merges_config_and_positional();
    return vk_test::finish("test_app_config");
}
