// Headless unit tests: app_config (pure CPU) ===================================
// The test executable runs with the repository root as working directory (see
// CMakeLists.txt VR_BUILD_TESTS block), so fixtures are referenced relative to it.
#include "vk_test.h"

#include <string_view>

import app_config;

namespace {
    void test_load_settings_applies_toml() {
        app_config::app_settings const settings = app_config::load_settings(VR_TEST_SOURCE_DIR "/tests/fixtures/config_full.toml");
        CHECK(settings.model == "Models/tri.gltf");
        CHECK(settings.render.msaa == 8);
        CHECK(settings.render.vsync);
        CHECK(!settings.render.skybox);
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
        CHECK(settings.render.unlit); // fixture: the flat render mode
        CHECK(settings.render.fxaa);
        CHECK(!settings.render.gpu_timings);
        CHECK(settings.render.gbuffer_debug);
        CHECK(settings.render.gbuffer_channel == 5);
        CHECK(settings.render.deferred);
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
        CHECK(settings.render.msaa == 0);
        CHECK(settings.render.skybox);
        CHECK(settings.render.shadow);
        CHECK(!settings.render.fxaa);
        CHECK(settings.render.gpu_timings); // default: pass timings are collected
        CHECK(!settings.render.gbuffer_debug);
        CHECK(settings.render.gbuffer_channel == 1);
        CHECK(!settings.render.deferred);
        CHECK(!settings.render.taa);
        CHECK(settings.render.shadow_cascades == 3); // default: three cascades
        CHECK(settings.render.shadow_cascade_blend > 0.09f && settings.render.shadow_cascade_blend < 0.11f);
        CHECK(settings.render.clustered_lights); // default: the cluster pass runs
        CHECK(settings.lighting.demo_lights == 0);
        CHECK(settings.render.ssao); // default: the deferred path traces screen-space AO
        CHECK(settings.render.ssao_samples == 8);
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
        CHECK(!settings.render.deferred);
        CHECK(!settings.render.taa);
        CHECK(settings.lighting.demo_lights == 0);
        CHECK(settings.lighting.env_size == 256);
        CHECK(settings.gui.show);
    }

    void test_resolve_from_argv_merges_config_and_positional() {
        char const* argv[] = {"vk_test", "Models/tri.gltf", "3"};
        app_config::app_settings const settings =
            app_config::resolve_from_argv(3, argv, VR_TEST_SOURCE_DIR "/tests/fixtures/config_full.toml");
        CHECK(settings.model == "Models/tri.gltf");
        CHECK(settings.grid_side == 3);
        CHECK(settings.render.msaa == 8); // still comes from the config file
    }
} // namespace

int main() {
    test_load_settings_applies_toml();
    test_example_config_matches_documentation();
    test_load_settings_missing_file_keeps_defaults();
    test_resolve_from_argv_merges_config_and_positional();
    return vk_test::finish("test_app_config");
}
