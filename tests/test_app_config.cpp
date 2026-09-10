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
        CHECK(settings.render.fxaa);
        CHECK(!settings.render.gpu_timings);
        CHECK(settings.render.gbuffer_debug);
        CHECK(settings.render.gbuffer_channel == 5);
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
    test_load_settings_missing_file_keeps_defaults();
    test_resolve_from_argv_merges_config_and_positional();
    return vk_test::finish("test_app_config");
}
