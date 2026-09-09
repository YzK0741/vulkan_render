// Headless unit tests: gltf_loader (pure CPU) =================================
// Loads the repository's DamagedHelmet sample (external .bin buffer + .jpg
// textures resolved relative to the glTF file) and checks error paths. The
// loader is pure CPU - no Vulkan anywhere.
#include "vk_test.h"

#include <cmath>
#include <cstddef>
#include <glm/glm.hpp>
#include <optional>
#include <string_view>
#include <vector>

import gltf_loader;

namespace {
    void test_khr_lights_punctual_minimal() {
        auto const result = gltf::load_model(VR_TEST_SOURCE_DIR "/tests/fixtures/lights_punctual_minimal.gltf");
        CHECK(result.has_value());
        if (!result.has_value()) {
            return;
        }
        gltf::scenes const& scenes = *result;
        CHECK(scenes.lights.size() == 3);
        if (scenes.lights.size() != 3) {
            return;
        }
        // point light: binary-clean values load exactly
        CHECK(scenes.lights[0].type == gltf::light_type::point);
        CHECK(scenes.lights[0].intensity == 2.0f);
        CHECK(scenes.lights[0].range == std::optional<float>(8.0f));
        CHECK(scenes.lights[0].color == glm::vec3(1.0f, 0.5f, 0.25f));
        // spot light: the optional cone angles are exported
        CHECK(scenes.lights[1].type == gltf::light_type::spot);
        CHECK(scenes.lights[1].spot_inner_cone.has_value());
        CHECK(scenes.lights[1].spot_outer_cone.has_value());
        if (scenes.lights[1].spot_outer_cone) {
            CHECK(std::abs(*scenes.lights[1].spot_outer_cone - 0.5f) < 1e-5f);
        }
        // directional light
        CHECK(scenes.lights[2].type == gltf::light_type::directional);
        CHECK(scenes.lights[2].intensity == 4.0f);

        // every KHR node attachment landed on a node (light_index into scenes.lights)
        CHECK(scenes.scene[0].nodes.size() == 4);
        std::size_t attached = 0;
        for (gltf::node const& node : scenes.scene[0].nodes) {
            if (node.light_index) {
                ++attached;
                CHECK(*node.light_index < scenes.lights.size());
            }
        }
        CHECK(attached == 3);
    }

    void test_load_damaged_helmet() {
        auto const result = gltf::load_model(VR_TEST_SOURCE_DIR "/gltf_model/DamagedHelmet.gltf");
        CHECK(result.has_value());
        if (!result.has_value()) {
            return;
        }
        gltf::scenes const& scenes = *result;

        // the sample has one material, several textures and one mesh with one primitive
        CHECK(scenes.materials.size() == 1);
        CHECK(!scenes.textures.empty());
        CHECK(!scenes.scene.empty());
        CHECK(!scenes.scene[0].nodes.empty());
        CHECK(!scenes.node_by_source.empty());

        // world AABB over every drawable primitive
        gltf::scene_bounds const bounds = gltf::compute_scene_bounds(scenes);
        CHECK(bounds.valid);
        CHECK(bounds.primitive_count >= 1);
        CHECK(bounds.min.x <= bounds.max.x);
        CHECK(bounds.min.y <= bounds.max.y);
        CHECK(bounds.min.z <= bounds.max.z);
        // the helmet fits in a modest box around the origin (loose sanity bound)
        CHECK(bounds.max.x - bounds.min.x < 100.0f);
        CHECK(bounds.max.y - bounds.min.y < 100.0f);
        CHECK(bounds.max.z - bounds.min.z < 100.0f);

        // the drawable iterator yields exactly the primitives the bounds counted
        std::size_t drawables = 0;
        for (auto it = scenes.begin(); it != gltf::scenes::end(); ++it) {
            ++drawables;
        }
        CHECK(drawables == bounds.primitive_count);

        // renderer-ready materials resolve one per source material
        std::vector<gltf::resolved_material> const resolved = gltf::resolve_materials(scenes);
        CHECK(resolved.size() == scenes.materials.size());
    }

    void test_async_load_matches_sync() {
        auto future = gltf::load_model_async(VR_TEST_SOURCE_DIR "/gltf_model/DamagedHelmet.gltf");
        auto const result = future.get();
        CHECK(result.has_value());
        if (result.has_value()) {
            gltf::scene_bounds const bounds = gltf::compute_scene_bounds(*result);
            CHECK(bounds.valid);
            CHECK(bounds.primitive_count >= 1);
        }
    }

    void test_missing_file_reports_file_not_found() {
        auto const result = gltf::load_model(VR_TEST_SOURCE_DIR "/tests/fixtures/definitely_missing.gltf");
        CHECK(!result.has_value());
        CHECK(result.error() == gltf::error_code::file_not_found);
    }
} // namespace

int main() {
    test_load_damaged_helmet();
    test_async_load_matches_sync();
    test_missing_file_reports_file_not_found();
    test_khr_lights_punctual_minimal();
    return vk_test::finish("test_gltf_loader");
}
