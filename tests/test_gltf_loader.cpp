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

    // A spec-legal file whose vertex attributes are NOT float: TEXCOORD_0 is normalized u8 (core
    // glTF) and NORMAL is signed normalized s16 (KHR_mesh_quantization). The loader's interleaved
    // vertex builder used to reinterpret both as float vectors, which reads 4x (u8) or 2x (s16) past
    // the accessor, so this pins the component-type conversion AND the element-count arithmetic.
    // Reading the interleaved output (rather than the raw accessor) is the point: the raw bytes are
    // correct either way - the corruption happens on the way into the vertex struct.
    void test_quantized_attributes() {
        auto const result = gltf::load_model(VR_TEST_SOURCE_DIR "/tests/fixtures/quantized_attributes.gltf");
        CHECK(result.has_value());
        if (!result.has_value()) {
            return;
        }
        std::vector<gltf::resolved_material> const materials = gltf::resolve_materials(*result);
        gltf::drawable_iterator it(*result, materials);
        CHECK(it != gltf::drawable_iterator());
        if (!(it != gltf::drawable_iterator())) {
            return;
        }
        gltf::vertex_view const vertex = it.get_vertex();
        CHECK(vertex.count == 3);
        CHECK(vertex.stride == 64); // position(12) normal(12) uv(8) joints(16) weights(16)
        if (vertex.count != 3 || vertex.stride != 64 || vertex.data.size() < 3u * vertex.stride) {
            return;
        }
        // field offsets inside the interleaved stride
        constexpr std::size_t off_position = 0;
        constexpr std::size_t off_normal = 12;
        constexpr std::size_t off_uv = 24;
        constexpr std::size_t off_weights = 48;
        auto const at = [&vertex](std::size_t const index, std::size_t const offset) {
            return vertex.data.data() + index * vertex.stride + offset;
        };
        auto const f32 = [&at](std::size_t const index, std::size_t const offset) {
            return *reinterpret_cast<float const*>(at(index, offset));
        };

        // POSITION: float vec3, taken verbatim (NOT run through the normalized conversion)
        CHECK(f32(0, off_position + 0) == -1.0f);
        CHECK(f32(0, off_position + 4) == -1.0f);
        CHECK(f32(1, off_position + 0) == 1.0f);
        CHECK(f32(2, off_position + 4) == 1.0f);

        // NORMAL: authored (0, 0, 32767) as signed normalized s16 -> +Z. Reading those six bytes as
        // a float vec3 would produce an absurd direction, and from the second vertex on the old code
        // ran past the 18-byte accessor entirely.
        for (std::size_t i = 0; i < 3; ++i) {
            CHECK(std::abs(f32(i, off_normal + 0)) < 1e-6f);
            CHECK(std::abs(f32(i, off_normal + 4)) < 1e-6f);
            CHECK(f32(i, off_normal + 8) == 1.0f); // 32767 / 32767
        }

        // TEXCOORD_0: normalized u8 (0,0) (255,0) (0,255) -> 0.0 / 1.0, not the raw byte value and
        // not whatever float bits sat past the 6-byte accessor
        CHECK(f32(0, off_uv + 0) == 0.0f);
        CHECK(f32(0, off_uv + 4) == 0.0f);
        CHECK(f32(1, off_uv + 0) == 1.0f); // 255 / 255
        CHECK(f32(1, off_uv + 4) == 0.0f);
        CHECK(f32(2, off_uv + 0) == 0.0f);
        CHECK(f32(2, off_uv + 4) == 1.0f);

        // the identity skin fallback stays intact for an unskinned primitive
        CHECK(f32(0, off_weights + 0) == 1.0f);
        CHECK(f32(0, off_weights + 4) == 0.0f);
        CHECK(f32(0, off_weights + 8) == 0.0f);
        CHECK(f32(0, off_weights + 12) == 0.0f);
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

// ONE SKIN, TWO NODES. glTF allows it (two meshes, or one mesh instanced twice) and the runtime's
// skin binding used to rig only the first node it found. That binding needs a device, so what this
// test holds still is the DATA it depends on: both nodes reach the scene carrying skin_index 0, and
// the skin is not folded away.
void test_one_skin_used_by_two_nodes() {
    auto const result = gltf::load_model(VR_TEST_SOURCE_DIR "/tests/fixtures/shared_skin_two_nodes.gltf");
    CHECK(result.has_value());
    if (!result.has_value()) {
        return;
    }
    gltf::scenes const& scenes = *result;
    CHECK(scenes.skins.size() == 1);
    std::size_t skinned = 0;
    for (auto const& [source, loader_node] : scenes.node_by_source) {
        (void)source;
        if (loader_node->skin_index && *loader_node->skin_index == 0) {
            ++skinned;
        }
    }
    CHECK(skinned == 2); // both users of the skin, not just the first
}

// THE FACE SDF'S HEAD FRAME: the name matcher, the matrix-to-frame extraction, and the fallback. All three are
// pure, which is the point of testing them here - the model this lane was written for has NO SKELETON at all, so
// the only way the bone half can have any evidence behind it is as arithmetic rather than as pixels.
void test_the_head_frame_matcher_rejects_lookalikes() {
    // WHAT COUNTS AS A HEAD BONE, and the negative cases are the test: every rig names its head bone something
    // like these four, and every character with a hat or a hairstyle also carries `headgear`, `overhead` or
    // `Forehead` in the same skeleton. A substring test would take those.
    CHECK(gltf::looks_like_head_joint("head"));
    CHECK(gltf::looks_like_head_joint("Head"));
    CHECK(gltf::looks_like_head_joint("Bip01 Head"));
    CHECK(gltf::looks_like_head_joint("J_Head"));
    CHECK(gltf::looks_like_head_joint("Head_Nub"));
    CHECK(gltf::looks_like_head_joint("頭_01"));
    CHECK(gltf::looks_like_head_joint("头"));
    CHECK(!gltf::looks_like_head_joint("headgear"));
    CHECK(!gltf::looks_like_head_joint("overhead"));
    CHECK(!gltf::looks_like_head_joint("Forehead"));
    CHECK(!gltf::looks_like_head_joint("neck"));
    CHECK(!gltf::looks_like_head_joint(""));
}

void test_the_head_frame_from_a_bone_and_when_there_is_none() {
    // THE FALLBACK IS THE REFERENCE'S OWN CONSTANTS, not an invention, so its exact values are the assertion.
    gltf::head_basis const fallback = gltf::head_basis_fallback();
    CHECK(fallback.front == glm::vec3(0.0f, 0.0f, -1.0f));
    CHECK(fallback.right == glm::vec3(-1.0f, 0.0f, 0.0f));
    CHECK(fallback.up == glm::vec3(0.0f, 1.0f, 0.0f));
    CHECK(!fallback.from_skeleton);

    // AN IDENTITY BONE. The reference negates `row3` for forward and `row1` for right, and for the identity
    // those rows are +Z and +X - so the negated frame is the fallback above, and it must come back flagged as a
    // REAL frame rather than as the fallback, because those two are different answers that look identical.
    gltf::head_basis const identity = gltf::head_basis_from_axes(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    CHECK(identity.from_skeleton);
    CHECK(identity.front == glm::vec3(0.0f, 0.0f, -1.0f));
    CHECK(identity.right == glm::vec3(-1.0f, 0.0f, 0.0f));
    CHECK(identity.up == glm::vec3(0.0f, 1.0f, 0.0f));

    // A DEGENERATE BONE IS THE FALLBACK. An unposed or missing bone arrives as a zero matrix, and normalising
    // that would put NaNs in the shader - i.e. a black or flickering face rather than an error.
    CHECK(!gltf::head_basis_from_axes(glm::vec3(0.0f), glm::vec3(0.0f)).from_skeleton);
    // ... and so are two PARALLEL axes, where there is no third one to build.
    CHECK(!gltf::head_basis_from_axes(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(-2.0f, 0.0f, 0.0f)).from_skeleton);

    // A NON-ORTHOGONAL BONE IS RE-ORTHOGONALISED rather than taken as it is: the returned frame's three axes
    // have to BE a frame, because the SDF's angle is taken between the light and them.
    gltf::head_basis const skewed = gltf::head_basis_from_axes(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.5f));
    CHECK(skewed.from_skeleton);
    CHECK(std::abs(glm::dot(skewed.front, skewed.right)) < 1e-4f);
    CHECK(std::abs(glm::dot(skewed.front, skewed.up)) < 1e-4f);
    CHECK(std::abs(glm::dot(skewed.right, skewed.up)) < 1e-4f);
}

void test_no_head_bone_is_found_where_there_is_none() {
    // THE TWO SKINNED FIXTURES HAVE JOINTS CALLED `pole`/`arm` AND `joint` - no head - so the finder must answer
    // "no head bone" rather than take the first joint it sees, which is the failure that would matter: a face
    // shaded from an arm's frame looks like a face shaded from a head's until the arm moves.
    auto const result = gltf::load_model(VR_TEST_SOURCE_DIR "/tests/fixtures/animated_skin_plane.gltf");
    CHECK(result.has_value());
    if (!result.has_value()) {
        return;
    }
    CHECK(result->skins.size() == 1);
    CHECK(!gltf::head_joint_of(*result, 0, 0).has_value());
    // OUT-OF-RANGE ASKS NOTHING RATHER THAN CRASHING: a caller that indexes a scene or a skin that this file
    // does not have gets "no head bone", which is the same answer it would get from a model with no skeleton.
    CHECK(!gltf::head_joint_of(*result, 99, 0).has_value());
    CHECK(!gltf::head_joint_of(*result, 0, 99).has_value());
}

int main() {
    test_load_damaged_helmet();
    test_async_load_matches_sync();
    test_missing_file_reports_file_not_found();
    test_khr_lights_punctual_minimal();
    test_one_skin_used_by_two_nodes();
    test_quantized_attributes();
    test_the_head_frame_matcher_rejects_lookalikes();
    test_the_head_frame_from_a_bone_and_when_there_is_none();
    test_no_head_bone_is_found_where_there_is_none();
    return vk_test::finish("test_gltf_loader");
}
