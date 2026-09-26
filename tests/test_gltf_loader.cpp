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

void test_a_head_bone_is_found_at_its_joint_index() {
    // THE SUCCESS PATH, which the other fixtures cannot exercise: both of them have joints called `pole`/`arm`
    // and `joint`, so until this file existed `head_joint_of` was only ever tested answering "no".
    //
    // THE FIXTURE IS BUILT SO THE ANSWER IS AN INDEX THAT IS NOT ZERO - its skin's joints are `neck` then
    // `Head`, so the head is joint 1. A finder that returned the first joint, or an asset node index instead of
    // a joint index, would pass a test written against joint 0 and fail here, and both of those are real
    // mistakes: the joint index is what picks a matrix out of the per-frame skin matrix array, so an off-by-one
    // shades a face from a neck.
    auto const result = gltf::load_model(VR_TEST_SOURCE_DIR "/tests/fixtures/skinned_head.gltf");
    CHECK(result.has_value());
    if (!result.has_value()) {
        return;
    }
    CHECK(result->skins.size() == 1);
    if (result->skins.size() != 1) {
        return;
    }
    CHECK(result->skins[0].joints.size() == 2);
    std::optional<std::size_t> const head = gltf::head_joint_of(*result, 0, 0);
    CHECK(head.has_value());
    CHECK(head == std::optional<std::size_t>{1});
    // AND IT IS AN INDEX INTO `joints`, NOT AN ASSET NODE INDEX: the two differ here on purpose (the head is
    // asset node 3 and joint 1), so a finder that returned the wrong one of them cannot pass both assertions.
    CHECK(result->skins[0].joints[1] != 1u);
}

void test_the_toon_family_matcher_reads_mmd_material_names() {
    // WHAT THIS GUARDS, and it is a failure with no symptom anywhere in the loader: a material whose name
    // matches no pattern is `toon_family::none`, and `none` is the family-independent toon path. A table that
    // cannot read a model's script therefore reports nothing at all - it just shades every material of that
    // model without its family, and the frame still looks like a toon character. That is exactly what
    // happened to the MMD character models this branch imports: their materials are named `颜`, `发`, `目`,
    // `眉`, `睫`, `口`, `齿`, `舌`, `鼻`, `鞋`, `裤`, while the table held the Japanese and two-character
    // spellings, so 25 of one model's 35 materials fell through to `none` - every `face` and every `hair`
    // material among them, which is what silently cost the hair its highlight and the eye its soft ramp.
    //
    // EVERY NAME BELOW IS TAKEN FROM A MODEL IN THIS REPOSITORY rather than invented. The first group is
    // `zhuangfy_toon.glb`'s own material list, in its own spellings.
    CHECK(gltf::toon_family_of("颜") == gltf::toon_family::face);
    CHECK(gltf::toon_family_of("肌上") == gltf::toon_family::skin);
    CHECK(gltf::toon_family_of("肌-手") == gltf::toon_family::skin);
    CHECK(gltf::toon_family_of("肌-耳") == gltf::toon_family::skin);
    CHECK(gltf::toon_family_of("肌下-隐藏") == gltf::toon_family::skin);
    CHECK(gltf::toon_family_of("目") == gltf::toon_family::eye);
    CHECK(gltf::toon_family_of("目白") == gltf::toon_family::eye);
    CHECK(gltf::toon_family_of("目HL") == gltf::toon_family::eye);
    CHECK(gltf::toon_family_of("眉") == gltf::toon_family::face);
    CHECK(gltf::toon_family_of("睫") == gltf::toon_family::face);
    CHECK(gltf::toon_family_of("舌") == gltf::toon_family::face);
    CHECK(gltf::toon_family_of("口线") == gltf::toon_family::face);
    CHECK(gltf::toon_family_of("鼻线") == gltf::toon_family::face);
    CHECK(gltf::toon_family_of("齿") == gltf::toon_family::face);
    CHECK(gltf::toon_family_of("二重") == gltf::toon_family::face);
    // `表情` is the expression overlay - blush and tears drawn over the same face geometry - and it is the
    // last name in these models to fall through, so it is asserted with them.
    CHECK(gltf::toon_family_of("表情") == gltf::toon_family::face);
    CHECK(gltf::toon_family_of("发") == gltf::toon_family::hair);
    CHECK(gltf::toon_family_of("前发饰") == gltf::toon_family::hair);
    CHECK(gltf::toon_family_of("发簪") == gltf::toon_family::hair);
    CHECK(gltf::toon_family_of("鞋") == gltf::toon_family::cloth);
    CHECK(gltf::toon_family_of("裤") == gltf::toon_family::cloth);
    CHECK(gltf::toon_family_of("裤-alpha") == gltf::toon_family::cloth);
    CHECK(gltf::toon_family_of("手") == gltf::toon_family::skin);

    // THE SAME WORD IN ANOTHER SCRIPT IS A DIFFERENT WORD TO A BYTE COMPARISON, so each spelling is asserted
    // on its own: a table that gained only one of a pair reads half the models it exists to serve.
    CHECK(gltf::toon_family_of("髪") == gltf::toon_family::hair);  // Japanese
    CHECK(gltf::toon_family_of("髮") == gltf::toon_family::hair);  // traditional
    CHECK(gltf::toon_family_of("顔") == gltf::toon_family::face);  // Japanese
    CHECK(gltf::toon_family_of("臉") == gltf::toon_family::face);  // traditional
    CHECK(gltf::toon_family_of("靴") == gltf::toon_family::cloth); // Japanese
    CHECK(gltf::toon_family_of("褲") == gltf::toon_family::cloth); // traditional
    CHECK(gltf::toon_family_of("襪") == gltf::toon_family::cloth);
    CHECK(gltf::toon_family_of("腳") == gltf::toon_family::skin);
    // ... AND THE FACE IS THE ONE PLACE THE LASH AND BROW LAYERS OF `zhuangfy_toon.glb` DISAGREE WITH THE
    // BARE CHARACTERS: `睫眉` is ONE material covering both, so it must still reach `face` through either.
    CHECK(gltf::toon_family_of("睫眉") == gltf::toon_family::face);

    // THE COLLISIONS THE GROUP ORDER RESOLVES, and both directions matter: `手套` (glove) must be cloth even
    // though it contains `手` (skin), and `袖口` (cuff) / `领口` (collar) must be cloth even though they
    // contain `口` (face). A table that gained the bare characters WITHOUT cloth ahead of skin and face
    // passes every assertion above and fails these three.
    CHECK(gltf::toon_family_of("手套") == gltf::toon_family::cloth);
    CHECK(gltf::toon_family_of("袖口") == gltf::toon_family::cloth);
    CHECK(gltf::toon_family_of("领口") == gltf::toon_family::cloth);

    // THE REFERENCE'S OWN NAMES ARE UNCHANGED, which is the widest risk in adding patterns: one that steals
    // a material the game data already classified moves a frame that was not supposed to move. These are
    // `actor_zhuangfy.glb`'s names, and `eyeshadow` and `tail` must stay `none` - they are the two names in
    // that file no family claims, and `eyeshadow` is the near miss that would catch a careless `eye` pattern.
    CHECK(gltf::toon_family_of("M_actor_zhuangfy_face_01") == gltf::toon_family::face);
    CHECK(gltf::toon_family_of("M_actor_zhuangfy_hair_01") == gltf::toon_family::hair);
    CHECK(gltf::toon_family_of("M_actor_zhuangfy_iris_01") == gltf::toon_family::eye);
    CHECK(gltf::toon_family_of("M_actor_zhuangfy_body_01") == gltf::toon_family::skin);
    CHECK(gltf::toon_family_of("M_actor_zhuangfy_cloth_01") == gltf::toon_family::cloth);
    CHECK(gltf::toon_family_of("M_S_actor_zhuangfy_eyebrow_01_lod0") == gltf::toon_family::face);
    CHECK(gltf::toon_family_of("M_S_actor_zhuangfy_hairshadow_01_lod0") == gltf::toon_family::hair);
    CHECK(gltf::toon_family_of("M_S_actor_zhuangfy_eyeshadow_01_lod0") == gltf::toon_family::none);
    CHECK(gltf::toon_family_of("M_S_actor_zhuangfy_tail_02_lod0") == gltf::toon_family::none);

    // AN EMPTY NAME ASKS NOTHING and an unrecognised one is `none` rather than the first pattern's family -
    // an unnamed material is common in these files and must keep the family-independent path.
    CHECK(gltf::toon_family_of("") == gltf::toon_family::none);
    CHECK(gltf::toon_family_of("biaoq") == gltf::toon_family::none);
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
    test_a_head_bone_is_found_at_its_joint_index();
    test_the_toon_family_matcher_reads_mmd_material_names();
    return vk_test::finish("test_gltf_loader");
}
