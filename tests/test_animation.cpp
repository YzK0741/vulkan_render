// Headless unit tests: vulkan.animation (pure CPU - format-neutral keyframe sampling)
// Exercises the glTF keyframe rules implemented by sample_channel / sample_node.
#include "vk_test.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <optional>
#include <string_view>
#include <vector>

import vulkan.animation;
import vulkan.animation.mmd_motion;

namespace {
    using namespace vulkan::animation;

    [[nodiscard]] bool approx(float a, float b, float const eps = 1e-4f) {
        return std::fabs(a - b) <= eps;
    }
    [[nodiscard]] bool approx(glm::vec3 const& a, glm::vec3 const& b, float const eps = 1e-4f) {
        return approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps);
    }
    [[nodiscard]] bool approx(glm::quat const& a, glm::quat const& b, float const eps = 1e-4f) {
        return approx(a.w, b.w, eps) && approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps);
    }

    void test_linear_translation_interpolates_and_clamps() {
        sampler const s = {.times = {0.0f, 2.0f}, .values = {0.0f, 0.0f, 0.0f, 10.0f, 0.0f, 0.0f}, .per_key = 3, .interp = interpolation::linear};
        channel_sample const mid = sample_channel(s, channel_path::translation, 1.0f);
        CHECK(mid.valid);
        CHECK(approx(mid.vec3, glm::vec3(5.0f, 0.0f, 0.0f)));
        // t outside the keyframe range clamps to the nearest key
        CHECK(approx(sample_channel(s, channel_path::translation, -1.0f).vec3, glm::vec3(0.0f, 0.0f, 0.0f)));
        CHECK(approx(sample_channel(s, channel_path::translation, 3.0f).vec3, glm::vec3(10.0f, 0.0f, 0.0f)));
    }

    void test_step_holds_previous_keyframe() {
        sampler const s = {.times = {0.0f, 2.0f}, .values = {0.0f, 0.0f, 0.0f, 10.0f, 0.0f, 0.0f}, .per_key = 3, .interp = interpolation::step};
        CHECK(approx(sample_channel(s, channel_path::translation, 1.0f).vec3, glm::vec3(0.0f, 0.0f, 0.0f)));  // holds the first key
        CHECK(approx(sample_channel(s, channel_path::translation, 2.0f).vec3, glm::vec3(10.0f, 0.0f, 0.0f))); // last key at the end
    }

    void test_rotation_linear_slerps_and_normalizes() {
        // keyframe values are stored (x, y, z, w): identity -> +90 degrees around Y
        sampler const s = {.times = {0.0f, 1.0f},
                           .values = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.70710678f, 0.0f, 0.70710678f},
                           .per_key = 4,
                           .interp = interpolation::linear};
        channel_sample const q0 = sample_channel(s, channel_path::rotation, 0.0f);
        CHECK(approx(q0.quat, glm::quat(1.0f, 0.0f, 0.0f, 0.0f)));
        // half-way = 45 degrees around Y: y = sin(22.5), w = cos(22.5)
        channel_sample const mid = sample_channel(s, channel_path::rotation, 0.5f);
        CHECK(approx(mid.quat.y, 0.38268343f, 1e-3f));
        CHECK(approx(mid.quat.w, 0.92387953f, 1e-3f));
        CHECK(approx(glm::length(mid.quat), 1.0f, 1e-4f)); // always normalized
        channel_sample const q1 = sample_channel(s, channel_path::rotation, 1.0f);
        CHECK(approx(q1.quat, glm::quat(0.70710678f, 0.0f, 0.70710678f, 0.0f)));
    }

    void test_weights_linear_per_target_scalars() {
        sampler const s = {.times = {0.0f, 1.0f}, .values = {0.0f, 0.0f, 1.0f, 0.5f}, .per_key = 2, .interp = interpolation::linear};
        channel_sample const mid = sample_channel(s, channel_path::weights, 0.5f);
        CHECK(mid.valid);
        CHECK(mid.scalars.size() == 2);
        CHECK(approx(mid.scalars[0], 0.5f));
        CHECK(approx(mid.scalars[1], 0.25f));
    }

    void test_cubic_translation_matches_lerp_for_zero_tangents() {
        // per cubic key: [in tangent (3), value (3), out tangent (3)]
        sampler const s = {.times = {0.0f, 1.0f},
                           .values = {0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                      0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                           .per_key = 3,
                           .interp = interpolation::cubic_spline};
        channel_sample const mid = sample_channel(s, channel_path::translation, 0.5f);
        CHECK(mid.valid);
        CHECK(approx(mid.vec3.x, 3.5f)); // zero tangents reduce the Hermite spline to a lerp
    }

    void test_sample_node_merges_channels_onto_base_pose() {
        clip const c = {.name = "test",
                        .samplers = {sampler{.times = {0.0f, 1.0f}, .values = {0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f}, .per_key = 3, .interp = interpolation::linear}},
                        .channels = {channel{.path = channel_path::translation, .sampler = 0, .target_node = 0}}};
        node_pose const base = {.translation = glm::vec3(1.0f, 2.0f, 3.0f)};

        node_pose const posed = sample_node(c, 0, base, 0.5f);
        CHECK(posed.any_channel);
        CHECK(posed.any_transform);
        CHECK(approx(posed.translation, glm::vec3(2.0f, 0.0f, 0.0f)));    // base overridden by the channel
        CHECK(approx(posed.rotation, glm::quat(1.0f, 0.0f, 0.0f, 0.0f))); // untouched paths keep the base
        CHECK(posed.weights.empty());

        node_pose const untouched = sample_node(c, 1, base, 0.5f); // node without a channel
        CHECK(!untouched.any_channel);
        CHECK(approx(untouched.translation, base.translation));
    }

    void test_broken_samplers_report_invalid() {
        sampler const empty = {.times = {}, .values = {}, .per_key = 3, .interp = interpolation::linear};
        CHECK(!sample_channel(empty, channel_path::translation, 0.0f).valid);
        sampler const short_values = {.times = {0.0f, 1.0f}, .values = {1.0f}, .per_key = 3, .interp = interpolation::linear};
        CHECK(!sample_channel(short_values, channel_path::translation, 0.5f).valid);
    }

    // --- vulkan.animation.mmd_motion (VMD) ------------------------------------------------
    // The two format details below are the ones that fail silently on a real file, so the
    // synthetic motion writes decoys exactly where a wrong reader would look: bytes 2 and 3 of
    // the interpolation block (which hold the physics flags, not Z's and R's first control
    // point) and the 9-byte property-keyframe header that precedes the IK state list.

    /** @brief a minimal well-formed VMD 0002 motion, built byte by byte */
    [[nodiscard]] std::vector<std::uint8_t> build_vmd() {
        std::vector<std::uint8_t> d;
        auto push = [&d](std::size_t const count, std::uint8_t const value) {
            for (std::size_t i = 0; i < count; ++i) {
                d.push_back(value);
            }
        };
        auto text = [&d](std::string_view const s, std::size_t const width) {
            for (char const c : s) {
                d.push_back(static_cast<std::uint8_t>(c));
            }
            for (std::size_t i = s.size(); i < width; ++i) {
                d.push_back(0);
            }
        };
        auto u32 = [&d](std::uint32_t const value) {
            d.push_back(static_cast<std::uint8_t>(value & 0xffu));
            d.push_back(static_cast<std::uint8_t>((value >> 8u) & 0xffu));
            d.push_back(static_cast<std::uint8_t>((value >> 16u) & 0xffu));
            d.push_back(static_cast<std::uint8_t>((value >> 24u) & 0xffu));
        };
        auto f32 = [&u32](float const value) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            u32(bits);
        };
        // MMD's default linear curve on all four channels, at the offsets the format uses; the
        // physics flags are decoys that a uniform-stride reader would mistake for Z and R.
        auto interp_block = [&](std::uint8_t const physics_lo, std::uint8_t const physics_hi) {
            std::vector<std::uint8_t> block(64, 0);
            block[0] = 20;
            block[4] = 20;
            block[8] = 107;
            block[12] = 107; // X
            block[1] = 20;
            block[5] = 20;
            block[9] = 107;
            block[13] = 107; // Y
            block[17] = 20;
            block[6] = 20;
            block[10] = 107;
            block[14] = 107; // Z
            block[18] = 20;
            block[7] = 20;
            block[11] = 107;
            block[15] = 107; // R
            block[2] = physics_lo;
            block[3] = physics_hi;
            for (std::uint8_t const byte : block) {
                d.push_back(byte);
            }
        };
        auto bone = [&](std::string_view const n, std::uint32_t const frame, float const px, float const py, float const pz) {
            text(n, 15);
            u32(frame);
            f32(px);
            f32(py);
            f32(pz);
            f32(0.0f);
            f32(0.0f);
            f32(0.0f);
            f32(1.0f); // identity quaternion, stored x, y, z, w
            interp_block(5, 6);
        };
        auto morph = [&](std::string_view const n, std::uint32_t const frame, float const weight) {
            text(n, 15);
            u32(frame);
            f32(weight);
        };

        text("Vocaloid Motion Data 0002", 30);
        text("test_model", 20);
        u32(3); // bone keyframes
        bone("B0", 0, 0.0f, 0.0f, 0.0f);
        bone("B0", 10, 1.0f, 2.0f, 3.0f);
        bone("B1", 0, 0.0f, 0.0f, 0.0f);
        u32(2); // morph keyframes
        morph("M0", 0, 0.0f);
        morph("M0", 10, 1.0f);
        u32(0);     // camera keyframes
        u32(0);     // light keyframes
        u32(0);     // self-shadow keyframes
        u32(1);     // property keyframes
        u32(7);     //   frame
        push(1, 1); // visible
        u32(1);     //   IK state count
        text("IK0", 20);
        push(1, 1); //   IK enabled
        return d;
    }

    void test_mmd_motion_parses_synthetic_vmd() {
        std::optional<mmd_motion> const parsed = parse_mmd_motion(build_vmd());
        CHECK(parsed.has_value());
        if (!parsed.has_value()) {
            return;
        }
        mmd_motion const& m = *parsed;

        CHECK(m.model_name == "test_model");
        // the last section must consume the file exactly; a nonzero count here means the parse
        // walked the sections wrongly and quietly left bytes behind
        CHECK(m.unparsed_bytes == 0);
        CHECK(m.bones.size() == 2);
        CHECK(m.bones[0].name == "B0");
        CHECK(m.bones[0].keys.size() == 2);
        CHECK(m.bones[1].name == "B1");
        CHECK(m.last_frame == 10);
        CHECK(m.morphs.size() == 1);
        CHECK(m.morphs[0].keys.size() == 2);

        // bytes 2 and 3 of the block are the physics flags, so Z's and R's first control point
        // live at 17 and 18; the decoys in bytes 2 and 3 fail these CHECKs if a reader goes back
        // to reading every channel at a uniform stride
        mmd_bone_key const& key = m.bones[0].keys[0];
        float const lo = 20.0f / 127.0f;
        float const hi = 107.0f / 127.0f;
        CHECK(approx(key.ease_x.x1, lo));
        CHECK(approx(key.ease_y.x1, lo));
        CHECK(approx(key.ease_z.x1, lo));
        CHECK(approx(key.ease_r.x1, lo));
        CHECK(approx(key.ease_x.y1, lo));
        CHECK(approx(key.ease_z.y1, lo));
        CHECK(approx(key.ease_x.x2, hi));
        CHECK(approx(key.ease_y.x2, hi));
        CHECK(approx(key.ease_z.x2, hi));
        CHECK(approx(key.ease_r.x2, hi));
        CHECK(approx(key.ease_x.y2, hi));
        CHECK(approx(key.ease_z.y2, hi));
        CHECK(key.physics_flags == 0x0605u);

        // the IK list is preceded by a property keyframe header (frame, visibility, state count)
        // and its names are 20 bytes wide; a bare "count then 21-byte records" reader reports a
        // garbage name and an impossible frame number here
        CHECK(m.ik.size() == 1);
        CHECK(m.ik[0].name == "IK0");
        CHECK(m.ik[0].frame == 7);
        CHECK(m.ik[0].enabled);

        // default-linear easing is symmetric, so frame 5 sits exactly halfway between the keys
        mmd_pose pose;
        CHECK(m.sample_bone("B0", 5.0f, pose));
        CHECK(approx(pose.translation, glm::vec3(0.5f, 1.0f, 1.5f)));
        CHECK(approx(pose.rotation, glm::quat(1.0f, 0.0f, 0.0f, 0.0f)));
        CHECK(m.sample_bone("B0", -5.0f, pose));
        CHECK(approx(pose.translation, glm::vec3(0.0f)));
        CHECK(m.sample_bone("B0", 100.0f, pose));
        CHECK(approx(pose.translation, glm::vec3(1.0f, 2.0f, 3.0f)));
        CHECK(!m.sample_bone("missing", 0.0f, pose));

        CHECK(approx(m.sample_morph("M0", 5.0f), 0.5f));
        CHECK(approx(m.sample_morph("missing", 5.0f), 0.0f));
    }

    void test_mmd_bezier_solver() {
        mmd_bezier const linear = mmd_bezier::linear();
        CHECK(approx(linear.evaluate(0.0f), 0.0f));
        CHECK(approx(linear.evaluate(1.0f), 1.0f));
        CHECK(approx(linear.evaluate(0.5f), 0.5f));
        // A curve the test motion does not contain: every key in the real motion is default
        // linear, so the solver is pinned here against values cross-checked with an independent
        // double-precision implementation.
        mmd_bezier const curve{0.1f, 0.9f, 0.9f, 0.1f};
        CHECK(approx(curve.evaluate(0.5f), 0.5f, 1e-6f));
        CHECK(approx(curve.evaluate(0.25f), 0.44682231f, 1e-6f));
        CHECK(approx(curve.evaluate(0.75f), 0.55317769f, 1e-6f));
    }

    void test_mmd_motion_rejects_bad_input() {
        std::vector<std::uint8_t> const valid = build_vmd();
        CHECK(!parse_mmd_motion(std::vector<std::uint8_t>{}).has_value());

        std::vector<std::uint8_t> wrong_signature = valid;
        wrong_signature[0] = 'X';
        CHECK(!parse_mmd_motion(wrong_signature).has_value());

        // Every prefix must either be rejected or parse cleanly; a prefix that is accepted with
        // bytes left over would mean the section walk had silently gone wrong.
        for (std::size_t cut = 0; cut < valid.size(); cut += 7) {
            std::vector<std::uint8_t> const truncated(valid.begin(), valid.begin() + static_cast<std::ptrdiff_t>(cut));
            std::optional<mmd_motion> const r = parse_mmd_motion(truncated);
            CHECK(!r.has_value() || r->unparsed_bytes == 0);
        }

        // trailing bytes are reported rather than ignored, so a malformed file cannot look clean
        std::vector<std::uint8_t> padded = valid;
        padded.push_back(0);
        padded.push_back(0);
        padded.push_back(0);
        std::optional<mmd_motion> const r = parse_mmd_motion(padded);
        CHECK(r.has_value());
        CHECK(r.has_value() && r->unparsed_bytes == 3);
    }

    // The 20 MB motion this parser was written against lives outside the repo, so the check that
    // needs it is opt-in: set VR_MMD_MOTION to its path to run it.
    void test_mmd_motion_real_file_when_available() {
        char const* const path = std::getenv("VR_MMD_MOTION");
        if (path == nullptr || *path == '\0') {
            std::println("  (VR_MMD_MOTION is not set - the real-motion check was skipped)");
            return;
        }
        std::optional<mmd_motion> const m = load_mmd_motion(path);
        CHECK_MSG(m.has_value(), "VR_MMD_MOTION is set but the file could not be parsed");
        if (!m.has_value()) {
            return;
        }
        std::size_t bone_keys = 0;
        for (auto const& track : m->bones) {
            bone_keys += track.keys.size();
        }
        std::size_t morph_keys = 0;
        for (auto const& track : m->morphs) {
            morph_keys += track.keys.size();
        }
        CHECK(m->bones.size() == 168);
        CHECK(bone_keys == 193525);
        CHECK(m->morphs.size() == 29);
        CHECK(morph_keys == 3349);
        CHECK(m->last_frame == 3954);
        CHECK(m->unparsed_bytes == 0);
        CHECK(m->ik.size() == 4);

        // Every alias entry must match a bone in the real motion. This is the check that catches a
        // mistyped byte in the alias table: the synthetic test only compares the table with a file
        // built from the same table, so it would agree with itself.
        std::vector<std::string> aliased;
        aliased.reserve(mmd_bone_aliases().size());
        for (mmd_bone_alias const& alias : mmd_bone_aliases()) {
            aliased.emplace_back(alias.joint_name);
        }
        mmd_retarget const retarget = build_mmd_retarget(*m, aliased);
        CHECK(mmd_bone_aliases().size() == 29);
        CHECK(retarget.mapped == mmd_bone_aliases().size());
        CHECK(retarget.unmapped == m->bones.size() - mmd_bone_aliases().size());
    }

    void test_mmd_name_escape_is_an_unambiguous_literal() {
        // 右足ＩＫ in Shift-JIS. The byte right after the first escape is 'E', itself a literal hex
        // digit, so without a separator the two merge into the single escape 0x89E - which is
        // exactly what a retarget table generated from this output would then contain.
        std::string const raw("\x89\x45\x91\xab\x82\x68\x82\x6a", 8);
        std::string const escaped = escape_mmd_name(raw);
        CHECK(escaped == "\\x89\"\"E\\x91\\xab\\x82h\\x82j");

        // The invariant behind that one case: no escape may be followed by a literal hex digit.
        // An escape spans i..i+3, so the character after it is at i+4.
        for (std::size_t i = 0; i + 4 < escaped.size(); ++i) {
            if (escaped[i] != '\\' || escaped[i + 1] != 'x') {
                continue;
            }
            char const after = escaped[i + 4];
            bool const hex = (after >= '0' && after <= '9') || (after >= 'a' && after <= 'f') || (after >= 'A' && after <= 'F');
            CHECK(!hex);
        }
    }

    /** @brief a VMD holding one keyframe per given bone name and nothing else */
    [[nodiscard]] std::vector<std::uint8_t> build_vmd_named(std::vector<std::string> const& names) {
        std::vector<std::uint8_t> d;
        auto text = [&d](std::string_view const s, std::size_t const width) {
            for (char const c : s) {
                d.push_back(static_cast<std::uint8_t>(c));
            }
            for (std::size_t i = s.size(); i < width; ++i) {
                d.push_back(0);
            }
        };
        auto u32 = [&d](std::uint32_t const value) {
            for (int byte = 0; byte < 4; ++byte) {
                d.push_back(static_cast<std::uint8_t>((value >> (8 * byte)) & 0xffu));
            }
        };
        // position and weight fields are floats: writing an integer into one stores its bit
        // pattern (u32(1) reads back as 1.4e-45), which looks like a zero and not like a mistake
        auto f32 = [&u32](float const value) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            u32(bits);
        };
        text("Vocaloid Motion Data 0002", 30);
        text("named", 20);
        u32(static_cast<std::uint32_t>(names.size()));
        std::uint32_t frame = 0;
        for (auto const& name : names) {
            text(name, 15);
            u32(frame);                             // frame: one key per entry, 30 frames apart
            f32(static_cast<float>(frame) / 30.0f); // position x advances with the frame
            u32(0);
            u32(0);
            u32(0);
            u32(0);
            u32(0);
            u32(0x3f800000u); // rotation x, y, z, w = 0, 0, 0, 1
            frame += 30u;
            for (int byte = 0; byte < 64; ++byte) {
                d.push_back(0); // interpolation block
            }
        }
        u32(0); // morph keyframes
        u32(0); // camera keyframes
        u32(0); // light keyframes
        u32(0); // self-shadow keyframes
        u32(0); // property keyframes
        return d;
    }

    void test_mmd_retarget_resolves_by_alias() {
        // センター, 頭, 首 - the third one deliberately has no joint to land on
        std::vector<std::uint8_t> const bytes = build_vmd_named({
            std::string("\x83\x5a\x83\x93\x83\x5e\x81\x5b", 8), // センター
            std::string("\x93\xaa", 2),                         // 頭
            std::string("\x8e\xf1", 2),                         // 首
        });
        std::optional<mmd_motion> const motion = parse_mmd_motion(bytes);
        CHECK(motion.has_value());
        if (!motion.has_value()) {
            return;
        }
        // order matters: the joint index is the position in this list, not the alias table's
        std::vector<std::string> const joints = {"mmd_head", "mmd_center"};
        mmd_retarget const retarget = build_mmd_retarget(*motion, joints);

        CHECK(retarget.joint_of_bone.size() == 3);
        CHECK(retarget.joint_of(0) == 1);  // センター -> joints[1]
        CHECK(retarget.joint_of(1) == 0);  // 頭 -> joints[0]
        CHECK(retarget.joint_of(2) == -1); // 首 -> mmd_neck, which the skeleton does not have
        CHECK(retarget.mapped == 2);
        CHECK(retarget.unmapped == 1);
        CHECK(retarget.unresolved_bones.size() == 1);
        CHECK(!retarget.unresolved_bones.empty() && retarget.unresolved_bones[0] == 2);
        CHECK(retarget.joint_of(99) == -1); // out of range reads as unmapped, never out of bounds

        // a skeleton whose joints carry none of the alias names maps nothing, and says so
        mmd_retarget const empty = build_mmd_retarget(*motion, {"root", "head"});
        CHECK(empty.mapped == 0);
        CHECK(empty.unmapped == 3);
        CHECK(empty.joint_of(0) == -1);
    }
    void test_mmd_bake_feeds_the_controller() {
        // センター twice: frame 0 at x=0 and frame 30 at x=1, so the bake has a span to sample
        std::string const centre("\x83\x5a\x83\x93\x83\x5e\x81\x5b", 8);
        std::vector<std::uint8_t> const bytes = build_vmd_named({centre, centre});
        std::optional<mmd_motion> const motion = parse_mmd_motion(bytes);
        CHECK(motion.has_value());
        if (!motion.has_value()) {
            return;
        }
        mmd_retarget const retarget = build_mmd_retarget(*motion, {"mmd_center"});
        CHECK(retarget.mapped == 1);

        clip const baked = bake_mmd_clip(*motion, retarget);
        CHECK(baked.samplers.size() == 2); // translation and rotation
        CHECK(baked.channels.size() == 2);
        CHECK(baked.channels[0].target_node == 0); // the joint the retarget resolved
        CHECK(baked.channels[1].target_node == 0);
        CHECK(baked.channels[0].path == channel_path::translation);
        CHECK(baked.channels[1].path == channel_path::rotation);
        // frame 30 is 1.0 s at MMD's source rate, so 0..30 is 31 samples one frame apart
        CHECK(baked.samplers[0].times.size() == 31);
        CHECK(approx(baked.samplers[0].times.back(), 1.0f));
        CHECK(baked.samplers[1].times.size() == 31);

        // Play it through the controller's own sampler, which is what consumes a clip.
        node_pose const base;
        CHECK(approx(sample_node(baked, 0, base, 0.0f).translation, glm::vec3(0.0f)));
        CHECK(approx(sample_node(baked, 0, base, 0.5f).translation, glm::vec3(0.5f, 0.0f, 0.0f)));
        CHECK(approx(sample_node(baked, 0, base, 1.0f).translation, glm::vec3(1.0f, 0.0f, 0.0f)));
        // past the last key it holds, like any other clip
        CHECK(approx(sample_node(baked, 0, base, 9.0f).translation, glm::vec3(1.0f, 0.0f, 0.0f)));
        // a node nothing targets stays untouched
        CHECK(!sample_node(baked, 7, base, 0.5f).any_channel);

        // Half the rate still spans the same motion, in half as many samples.
        clip const coarse =
            bake_mmd_clip(*motion, retarget, mmd_bake_options{.frames_per_second = 15.0f});
        CHECK(coarse.samplers[0].times.size() == 16);
        CHECK(approx(coarse.samplers[0].times.back(), 1.0f));
        CHECK(approx(sample_node(coarse, 0, base, 0.5f).translation, glm::vec3(0.5f, 0.0f, 0.0f)));
    }
} // namespace

int main() {
    test_linear_translation_interpolates_and_clamps();
    test_step_holds_previous_keyframe();
    test_rotation_linear_slerps_and_normalizes();
    test_weights_linear_per_target_scalars();
    test_cubic_translation_matches_lerp_for_zero_tangents();
    test_sample_node_merges_channels_onto_base_pose();
    test_broken_samplers_report_invalid();
    test_mmd_motion_parses_synthetic_vmd();
    test_mmd_bezier_solver();
    test_mmd_motion_rejects_bad_input();
    test_mmd_motion_real_file_when_available();
    test_mmd_name_escape_is_an_unambiguous_literal();
    test_mmd_retarget_resolves_by_alias();
    test_mmd_bake_feeds_the_controller();
    return vk_test::finish("test_animation");
}
