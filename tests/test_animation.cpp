// Headless unit tests: vulkan.animation (pure CPU - format-neutral keyframe sampling)
// Exercises the glTF keyframe rules implemented by sample_channel / sample_node.
#include "vk_test.h"

#include <cmath>
#include <cstddef>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import vulkan.animation;

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
} // namespace

int main() {
    test_linear_translation_interpolates_and_clamps();
    test_step_holds_previous_keyframe();
    test_rotation_linear_slerps_and_normalizes();
    test_weights_linear_per_target_scalars();
    test_cubic_translation_matches_lerp_for_zero_tangents();
    test_sample_node_merges_channels_onto_base_pose();
    test_broken_samplers_report_invalid();
    return vk_test::finish("test_animation");
}
