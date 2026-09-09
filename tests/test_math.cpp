// Headless unit tests: vulkan.math module (pure CPU - IBL precompute helpers) ===
#include "vk_test.h"

#include <cmath>
#include <cstddef>
#include <vector>

import vulkan.math;

namespace {
    constexpr std::size_t cubemap_float_count(int size) {
        return static_cast<std::size_t>(6) * static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4;
    }

    void test_environment_cubemap_shape() {
        std::vector<float> const env = vulkan::generate_environment_cubemap(16);
        CHECK(env.size() == cubemap_float_count(16));
        bool any_positive = false;
        bool all_finite = true;
        for (float const value : env) {
            all_finite &= std::isfinite(value) != 0;
            any_positive |= value > 0.0f;
        }
        CHECK(all_finite);
        CHECK(any_positive); // a light source exists somewhere in the procedural sky
    }

    void test_irradiance_map_shape() {
        std::vector<float> const env = vulkan::generate_environment_cubemap(8);
        std::vector<float> const irr = vulkan::generate_irradiance_map(env, 8, 4);
        CHECK(irr.size() == cubemap_float_count(4));
    }

    void test_brdf_lut_shape() {
        // RG32F: scale + bias per texel
        std::vector<float> const lut = vulkan::generate_brdf_lut(16);
        CHECK(lut.size() == static_cast<std::size_t>(16) * 16 * 2);
    }
} // namespace

int main() {
    test_environment_cubemap_shape();
    test_irradiance_map_shape();
    test_brdf_lut_shape();
    return vk_test::finish("test_math");
}
