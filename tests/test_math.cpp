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

    // The prefiltered environment must be SMOOTH, not just average the right colour: one coarse-level
    // texel covers a large area on screen, so an isolated bright texel (the sun disc landing on some
    // Monte-Carlo taps and missing their neighbours) reads as a big highlight patch sweeping across a
    // rotating metal surface. The variant this replaced had such texels - at 16^2 the brightest texel
    // was 7x the level mean and 2.3x its own 3x3 neighbourhood. Sampling a source mip per tap fixes
    // it, and these bounds are the regression test: they fail loudly on the old behaviour and hold
    // with room to spare on the new one.
    void test_prefiltered_environment_is_smooth() {
        constexpr int env_size = 256;
        constexpr int mip_count = 5;
        std::vector<float> const env = vulkan::generate_environment_cubemap(env_size);
        std::vector<float> const prefiltered = vulkan::prefilter_environment(env, env_size, mip_count);
        std::size_t cursor = 0;
        for (int mip = 0; mip < mip_count; ++mip) {
            int const size = std::max(1, env_size >> mip);
            auto const luminance = [&](int const face, int const x, int const y) {
                std::size_t const at = cursor + (static_cast<std::size_t>(face) * size * size + static_cast<std::size_t>(y) * size + x) * 4;
                return 0.2126 * prefiltered[at] + 0.7152 * prefiltered[at + 1] + 0.0722 * prefiltered[at + 2];
            };
            double sum = 0.0;
            double brightest = 0.0;
            double worst_local_ratio = 0.0;
            int counted = 0;
            for (int face = 0; face < 6; ++face) {
                for (int y = 1; y < size - 1; ++y) {
                    for (int x = 1; x < size - 1; ++x) {
                        double const centre = luminance(face, x, y);
                        double const neighbourhood = (luminance(face, x - 1, y - 1) + luminance(face, x, y - 1) + luminance(face, x + 1, y - 1) +
                                                      luminance(face, x - 1, y) + luminance(face, x + 1, y) +
                                                      luminance(face, x - 1, y + 1) + luminance(face, x, y + 1) + luminance(face, x + 1, y + 1)) /
                                                     8.0;
                        sum += centre;
                        counted += 1;
                        brightest = std::max(brightest, centre);
                        if (neighbourhood > 1e-6) {
                            worst_local_ratio = std::max(worst_local_ratio, centre / neighbourhood);
                        }
                    }
                }
            }
            double const mean = sum / static_cast<double>(counted);
            // Levels 0 and 1 keep the sun disc itself (a real, smooth feature); the coarse levels are
            // where an unsmoothed disc shows up as a hotspot.
            // The bounds separate the two variants by measurement, not by taste: brightest/mean per
            // coarse level was 3.93 / 3.67 / 7.10 with the old level-0 sampling and 3.10 / 2.4 / 2.5
            // with the source-mip one, the local ratio 1.20 / 1.81 / 2.27 against 1.1 / 1.1 / 1.3.
            if (mip >= 2) {
                CHECK(brightest <= 3.5 * mean);
                CHECK(worst_local_ratio <= 1.5);
            }
            cursor += static_cast<std::size_t>(6) * size * size * 4;
        }
        CHECK(cursor == prefiltered.size());
    }
} // namespace

int main() {
    test_environment_cubemap_shape();
    test_irradiance_map_shape();
    test_brdf_lut_shape();
    test_prefiltered_environment_is_smooth();
    return vk_test::finish("test_math");
}
