module;

#include <glm/glm.hpp>

module vulkan.math;

namespace vulkan {
    namespace {
        constexpr float k_pi = 3.14159265359f;

        // Cubemap face direction: texel (u, v) in [-1, 1] -> unit direction (Vulkan/GL cubemap convention)
        glm::vec3 cube_face_direction(int const face, float const u, float const v) {
            switch (face) {
            case 0:
                return glm::normalize(glm::vec3(1.0f, -v, -u)); // +X
            case 1:
                return glm::normalize(glm::vec3(-1.0f, -v, u)); // -X
            case 2:
                return glm::normalize(glm::vec3(u, 1.0f, v)); // +Y
            case 3:
                return glm::normalize(glm::vec3(u, -1.0f, -v)); // -Y
            case 4:
                return glm::normalize(glm::vec3(u, -v, 1.0f)); // +Z
            default:
                return glm::normalize(glm::vec3(-u, -v, -1.0f)); // -Z
            }
        }

        // Procedural environment (HDR): gradient sky/ground + sun disc.
        // Keep in sync with shaders/skybox.frag sky_color(): the visible sky is computed
        // analytically per-pixel (no cubemap sampling), so the IBL cubemap baked from this
        // function must produce exactly the same colors for reflections to match the sky.
        glm::vec3 environment_color(glm::vec3 const& dir) {
            float const t = std::clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f); // 0 nadir, 1 zenith
            auto const smooth = [](float const e0, float const e1, float const x) {
                float const u = std::clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
                return u * u * (3.0f - 2.0f * u);
            };
            constexpr glm::vec3 ground = glm::vec3(0.05f, 0.05f, 0.07f) * 0.75f;
            constexpr glm::vec3 horizon = glm::vec3(0.17f, 0.20f, 0.27f) * 0.75f;
            constexpr glm::vec3 sky = glm::vec3(0.28f, 0.45f, 0.75f) * 0.75f;
            float const g = smooth(0.28f, 0.50f, t); // ground -> horizon
            float const s = smooth(0.50f, 0.92f, t); // horizon -> sky
            glm::vec3 env = ground + (horizon - ground) * g;
            env += (sky - env) * s;
            glm::vec3 const sun_dir = glm::normalize(glm::vec3(0.3f, 1.0f, 0.5f));
            float const sun = smooth(0.98f, 1.0f, glm::dot(dir, sun_dir)); // soft-edged disc
            env += glm::vec3(1.0f, 0.95f, 0.85f) * sun * 1.5f;             // visible sun for metallic highlights
            return env;
        }

        // Cubemap face mapping: direction -> (face, u, v) with u/v in [-1, 1]. This is the one owner of
        // the convention (cube_face_direction() above is its inverse) and every sampler here goes
        // through it, so the bakes cannot drift apart from each other or from the GPU's own lookup.
        void cube_face_uv(glm::vec3 const& dir, int& face, float& u, float& v) {
            float const ax = std::abs(dir.x);
            float const ay = std::abs(dir.y);
            float const az = std::abs(dir.z);
            if (ax >= ay && ax >= az) {
                face = dir.x >= 0.0f ? 0 : 1;
                u = face == 0 ? -dir.z : dir.z;
                v = -dir.y;
            } else if (ay >= ax && ay >= az) {
                face = dir.y >= 0.0f ? 2 : 3;
                u = dir.x;
                v = face == 2 ? dir.z : -dir.z;
            } else {
                face = dir.z >= 0.0f ? 4 : 5;
                u = face == 4 ? dir.x : -dir.x;
                v = -dir.y;
            }
        }

        // Nearest-neighbor fetch of level 0. The irradiance bake and the base level use this; the
        // prefilter samples a whole source mip chain instead - see prefilter_environment.
        glm::vec3 sample_cubemap(std::span<float const> const data, int const size, glm::vec3 const& dir) {
            int face = 0;
            float u = 0.0f;
            float v = 0.0f;
            cube_face_uv(dir, face, u, v);
            int const px = std::clamp(static_cast<int>((u * 0.5f + 0.5f) * static_cast<float>(size)), 0, size - 1);
            int const py = std::clamp(static_cast<int>((v * 0.5f + 0.5f) * static_cast<float>(size)), 0, size - 1);
            size_t const offset = (static_cast<size_t>(face) * size * size + static_cast<size_t>(py) * size + px) * 4;
            return glm::vec3(data[offset], data[offset + 1], data[offset + 2]);
        }

        // Van der Corput sequence (second component of Hammersley)
        float radical_inverse_vdc(uint32_t bits) {
            bits = (bits << 16u) | (bits >> 16u);
            bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
            bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
            bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
            bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
            return static_cast<float>(bits) * 2.3283064365386963e-10f;
        }

        glm::vec2 hammersley(uint32_t const i, uint32_t const n) {
            return glm::vec2(static_cast<float>(i) / static_cast<float>(n), radical_inverse_vdc(i));
        }

        // GGX importance sampling: build the half vector from Hammersley samples
        glm::vec3 importance_sample_ggx(glm::vec2 const& xi, glm::vec3 const& n, float const roughness) {
            float const a = roughness * roughness;
            float const phi = 2.0f * k_pi * xi.x;
            float const cos_theta = std::sqrt((1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y));
            float const sin_theta = std::sqrt(std::max(1.0f - cos_theta * cos_theta, 0.0f));
            glm::vec3 const h(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
            glm::vec3 const up = std::abs(n.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
            glm::vec3 const tangent = glm::normalize(glm::cross(up, n));
            glm::vec3 const bitangent = glm::cross(n, tangent);
            return glm::normalize(tangent * h.x + bitangent * h.y + n * h.z);
        }

        // IEEE 754 binary32 -> binary16 (truncated; plenty for ambient light)
        uint16_t float_to_half(float const value) {
            uint32_t const bits = std::bit_cast<uint32_t>(value);
            uint16_t const sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
            int32_t const exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
            uint32_t const mantissa = bits & 0x7FFFFFu;
            if (exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00u); // infinity
            }
            if (exponent <= 0) {
                return sign; // subnormal/zero -> 0
            }
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
        }
    } // namespace

    std::vector<float> generate_environment_cubemap(int const size) {
        std::vector<float> data(static_cast<size_t>(6) * size * size * 4);
        for (int face = 0; face < 6; ++face) {
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    float const u = (static_cast<float>(x) + 0.5f) / static_cast<float>(size) * 2.0f - 1.0f;
                    float const v = (static_cast<float>(y) + 0.5f) / static_cast<float>(size) * 2.0f - 1.0f;
                    glm::vec3 const color = environment_color(cube_face_direction(face, u, v));
                    size_t const offset = (static_cast<size_t>(face) * size * size + static_cast<size_t>(y) * size + x) * 4;
                    data[offset + 0] = color.r;
                    data[offset + 1] = color.g;
                    data[offset + 2] = color.b;
                    data[offset + 3] = 1.0f;
                }
            }
        }
        return data;
    }

    // One box-filtered source mip chain: level k is 2^k times smaller than the environment. The
    // prefilter reads a level per sample instead of always reading level 0, which is what keeps
    // the coarse levels smooth (see prefilter_environment).
    std::vector<std::vector<float>> build_environment_pyramid(std::span<float const> const env, int const env_size, int const levels) {
        std::vector<std::vector<float>> pyramid;
        pyramid.reserve(static_cast<std::size_t>(levels));
        pyramid.emplace_back(env.begin(), env.end());
        for (int level = 1; level < levels; ++level) {
            int const source_size = std::max(1, env_size >> (level - 1));
            int const target_size = std::max(1, env_size >> level);
            std::vector<float> const& source = pyramid.back();
            std::vector<float> target(static_cast<std::size_t>(6) * target_size * target_size * 4, 0.0f);
            for (int face = 0; face < 6; ++face) {
                for (int y = 0; y < target_size; ++y) {
                    for (int x = 0; x < target_size; ++x) {
                        glm::vec4 sum(0.0f);
                        for (int dy = 0; dy < 2; ++dy) {
                            for (int dx = 0; dx < 2; ++dx) {
                                int const sx = std::min(x * 2 + dx, source_size - 1);
                                int const sy = std::min(y * 2 + dy, source_size - 1);
                                std::size_t const at = (static_cast<std::size_t>(face) * source_size * source_size + static_cast<std::size_t>(sy) * source_size + sx) * 4;
                                sum += glm::vec4(source[at], source[at + 1], source[at + 2], source[at + 3]);
                            }
                        }
                        std::size_t const at = (static_cast<std::size_t>(face) * target_size * target_size + static_cast<std::size_t>(y) * target_size + x) * 4;
                        target[at + 0] = sum.x * 0.25f;
                        target[at + 1] = sum.y * 0.25f;
                        target[at + 2] = sum.z * 0.25f;
                        target[at + 3] = 1.0f;
                    }
                }
            }
            pyramid.push_back(std::move(target));
        }
        return pyramid;
    }

    // Bilinear fetch of one pyramid level, inside the face of @p dir. Filtering stops at the face
    // edge (the environment is a smooth analytic gradient and the sun disc sits well inside a
    // face, so nothing visible crosses a seam).
    glm::vec3 sample_cubemap_level(std::span<float const> const level, int const size, glm::vec3 const& dir) {
        int face = 0;
        float u = 0.0f;
        float v = 0.0f;
        cube_face_uv(dir, face, u, v);
        float const fx = std::clamp((u * 0.5f + 0.5f) * static_cast<float>(size) - 0.5f, 0.0f, static_cast<float>(size - 1));
        float const fy = std::clamp((v * 0.5f + 0.5f) * static_cast<float>(size) - 0.5f, 0.0f, static_cast<float>(size - 1));
        int const x0 = static_cast<int>(fx);
        int const y0 = static_cast<int>(fy);
        int const x1 = std::min(x0 + 1, size - 1);
        int const y1 = std::min(y0 + 1, size - 1);
        float const tx = fx - static_cast<float>(x0);
        float const ty = fy - static_cast<float>(y0);
        auto const fetch = [&](int const x, int const y) {
            std::size_t const at = (static_cast<std::size_t>(face) * size * size + static_cast<std::size_t>(y) * size + x) * 4;
            return glm::vec3(level[at], level[at + 1], level[at + 2]);
        };
        glm::vec3 const top = glm::mix(fetch(x0, y0), fetch(x1, y0), tx);
        glm::vec3 const bottom = glm::mix(fetch(x0, y1), fetch(x1, y1), tx);
        return glm::mix(top, bottom, ty);
    }

    // Trilinear fetch across the pyramid: one bilinear fetch per level, blended by the fraction.
    glm::vec3 sample_environment_trilinear(std::vector<std::vector<float>> const& pyramid, int const env_size, glm::vec3 const& dir, float const lod) {
        float const clamped = std::clamp(lod, 0.0f, static_cast<float>(pyramid.size() - 1));
        int const low = static_cast<int>(clamped);
        int const high = std::min(low + 1, static_cast<int>(pyramid.size()) - 1);
        float const blend = clamped - static_cast<float>(low);
        glm::vec3 const a = sample_cubemap_level(pyramid[static_cast<std::size_t>(low)], std::max(1, env_size >> low), dir);
        glm::vec3 const b = sample_cubemap_level(pyramid[static_cast<std::size_t>(high)], std::max(1, env_size >> high), dir);
        return glm::mix(a, b, blend);
    }

    std::vector<float> prefilter_environment(std::span<float const> const env, int const env_size, int const mip_count) {
        // Why the samples read a source mip instead of level 0: this environment carries a hard
        // sun disc, and a narrow GGX lobe either lands on it or misses it. With 64 taps of level 0
        // a coarse level ended up with isolated texels several times brighter than the rest of the
        // level (measured at 16^2: brightest texel 1.22 against a level mean of 0.17, i.e. 7x the
        // mean, and 2.3x its own 3x3 neighbourhood). On screen ONE coarse texel covers a large
        // area, so those texels read as big highlight patches sweeping across a rotating metal
        // surface - the artifact this replaces. Averaging each tap over the solid angle it
        // actually represents (Karis) removes them: same measurement, brightest texel 0.42
        // (2.5x the mean, 1.3x its neighbourhood) with every level mean unchanged - the fix is
        // variance, not energy. It is also FASTER than the 64-tap-of-level-0 version (535 ms
        // against 964 ms for 256^2 x 5 levels here): level 0 needs no samples at all, its texels
        // ARE the mirror reflection, and the coarse levels take half the taps of the level before.
        std::vector<std::vector<float>> const pyramid = build_environment_pyramid(env, env_size, mip_count);
        float const texel_solid_angle = 4.0f * k_pi / (6.0f * static_cast<float>(env_size) * static_cast<float>(env_size));
        std::vector<float> result;
        for (int mip = 0; mip < mip_count; ++mip) {
            int const mip_size = std::max(1, env_size >> mip);
            float const roughness = static_cast<float>(mip) / static_cast<float>(mip_count - 1);
            // 128 taps at the first roughness level, halved per level (32 is the floor); level 0 needs
            // none - see the comment above. That is still about the cost of the naive version this
            // replaces, because the source mips do the averaging and level 0 does no work at all.
            uint32_t const sample_count = mip == 0 ? 1u : std::max(32u, 128u >> (mip - 1));
            float const alpha = roughness * roughness;
            std::vector<float> mip_data(static_cast<size_t>(6) * mip_size * mip_size * 4, 0.0f);
            for (int face = 0; face < 6; ++face) {
                for (int y = 0; y < mip_size; ++y) {
                    for (int x = 0; x < mip_size; ++x) {
                        float const u = (static_cast<float>(x) + 0.5f) / static_cast<float>(mip_size) * 2.0f - 1.0f;
                        float const v = (static_cast<float>(y) + 0.5f) / static_cast<float>(mip_size) * 2.0f - 1.0f;
                        glm::vec3 const n = cube_face_direction(face, u, v);
                        glm::vec3 color(0.0f);
                        if (mip == 0) {
                            // roughness 0 is a mirror: every sample has H == N and therefore L == N, so
                            // this level IS the environment (at lod 0 the trilinear fetch is bilinear)
                            color = sample_environment_trilinear(pyramid, env_size, n, 0.0f);
                        } else {
                            glm::vec3 sum(0.0f);
                            float total_weight = 0.0f;
                            for (uint32_t i = 0; i < sample_count; ++i) {
                                glm::vec3 const h = importance_sample_ggx(hammersley(i, sample_count), n, roughness);
                                glm::vec3 const l = glm::normalize(2.0f * glm::dot(n, h) * h - n);
                                float const ndotl = glm::dot(n, l);
                                if (ndotl <= 0.0f) {
                                    continue;
                                }
                                // GGX pdf of this sample (V == N here, so VoH == NoH), then the source mip
                                // whose texel footprint matches the solid angle the sample stands for
                                float const alpha2 = alpha * alpha;
                                float const ndoth = std::max(glm::dot(n, h), 0.0f);
                                float const denominator = ndoth * ndoth * (alpha2 - 1.0f) + 1.0f;
                                float const distribution = alpha2 / std::max(k_pi * denominator * denominator, 1e-8f);
                                float const pdf = distribution * ndoth / std::max(4.0f * ndoth, 1e-6f) + 1e-4f;
                                float const sample_solid_angle = 1.0f / (static_cast<float>(sample_count) * pdf);
                                float const lod = std::max(0.5f * std::log2(sample_solid_angle / texel_solid_angle), 0.0f);
                                sum += sample_environment_trilinear(pyramid, env_size, l, lod) * ndotl;
                                total_weight += ndotl;
                            }
                            color = total_weight > 0.0f ? sum / total_weight : glm::vec3(0.0f);
                        }
                        size_t const offset = (static_cast<size_t>(face) * mip_size * mip_size + static_cast<size_t>(y) * mip_size + x) * 4;
                        mip_data[offset + 0] = color.r;
                        mip_data[offset + 1] = color.g;
                        mip_data[offset + 2] = color.b;
                        mip_data[offset + 3] = 1.0f;
                    }
                }
            }
            result.insert(result.end(), mip_data.begin(), mip_data.end());
        }
        return result;
    }

    std::vector<float> generate_irradiance_map(std::span<float const> const env, int const env_size, int const irr_size) {
        std::vector<float> result(static_cast<size_t>(6) * irr_size * irr_size * 4, 0.0f);
        // Loop-invariant constant: deliberately at function scope (not inside the loops)
        constexpr uint32_t sample_count = 512; // NOLINT (some toolchains flag the constant when scoped to the inner loop)
        for (int face = 0; face < 6; ++face) {
            for (int y = 0; y < irr_size; ++y) {
                for (int x = 0; x < irr_size; ++x) {
                    float const u = (static_cast<float>(x) + 0.5f) / static_cast<float>(irr_size) * 2.0f - 1.0f;
                    float const v = (static_cast<float>(y) + 0.5f) / static_cast<float>(irr_size) * 2.0f - 1.0f;
                    glm::vec3 const n = cube_face_direction(face, u, v);
                    glm::vec3 const up = std::abs(n.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
                    glm::vec3 const tangent = glm::normalize(glm::cross(up, n));
                    glm::vec3 const bitangent = glm::cross(n, tangent);
                    glm::vec3 sum(0.0f);
                    float total_weight = 0.0f;
                    for (uint32_t i = 0; i < sample_count; ++i) {
                        glm::vec2 const xi = hammersley(i, sample_count);
                        float const phi = 2.0f * k_pi * xi.x;
                        float const cos_theta = std::sqrt(xi.y);
                        float const sin_theta = std::sqrt(std::max(1.0f - xi.y, 0.0f));
                        glm::vec3 const local(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
                        glm::vec3 const l = glm::normalize(tangent * local.x + bitangent * local.y + n * local.z);
                        sum += sample_cubemap(env, env_size, l) * cos_theta;
                        total_weight += cos_theta;
                    }
                    glm::vec3 const color = total_weight > 0.0f ? sum / total_weight : glm::vec3(0.0f);
                    size_t const offset = (static_cast<size_t>(face) * irr_size * irr_size + static_cast<size_t>(y) * irr_size + x) * 4;
                    result[offset + 0] = color.r;
                    result[offset + 1] = color.g;
                    result[offset + 2] = color.b;
                    result[offset + 3] = 1.0f;
                }
            }
        }
        return result;
    }

    std::vector<float> generate_brdf_lut(int const size) {
        std::vector<float> result(static_cast<size_t>(size) * size * 2);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                float const ndotv = (static_cast<float>(x) + 0.5f) / static_cast<float>(size);
                float const roughness = (static_cast<float>(y) + 0.5f) / static_cast<float>(size);
                constexpr glm::vec4 c0(-1.0f, -0.0275f, -0.572f, 0.022f);
                constexpr glm::vec4 c1(1.0f, 0.0425f, 1.04f, -0.04f);
                glm::vec4 const r = roughness * c0 + c1;
                float const a004 = std::min(r.x * r.x, std::exp2(-9.28f * ndotv)) * r.x + r.y;
                result[static_cast<size_t>(y) * size * 2 + static_cast<size_t>(x) * 2 + 0] = -1.04f * a004 + r.z;
                result[static_cast<size_t>(y) * size * 2 + static_cast<size_t>(x) * 2 + 1] = 1.04f * a004 + r.w;
            }
        }
        return result;
    }

    std::vector<unsigned char> to_half_rgba(std::span<float const> const data) {
        std::vector<unsigned char> out(data.size() * 2);
        for (size_t i = 0; i < data.size(); ++i) {
            uint16_t const h = float_to_half(data[i]);
            out[i * 2 + 0] = static_cast<unsigned char>(h & 0xFFu);
            out[i * 2 + 1] = static_cast<unsigned char>(h >> 8);
        }
        return out;
    }

    std::vector<unsigned char> to_half_rg(std::span<float const> const data) {
        std::vector<unsigned char> out(data.size() * 2);
        for (size_t i = 0; i < data.size(); ++i) {
            uint16_t const h = float_to_half(data[i]);
            out[i * 2 + 0] = static_cast<unsigned char>(h & 0xFFu);
            out[i * 2 + 1] = static_cast<unsigned char>(h >> 8);
        }
        return out;
    }

    // ---- async wrappers (see math.cppm): delegate to the synchronous functions on a
    //      std::async thread; the caller consumes the future when the result is needed ----

    std::future<std::vector<float>> generate_environment_cubemap_async(int const size) {
        return std::async(std::launch::async, [size] { return generate_environment_cubemap(size); });
    }

    std::future<std::vector<float>> prefilter_environment_async(std::span<float const> const env, int const env_size, int const mip_count) {
        return std::async(std::launch::async, [env, env_size, mip_count] { return prefilter_environment(env, env_size, mip_count); });
    }

    std::future<std::vector<float>> generate_irradiance_map_async(std::span<float const> const env, int const env_size, int const irr_size) {
        return std::async(std::launch::async, [env, env_size, irr_size] { return generate_irradiance_map(env, env_size, irr_size); });
    }

    std::future<std::vector<float>> generate_brdf_lut_async(int const size) {
        return std::async(std::launch::async, [size] { return generate_brdf_lut(size); });
    }
} // namespace vulkan
