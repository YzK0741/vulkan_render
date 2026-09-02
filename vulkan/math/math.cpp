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

        // Nearest-neighbor cubemap sampling (smooth enough after summed-area averaging)
        glm::vec3 sample_cubemap(std::span<float const> const data, int const size, glm::vec3 const& dir) {
            float const ax = std::abs(dir.x);
            float const ay = std::abs(dir.y);
            float const az = std::abs(dir.z);
            int face = 0;
            float u = 0.0f;
            float v = 0.0f;
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

    std::vector<float> prefilter_environment(std::span<float const> const env, int const env_size, int const mip_count) {
        std::vector<float> result;
        for (int mip = 0; mip < mip_count; ++mip) {
            int const mip_size = std::max(1, env_size >> mip);
            float const roughness = static_cast<float>(mip) / static_cast<float>(mip_count - 1);
            uint32_t const sample_count = static_cast<uint32_t>(std::max(4, 64 >> mip));
            std::vector<float> mip_data(static_cast<size_t>(6) * mip_size * mip_size * 4, 0.0f);
            for (int face = 0; face < 6; ++face) {
                for (int y = 0; y < mip_size; ++y) {
                    for (int x = 0; x < mip_size; ++x) {
                        float const u = (static_cast<float>(x) + 0.5f) / static_cast<float>(mip_size) * 2.0f - 1.0f;
                        float const v = (static_cast<float>(y) + 0.5f) / static_cast<float>(mip_size) * 2.0f - 1.0f;
                        glm::vec3 const n = cube_face_direction(face, u, v);
                        glm::vec3 sum(0.0f);
                        float total_weight = 0.0f;
                        for (uint32_t i = 0; i < sample_count; ++i) {
                            glm::vec3 const h = importance_sample_ggx(hammersley(i, sample_count), n, roughness);
                            glm::vec3 const l = glm::normalize(2.0f * glm::dot(n, h) * h - n);
                            float const ndotl = glm::dot(n, l);
                            if (ndotl > 0.0f) {
                                sum += sample_cubemap(env, env_size, l) * ndotl;
                                total_weight += ndotl;
                            }
                        }
                        glm::vec3 const color = total_weight > 0.0f ? sum / total_weight : glm::vec3(0.0f);
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
