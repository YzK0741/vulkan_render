module;

#include <glm/glm.hpp>

module vulkan.math;

namespace vulkan {
    namespace {
        constexpr float k_pi = 3.14159265359f;

        // Cubemap face direction: texel (u, v) in [-1, 1] -> unit direction (Vulkan/GL cubemap convention)
        glm::vec3 cube_face_direction(const int face, const float u, const float v) {
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

        // Procedural environment (HDR): gradient sky/ground + sun disc
        glm::vec3 environment_color(const glm::vec3& dir) {
            const float t = std::clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
            constexpr glm::vec3 ground = glm::vec3(0.03f, 0.03f, 0.05f) * 0.75f;
            constexpr glm::vec3 horizon = glm::vec3(0.16f, 0.19f, 0.26f) * 0.75f;
            constexpr glm::vec3 sky = glm::vec3(0.28f, 0.45f, 0.75f) * 0.75f;
            glm::vec3 env = t < 0.5f ? glm::mix(ground, horizon, t * 2.0f) : glm::mix(horizon, sky, (t - 0.5f) * 2.0f);
            const glm::vec3 sun_dir = glm::normalize(glm::vec3(0.3f, 1.0f, 0.5f));
            const float sun = std::pow(std::max(glm::dot(dir, sun_dir), 0.0f), 64.0f);
            env += glm::vec3(1.0f, 0.95f, 0.85f) * sun * 1.5f; // slightly stronger sun disc for clearer metallic highlights
            return env;
        }

        // Nearest-neighbor cubemap sampling (smooth enough after summed-area averaging)
        glm::vec3 sample_cubemap(const std::vector<float>& data, const int size, const glm::vec3& dir) {
            const float ax = std::abs(dir.x);
            const float ay = std::abs(dir.y);
            const float az = std::abs(dir.z);
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
            const int px = std::clamp(static_cast<int>((u * 0.5f + 0.5f) * static_cast<float>(size)), 0, size - 1);
            const int py = std::clamp(static_cast<int>((v * 0.5f + 0.5f) * static_cast<float>(size)), 0, size - 1);
            const size_t offset = (static_cast<size_t>(face) * size * size + static_cast<size_t>(py) * size + px) * 4;
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

        glm::vec2 hammersley(const uint32_t i, const uint32_t n) {
            return glm::vec2(static_cast<float>(i) / static_cast<float>(n), radical_inverse_vdc(i));
        }

        // GGX importance sampling: build the half vector from Hammersley samples
        glm::vec3 importance_sample_ggx(const glm::vec2& xi, const glm::vec3& n, const float roughness) {
            const float a = roughness * roughness;
            const float phi = 2.0f * k_pi * xi.x;
            const float cos_theta = std::sqrt((1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y));
            const float sin_theta = std::sqrt(std::max(1.0f - cos_theta * cos_theta, 0.0f));
            const glm::vec3 h(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
            const glm::vec3 up = std::abs(n.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
            const glm::vec3 tangent = glm::normalize(glm::cross(up, n));
            const glm::vec3 bitangent = glm::cross(n, tangent);
            return glm::normalize(tangent * h.x + bitangent * h.y + n * h.z);
        }

        // IEEE 754 binary32 -> binary16 (truncated; plenty for ambient light)
        uint16_t float_to_half(const float value) {
            const uint32_t bits = std::bit_cast<uint32_t>(value);
            const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
            const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
            const uint32_t mantissa = bits & 0x7FFFFFu;
            if (exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00u); // infinity
            }
            if (exponent <= 0) {
                return sign; // subnormal/zero -> 0
            }
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
        }
    } // namespace

    std::vector<float> generate_environment_cubemap(const int size) {
        std::vector<float> data(static_cast<size_t>(6) * size * size * 4);
        for (int face = 0; face < 6; ++face) {
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(size) * 2.0f - 1.0f;
                    const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(size) * 2.0f - 1.0f;
                    const glm::vec3 color = environment_color(cube_face_direction(face, u, v));
                    const size_t offset = (static_cast<size_t>(face) * size * size + static_cast<size_t>(y) * size + x) * 4;
                    data[offset + 0] = color.r;
                    data[offset + 1] = color.g;
                    data[offset + 2] = color.b;
                    data[offset + 3] = 1.0f;
                }
            }
        }
        return data;
    }

    std::vector<float> prefilter_environment(const std::vector<float>& env, const int env_size, const int mip_count) {
        std::vector<float> result;
        for (int mip = 0; mip < mip_count; ++mip) {
            const int mip_size = std::max(1, env_size >> mip);
            const float roughness = static_cast<float>(mip) / static_cast<float>(mip_count - 1);
            const uint32_t sample_count = static_cast<uint32_t>(std::max(4, 64 >> mip));
            std::vector<float> mip_data(static_cast<size_t>(6) * mip_size * mip_size * 4, 0.0f);
            for (int face = 0; face < 6; ++face) {
                for (int y = 0; y < mip_size; ++y) {
                    for (int x = 0; x < mip_size; ++x) {
                        const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(mip_size) * 2.0f - 1.0f;
                        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(mip_size) * 2.0f - 1.0f;
                        const glm::vec3 n = cube_face_direction(face, u, v);
                        glm::vec3 sum(0.0f);
                        float total_weight = 0.0f;
                        for (uint32_t i = 0; i < sample_count; ++i) {
                            const glm::vec3 h = importance_sample_ggx(hammersley(i, sample_count), n, roughness);
                            const glm::vec3 l = glm::normalize(2.0f * glm::dot(n, h) * h - n);
                            const float ndotl = glm::dot(n, l);
                            if (ndotl > 0.0f) {
                                sum += sample_cubemap(env, env_size, l) * ndotl;
                                total_weight += ndotl;
                            }
                        }
                        const glm::vec3 color = total_weight > 0.0f ? sum / total_weight : glm::vec3(0.0f);
                        const size_t offset = (static_cast<size_t>(face) * mip_size * mip_size + static_cast<size_t>(y) * mip_size + x) * 4;
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

    std::vector<float> generate_irradiance_map(const std::vector<float>& env, const int env_size, const int irr_size) {
        std::vector<float> result(static_cast<size_t>(6) * irr_size * irr_size * 4, 0.0f);
        // Loop-invariant constant: deliberately at function scope (not inside the loops)
        constexpr uint32_t sample_count = 512; // NOLINT (some toolchains flag the constant when scoped to the inner loop)
        for (int face = 0; face < 6; ++face) {
            for (int y = 0; y < irr_size; ++y) {
                for (int x = 0; x < irr_size; ++x) {
                    const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(irr_size) * 2.0f - 1.0f;
                    const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(irr_size) * 2.0f - 1.0f;
                    const glm::vec3 n = cube_face_direction(face, u, v);
                    const glm::vec3 up = std::abs(n.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
                    const glm::vec3 tangent = glm::normalize(glm::cross(up, n));
                    const glm::vec3 bitangent = glm::cross(n, tangent);
                    glm::vec3 sum(0.0f);
                    float total_weight = 0.0f;
                    for (uint32_t i = 0; i < sample_count; ++i) {
                        const glm::vec2 xi = hammersley(i, sample_count);
                        const float phi = 2.0f * k_pi * xi.x;
                        const float cos_theta = std::sqrt(xi.y);
                        const float sin_theta = std::sqrt(std::max(1.0f - xi.y, 0.0f));
                        const glm::vec3 local(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
                        const glm::vec3 l = glm::normalize(tangent * local.x + bitangent * local.y + n * local.z);
                        sum += sample_cubemap(env, env_size, l) * cos_theta;
                        total_weight += cos_theta;
                    }
                    const glm::vec3 color = total_weight > 0.0f ? sum / total_weight : glm::vec3(0.0f);
                    const size_t offset = (static_cast<size_t>(face) * irr_size * irr_size + static_cast<size_t>(y) * irr_size + x) * 4;
                    result[offset + 0] = color.r;
                    result[offset + 1] = color.g;
                    result[offset + 2] = color.b;
                    result[offset + 3] = 1.0f;
                }
            }
        }
        return result;
    }

    std::vector<float> generate_brdf_lut(const int size) {
        std::vector<float> result(static_cast<size_t>(size) * size * 2);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                const float ndotv = (static_cast<float>(x) + 0.5f) / static_cast<float>(size);
                const float roughness = (static_cast<float>(y) + 0.5f) / static_cast<float>(size);
                constexpr glm::vec4 c0(-1.0f, -0.0275f, -0.572f, 0.022f);
                constexpr glm::vec4 c1(1.0f, 0.0425f, 1.04f, -0.04f);
                const glm::vec4 r = roughness * c0 + c1;
                const float a004 = std::min(r.x * r.x, std::exp2(-9.28f * ndotv)) * r.x + r.y;
                result[static_cast<size_t>(y) * size * 2 + static_cast<size_t>(x) * 2 + 0] = -1.04f * a004 + r.z;
                result[static_cast<size_t>(y) * size * 2 + static_cast<size_t>(x) * 2 + 1] = 1.04f * a004 + r.w;
            }
        }
        return result;
    }

    std::vector<unsigned char> to_half_rgba(const std::vector<float>& data) {
        std::vector<unsigned char> out(data.size() * 2);
        for (size_t i = 0; i < data.size(); ++i) {
            const uint16_t h = float_to_half(data[i]);
            out[i * 2 + 0] = static_cast<unsigned char>(h & 0xFFu);
            out[i * 2 + 1] = static_cast<unsigned char>(h >> 8);
        }
        return out;
    }

    std::vector<unsigned char> to_half_rg(const std::vector<float>& data) {
        std::vector<unsigned char> out(data.size() * 2);
        for (size_t i = 0; i < data.size(); ++i) {
            const uint16_t h = float_to_half(data[i]);
            out[i * 2 + 0] = static_cast<unsigned char>(h & 0xFFu);
            out[i * 2 + 1] = static_cast<unsigned char>(h >> 8);
        }
        return out;
    }
} // namespace vulkan
