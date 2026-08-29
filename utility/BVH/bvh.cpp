module;

#include <glm/glm.hpp>

module utility.bvh;

bool hit(glm::vec3 const& min, glm::vec3 const& max, glm::vec3 const& start, glm::vec3 const& direction, const float t_min, const float t_max) {
    float near = t_min;
    float far = t_max;

    for (int axis = 0; axis < 3; axis++) {
        if (glm::abs(direction[axis]) < 1e-8) {
            if (start[axis] < min[axis] || start[axis] > max[axis]) {
                return false;
            }
        } else {
            const float inv_d = 1.0f / direction[axis];
            float t1 = (min[axis] - start[axis]) * inv_d;
            float t2 = (max[axis] - start[axis]) * inv_d;

            if (t1 > t2) {
                std::swap(t1, t2);
            }

            near = std::max(near, t1);
            far = std::min(far, t2);

            if (near > far) {
                return false;
            }
        }
    }
    return true;
}

namespace {
    utility::morton_code morton_encode(const uint32_t x, const uint32_t y, const uint32_t z) {
        utility::morton_code code;
        auto* const out = code.data.data();

        auto spread = [](const uint32_t n) -> __uint128_t {
            __uint128_t output = 0;
            for (int i = 0; i < 32; i++) {
                const __uint128_t bit = (n >> i) & 1u; // NOLINT(*-signed-bitwise)
                output |= (bit << (3 * i));            // NOLINT(*-signed-bitwise)
            }
            return output;
        };

        const __uint128_t x_spread = spread(x);
        const __uint128_t y_spread = spread(y);
        const __uint128_t z_spread = spread(z);

        __uint128_t result = 0;
        result |= static_cast<__uint128_t>(x_spread);
        result |= static_cast<__uint128_t>(y_spread) << 1;
        result |= static_cast<__uint128_t>(z_spread) << 2;

        memcpy(out, &result, 12);
        return code;
    }
} // namespace

namespace utility {

    std::expected<morton_code, std::string> generate_morton_from_midpoint(glm::vec3 const& midpoint, const float scale) {
        using fail = std::unexpected<std::string>;

        if (midpoint.x < 0.0 || midpoint.y < 0.0 || midpoint.z < 0.0 || midpoint.x > 1.0 || midpoint.y > 1.0 || midpoint.z > 1.0) {
            return fail("position must be in [0, 1]");
        }

        const auto x = static_cast<uint32_t>(midpoint.x * scale);
        const auto y = static_cast<uint32_t>(midpoint.y * scale);
        const auto z = static_cast<uint32_t>(midpoint.z * scale);

        morton_code code = morton_encode(x, y, z);

        return code;
    }

    bool frustum::in(glm::vec3 const& min, glm::vec3 const& max) {
        return std::ranges::all_of(this->planes, [&min, &max](glm::vec4 const& p) {
            glm::vec3 pos = {};
            pos.x = p.x >= 0 ? max.x : min.x;
            pos.y = p.y >= 0 ? max.y : min.y;
            pos.z = p.z >= 0 ? max.z : min.z;

            return glm::dot(pos, glm::vec3(p.x, p.y, p.z)) + p.w >= 0;
        });
    }
} // namespace utility
