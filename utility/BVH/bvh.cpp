//
// Created by 23530 on 2026/8/9.
//
module;

#include <array>
#include <expected>
#include <string>
#include <algorithm>
#include <glm/glm.hpp>

module utility.bvh;

namespace
{
    utility::morton_code morton_encode(const uint32_t x, const uint32_t y, const uint32_t z) {
        utility::morton_code code;
        const auto out = code.data.data();

        auto spread = [](const uint32_t n) -> __uint128_t {
            __uint128_t output = 0;
            for (int i = 0; i < 32; i++) {
                const uint64_t bit = (n >> i) & 1;
                output |= (bit << (3 * i));
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
}

namespace utility
{
    std::expected<morton_code, std::string> generate_morton_from_midpoint(glm::vec3 const& midpoint, const float scale)
    {
        using fail = std::unexpected<std::string>;

        if (midpoint.x < 0.0 || midpoint.y < 0.0 || midpoint.z < 0.0 || midpoint.x > 1.0 || midpoint.y > 1.0 || midpoint.z > 1.0)
        {
            return fail("position must be in [0, 1]");
        }

        const auto x = static_cast<uint32_t>(midpoint.x * scale);
        const auto y = static_cast<uint32_t>(midpoint.y * scale);
        const auto z = static_cast<uint32_t>(midpoint.z * scale);

        morton_code code = morton_encode(x, y, z);

        return code;
    }

    bool frustum::in(glm::vec3 const& min, glm::vec3 const& max)
    {
        return std::ranges::all_of(this->planes, [&min, &max](glm::vec4 const& p)
        {
            glm::vec3 pos = {};
            pos.x = p.x >= 0? max.x : min.x;
            pos.y = p.y >= 0? max.y : min.y;
            pos.z = p.z >= 0? max.z : min.z;

            if (glm::dot(pos, glm::vec3(p.x, p.y, p.z)) + p.w < 0) return false;
            return true;
        });
    }
}
