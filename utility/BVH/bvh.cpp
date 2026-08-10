//
// Created by 23530 on 2026/8/9.
//
module;

#include <array>
#include <expected>
#include <string>
#include <glm/glm.hpp>

module utility.bvh;

namespace
{
    utility::morton_code morton_encode(const uint32_t x, const uint32_t y, const uint32_t z) {
        utility::morton_code code;
        const auto out = code.data.data();

        // 将32位扩展到96位（隔两位插入0）
        auto spread = [](__uint128_t n) -> __uint128_t {
            n = (n | (n << 16)) & 0x0000FFFF0000FFFFULL;
            n = (n | (n << 8))  & 0x00FF00FF00FF00FFULL;
            n = (n | (n << 4))  & 0x0F0F0F0F0F0F0F0FULL;
            n = (n | (n << 2))  & 0x3333333333333333ULL;
            n = (n | (n << 1))  & 0x5555555555555555ULL;
            return n;
        };

        const uint64_t x_spread = spread(x);
        const uint64_t y_spread = spread(y);
        const uint64_t z_spread = spread(z);


        __uint128_t result = 0;
        result |= static_cast<__uint128_t>(x_spread);
        result |= static_cast<__uint128_t>(y_spread) << 1;
        result |= static_cast<__uint128_t>(z_spread) << 2;

        // 复制到输出数组（12字节）
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
}