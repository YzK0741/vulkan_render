//
// Created by 23530 on 2026/8/8.
//
module;

#include <expected>
#include <vector>
#include <glm/glm.hpp>

export module utility.bvh;
export import utility.data_block;

namespace utility
{
    export struct aabb_box
    {
        glm::vec3 min;
        glm::vec3 max;

        [[nodiscard]] glm::vec3 get_midpoint() const noexcept;
    };

    using morton_code = data_block<12>;

    std::expected<morton_code, std::string> generate_morton_from_aabb(aabb_box const& box, float scale);
}