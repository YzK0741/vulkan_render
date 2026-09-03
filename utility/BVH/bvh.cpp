module;

#include <glm/glm.hpp>

module utility.bvh;

bool hit(glm::vec3 const& min, glm::vec3 const& max, glm::vec3 const& start, glm::vec3 const& direction, float const t_min, float const t_max) {
    float near = t_min;
    float far = t_max;

    for (int axis = 0; axis < 3; axis++) {
        if (glm::abs(direction[axis]) < 1e-8) {
            if (start[axis] < min[axis] || start[axis] > max[axis]) {
                return false;
            }
        } else {
            float const inv_d = 1.0f / direction[axis];
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
    utility::morton_code morton_encode(uint32_t const x, uint32_t const y, uint32_t const z) {
        utility::morton_code code;
        auto* const out = code.data.data();

        auto spread = [](uint32_t const n) -> __uint128_t {
            __uint128_t output = 0;
            for (int i = 0; i < 32; i++) {
                __uint128_t const bit = (n >> i) & 1u; // NOLINT(*-signed-bitwise)
                output |= (bit << (3 * i));            // NOLINT(*-signed-bitwise)
            }
            return output;
        };

        __uint128_t const x_spread = spread(x);
        __uint128_t const y_spread = spread(y);
        __uint128_t const z_spread = spread(z);

        __uint128_t result = 0;
        result |= static_cast<__uint128_t>(x_spread);
        result |= static_cast<__uint128_t>(y_spread) << 1;
        result |= static_cast<__uint128_t>(z_spread) << 2;

        memcpy(out, &result, 12);
        return code;
    }
} // namespace

namespace utility {

    std::expected<morton_code, std::string> generate_morton_from_midpoint(glm::vec3 const& midpoint, float const scale) {
        using fail = std::unexpected<std::string>;

        if (midpoint.x < 0.0 || midpoint.y < 0.0 || midpoint.z < 0.0 || midpoint.x > 1.0 || midpoint.y > 1.0 || midpoint.z > 1.0) {
            return fail("position must be in [0, 1]");
        }

        auto const x = static_cast<uint32_t>(midpoint.x * scale);
        auto const y = static_cast<uint32_t>(midpoint.y * scale);
        auto const z = static_cast<uint32_t>(midpoint.z * scale);

        morton_code code = morton_encode(x, y, z);

        return code;
    }

    bool frustum::in(glm::vec3 const& min, glm::vec3 const& max) const {
        return std::ranges::all_of(this->planes, [&min, &max](glm::vec4 const& p) {
            glm::vec3 pos = {};
            pos.x = p.x >= 0 ? max.x : min.x;
            pos.y = p.y >= 0 ? max.y : min.y;
            pos.z = p.z >= 0 ? max.z : min.z;

            return glm::dot(pos, glm::vec3(p.x, p.y, p.z)) + p.w >= 0;
        });
    }

    frustum make_frustum(glm::mat4 const& view_proj) {
        // view_proj is a column-major glm mat4 meaning v_clip = view_proj * v_world, i.e. the
        // i-th ROW of the matrix is (view_proj[0][i], view_proj[1][i], view_proj[2][i],
        // view_proj[3][i]). Grab the four rows first, then combine them Gribb-Hartmann style:
        //   left   = row3 + row0        right = row3 - row0
        //   bottom = row3 + row1        top   = row3 - row1
        //   near   = row2               far   = row3 - row2   (depth [0,1] + RH_ZO: see below)
        // Planes face inward: point p is inside when dot(xyz, p) + w >= 0 (frustum::in test).
        glm::vec4 const row0(view_proj[0][0], view_proj[1][0], view_proj[2][0], view_proj[3][0]);
        glm::vec4 const row1(view_proj[0][1], view_proj[1][1], view_proj[2][1], view_proj[3][1]);
        glm::vec4 const row2(view_proj[0][2], view_proj[1][2], view_proj[2][2], view_proj[3][2]);
        glm::vec4 const row3(view_proj[0][3], view_proj[1][3], view_proj[2][3], view_proj[3][3]);

        frustum result = {};
        auto const inward = [](glm::vec4 const& plane) {
            // normalize the (a,b,c) part so w becomes a real signed distance; keep orientation
            float const length = glm::length(glm::vec3(plane));
            return length > 1e-8f ? plane / length : plane;
        };
        result.planes[0] = inward(row3 + row0); // left
        result.planes[1] = inward(row3 - row0); // right
        result.planes[2] = inward(row3 + row1); // bottom
        result.planes[3] = inward(row3 - row1); // top
        // The renderer uses perspectiveRH_ZO with a y-flip; clip z in [0,1]. For row-major clip
        // coords with M*v convention, near plane = row2 (z_w = 0 -> row2 dot v = 0) already has
        // the correct inward orientation for RH_ZO, and far = row3 - row2.
        result.planes[4] = inward(row2);        // near
        result.planes[5] = inward(row3 - row2); // far
        return result;
    }
} // namespace utility
