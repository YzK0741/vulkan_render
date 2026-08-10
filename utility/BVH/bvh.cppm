//
// Created by 23530 on 2026/8/8.
//
module;

#include <algorithm>
#include <expected>
#include <ranges>
#include <vector>
#include <glm/glm.hpp>

export module utility.bvh;
export import utility.data_block;

namespace utility
{
    export template<typename T>
    struct aabb_box
    {
        glm::vec3 min;
        glm::vec3 max;
        T extra_data;
        [[nodiscard]] glm::vec3 get_midpoint() const noexcept
        {
            glm::vec3 midpoint;
            midpoint.x = (min.x + max.x) * 0.5f;
            midpoint.y = (min.y + max.y) * 0.5f;
            midpoint.z = (min.z + max.z) * 0.5f;
            return midpoint;
        }
        aabb_box get_common(aabb_box const& other) const noexcept
        {
            aabb_box common;
            common.min = glm::min(other.min, this->min);
            common.max = glm::max(other.max, this->max);
            return common;
        }

        aabb_box operator&(aabb_box const& other) const noexcept
        {
            return get_common(other);
        }
    };

    using morton_code = data_block<12>;

    std::expected<morton_code, std::string> generate_morton_from_midpoint(glm::vec3 const& midpoint, float scale);

    export template<typename T>
    struct bvh_node
    {
        bvh_node* left;
        bvh_node* right;
        morton_code code;
        struct
        {
            glm::vec3 min;
            glm::vec3 max;
            [[nodiscard]] glm::vec3 get_midpoint() const noexcept
            {
                glm::vec3 midpoint;
                midpoint.x = (min.x + max.x) * 0.5f;
                midpoint.y = (min.y + max.y) * 0.5f;
                midpoint.z = (min.z + max.z) * 0.5f;
                return midpoint;
            }
        } aabb;
        T* extra_data;
        [[nodiscard]] bool is_leaf() const noexcept{return this->left == nullptr && this->right == nullptr;}
        ~bvh_node()
        {
            if (this->left != nullptr && !this->left->is_leaf())
            {
                delete this->left;
            }
            if (this->right != nullptr && !this->right->is_leaf())
            {
                delete this->right;
            }
        }
    };

    export template<typename T>
    class bvh
    {
        std::vector<bvh_node<T>> leaves;
        std::unique_ptr<bvh_node<T>> root;
        static std::expected<bvh, std::string> make(std::vector<aabb_box<T>> const& datas)
        {
            using fail = std::unexpected<std::string>;

            if (datas.empty())
            {
                return fail("data is empty");
            }

            std::vector<bvh_node<T>> leaves;
            leaves.reserve(datas.size());
            for (auto const& data : datas)
            {
                auto morton = generate_morton_from_midpoint(data.get_midpoint());
                bvh_node<T> node = {};
                node.aabb.min = data.min;
                node.aabb.max = data.max;
                node.extra_data = data.extra_data;
                if (!morton)
                {
                    return fail(morton.error());
                }
                node.code = std::move(morton).value();
                leaves.push_back(node);
            }
            std::ranges::sort(leaves, [](auto a, auto b){return a.code < b.code;});

            auto leave_it = leaves.begin();

            std::vector<std::vector<bvh_node<T>*>> layers(1);

            while (leave_it != leaves.end())
            {
                if (leave_it + 1 == leaves.end())
                {
                    layers[0].push_back(&*leave_it);
                    ++leave_it;
                } else
                {
                    auto* node = new bvh_node<T>();
                    node->left = &*leave_it;
                    ++leave_it;
                    node->right = &*leave_it;
                    ++leave_it;
                    node->aabb = node->left->aabb & node->right->aabb;
                    auto code = generate_morton_from_midpoint(node->aabb.get_midpoint());
                    if (!code)
                    {
                        return fail(code.error());
                    }
                    node->code = std::move(code).value();
                    layers[0].push_back(node);
                }
            }

            while (layers.back().size() != 1)
            {
                auto layer_it = layers.back().begin();
                auto layer_end = layers.back().end();
                layers.emplace_back();
                while (layer_it != layer_end)
                {
                    if (layer_it + 1 == layer_end)
                    {
                        layers.back().push_back(&*layer_it);
                        ++layer_it;
                    } else
                    {
                        auto* node = new bvh_node<T>();
                        node->left = &*layer_it;
                        ++layer_it;
                        node->right = &*layer_it;
                        ++layer_it;
                        node->aabb = node->left->aabb & node->right->aabb;
                        auto code = generate_morton_from_midpoint(node->aabb.get_midpoint());
                        if (!code)
                        {
                            return fail(code.error());
                        }
                        node->code = std::move(code).value();
                        layers.back().push_back(node);
                    }
                }
            }

            bvh result;
            result.leaves = std::move(leaves);
            result.root = std::move(layers.back()[0]);
            return result;
        }
    };

}