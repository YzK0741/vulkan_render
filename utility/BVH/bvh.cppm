//
// Created by 23530 on 2026/8/8.
//
module;

#include <algorithm>
#include <expected>
#include <queue>
#include <ranges>
#include <stack>
#include <vector>
#include <glm/glm.hpp>

export module utility.bvh;
export import utility.data_block;

namespace
{
    bool hit(glm::vec3 const& min, glm::vec3 const& max , glm::vec3 const& start, glm::vec3 const& direction, const float t_min = 0.01f, const float t_max = std::numeric_limits<float>::infinity())
    {
        float near = t_min;
        float far = t_max;

        for (int axis = 0; axis < 3; axis++)
        {
            if (glm::abs(direction[axis]) < 1e-8)
            {
                if (start[axis] < min[axis] || start[axis] > max[axis])
                {
                    return false;
                }
            }
            else
            {
                const float invD = 1.0f / direction[axis];
                float t1 = (min[axis] - start[axis]) * invD;
                float t2 = (max[axis] - start[axis]) * invD;

                if (t1 > t2) std::swap(t1, t2);

                near = std::max(near, t1);
                far = std::min(far, t2);

                if (near > far) {
                    return false;
                }
            }
        }
        return true;
    }
}

namespace utility
{
    export template<typename T>
    struct aabb_box
    {
        glm::vec3 min = {};
        glm::vec3 max = {};
        T* extra_data = nullptr;
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

        float surface_area()
        {
            const float w = this->max.y - this->min.y;
            const float h = this->max.x - this->min.x;
            const float l = this->max.z - this->min.z;
            return 2.0f * (w * h + w * l + h * l);
        }

        aabb_box operator&(aabb_box const& other) const noexcept
        {
            return this->get_common(other);
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
            float surface_area()
            {
                const float w = this->max.y - this->min.y;
                const float h = this->max.x - this->min.x;
                const float l = this->max.z - this->min.z;
                return 2.0f * (w * h + w * l + l * h);
            }
        } aabb;

        using data_type = std::conditional_t<std::is_void_v<T>, std::monostate, T>;

        data_type* extra_data;
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

    export struct frustum
    {
        glm::vec4 planes[6];
        glm::vec3 corners[8];
        bool in(glm::vec3 const& min, glm::vec3 const& max);
    };

    export template<typename T>
    class bvh
    {
        std::vector<bvh_node<T>> leaves;
        std::unique_ptr<bvh_node<T>> root;

        static std::unique_ptr<bvh_node<T>> build_from_leaves(std::vector<bvh_node<T>> const& leaves)
        {
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
                layers.emplace_back();
                auto layer_it = (layers.back() - 1).begin();
                auto layer_end = (layers.back() - 1).end();
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

            return {std::move(layers.back()[0])};
        }

    public:
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
            std::ranges::sort(leaves, [](auto const& a, auto const& b){return a.code < b.code;});

            bvh result;

            result.root = build_from_leaves(leaves);

            result.leaves = std::move(leaves);
            return result;
        }

        void rebuild()
        {
            this->root.release();
            std::ranges::sort(this->leaves, [](auto const& a, auto const& b){return a.code < b.code;});
            this->root = build_from_leaves(this->leaves);
        }

        void add(aabb_box<T> const& box)
        {
            bvh_node<T> node;
            node.aabb.min = box.min;
            node.aabb.max = box.max;
            node.code = generate_morton_from_midpoint(box.get_midpoint());
            this->leaves.push_back(node);
        }

        void get_hit(glm::vec3 const& start, glm::vec3 const& direction, float t_min = 0.01f, float t_max = std::numeric_limits<float>::infinity())
        {
            std::stack<bvh_node<T>*> nodes_to_access;
            std::priority_queue<bvh_node<T>*> hit_list;
            bvh_node<T>* root = this->root.get();
            if (auto const& aabb = root->aabb; hit(aabb.min, aabb.max, start, direction, t_min, t_max))
            {
                if (root->left != nullptr)
                {
                    nodes_to_access.push(root->left);
                }
                if (root->right != nullptr)
                {
                    nodes_to_access.push(root->right);
                }
            }

            while (!nodes_to_access.empty())
            {
                auto* node = nodes_to_access.top();
                nodes_to_access.pop();
                if (auto aabb = node->aabb; hit(aabb.min, aabb.max, start, direction, t_min, t_max))
                {
                    if (node->is_leaf())
                    {
                        hit_list.push_back(node);
                    }
                    else
                    {
                        if (node->left != nullptr)
                        {
                            nodes_to_access.push(node->left);
                        }
                        if (node->right != nullptr)
                        {
                            nodes_to_access.push(node->right);
                        }
                    }
                }
            }
        }

        std::vector<bvh_node<T>*> frustum_cull(frustum const& f)
        {
            std::vector<bvh_node<T>*> result;

            if (root == nullptr)
            {
                return result;
            }

            // 快速拒绝：根节点都不在视锥体内
            if (!f.in(root->aabb.min, root->aabb.max))
            {
                return result;
            }

            std::stack<bvh_node<T>*> nodes_to_process;
            nodes_to_process.push(root);

            while (!nodes_to_process.empty())
            {
                bvh_node<T>* node = nodes_to_process.top();
                nodes_to_process.pop();

                // 先检查当前节点的AABB
                if (!f.in(node->aabb.min, node->aabb.max))
                {
                    continue;
                }

                if (node->is_leaf())
                {
                    result.push_back(node);
                    continue;
                }

                if (node->left != nullptr)
                {
                    nodes_to_process.push(node->left);
                }
                if (node->right != nullptr)
                {
                    nodes_to_process.push(node->right);
                }
            }

            return result;
        }
    };

}