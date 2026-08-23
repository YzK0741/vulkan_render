//
// Created by 23530 on 2026/8/8.
//
module;

#include <glm/glm.hpp>

export module utility.bvh;
export import std;

export import utility.data_block;

/**
 * @file bvh.cppm
 * @defgroup bvh Bounding Volume Hierarchy
 * @ingroup utility
 * @brief BVH acceleration structure: AABB boxes, morton-code building, ray hit test and frustum culling
 * @note
 *      - leaf nodes live in an internal std::vector, internal nodes are heap allocated
 *      - make() builds the tree, rebuild() rebuilds it after leaves change
 */
/**
 * @ingroup bvh
 * @brief ray vs AABB slab intersection test
 * @param min AABB min corner
 * @param max AABB max corner
 * @param start ray origin
 * @param direction ray direction
 * @param t_min minimum hit distance
 * @param t_max maximum hit distance
 * @return true if the ray hits the AABB within [t_min, t_max]
 */
bool hit(glm::vec3 const& min, glm::vec3 const& max , glm::vec3 const& start, glm::vec3 const& direction, float t_min = 0.01f, float t_max = std::numeric_limits<float>::infinity());

namespace utility
{
    /**
     * @ingroup bvh
     * @brief axis-aligned bounding box with an optional user data pointer
     * @tparam T type of the attached user data
     */
    export template<typename T>
    struct aabb_box
    {
        glm::vec3 min = {};
        glm::vec3 max = {};
        T* extra_data = nullptr;

        bool is_valid()
        {
            return !(this->min.x > this->max.x || this->min.y > this->max.y || this->min.z > this->max.z);
        }
        [[nodiscard]] glm::vec3 get_midpoint() const noexcept
        {
            glm::vec3 midpoint;
            midpoint.x = (min.x + max.x) * 0.5f;
            midpoint.y = (min.y + max.y) * 0.5f;
            midpoint.z = (min.z + max.z) * 0.5f;
            return midpoint;
        }
        aabb_box get_intersection(aabb_box const& other) const noexcept
        {
            aabb_box intersection;
            intersection.max = glm::min(other.max, this->max);
            intersection.min = glm::max(other.min, this->min);
            return intersection;
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

        explicit operator bool()
        {
            return this->is_valid();
        }

        aabb_box operator&(aabb_box const& other) const noexcept
        {
            return this->get_intersection(other);
        }

        aabb_box operator|(aabb_box const& other) const noexcept
        {
            return this->get_common(other);
        }
    };

    /**
     * @ingroup bvh
     * @brief 96-bit morton code used to order leaves
     */
    using morton_code = data_block<12>;

    /**
     * @ingroup bvh
     * @brief generate a morton code from a normalized midpoint
     * @param midpoint point in [0, 1]^3
     * @param scale quantization scale per axis
     * @return morton_code on success, error message if midpoint is out of [0, 1]
     */
    std::expected<morton_code, std::string> generate_morton_from_midpoint(glm::vec3 const& midpoint, float scale);

    /**
     * @ingroup bvh
     * @brief node of the BVH tree; leaves reference internal storage, internal nodes own heap children
     * @tparam T type of the attached user data
     */
    export template<typename T>
    struct bvh_node
    {
        bvh_node* left = nullptr;
        bvh_node* right = nullptr;
        morton_code code = {};
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

    /**
     * @ingroup bvh
     * @brief view frustum defined by six planes
     */
    export struct frustum
    {
        glm::vec4 planes[6];
        glm::vec3 corners[8];
        bool in(glm::vec3 const& min, glm::vec3 const& max);
    };

    /**
     * @ingroup bvh
     * @brief bounding volume hierarchy built with morton codes
     * @tparam T type of the user data attached to each leaf
     * @note
     *      - make() builds the tree from AABBs
     *      - add() inserts a leaf, call rebuild() afterwards
     *      - get_hit() collects hit leaves along a ray
     *      - frustum_cull() returns leaves inside a frustum
     */
    export template<typename T>
    class bvh
    {
        std::vector<bvh_node<T>> leaves;
        std::unique_ptr<bvh_node<T>> root;

        /**
         * @ingroup bvh
         * @brief build the internal tree bottom-up from the given leaves
         * @param leaves leaf nodes sorted by morton code
         * @return the root node on success, error message on failure
         */
        static std::expected<std::unique_ptr<bvh_node<T>>, std::string> build_from_leaves(std::vector<bvh_node<T>> const& leaves)
        {
            using fail = std::unexpected<std::string>;
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
                    node->aabb = node->left->aabb | node->right->aabb;
                    auto code = generate_morton_from_midpoint(node->aabb.get_midpoint(), 1.0f);
                    if (!code)
                    {
                        delete node;
                        std::ranges::for_each(layers.back(), [](auto* n)
                        {
                            if (n != nullptr && !n->is_leaf())
                            {
                                delete n;
                            }
                        });
                        return fail(code.error());
                    }
                    node->code = std::move(code).value();
                    layers[0].push_back(node);
                }
            }

            while (layers.back().size() != 1)
            {
                layers.emplace_back();
                auto layer_it = (layers.end() - 2)->begin();
                auto layer_end = (layers.end() - 2)->end();
                while (layer_it != layer_end)
                {
                    if (layer_it + 1 == layer_end)
                    {
                        layers.back().push_back(*layer_it);
                        ++layer_it;
                    } else
                    {
                        auto* node = new bvh_node<T>();
                        node->left = &*layer_it;
                        ++layer_it;
                        node->right = &*layer_it;
                        ++layer_it;
                        node->aabb = node->left->aabb | node->right->aabb;
                        auto code = generate_morton_from_midpoint(node->aabb.get_midpoint(), 1.0f);
                        if (!code)
                        {
                            delete node;
                            std::ranges::for_each(layers.back(), [](auto* n)
                            {
                                if (n != nullptr && !n->is_leaf())
                                {
                                    delete n;
                                }
                            });
                            return fail(code.error());
                        }
                        node->code = std::move(code).value();
                        layers.back().push_back(node);
                    }
                }
            }

            if (layers.size() == 1)
            {
                // 只有一个元素时，layers[0][0] 指向的是 leaves 的 vector 元素，
                // 直接交给 unique_ptr 会在析构时 delete 它（不是 new 出来的）→ 双重释放 UB。
                // 在堆上拷贝一份作为根节点。
                return {std::make_unique<bvh_node<T>>(*layers.back()[0])};
            }

            return {std::move(layers.back()[0])};
        }

    public:
        /**
         * @ingroup bvh
         * @brief build a BVH from the given AABBs
         * @param datas AABBs to build the tree from
         * @return the built bvh on success, error message on failure
         */
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
                auto morton = generate_morton_from_midpoint(data.get_midpoint(), 1.0f);
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

            auto build_result = build_from_leaves(leaves);

            if (!build_result)
            {
                return fail(build_result.error());
            }

            result.root = std::move(build_result).value();

            result.leaves = std::move(leaves);
            return result;
        }

        /**
         * @ingroup bvh
         * @brief rebuild the tree from the current leaves
         */
        void rebuild()
        {
            std::ranges::sort(this->leaves, [](auto const& a, auto const& b){return a.code < b.code;});
            auto build_result = build_from_leaves(this->leaves);

            if (!build_result)
            {
                return;
            }

            this->root = std::move(build_result).value();
        }

        /**
         * @ingroup bvh
         * @brief add an AABB as a new leaf
         * @param box the AABB to add
         * @return {} on success, error message on failure
         * @note the tree is not rebuilt automatically, call rebuild() afterwards
         */
        std::expected<void, std::string> add(aabb_box<T> const& box)
        {
            using fail = std::unexpected<std::string>;

            bvh_node<T> node = {};
            node.aabb.min = box.min;
            node.aabb.max = box.max;
            node.extra_data = box.extra_data;

            auto morton = generate_morton_from_midpoint(box.get_midpoint(), 1.0f);
            if (!morton)
            {
                return fail(morton.error());
            }
            node.code = std::move(morton).value();

            this->leaves.push_back(node);
            return {};
        }

        /**
         * @ingroup bvh
         * @brief collect leaf nodes hit by a ray
         * @param start ray origin
         * @param direction ray direction
         * @param t_min minimum hit distance
         * @param t_max maximum hit distance
         */
        void get_hit(glm::vec3 const& start, glm::vec3 const& direction, float t_min = 0.01f, float t_max = std::numeric_limits<float>::infinity())
        {
            std::stack<bvh_node<T>*> nodes_to_access;
            std::priority_queue<bvh_node<T>*> hit_list;
            bvh_node<T>* root = this->root.get();
            if (root == nullptr)
            {
                return;
            }
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
                        hit_list.push(node);
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

        /**
         * @ingroup bvh
         * @brief return leaf nodes intersecting the frustum
         * @param f the frustum
         * @return leaf node pointers inside the frustum
         */
        std::vector<bvh_node<T>*> frustum_cull(frustum const& f)
        {
            std::vector<bvh_node<T>*> result;

            if (root == nullptr)
            {
                return result;
            }

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