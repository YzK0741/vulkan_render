module;

#include <glm/glm.hpp>

export module utility.bvh;
export import vstd;

export import utility.data_block;

/**
 * @file bvh.cppm
 * @defgroup bvh Bounding Volume Hierarchy
 * @ingroup utility
 * @brief BVH acceleration structure: AABB boxes, morton-code building, ray hit test and frustum culling
 * @note
 *      - leaf nodes live in an internal std::vector, internal nodes are heap allocated
 *      - make() builds the tree, rebuild() rebuilds it after leaves change
 *      - consumed by the renderer: the runtime builds a per-frame bvh over scene leaf world
 *        AABBs and frustum_culls the main pass against the camera frustum
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
bool hit(glm::vec3 const& min, glm::vec3 const& max, glm::vec3 const& start, glm::vec3 const& direction, float t_min = 0.01f, float t_max = std::numeric_limits<float>::infinity());

namespace utility {
    /**
     * @ingroup bvh
     * @brief axis-aligned bounding box with an optional user data pointer
     * @tparam T type of the attached user data
     */
    export template <typename T>
    struct aabb_box {
        glm::vec3 min = {};
        glm::vec3 max = {};
        T* extra_data = nullptr;

        [[nodiscard]] bool is_valid() const {
            return this->min.x <= this->max.x && this->min.y <= this->max.y && this->min.z <= this->max.z;
        }
        [[nodiscard]] glm::vec3 get_midpoint() const noexcept {
            glm::vec3 midpoint;
            midpoint.x = (min.x + max.x) * 0.5f;
            midpoint.y = (min.y + max.y) * 0.5f;
            midpoint.z = (min.z + max.z) * 0.5f;
            return midpoint;
        }
        aabb_box get_intersection(aabb_box const& other) const noexcept {
            aabb_box intersection;
            intersection.max = glm::min(other.max, this->max);
            intersection.min = glm::max(other.min, this->min);
            return intersection;
        }
        aabb_box get_common(aabb_box const& other) const noexcept {
            aabb_box common;
            common.min = glm::min(other.min, this->min);
            common.max = glm::max(other.max, this->max);
            return common;
        }

        [[nodiscard]] float surface_area() const {
            float const w = this->max.y - this->min.y;
            float const h = this->max.x - this->min.x;
            float const l = this->max.z - this->min.z;
            return 2.0f * (w * h + w * l + h * l);
        }

        explicit operator bool() const {
            return this->is_valid();
        }

        aabb_box operator&(aabb_box const& other) const noexcept {
            return this->get_intersection(other);
        }

        aabb_box operator|(aabb_box const& other) const noexcept {
            return this->get_common(other);
        }
    };

    /**
     * @ingroup bvh
     * @brief 96-bit morton code used to order leaves
     */
    using morton_code = data_block<12>;

    // Quantization grid for morton codes: normalized midpoints ([0,1] per axis) are scaled by
    // 2^21 before truncation, so codes carry ~21 bits per axis instead of collapsing to {0,1}
    // (scale 1.0 would make the morton sort a no-op and the BVH degenerate to insertion order).
    inline constexpr float morton_quantization = 2097152.0f; // 2^21

    /**
     * @ingroup bvh
     * @brief whether every component of @p value is finite (neither NaN nor infinite)
     * @note std::isfinite per component rather than glm's: the vector form lives in
     *       <glm/gtx/compatibility.hpp> and this module pulls in only the core <glm/glm.hpp>. Used
     *       where a non-finite value would otherwise reach a float -> integer conversion, which is
     *       undefined behaviour - every comparison against a NaN is false, so a range check written
     *       the obvious way lets it through.
     */
    [[nodiscard]] inline bool all_finite(glm::vec3 const& value) noexcept {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    }

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
    export template <typename T>
    struct bvh_node {
        bvh_node* left = nullptr;
        bvh_node* right = nullptr;
        morton_code code = {};
        // bounding volume of this node (leaf or merged subtree); reuses aabb_box so the min/max
        // accessors, midpoint, surface area and the merge operator live in exactly one place
        aabb_box<T> aabb = {};

        using data_type = std::conditional_t<std::is_void_v<T>, std::monostate, T>;

        // nullptr rather than indeterminate: the build's INTERNAL nodes never have user data, and an
        // uninitialized pointer is the one member of this struct that a reader could dereference by
        // accident (the leaves' pointer is always assigned from the caller's aabb_box)
        data_type* extra_data = nullptr;

        // A node is a node of a graph whose links are raw pointers into storage the containing
        // bvh<T> owns: internal nodes are heap-allocated and delete their non-leaf children, while
        // leaf children point INTO bvh<T>::leaves. Copying that would alias the same subtree (two
        // destructors deleting it) and, for the leaf form, silently outlive nothing at all - so the
        // shallow copy the compiler would generate is never the right operation. Deleted rather than
        // documented: the hazard is a double free, and a compile error is a better teacher than a
        // comment. bvh<T>::add() builds leaves in place so no copy is needed anywhere.
        bvh_node(bvh_node const&) = delete;
        bvh_node& operator=(bvh_node const&) = delete;
        bvh_node(bvh_node&&) = default;
        bvh_node& operator=(bvh_node&&) = default;
        bvh_node() = default;

        /** @brief construct a LEAF in place (see the deleted copy operations above) */
        [[nodiscard]] static bvh_node make_leaf(aabb_box<T> const& box) {
            bvh_node leaf;
            leaf.aabb.min = box.min;
            leaf.aabb.max = box.max;
            leaf.extra_data = box.extra_data;
            leaf.code = {}; // recomputed by rebuild()
            return leaf;
        }

        [[nodiscard]] bool is_leaf() const noexcept {
            return this->left == nullptr && this->right == nullptr;
        }
        ~bvh_node() {
            if (this->left != nullptr && !this->left->is_leaf()) {
                delete this->left;
            }
            if (this->right != nullptr && !this->right->is_leaf()) {
                delete this->right;
            }
        }
    };

    /**
     * @ingroup bvh
     * @brief view frustum defined by six planes
     */
    export struct frustum {
        glm::vec4 planes[6];
        glm::vec3 corners[8];
        bool in(glm::vec3 const& min, glm::vec3 const& max) const;
    };

    /**
     * @ingroup bvh
     * @brief extract the six frustum planes from a combined view-projection matrix
     * @param view_proj the camera's proj * view (row-major, as built by make_orbit_camera_ubo)
     * @return frustum whose planes are in world space; plane i has inward-facing normal
     *         (glm::vec3(plane) = normal, plane.w = signed distance), so a point is inside when
     *         dot(normal, p) + plane.w >= 0 (matches frustum::in's p-vertex test)
     * @note Gribb-Hartmann plane extraction; rows of a row-major matrix are the clip-space plane
     *       coefficients. Corners are not filled (unused by frustum::in).
     */
    export frustum make_frustum(glm::mat4 const& view_proj);

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
    export template <typename T>
    class bvh {
        // Leaf storage is a DEQUE, not a vector: internal (heap) nodes point at leaf elements,
        // and add() inserts new leaves without invalidating those addresses. A vector would
        // reallocate on growth and leave every internal node dangling until the next rebuild -
        // add() + rebuild() is a documented usage, so leaf addresses must be stable. Deque
        // swaps also keep references valid, which make() relies on when it hands its built tree
        // to the result.
        std::deque<bvh_node<T>> leaves;
        std::unique_ptr<bvh_node<T>> root;
        // normalization for morton coding: world-space midpoints are mapped into [0,1]^3 by
        // (mid - origin) * scale so the morton quantizer works for AABBs anywhere in space
        glm::vec3 origin = glm::vec3(0.0f);
        glm::vec3 scale = glm::vec3(1.0f);

        /** @brief map a world-space midpoint into [0,1]^3 for morton coding
         *  @note glm::clamp PROPAGATES NaN rather than absorbing it (min(max(NaN,0),1) is NaN), so a
         *        NaN component would sail through the clamp and only be rejected - or become UB - in
         *        generate_morton_from_midpoint(). A degenerate scene extent (every AABB at one point,
         *        or an empty set) makes normalize() produce NaN/inf here, so map non-finite results to
         *        the origin instead: those leaves then share one code and stay in insertion order,
         *        which is a sound order for the build rather than a rejected input. */
        [[nodiscard]] glm::vec3 normalize(glm::vec3 const& midpoint) const {
            glm::vec3 const normalized = (midpoint - this->origin) * this->scale; // NOLINT
            return all_finite(normalized) ? glm::clamp(normalized, glm::vec3(0.0f), glm::vec3(1.0f)) : glm::vec3(0.0f);
        }
        /** @brief compute origin/scale from a set of leaf AABBs (scene extent) */
        void set_extent(std::deque<bvh_node<T>> const& leaf_nodes) {
            glm::vec3 min = glm::vec3(std::numeric_limits<float>::infinity());
            glm::vec3 max = glm::vec3(-std::numeric_limits<float>::infinity());
            for (bvh_node<T> const& leaf : leaf_nodes) {
                min = glm::min(min, leaf.aabb.min);
                max = glm::max(max, leaf.aabb.max);
            }
            this->origin = min;
            glm::vec3 const extent = glm::max(max - min, glm::vec3(1e-6f));
            this->scale = 1.0f / extent;
        }

        /**
         * @ingroup bvh
         * @brief build the internal tree bottom-up from the given leaves
         * @param leaves leaf nodes sorted by morton code
         * @param origin normalization origin (scene AABB min)
         * @param scale normalization scale (1 / scene extent)
         * @return the root node on success, error message on failure
         */
        static std::expected<std::unique_ptr<bvh_node<T>>, std::string> build_from_leaves( // NOLINT(*-function-cognitive-complexity)
            std::deque<bvh_node<T>>& leaves,
            glm::vec3 const& origin,
            glm::vec3 const& scale) {
            using fail = std::unexpected<std::string>;
            // Same sanitizing guard as bvh<T>::normalize(): glm::clamp propagates NaN, so without it
            // a NaN component would reach generate_morton_from_midpoint(), whose own finite check
            // would then turn a degenerate extent into a failed BUILD. Mapping it to the origin
            // instead keeps the build succeeding with a sound order (see that member's note).
            auto const normalize = [&origin, &scale](glm::vec3 const& midpoint) {
                glm::vec3 const normalized = (midpoint - origin) * scale; // NOLINT
                return all_finite(normalized) ? glm::clamp(normalized, glm::vec3(0.0f), glm::vec3(1.0f)) : glm::vec3(0.0f);
            };
            auto leave_it = leaves.begin();

            std::vector<std::vector<bvh_node<T>*>> layers(1);

            while (leave_it != leaves.end()) {
                if (leave_it + 1 == leaves.end()) {
                    layers[0].push_back(&*leave_it);
                    ++leave_it;
                } else {
                    auto* node = new bvh_node<T>();
                    node->left = &*leave_it;
                    ++leave_it;
                    node->right = &*leave_it;
                    ++leave_it;
                    node->aabb = node->left->aabb | node->right->aabb;
                    auto code = generate_morton_from_midpoint(normalize(node->aabb.get_midpoint()), morton_quantization);
                    if (!code) {
                        delete node;
                        std::ranges::for_each(layers.back(), [](auto* n) {
                            if (n != nullptr && !n->is_leaf()) {
                                delete n;
                            }
                        });
                        return fail(code.error());
                    }
                    node->code = std::move(code).value();
                    layers[0].push_back(node);
                }
            }

            while (layers.back().size() != 1) {
                layers.emplace_back();
                auto layer_it = (layers.end() - 2)->begin();
                auto layer_end = (layers.end() - 2)->end();
                while (layer_it != layer_end) {
                    if (layer_it + 1 == layer_end) {
                        layers.back().push_back(*layer_it);
                        ++layer_it;
                    } else {
                        auto* node = new bvh_node<T>();
                        node->left = *layer_it;
                        ++layer_it;
                        node->right = *layer_it;
                        ++layer_it;
                        node->aabb = node->left->aabb | node->right->aabb;
                        auto code = generate_morton_from_midpoint(normalize(node->aabb.get_midpoint()), morton_quantization);
                        if (!code) {
                            delete node;
                            std::ranges::for_each(layers.back(), [](auto* n) {
                                if (n != nullptr && !n->is_leaf()) {
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

            if (layers.size() == 1) {
                // Single element on the bottom layer - two cases:
                //  - the sole LEAF points at an element of `leaves`: it was not new'd, handing
                //    it to a unique_ptr would double-free on destruction, so copy it to the heap.
                //  - ONE heap-allocated internal node (exactly two leaves paired into it): take
                //    ownership directly - copying it would leak the original.
                bvh_node<T>* const sole = layers.back()[0];
                if (sole->is_leaf()) {
                    // A fresh childless node carrying the same data, NOT a copy of the node: a node
                    // would copy the raw child links too, and this one is handed to a unique_ptr
                    // whose destructor deletes non-leaf children - i.e. it would delete the
                    // deque-owned leaf it was copied from. That made the correctness of the old
                    // `new bvh_node<T>(*sole)` depend on the leaf happening to have no links, which
                    // is exactly the kind of invariant the deleted copy operations now enforce.
                    auto heap_leaf = std::make_unique<bvh_node<T>>();
                    heap_leaf->aabb = sole->aabb;
                    heap_leaf->extra_data = sole->extra_data;
                    heap_leaf->code = sole->code;
                    return heap_leaf;
                }
                return std::unique_ptr<bvh_node<T>>(sole);
            }

            return std::unique_ptr<bvh_node<T>>(layers.back()[0]);
        }

    public:
        /**
         * @ingroup bvh
         * @brief build a BVH from the given AABBs
         * @param datas AABBs to build the tree from
         * @return the built bvh on success, error message on failure
         */
        static std::expected<bvh, std::string> make(std::vector<aabb_box<T>> const& datas) {
            using fail = std::unexpected<std::string>;

            if (datas.empty()) {
                return fail("data is empty");
            }

            std::deque<bvh_node<T>> leaves;
            for (auto const& data : datas) {
                // make_leaf, not a local + push_back: bvh_node is non-copyable by design (its links
                // are raw pointers into this deque, so a copy would alias a subtree)
                leaves.push_back(bvh_node<T>::make_leaf(data));
            }

            bvh result;
            result.set_extent(leaves); // morton normalization needs the scene extent first
            for (bvh_node<T>& leaf : leaves) {
                auto morton = generate_morton_from_midpoint(result.normalize(leaf.aabb.get_midpoint()), morton_quantization);
                if (!morton) {
                    return fail(morton.error());
                }
                leaf.code = std::move(morton).value();
            }
            std::ranges::sort(leaves, [](auto const& a, auto const& b) { return a.code < b.code; });

            auto build_result = build_from_leaves(leaves, result.origin, result.scale);

            if (!build_result) {
                return fail(build_result.error());
            }

            result.root = std::move(build_result).value();

            // swap, not move: the built tree's internal nodes point at these leaf addresses,
            // and deque::swap keeps every element at its address (a move may not, per the
            // standard's guarantees for deque).
            result.leaves.swap(leaves);
            return result;
        }

        /**
         * @ingroup bvh
         * @brief rebuild the tree from the current leaves
         */
        void rebuild() {
            if (this->leaves.empty()) {
                this->root = nullptr;
                return;
            }
            this->set_extent(this->leaves);
            for (bvh_node<T>& leaf : this->leaves) {
                auto morton = generate_morton_from_midpoint(this->normalize(leaf.aabb.get_midpoint()), morton_quantization);
                if (!morton) {
                    return;
                }
                leaf.code = std::move(morton).value();
            }
            std::ranges::sort(this->leaves, [](auto const& a, auto const& b) { return a.code < b.code; });
            auto build_result = build_from_leaves(this->leaves, this->origin, this->scale);

            if (!build_result) {
                return;
            }

            this->root = std::move(build_result).value();
        }

        /**
         * @ingroup bvh
         * @brief add an AABB as a new leaf
         * @param box the AABB to add
         * @return {} on success, error message on failure
         * @note the tree is not rebuilt automatically, call rebuild() afterwards (rebuild
         *       recomputes the scene extent, so out-of-range boxes added here are fine)
         */
        std::expected<void, std::string> add(aabb_box<T> const& box) {
            // constructed in place: bvh_node is deliberately non-copyable (see its deleted copy ops)
            this->leaves.push_back(bvh_node<T>::make_leaf(box));
            return {};
        }

        /**
         * @ingroup bvh
         * @brief collect leaf nodes hit by a ray
         * @param start ray origin
         * @param direction ray direction
         * @param t_min minimum hit distance
         * @param t_max maximum hit distance
         * @return the hit leaf nodes (unsorted; compute the exact entry distance per hit if needed)
         */
        std::vector<bvh_node<T>*> get_hit(glm::vec3 const& start, glm::vec3 const& direction, float t_min = 0.01f, float t_max = std::numeric_limits<float>::infinity()) const {
            std::vector<bvh_node<T>*> hits;
            bvh_node<T>* root = this->root.get();
            if (root == nullptr) {
                return hits;
            }
            if (!hit(root->aabb.min, root->aabb.max, start, direction, t_min, t_max)) {
                return hits;
            }
            if (root->is_leaf()) {
                hits.push_back(root); // single-leaf tree: the root itself is the hit
                return hits;
            }

            std::stack<bvh_node<T>*> nodes_to_access;
            if (root->left != nullptr) {
                nodes_to_access.push(root->left);
            }
            if (root->right != nullptr) {
                nodes_to_access.push(root->right);
            }
            while (!nodes_to_access.empty()) {
                bvh_node<T>* node = nodes_to_access.top();
                nodes_to_access.pop();
                if (hit(node->aabb.min, node->aabb.max, start, direction, t_min, t_max)) {
                    if (node->is_leaf()) {
                        hits.push_back(node);
                    } else {
                        if (node->left != nullptr) {
                            nodes_to_access.push(node->left);
                        }
                        if (node->right != nullptr) {
                            nodes_to_access.push(node->right);
                        }
                    }
                }
            }
            return hits;
        }

        /**
         * @ingroup bvh
         * @brief return leaf nodes intersecting the frustum
         * @param f the frustum
         * @return leaf node pointers inside the frustum
         */
        std::vector<bvh_node<T>*> frustum_cull(frustum const& f) const {
            std::vector<bvh_node<T>*> result;

            if (root == nullptr) {
                return result;
            }

            if (!f.in(root->aabb.min, root->aabb.max)) {
                return result;
            }

            std::stack<bvh_node<T>*> nodes_to_process;
            nodes_to_process.push(this->root.get());

            while (!nodes_to_process.empty()) {
                bvh_node<T>* node = nodes_to_process.top();
                nodes_to_process.pop();

                if (!f.in(node->aabb.min, node->aabb.max)) {
                    continue;
                }

                if (node->is_leaf()) {
                    result.push_back(node);
                    continue;
                }

                if (node->left != nullptr) {
                    nodes_to_process.push(node->left);
                }
                if (node->right != nullptr) {
                    nodes_to_process.push(node->right);
                }
            }
            return result;
        }
    };
} // namespace utility