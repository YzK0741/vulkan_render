module;

#include <glm/glm.hpp>

export module vulkan.runtime.scene_tree;
export import std;

/**
 * @file scene_tree.cppm
 * @defgroup vulkan_runtime_scene_tree Vulkan Runtime Scene Tree
 * @brief scene organization: a transform hierarchy of scene_node objects with
 *        drawable leaves.
 *
 * Design notes (see docs/scene_tree_design.md):
 *   - the runtime today stores models as a FLAT per-pipeline list with baked
 *     world matrices; this module introduces the missing hierarchy: each node
 *     carries a LOCAL transform, children and optionally one drawable leaf
 *   - world matrices are accumulated by a depth-first walk each frame and
 *     pushed back into the leaf through drawable::set_world()
 *   - deliberately pure CPU + glm only (no vulkan.model / vulkan.core import):
 *     the loader keeps its own pure-CPU hierarchy types, and the existing
 *     model classes will later be refactored into this module's primitive /
 *     drawable concept ("model becomes a primitive merged into scene_tree")
 */
namespace vulkan::scene_tree {
    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief abstract drawable leaf of a scene node (the future "primitive"
     *        that will absorb the current vulkan.model hierarchy)
     * @note pure interface: implementations own their GPU geometry and record
     *       their draw commands; scene_tree only feeds them their accumulated
     *       world transform every frame
     */
    export class drawable {
    public:
        virtual ~drawable() = default;

        /**
         * @brief store the accumulated world transform of the owning node
         * @param world parent_world * node.local (computed by update_world)
         */
        virtual void set_world(glm::mat4 const& world) = 0;
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief one node of the scene tree: a local transform, child nodes and an
     *        optional drawable leaf. Value semantics: children are owned inline
     *        (copying a node copies its subtree).
     */
    export struct scene_node {
        std::string name = {};             // debugging / future animation lookup
        glm::mat4 local = glm::mat4(1.0f); // local transform (T*R*S or full matrix)
        std::vector<scene_node> children = {};
        std::unique_ptr<drawable> drawable_leaf = {}; // null for transform-only nodes

        scene_node() = default;
        // unique_ptr makes the node non-copyable; define an explicit clone for subtree copies
        scene_node(scene_node&&) noexcept = default;
        scene_node& operator=(scene_node&&) noexcept = default;
        scene_node(scene_node const&) = delete;
        scene_node& operator=(scene_node const&) = delete;

        /** @brief deep-copy this subtree (children and all) */
        [[nodiscard]] scene_node clone() const;
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief a named scene: a list of root nodes (mirrors gltf::scene's shape)
     */
    export struct scene {
        std::string name = {};
        std::vector<scene_node> roots = {};
    };

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief depth-first walk that accumulates world transforms and pushes them
     *        into every drawable leaf: world(child) = world(parent) * child.local
     * @param node subtree root to walk (call once per scene root with mat4(1))
     * @param parent_world accumulated world of this node's parent
     */
    export void update_world(scene_node& node, glm::mat4 const& parent_world);

    /**
     * @ingroup vulkan_runtime_scene_tree
     * @brief walk the subtree and call @p visit on every drawable leaf
     * @tparam F invocable(scene_node const&, glm::mat4 const& world)
     */
    export template <class F>
    void visit_drawables(scene_node const& node, glm::mat4 const& parent_world, F&& visit) {
        glm::mat4 const world = parent_world * node.local;
        if (node.drawable_leaf) {
            visit(node, world);
        }
        for (scene_node const& child : node.children) {
            visit_drawables(child, world, visit);
        }
    }
} // namespace vulkan::scene_tree
