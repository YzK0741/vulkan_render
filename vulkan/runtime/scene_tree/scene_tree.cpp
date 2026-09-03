module;

#include <glm/glm.hpp>

module vulkan.runtime.scene_tree;

namespace vulkan::scene_tree {
    void update_world(scene_node& node, glm::mat4 const& parent_world) {
        glm::mat4 const world = parent_world * node.local;
        if (node.drawable_leaf) {
            node.drawable_leaf->set_world(world);
        }
        for (scene_node& child : node.children) {
            update_world(child, world);
        }
    }

    scene_node scene_node::clone() const {
        scene_node copy;
        copy.name = this->name;
        copy.local = this->local;
        copy.children.reserve(this->children.size());
        for (scene_node const& child : this->children) {
            copy.children.push_back(child.clone());
        }
        // drawable leaves are intentionally NOT cloned: their ownership is
        // external (the runtime registers them once); copy preserves null
        return copy;
    }
} // namespace vulkan::scene_tree
