module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>

module vulkan.runtime.scene_tree;
import vulkan.core;

namespace vulkan::scene_tree {
    void update_world(scene_node& node, glm::mat4 const& parent_world) {
        glm::mat4 const world = parent_world * node.local;
        if (node.primitive_leaf) {
            node.primitive_leaf->set_world(world);
        }
        for (scene_node& child : node.children) {
            update_world(child, world);
        }
    }

    // DFS pre-order over every root of the scene: the iterator holds an explicit stack of
    // {node, depth} frames; ++ pops the visited node and pushes its children (reversed so the
    // first child is visited next). Only valid while the tree structure is frozen (see
    // scene_iterator in scene_tree.cppm): no make/import/clear / child-vector pushes while
    // iterators live.
    scene_iterator::scene_iterator(scene& owner) {
        for (auto it = owner.roots.rbegin(); it != owner.roots.rend(); ++it) {
            this->stack.emplace_back(&*it, 0); // roots sit at depth 0
        }
        this->exhausted = this->stack.empty();
    }

    scene_node& scene_iterator::operator*() const noexcept {
        return *this->stack.back().first;
    }

    scene_node* scene_iterator::operator->() const noexcept {
        return this->stack.back().first;
    }

    scene_iterator& scene_iterator::operator++() {
        scene_node* const current = this->stack.back().first;
        std::size_t const child_depth = this->stack.back().second + 1;
        this->stack.pop_back();
        for (auto it = current->children.rbegin(); it != current->children.rend(); ++it) {
            this->stack.emplace_back(&*it, child_depth);
        }
        if (this->stack.empty()) {
            this->exhausted = true;
        }
        return *this;
    }

    scene_node scene_node::clone() const {
        scene_node copy;
        copy.name = this->name;
        copy.local = this->local;
        copy.children.reserve(this->children.size());
        for (scene_node const& child : this->children) {
            copy.children.push_back(child.clone());
        }
        // primitive leaves are intentionally NOT cloned: their ownership is
        // external (the runtime registers them once); copy preserves null
        return copy;
    }

    scene_node& scene_node::add_child() {
        this->children.emplace_back();
        return this->children.back();
    }

    scene_node& scene_node::add_child(scene_node child) {
        this->children.push_back(std::move(child));
        return this->children.back();
    }

    primitive* scene_node::attach(std::unique_ptr<primitive> leaf) {
        if (leaf == nullptr) {
            return nullptr;
        }
        primitive* const result = leaf.get();
        this->primitive_leaf = std::move(leaf);
        return result;
    }

    // iterative DFS (explicit stack, mirrors import_scene's traversal style): visit the node,
    // then its children in order; first name match wins (pre-order).
    scene_node* scene_node::find_node(std::string_view const name) noexcept {
        if (name.empty()) {
            return nullptr; // empty names are the unnamed default; never match
        }
        std::vector<scene_node*> stack;
        stack.push_back(this);
        while (!stack.empty()) {
            scene_node* const current = stack.back();
            stack.pop_back();
            if (current->name == name) {
                return current;
            }
            // push children reversed so the first child is visited next (DFS pre-order)
            for (auto it = current->children.rbegin(); it != current->children.rend(); ++it) {
                stack.push_back(&*it);
            }
        }
        return nullptr;
    }

    scene_node const* scene_node::find_node(std::string_view const name) const noexcept {
        if (name.empty()) {
            return nullptr;
        }
        std::vector<scene_node const*> stack;
        stack.push_back(this);
        while (!stack.empty()) {
            scene_node const* const current = stack.back();
            stack.pop_back();
            if (current->name == name) {
                return current;
            }
            for (auto it = current->children.rbegin(); it != current->children.rend(); ++it) {
                stack.push_back(&*it);
            }
        }
        return nullptr;
    }

    scene_node& scene::add_root() {
        this->roots.emplace_back();
        return this->roots.back();
    }

    scene_node& scene::add_root(scene_node root) {
        this->roots.push_back(std::move(root));
        return this->roots.back();
    }

    scene_node* scene::find_node(std::string_view const name) noexcept {
        for (scene_node& root : this->roots) {
            if (scene_node* const found = root.find_node(name)) {
                return found;
            }
        }
        return nullptr;
    }

    scene_node const* scene::find_node(std::string_view const name) const noexcept {
        for (scene_node const& root : this->roots) {
            if (scene_node const* const found = root.find_node(name)) {
                return found;
            }
        }
        return nullptr;
    }
} // namespace vulkan::scene_tree

namespace vulkan {
    void primitive::set_world(glm::mat4 const& world) {
        // the accumulated world transform written by a scene tree walk (scene_tree::primitive
        // interface); draw() pushes push.model verbatim, so this is all the leaf needs
        this->push.model = world;
    }

    // shared recording for draw strategies that render this object's own geometry with its
    // push constants (normal_draw_primitive; instanced_draw_primitive overrides both pieces).
    // The push-constant layout is the environment's shared scene layout - every pipeline shares
    // it, so pushing does not depend on which pipeline is currently bound.
    void primitive::bind_geometry_and_push(VkCommandBuffer const command_buffer, render_environment const& env) const {
        constexpr VkDeviceSize vertex_offset = 0;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &this->vertex_detail->buffer, &vertex_offset);
        vkCmdBindIndexBuffer(command_buffer, this->index_detail->buffer, 0, this->index_type);

        vkCmdPushConstants(command_buffer,
                           env.layout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(this->push),
                           &this->push);
    }

    // Default-semantics draws (normal / instanced / static): request the recording session's
    // default pipeline - bind_default() no-ops when it is already bound, so consecutive leaves
    // of the same pass share one bind. Cull mode stays per draw (dynamic state, pipeline
    // independent): double-sided materials keep back faces.
    void normal_draw_primitive::draw(VkCommandBuffer const command_buffer, render_environment& env) const {
        env.bind_default(command_buffer);
        vkCmdSetCullMode(command_buffer, this->double_sided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);
        this->bind_geometry_and_push(command_buffer, env);
        vkCmdDrawIndexed(command_buffer, this->index_count, 1, 0, 0, 0);
    }

    void normal_draw_primitive::destroy(vma_allocator&) noexcept {
        // geometry is owned by the vk_buffer members and released when this primitive (the tree
        // node's leaf) is destroyed; here we only drop the cached accessors so a dangling detail
        // pointer can never be used after the owner went away
        this->vertex_buffer.reset();
        this->vertex_detail = nullptr;
        this->index_buffer.reset();
        this->index_detail = nullptr;
        this->index_count = 0;
        this->vertex_count = 0;
    }

    bool normal_draw_primitive::is_valid() const noexcept {
        return this->vertex_detail != nullptr && this->index_detail != nullptr &&
               this->index_count != 0;
    }

    void instanced_draw_primitive::draw(VkCommandBuffer const command_buffer, render_environment& env) const {
        env.bind_default(command_buffer);
        // geometry belongs to source: bind ITS buffers, then draw it instance_count times;
        // push flag bit0 makes pbr.vert pick instances[gl_InstanceIndex] per instance
        primitive const& geometry_source = *this->source;
        vkCmdSetCullMode(command_buffer, this->double_sided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);
        constexpr VkDeviceSize vertex_offset = 0;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &geometry_source.vertex_detail->buffer, &vertex_offset);
        vkCmdBindIndexBuffer(command_buffer, geometry_source.index_detail->buffer, 0, geometry_source.index_type);

        vkCmdPushConstants(command_buffer,
                           env.layout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(this->push),
                           &this->push);
        vkCmdDrawIndexed(command_buffer, geometry_source.index_count, this->instance_count, 0, 0, 0);
    }

    void instanced_draw_primitive::destroy([[maybe_unused]] vma_allocator& vma) noexcept {
        // owns nothing: the instance transform buffer is runtime-owned, geometry is source's
    }

    bool instanced_draw_primitive::is_valid() const noexcept {
        return this->source != nullptr && this->source->is_valid() && this->instance_count != 0;
    }

    void static_draw_primitive::draw(VkCommandBuffer const command_buffer, render_environment& env) const {
        env.bind_default(command_buffer);
        // ONE bind for the whole merged geometry, then one offset draw per chunk (each chunk
        // pushes its own material_index — the batch shares push.model, set by update_world)
        constexpr VkDeviceSize vertex_offset_bytes = 0;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &this->vertex_detail->buffer, &vertex_offset_bytes);
        vkCmdBindIndexBuffer(command_buffer, this->index_detail->buffer, 0, this->index_type);

        if (this->chunks.empty()) {
            // degenerate: draw the whole merged range once (plain normal draw)
            vkCmdSetCullMode(command_buffer, this->double_sided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);
            vkCmdPushConstants(command_buffer,
                               env.layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0,
                               sizeof(this->push),
                               &this->push);
            vkCmdDrawIndexed(command_buffer, this->index_count, 1, 0, 0, 0);
            return;
        }

        // chunked: per chunk set the cull mode + material_index (push.material_index is the
        // first field, so only that slice needs re-pushing; model stays from the base push)
        for (chunk_record const& chunk : this->chunks) {
            vkCmdSetCullMode(command_buffer, chunk.double_sided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);
            material_push_constants const chunk_push = [&] {
                material_push_constants p = this->push; // model + flags already correct
                p.material_index = chunk.material_index;
                return p;
            }();
            vkCmdPushConstants(command_buffer,
                               env.layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0,
                               sizeof(chunk_push),
                               &chunk_push);
            vkCmdDrawIndexed(command_buffer,
                             chunk.index_count,
                             1,
                             chunk.first_index,
                             static_cast<int32_t>(chunk.vertex_offset),
                             0);
        }
    }

    void static_draw_primitive::destroy(vma_allocator&) noexcept {
        // owns the merged buffers: releasing them (and the cached detail pointers) frees the
        // GPU memory when the last copy of the vk_buffer members goes away
        this->vertex_buffer.reset();
        this->vertex_detail = nullptr;
        this->index_buffer.reset();
        this->index_detail = nullptr;
        this->index_count = 0;
        this->vertex_count = 0;
        this->chunks.clear();
    }

    bool static_draw_primitive::is_valid() const noexcept {
        if (this->vertex_detail == nullptr || this->index_detail == nullptr) {
            return false;
        }
        return this->chunks.empty()
                   ? this->index_count != 0 // degenerate whole-range draw
                   : std::all_of(this->chunks.begin(), this->chunks.end(), [](chunk_record const& c) { return c.index_count != 0; });
    }

    camera_ubo make_orbit_camera_ubo(
        float const yaw,
        float const pitch,
        float const distance,
        glm::vec3 const& target,
        float const scene_radius,
        float const aspect) {
        // Orbit camera: the eye orbits the target point spherically
        float const cp = std::cos(pitch);
        glm::vec3 const eye(target + glm::vec3(distance * cp * std::sin(yaw),
                                               distance * std::sin(pitch),
                                               distance * cp * std::cos(yaw)));

        // RH_ZO: right-handed + depth [0,1] (Vulkan convention). The far plane always covers the
        // whole scene: the farthest visible point sits at target + scene_radius, i.e. at most
        // distance + scene_radius from the eye, so far >= distance + scene_radius (with margin).
        // Zooming in (small distance) must NOT shrink the far plane below that - a distance-follow
        // far (e.g. 8 * distance) clips the scene's far side exactly when the camera gets close,
        // making objects vanish. Zooming out keeps the historic generous far (max with 8*distance).
        float const far_plane = std::max(100.0f, std::max(distance + 2.0f * scene_radius, 8.0f * distance));
        glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(45.0f), aspect, 0.1f, far_plane);
        // glm's projection follows the OpenGL convention (NDC y up), but Vulkan framebuffers are y-down:
        // flip the projection's Y, otherwise glTF's CCW front-face winding becomes CW in the framebuffer
        // and is culled by the pipeline's CULL_BACK, leaving only the object's interior visible.
        proj[1][1] *= -1.0f;

        camera_ubo ubo;
        ubo.view = glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj = proj;
        ubo.camera_pos = eye;
        return ubo;
    }

    light_ubo make_directional_light_ubo(glm::vec3 const& scene_center, float const scene_radius) {
        // The light direction must match the analytic sky sun (see skybox.frag): the PBR direct
        // light, the visible sun disc and the shadow map all share this single fixed direction.
        // light_dir points TOWARD the sun in the sky (pbr.frag treats it as the surface-to-light
        // vector), so the sun's rays travel -light_dir and the shadow camera must sit UP-SUN.
        glm::vec3 const light_dir = glm::normalize(glm::vec3(0.3f, 1.0f, 0.5f));

        // Orthographic shadow frustum framing the scene's bounding sphere:
        //  - the light sits up-sun at scene_center + dir * 2r (above the scene for a sky sun),
        //    looking down along -dir at scene_center, matching the rays of the real sun
        //  - the sphere spans [r, 3r] along the light direction (center at 2r), so the near/far
        //    planes with a margin around it cover every caster
        //  - the ortho box half-extent is the sphere radius (plus margin): any point of the
        //    sphere projects within it, so nothing casts outside the shadow map
        float const r = scene_radius;
        glm::vec3 const eye = scene_center + light_dir * (2.0f * r);
        glm::mat4 const view = glm::lookAt(eye, scene_center, glm::vec3(0.0f, 1.0f, 0.0f));

        float const half = r * 1.1f;
        glm::mat4 proj = glm::orthoRH_ZO(-half, half, -half, half, r * 0.5f, r * 3.5f);
        // Same y-flip convention as the camera projection (see make_orbit_camera_ubo): Vulkan
        // framebuffers are y-down, so the light view-proj must flip Y too, otherwise the shadow
        // pass renders the scene mirrored and the sampled shadow UVs would not match it.
        proj[1][1] *= -1.0f;

        light_ubo ubo;
        ubo.light_view_proj = proj * view;
        ubo.light_dir = glm::vec4(light_dir, 0.0f);
        ubo.shadow_enabled = 1.0f; // shadows on by default; runtime::set_shadow_enabled flips it
        ubo.pad[0] = 0.0f;
        ubo.pad[1] = 0.0f;
        ubo.pad[2] = 0.0f;
        return ubo;
    }
} // namespace vulkan
