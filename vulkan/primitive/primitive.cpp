module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>

module vulkan.primitive;
namespace vulkan {
    void primitive::set_world(glm::mat4 const& world) {
        // the accumulated world transform written by a scene tree walk (scene_tree::primitive
        // interface); draw() pushes push.model verbatim, so this is all the leaf needs
        this->push.model = world;
    }

    // shared recording for draw strategies that render this object's own geometry with its
    // push constants (normal_draw_primitive; instanced_draw_primitive overrides both pieces),
    // recorded onto the environment's command buffer. The push-constant layout is the
    // environment's shared scene layout - every pipeline shares it, so pushing does not depend
    // on which pipeline is currently bound.
    void primitive::bind_geometry_and_push(render_environment const& env) const {
        VkCommandBuffer const command_buffer = env.command_buffer;
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
    // independent): double-sided materials keep back faces. Transparent (alphaMode BLEND)
    // leaves disable depth writes so they blend onto whatever is behind them. All commands
    // record onto env.command_buffer.
    void normal_draw_primitive::draw(render_environment& env) const {
        env.bind_default();
        env.set_depth_write(!this->transparent);
        VkCommandBuffer const command_buffer = env.command_buffer;
        env.set_cull_mode(this->double_sided);
        this->bind_geometry_and_push(env);
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

    void instanced_draw_primitive::draw(render_environment& env) const {
        env.bind_default();
        env.set_depth_write(!this->transparent);
        VkCommandBuffer const command_buffer = env.command_buffer;
        // geometry belongs to source: bind ITS buffers, then draw it instance_count times;
        // push flag bit0 makes pbr.vert pick instances[gl_InstanceIndex] per instance
        primitive const& geometry_source = *this->source;
        env.set_cull_mode(this->double_sided);
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

    void static_draw_primitive::draw(render_environment& env) const {
        env.bind_default();
        env.set_depth_write(!this->transparent);
        VkCommandBuffer const command_buffer = env.command_buffer;
        // ONE bind for the whole merged geometry, then one offset draw per chunk (each chunk
        // pushes its own material_index — the batch shares push.model, set by update_world)
        constexpr VkDeviceSize vertex_offset_bytes = 0;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &this->vertex_detail->buffer, &vertex_offset_bytes);
        vkCmdBindIndexBuffer(command_buffer, this->index_detail->buffer, 0, this->index_type);

        // chunked: per chunk set the cull mode + material_index (push.material_index is the
        // first field, so only that slice needs re-pushing; model stays from the base push).
        // The chunk table is validated at make_static_draw() time (in-range index windows and
        // vertex references), so no draw can go out of bounds.
        for (chunk_record const& chunk : this->chunks) {
            env.set_cull_mode(chunk.double_sided);
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
        if (this->vertex_detail == nullptr || this->index_detail == nullptr || this->chunks.empty()) {
            return false; // a validated, non-empty chunk table is required (see make_static_draw)
        }
        return std::ranges::all_of(this->chunks, [](chunk_record const& c) { return c.index_count != 0; });
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

    light_ubo make_directional_light_ubo(glm::vec3 const& scene_center, float const scene_radius, float const shadow_map_size) {
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
        // Default: ONE cascade covering the whole scene sphere - the fit runtime::update_shadow_frustum
        // replaces on the first frame (and the cascaded version fills the same slots with one fitted
        // matrix per cascade). Every lane gets the same matrix here so a frame that renders before the
        // first fit still samples something sound.
        for (auto& matrix : ubo.light_view_proj) {
            matrix = proj * view;
        }
        ubo.light_dir = glm::vec4(light_dir, 1.0f / shadow_map_size); // w: uv texel size for the pcf taps
        ubo.shadow_enabled = 1.0f;                                    // shadows on by default; runtime::set_shadow_enabled flips it
        ubo.brdf_model = 0.0f;                                        // defaults: GGX + joint Smith, Lambert (see light_ubo docs)
        ubo.diffuse_model = 0.0f;
        float const default_texel_world = (2.0f * half) / shadow_map_size;
        ubo.cascade_texel_world = glm::vec4(default_texel_world); // world size of one shadow texel
        ubo.cascade_count = 1.0f;
        return ubo;
    }
} // namespace vulkan
