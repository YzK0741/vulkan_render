module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>

module vulkan.model;
import vulkan.core;

namespace vulkan {
    void primitive::set_world(glm::mat4 const& world) {
        // the accumulated world transform written by a scene tree walk (scene_tree::primitive
        // interface); draw() pushes push.model verbatim, so this is all the leaf needs
        this->push.model = world;
    }

    // shared recording for draw strategies that render this object's own geometry with its
    // push constants (normal_draw_primitive; instanced_draw_primitive overrides both pieces)
    void primitive::bind_geometry_and_push(VkCommandBuffer const command_buffer) const {
        constexpr VkDeviceSize vertex_offset = 0;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &this->vertex_detail->buffer, &vertex_offset);
        vkCmdBindIndexBuffer(command_buffer, this->index_detail->buffer, 0, this->index_type);

        vkCmdPushConstants(command_buffer,
                           this->pipeline->get_pipeline_layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(this->push),
                           &this->push);
    }

    void normal_draw_primitive::draw(VkCommandBuffer const command_buffer) const {
        // double-sided materials keep back faces (dynamic cull mode, Vulkan 1.3 core)
        vkCmdSetCullMode(command_buffer, this->double_sided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);
        this->bind_geometry_and_push(command_buffer);
        vkCmdDrawIndexed(command_buffer, this->index_count, 1, 0, 0, 0);
    }

    void normal_draw_primitive::destroy(vma_allocator& vma) noexcept {
        if (this->vertex_buffer_handle != 0) {
            vma.free_buffer(this->vertex_buffer_handle);
        }
        if (this->index_buffer_handle != 0) {
            vma.free_buffer(this->index_buffer_handle);
        }
        this->vertex_buffer_handle = 0;
        this->vertex_detail = nullptr;
        this->index_buffer_handle = 0;
        this->index_detail = nullptr;
        this->index_count = 0;
        this->vertex_count = 0;
    }

    bool normal_draw_primitive::is_valid() const noexcept {
        return this->vertex_detail != nullptr && this->index_detail != nullptr &&
               this->index_count != 0 && this->pipeline != nullptr;
    }

    void instanced_draw_primitive::draw(VkCommandBuffer const command_buffer) const {
        // geometry belongs to source: bind ITS buffers, then draw it instance_count times;
        // push flag bit0 makes pbr.vert pick instances[gl_InstanceIndex] per instance
        primitive const& geometry_source = *this->source;
        vkCmdSetCullMode(command_buffer, this->double_sided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);
        constexpr VkDeviceSize vertex_offset = 0;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &geometry_source.vertex_detail->buffer, &vertex_offset);
        vkCmdBindIndexBuffer(command_buffer, geometry_source.index_detail->buffer, 0, geometry_source.index_type);

        vkCmdPushConstants(command_buffer,
                           this->pipeline->get_pipeline_layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(this->push),
                           &this->push);
        vkCmdDrawIndexed(command_buffer, geometry_source.index_count, this->instance_count, 0, 0, 0);
    }

    void instanced_draw_primitive::destroy(vma_allocator&) noexcept {
        // owns nothing: the instance transform buffer is runtime-owned, geometry is source's
    }

    bool instanced_draw_primitive::is_valid() const noexcept {
        return this->source != nullptr && this->source->is_valid() && this->instance_count != 0 &&
               this->pipeline != nullptr;
    }

    camera_ubo make_orbit_camera_ubo(
        float const yaw,
        float const pitch,
        float const distance,
        glm::vec3 const& target,
        float const aspect) {
        // Orbit camera: the eye orbits the target point spherically
        float const cp = std::cos(pitch);
        glm::vec3 const eye(target + glm::vec3(distance * cp * std::sin(yaw),
                                               distance * std::sin(pitch),
                                               distance * cp * std::cos(yaw)));

        // RH_ZO: right-handed + depth [0,1] (Vulkan convention)
        glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(45.0f), aspect, 0.1f, 100.0f);
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
        glm::vec3 const light_dir = glm::normalize(glm::vec3(0.3f, 1.0f, 0.5f));

        // Orthographic shadow frustum framing the scene's bounding sphere:
        //  - the light sits at scene_center - dir * 2r, looking back at scene_center
        //  - the sphere spans [r, 3r] along the light direction (center at 2r), so the near/far
        //    planes with a margin around it cover every caster
        //  - the ortho box half-extent is the sphere radius (plus margin): any point of the
        //    sphere projects within it, so nothing casts outside the shadow map
        float const r = scene_radius;
        glm::vec3 const eye = scene_center - light_dir * (2.0f * r);
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
        return ubo;
    }
} // namespace vulkan
