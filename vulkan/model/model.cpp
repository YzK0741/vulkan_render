module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>

module vulkan.model;
import vulkan.core;

namespace vulkan {
    // shared recording for draw strategies that render this object's own geometry with its
    // push constants (normal_draw_model; instanced_draw_model overrides both pieces)
    void model::bind_geometry_and_push(VkCommandBuffer const command_buffer) const {
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

    void normal_draw_model::draw(VkCommandBuffer const command_buffer) const {
        this->bind_geometry_and_push(command_buffer);
        vkCmdDrawIndexed(command_buffer, this->index_count, 1, 0, 0, 0);
    }

    void normal_draw_model::destroy(vma_allocator& vma) noexcept {
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

    bool normal_draw_model::is_valid() const noexcept {
        return this->vertex_detail != nullptr && this->index_detail != nullptr &&
               this->index_count != 0 && this->pipeline != nullptr;
    }

    void instanced_draw_model::draw(VkCommandBuffer const command_buffer) const {
        // geometry belongs to source: bind ITS buffers, then draw it instance_count times;
        // push flag bit0 makes pbr.vert pick instances[gl_InstanceIndex] per instance
        model const& geometry_source = *this->source;
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

    void instanced_draw_model::destroy(vma_allocator&) noexcept {
        // owns nothing: the instance transform buffer is runtime-owned, geometry is source's
    }

    bool instanced_draw_model::is_valid() const noexcept {
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
        // and is culled by the pipeline's CULL_BACK, leaving only the model's interior visible.
        proj[1][1] *= -1.0f;

        camera_ubo ubo;
        ubo.view = glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj = proj;
        ubo.camera_pos = eye;
        return ubo;
    }
} // namespace vulkan
