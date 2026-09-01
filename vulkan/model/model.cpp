module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>

module vulkan.model;
import vulkan.core;

namespace vulkan {
    void model::draw(VkCommandBuffer const command_buffer) const {
        // The pipeline and the shared scene descriptor set are bound by the runtime once per
        // frame; a model only binds its geometry, pushes its material constants and draws.
        constexpr VkDeviceSize vertex_offset = 0;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &this->vertex_detail->buffer, &vertex_offset);
        vkCmdBindIndexBuffer(command_buffer, this->index_detail->buffer, 0, this->index_type);

        vkCmdPushConstants(command_buffer,
                           this->pipeline->get_pipeline_layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(this->push),
                           &this->push);
        vkCmdDrawIndexed(command_buffer, this->index_count, 1, 0, 0, 0);
    }

    void model::destroy(vma_allocator& vma) noexcept {
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

    bool model::is_valid() const noexcept {
        return this->vertex_detail != nullptr && this->index_detail != nullptr &&
               this->index_count != 0 && this->pipeline != nullptr;
    }

    camera_ubo make_orbit_camera_ubo(
        float const yaw,
        float const pitch,
        float const distance,
        float const aspect) {
        // Orbit camera position: scene centered at the origin, camera orbits spherically
        float const cp = std::cos(pitch);
        glm::vec3 const eye(distance * cp * std::sin(yaw),
                            distance * std::sin(pitch),
                            distance * cp * std::cos(yaw));

        // RH_ZO: right-handed + depth [0,1] (Vulkan convention)
        glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        // glm's projection follows the OpenGL convention (NDC y up), but Vulkan framebuffers are y-down:
        // flip the projection's Y, otherwise glTF's CCW front-face winding becomes CW in the framebuffer
        // and is culled by the pipeline's CULL_BACK, leaving only the model's interior visible.
        proj[1][1] *= -1.0f;

        camera_ubo ubo;
        ubo.view = glm::lookAt(eye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj = proj;
        ubo.camera_pos = eye;
        return ubo;
    }
} // namespace vulkan
