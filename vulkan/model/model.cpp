module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>

module vulkan.model;
import vulkan.core;
import utility;

namespace {
    // Upload one RGBA texture via vma (GPU-only + staging); returns the image handle (0 on failure)
    uint64_t upload_texture(vulkan::core& core, const vulkan::texture_input& texture) {
        vulkan::image_create_info create_info = {};
        create_info.width = texture.width;
        create_info.height = texture.height;
        create_info.mip_levels = 1;
        create_info.array_layers = 1;
        create_info.format = texture.format;
        return core.vma.create_image(texture.data.data(), texture.data.size_bytes(), create_info, vulkan::image_type::texture_2d);
    }
} // namespace

namespace vulkan {
    model make_model(vulkan::core& core, const vk_pipeline& pipeline, const model_create_info& info) {
        model result;
        result.pipeline = &pipeline; // points into the runtime's pipeline cache, valid for its lifetime
        result.model_matrix = info.model_matrix;

        // ---- 1. Geometry buffers ----
        result.vertex_buffer_handle = core.vma.create_buffer(info.vertex_data.data(), info.vertex_data.size_bytes(), vulkan::buffer_type::vertex);
        if (result.vertex_buffer_handle == 0) {
            utility::panic("failed to create vertex buffer");
        }
        result.vertex_detail = core.vma.get_buffer_detail(result.vertex_buffer_handle);
        if (result.vertex_detail == nullptr) {
            utility::panic("failed to get vertex buffer detail");
        }

        result.index_buffer_handle = core.vma.create_buffer(info.index_data.data(), info.index_data.size_bytes(), vulkan::buffer_type::index);
        if (result.index_buffer_handle == 0) {
            utility::panic("failed to create index buffer");
        }
        result.index_detail = core.vma.get_buffer_detail(result.index_buffer_handle);
        if (result.index_detail == nullptr) {
            utility::panic("failed to get index buffer detail");
        }

        result.index_type = info.index_type;
        result.index_count = info.index_count;
        result.vertex_count = info.vertex_count;

        // ---- 2. Set 1 material: 5 textures (binding 0-4), missing ones fall back to a 1x1 white image ----
        const std::array<texture_input, 5> textures = {
            info.albedo,
            info.metallic_roughness,
            info.normal,
            info.occlusion,
            info.emissive,
        };
        bool need_white = false;
        for (const auto& texture : textures) {
            if (!texture.valid) {
                need_white = true;
                break;
            }
        }

        // material push constants: texture-presence flags, factors keep the PBR identity defaults
        result.push.flags = 0;
        if (info.normal.valid) {
            result.push.flags |= 1u;
        }
        if (info.occlusion.valid) {
            result.push.flags |= 2u;
        }
        if (info.emissive.valid) {
            result.push.flags |= 4u;
        }

        uint64_t white_handle = 0;
        vk_image_view white_view = {};
        if (need_white) {
            constexpr std::array<unsigned char, 4> white_pixels = {255, 255, 255, 255};
            vulkan::image_create_info white_info = {};
            white_info.width = 1;
            white_info.height = 1;
            white_info.mip_levels = 1;
            white_info.array_layers = 1;
            white_info.format = VK_FORMAT_R8G8B8A8_UNORM;
            white_handle = core.vma.create_image(white_pixels.data(), white_pixels.size(), white_info, vulkan::image_type::texture_2d);
            const auto* white_detail = core.vma.get_image_detail(white_handle);
            if (white_detail == nullptr) {
                utility::panic("failed to create white fallback texture");
            }
            white_view = core.make_image_view(white_detail->image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
        }

        std::array<VkImageView, 5> texture_view_handles = {};
        for (int i = 0; i < 5; ++i) {
            if (textures[i].valid) {
                const uint64_t handle = upload_texture(core, textures[i]);
                if (handle == 0) {
                    utility::panic("failed to create material texture");
                }
                const auto* detail = core.vma.get_image_detail(handle);
                if (detail == nullptr) {
                    utility::panic("failed to get material texture detail");
                }
                result.image_handles.push_back(handle);
                result.image_views.push_back(core.make_image_view(detail->image, textures[i].format, VK_IMAGE_VIEW_TYPE_2D));
                texture_view_handles[i] = *result.image_views.back();
            } else {
                texture_view_handles[i] = *white_view;
            }
        }
        if (need_white) {
            result.image_handles.push_back(white_handle);
            result.image_views.push_back(std::move(white_view));
        }

        // ---- 3. Set 1 IBL (binding 5-7): prefiltered env cubemap, irradiance cubemap, BRDF LUT ----
        std::array<VkImageView, 3> ibl_view_handles = {};
        const bool has_ibl = info.ibl.env_size > 0;
        if (has_ibl) {
            vulkan::image_create_info env_info = {};
            env_info.width = info.ibl.env_size;
            env_info.height = info.ibl.env_size;
            env_info.mip_levels = info.ibl.env_mip_count;
            env_info.array_layers = 6;
            env_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            const uint64_t env_handle = core.vma.create_image(info.ibl.prefiltered_env.data(), info.ibl.prefiltered_env.size_bytes(), env_info, vulkan::image_type::texture_cubemap);
            if (env_handle == 0) {
                utility::panic("failed to create prefiltered environment image");
            }
            const auto* env_detail = core.vma.get_image_detail(env_handle);
            if (env_detail == nullptr) {
                utility::panic("failed to get environment image detail");
            }
            result.image_handles.push_back(env_handle);
            result.image_views.push_back(core.make_image_view(env_detail->image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_CUBE));
            ibl_view_handles[0] = *result.image_views.back();

            vulkan::image_create_info irr_info = {};
            irr_info.width = info.ibl.irr_size;
            irr_info.height = info.ibl.irr_size;
            irr_info.mip_levels = 1;
            irr_info.array_layers = 6;
            irr_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            const uint64_t irr_handle = core.vma.create_image(info.ibl.irradiance.data(), info.ibl.irradiance.size_bytes(), irr_info, vulkan::image_type::texture_cubemap);
            if (irr_handle == 0) {
                utility::panic("failed to create irradiance image");
            }
            const auto* irr_detail = core.vma.get_image_detail(irr_handle);
            if (irr_detail == nullptr) {
                utility::panic("failed to get irradiance image detail");
            }
            result.image_handles.push_back(irr_handle);
            result.image_views.push_back(core.make_image_view(irr_detail->image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_CUBE));
            ibl_view_handles[1] = *result.image_views.back();

            vulkan::image_create_info lut_info = {};
            lut_info.width = info.ibl.lut_size;
            lut_info.height = info.ibl.lut_size;
            lut_info.mip_levels = 1;
            lut_info.array_layers = 1;
            lut_info.format = VK_FORMAT_R16G16_SFLOAT;
            const uint64_t lut_handle = core.vma.create_image(info.ibl.brdf_lut.data(), info.ibl.brdf_lut.size_bytes(), lut_info, vulkan::image_type::texture_2d);
            if (lut_handle == 0) {
                utility::panic("failed to create BRDF LUT image");
            }
            const auto* lut_detail = core.vma.get_image_detail(lut_handle);
            if (lut_detail == nullptr) {
                utility::panic("failed to get BRDF LUT image detail");
            }
            result.image_handles.push_back(lut_handle);
            result.image_views.push_back(core.make_image_view(lut_detail->image, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_VIEW_TYPE_2D));
            ibl_view_handles[2] = *result.image_views.back();

            result.env_sampler = core.make_sampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, static_cast<float>(info.ibl.env_mip_count - 1));
        }

        // ---- 4. Set 1 descriptor writes: 5 textures + 3 IBL, all combined image samplers ----
        result.texture_sampler = core.make_sampler(VK_SAMPLER_ADDRESS_MODE_REPEAT, 0.25f);
        result.material_set = core.make_descriptor_set(pipeline.get_descriptor_set_layouts()[1]);

        std::array<VkDescriptorImageInfo, 8> image_infos = {};
        std::array<VkWriteDescriptorSet, 8> writes = {};
        const VkSampler texture_sampler_handle = *result.texture_sampler;
        for (int binding = 0; binding < 5; ++binding) {
            image_infos[binding] = {.sampler = texture_sampler_handle, .imageView = texture_view_handles[binding], .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[binding].dstSet = *result.material_set;
            writes[binding].dstBinding = static_cast<uint32_t>(binding);
            writes[binding].descriptorCount = 1;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[binding].pImageInfo = &image_infos[binding];
        }
        uint32_t write_count = 5;
        if (has_ibl) {
            const VkSampler env_sampler_handle = *result.env_sampler;
            for (int i = 0; i < 3; ++i) {
                const int binding = 5 + i;
                image_infos[binding] = {.sampler = env_sampler_handle, .imageView = ibl_view_handles[i], .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[binding].dstSet = *result.material_set;
                writes[binding].dstBinding = static_cast<uint32_t>(binding);
                writes[binding].descriptorCount = 1;
                writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[binding].pImageInfo = &image_infos[binding];
                ++write_count;
            }
        }
        vkUpdateDescriptorSets(core.device, write_count, writes.data(), 0, nullptr);

        // ---- 5. Set 0: per-frame camera UBO ----
        result.ubo_size = sizeof(camera_ubo);
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            camera_ubo initial = {};
            const uint64_t handle = core.vma.create_buffer(std::span(&initial, 1), vulkan::buffer_type::uniform_coherent);
            if (handle == 0) {
                utility::panic("failed to create ubo buffer");
            }
            const auto* detail = core.vma.get_buffer_detail(handle);
            if (detail == nullptr) {
                utility::panic("failed to get ubo buffer detail");
            }
            result.ubo_buffer_handles.push_back(handle);
            result.ubo_mapped.push_back(detail->allocation_info.pMappedData);

            auto set = core.make_descriptor_set(pipeline.get_descriptor_set_layouts()[0]);
            const VkDescriptorBufferInfo ubo_info{detail->buffer, 0, result.ubo_size};
            VkWriteDescriptorSet write = {};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = *set;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.pBufferInfo = &ubo_info;
            vkUpdateDescriptorSets(core.device, 1, &write, 0, nullptr);
            result.ubo_sets.push_back(std::move(set));
        }
        return result;
    }

    void model::draw(const VkCommandBuffer command_buffer, const uint32_t frame_slot) const { // NOLINT(*-misplaced-const)
        this->pipeline->begin_pipeline(command_buffer);

        // pOffsets must be a valid pointer (unlike vkCmdBindVertexBuffers2, NULL is not allowed here)
        constexpr VkDeviceSize vertex_offset = 0;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &this->vertex_detail->buffer, &vertex_offset);
        vkCmdBindIndexBuffer(command_buffer, this->index_detail->buffer, 0, this->index_type);

        const std::array<VkDescriptorSet, 2> descriptor_sets = {*this->ubo_sets[frame_slot], *this->material_set};
        vkCmdBindDescriptorSets(command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                this->pipeline->get_pipeline_layout(),
                                0,
                                descriptor_sets.size(),
                                descriptor_sets.data(),
                                0,
                                nullptr);
        vkCmdPushConstants(command_buffer,
                           this->pipeline->get_pipeline_layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(this->push),
                           &this->push);
        vkCmdDrawIndexed(command_buffer, this->index_count, 1, 0, 0, 0);
    }

    void model::update_camera_ubo(const uint32_t frame_slot, const camera_ubo& ubo) const noexcept {
        if (frame_slot < this->ubo_mapped.size() && this->ubo_mapped[frame_slot] != nullptr) {
            std::memcpy(this->ubo_mapped[frame_slot], &ubo, this->ubo_size);
        }
    }

    void model::destroy(vma_allocator& vma) noexcept {
        for (const uint64_t handle : this->ubo_buffer_handles) {
            vma.free_buffer(handle);
        }
        this->ubo_buffer_handles.clear();
        for (const uint64_t handle : this->image_handles) {
            vma.free_image(handle);
        }
        this->image_handles.clear();
        this->vertex_buffer_handle = 0;
        this->vertex_detail = nullptr;
        this->index_buffer_handle = 0;
        this->index_detail = nullptr;
        this->index_count = 0;
        this->vertex_count = 0;
    }

    bool model::is_valid() const noexcept {
        return this->vertex_detail != nullptr && this->index_detail != nullptr &&
               this->index_count != 0 && this->material_set.get() != VK_NULL_HANDLE &&
               !this->ubo_sets.empty();
    }

    camera_ubo make_orbit_camera_ubo(
        const float yaw,
        const float pitch,
        const float distance,
        const glm::mat4& model,
        const float aspect) {
        // Orbit camera position: model centered at the origin, camera orbits spherically
        const float cp = std::cos(pitch);
        const glm::vec3 eye(distance * cp * std::sin(yaw),
                            distance * std::sin(pitch),
                            distance * cp * std::cos(yaw));

        // RH_ZO: right-handed + depth [0,1] (Vulkan convention)
        glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        // glm's projection follows the OpenGL convention (NDC y up), but Vulkan framebuffers are y-down:
        // flip the projection's Y, otherwise glTF's CCW front-face winding becomes CW in the framebuffer
        // and is culled by the pipeline's CULL_BACK, leaving only the model's interior visible.
        proj[1][1] *= -1.0f;

        camera_ubo ubo;
        ubo.model = model;
        ubo.view = glm::lookAt(eye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj = proj;
        ubo.camera_pos = eye;
        return ubo;
    }
} // namespace vulkan
