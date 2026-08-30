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

    namespace {
        constexpr float k_pi = 3.14159265359f;

        // Cubemap face direction: texel (u, v) in [-1, 1] -> unit direction (Vulkan/GL cubemap convention)
        glm::vec3 cube_face_direction(const int face, const float u, const float v) {
            switch (face) {
            case 0:
                return glm::normalize(glm::vec3(1.0f, -v, -u)); // +X
            case 1:
                return glm::normalize(glm::vec3(-1.0f, -v, u)); // -X
            case 2:
                return glm::normalize(glm::vec3(u, 1.0f, v)); // +Y
            case 3:
                return glm::normalize(glm::vec3(u, -1.0f, -v)); // -Y
            case 4:
                return glm::normalize(glm::vec3(u, -v, 1.0f)); // +Z
            default:
                return glm::normalize(glm::vec3(-u, -v, -1.0f)); // -Z
            }
        }

        // Procedural environment (HDR): gradient sky/ground + sun disc
        glm::vec3 environment_color(const glm::vec3& dir) {
            const float t = std::clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
            const glm::vec3 ground = glm::vec3(0.03f, 0.03f, 0.05f) * 0.75f;
            const glm::vec3 horizon = glm::vec3(0.16f, 0.19f, 0.26f) * 0.75f;
            const glm::vec3 sky = glm::vec3(0.28f, 0.45f, 0.75f) * 0.75f;
            glm::vec3 env = t < 0.5f ? glm::mix(ground, horizon, t * 2.0f) : glm::mix(horizon, sky, (t - 0.5f) * 2.0f);
            const glm::vec3 sun_dir = glm::normalize(glm::vec3(0.3f, 1.0f, 0.5f));
            const float sun = std::pow(std::max(glm::dot(dir, sun_dir), 0.0f), 64.0f);
            env += glm::vec3(1.0f, 0.95f, 0.85f) * sun * 1.5f; // slightly stronger sun disc for clearer metallic highlights
            return env;
        }

        // Nearest-neighbor cubemap sampling (smooth enough after summed-area averaging)
        glm::vec3 sample_cubemap(const std::vector<float>& data, const int size, const glm::vec3& dir) {
            const float ax = std::abs(dir.x), ay = std::abs(dir.y), az = std::abs(dir.z);
            int face = 0;
            float u = 0.0f;
            float v = 0.0f;
            if (ax >= ay && ax >= az) {
                face = dir.x >= 0.0f ? 0 : 1;
                u = face == 0 ? -dir.z : dir.z;
                v = -dir.y;
            } else if (ay >= ax && ay >= az) {
                face = dir.y >= 0.0f ? 2 : 3;
                u = dir.x;
                v = face == 2 ? dir.z : -dir.z;
            } else {
                face = dir.z >= 0.0f ? 4 : 5;
                u = face == 4 ? dir.x : -dir.x;
                v = -dir.y;
            }
            const int px = std::clamp(static_cast<int>((u * 0.5f + 0.5f) * size), 0, size - 1);
            const int py = std::clamp(static_cast<int>((v * 0.5f + 0.5f) * size), 0, size - 1);
            const size_t offset = (static_cast<size_t>(face) * size * size + static_cast<size_t>(py) * size + px) * 4;
            return glm::vec3(data[offset], data[offset + 1], data[offset + 2]);
        }

        // Van der Corput sequence (second component of Hammersley)
        float radical_inverse_vdc(uint32_t bits) {
            bits = (bits << 16u) | (bits >> 16u);
            bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
            bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
            bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
            bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
            return static_cast<float>(bits) * 2.3283064365386963e-10f;
        }

        glm::vec2 hammersley(const uint32_t i, const uint32_t n) {
            return glm::vec2(static_cast<float>(i) / static_cast<float>(n), radical_inverse_vdc(i));
        }

        // GGX importance sampling: build the half vector from Hammersley samples
        glm::vec3 importance_sample_ggx(const glm::vec2& xi, const glm::vec3& n, const float roughness) {
            const float a = roughness * roughness;
            const float phi = 2.0f * k_pi * xi.x;
            const float cos_theta = std::sqrt((1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y));
            const float sin_theta = std::sqrt(std::max(1.0f - cos_theta * cos_theta, 0.0f));
            const glm::vec3 h(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
            const glm::vec3 up = std::abs(n.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
            const glm::vec3 tangent = glm::normalize(glm::cross(up, n));
            const glm::vec3 bitangent = glm::cross(n, tangent);
            return glm::normalize(tangent * h.x + bitangent * h.y + n * h.z);
        }

        // IEEE 754 binary32 -> binary16 (truncated; plenty for ambient light)
        uint16_t float_to_half(const float value) {
            const uint32_t bits = std::bit_cast<uint32_t>(value);
            const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
            const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
            const uint32_t mantissa = bits & 0x7FFFFFu;
            if (exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00u); // infinity
            }
            if (exponent <= 0) {
                return sign; // subnormal/zero -> 0
            }
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
        }
    } // namespace

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

    std::vector<float> generate_environment_cubemap(const int size) {
        std::vector<float> data(static_cast<size_t>(6) * size * size * 4);
        for (int face = 0; face < 6; ++face) {
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    const float u = (static_cast<float>(x) + 0.5f) / size * 2.0f - 1.0f;
                    const float v = (static_cast<float>(y) + 0.5f) / size * 2.0f - 1.0f;
                    const glm::vec3 color = environment_color(cube_face_direction(face, u, v));
                    const size_t offset = (static_cast<size_t>(face) * size * size + static_cast<size_t>(y) * size + x) * 4;
                    data[offset + 0] = color.r;
                    data[offset + 1] = color.g;
                    data[offset + 2] = color.b;
                    data[offset + 3] = 1.0f;
                }
            }
        }
        return data;
    }

    std::vector<float> prefilter_environment(const std::vector<float>& env, const int env_size, const int mip_count) {
        std::vector<float> result;
        for (int mip = 0; mip < mip_count; ++mip) {
            const int mip_size = std::max(1, env_size >> mip);
            const float roughness = static_cast<float>(mip) / static_cast<float>(mip_count - 1);
            const uint32_t sample_count = static_cast<uint32_t>(std::max(4, 64 >> mip));
            std::vector<float> mip_data(static_cast<size_t>(6) * mip_size * mip_size * 4, 0.0f);
            for (int face = 0; face < 6; ++face) {
                for (int y = 0; y < mip_size; ++y) {
                    for (int x = 0; x < mip_size; ++x) {
                        const float u = (static_cast<float>(x) + 0.5f) / mip_size * 2.0f - 1.0f;
                        const float v = (static_cast<float>(y) + 0.5f) / mip_size * 2.0f - 1.0f;
                        const glm::vec3 n = cube_face_direction(face, u, v);
                        glm::vec3 sum(0.0f);
                        float total_weight = 0.0f;
                        for (uint32_t i = 0; i < sample_count; ++i) {
                            const glm::vec3 h = importance_sample_ggx(hammersley(i, sample_count), n, roughness);
                            const glm::vec3 l = glm::normalize(2.0f * glm::dot(n, h) * h - n);
                            const float ndotl = glm::dot(n, l);
                            if (ndotl > 0.0f) {
                                sum += sample_cubemap(env, env_size, l) * ndotl;
                                total_weight += ndotl;
                            }
                        }
                        const glm::vec3 color = total_weight > 0.0f ? sum / total_weight : glm::vec3(0.0f);
                        const size_t offset = (static_cast<size_t>(face) * mip_size * mip_size + static_cast<size_t>(y) * mip_size + x) * 4;
                        mip_data[offset + 0] = color.r;
                        mip_data[offset + 1] = color.g;
                        mip_data[offset + 2] = color.b;
                        mip_data[offset + 3] = 1.0f;
                    }
                }
            }
            result.insert(result.end(), mip_data.begin(), mip_data.end());
        }
        return result;
    }

    std::vector<float> generate_irradiance_map(const std::vector<float>& env, const int env_size, const int irr_size) {
        std::vector<float> result(static_cast<size_t>(6) * irr_size * irr_size * 4, 0.0f);
        constexpr uint32_t sample_count = 512;
        for (int face = 0; face < 6; ++face) {
            for (int y = 0; y < irr_size; ++y) {
                for (int x = 0; x < irr_size; ++x) {
                    const float u = (static_cast<float>(x) + 0.5f) / irr_size * 2.0f - 1.0f;
                    const float v = (static_cast<float>(y) + 0.5f) / irr_size * 2.0f - 1.0f;
                    const glm::vec3 n = cube_face_direction(face, u, v);
                    const glm::vec3 up = std::abs(n.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
                    const glm::vec3 tangent = glm::normalize(glm::cross(up, n));
                    const glm::vec3 bitangent = glm::cross(n, tangent);
                    glm::vec3 sum(0.0f);
                    float total_weight = 0.0f;
                    for (uint32_t i = 0; i < sample_count; ++i) {
                        const glm::vec2 xi = hammersley(i, sample_count);
                        const float phi = 2.0f * k_pi * xi.x;
                        const float cos_theta = std::sqrt(xi.y);
                        const float sin_theta = std::sqrt(std::max(1.0f - xi.y, 0.0f));
                        const glm::vec3 local(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
                        const glm::vec3 l = glm::normalize(tangent * local.x + bitangent * local.y + n * local.z);
                        sum += sample_cubemap(env, env_size, l) * cos_theta;
                        total_weight += cos_theta;
                    }
                    const glm::vec3 color = total_weight > 0.0f ? sum / total_weight : glm::vec3(0.0f);
                    const size_t offset = (static_cast<size_t>(face) * irr_size * irr_size + static_cast<size_t>(y) * irr_size + x) * 4;
                    result[offset + 0] = color.r;
                    result[offset + 1] = color.g;
                    result[offset + 2] = color.b;
                    result[offset + 3] = 1.0f;
                }
            }
        }
        return result;
    }

    std::vector<float> generate_brdf_lut(const int size) {
        std::vector<float> result(static_cast<size_t>(size) * size * 2);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                const float ndotv = (static_cast<float>(x) + 0.5f) / size;
                const float roughness = (static_cast<float>(y) + 0.5f) / size;
                const glm::vec4 c0(-1.0f, -0.0275f, -0.572f, 0.022f);
                const glm::vec4 c1(1.0f, 0.0425f, 1.04f, -0.04f);
                const glm::vec4 r = roughness * c0 + c1;
                const float a004 = std::min(r.x * r.x, std::exp2(-9.28f * ndotv)) * r.x + r.y;
                result[static_cast<size_t>(y) * size * 2 + x * 2 + 0] = -1.04f * a004 + r.z;
                result[static_cast<size_t>(y) * size * 2 + x * 2 + 1] = 1.04f * a004 + r.w;
            }
        }
        return result;
    }

    std::vector<unsigned char> to_half_rgba(const std::vector<float>& data) {
        std::vector<unsigned char> out(data.size() * 2);
        for (size_t i = 0; i < data.size(); ++i) {
            const uint16_t h = float_to_half(data[i]);
            out[i * 2 + 0] = static_cast<unsigned char>(h & 0xFFu);
            out[i * 2 + 1] = static_cast<unsigned char>(h >> 8);
        }
        return out;
    }

    std::vector<unsigned char> to_half_rg(const std::vector<float>& data) {
        std::vector<unsigned char> out(data.size() * 2);
        for (size_t i = 0; i < data.size(); ++i) {
            const uint16_t h = float_to_half(data[i]);
            out[i * 2 + 0] = static_cast<unsigned char>(h & 0xFFu);
            out[i * 2 + 1] = static_cast<unsigned char>(h >> 8);
        }
        return out;
    }
} // namespace vulkan
