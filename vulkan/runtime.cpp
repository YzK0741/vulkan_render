module;

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

module vulkan.runtime;

import utility;

namespace {
    vulkan::runtime* runtime_from_window(GLFWwindow* window) {
        return static_cast<vulkan::runtime*>(glfwGetWindowUserPointer(window));
    }

    void mouse_button_callback(GLFWwindow* window, const int button, const int action, const int) {
        auto* runtime = runtime_from_window(window);
        if (button != GLFW_MOUSE_BUTTON_LEFT) {
            return;
        }
        if (action == GLFW_PRESS) {
            runtime->camera.dragging = true;
            glfwGetCursorPos(window, &runtime->camera.last_x, &runtime->camera.last_y);
        } else if (action == GLFW_RELEASE) {
            runtime->camera.dragging = false;
        }
    }

    void cursor_pos_callback(GLFWwindow* window, const double x, const double y) {
        auto& camera = runtime_from_window(window)->camera;
        if (!camera.dragging) {
            return;
        }
        constexpr float sensitivity = 0.005f;
        const float dx = static_cast<float>(x - camera.last_x);
        const float dy = static_cast<float>(y - camera.last_y);
        camera.last_x = x;
        camera.last_y = y;
        camera.yaw += dx * sensitivity; // drag direction matches the model rotation
        camera.pitch -= dy * sensitivity;
        camera.pitch = std::clamp(camera.pitch, -1.5f, 1.5f); // avoid flipping
    }

    void scroll_callback(GLFWwindow* window, const double, const double yoffset) {
        auto& camera = runtime_from_window(window)->camera;
        camera.distance *= std::pow(0.9f, static_cast<float>(yoffset));
        camera.distance = std::clamp(camera.distance, 0.5f, 20.0f);
    }
} // namespace

namespace vulkan {
    runtime::runtime()
        : filtered_core{vulkan_core} {
        glfwSetWindowUserPointer(this->vulkan_core.window, this);
        glfwSetMouseButtonCallback(this->vulkan_core.window, mouse_button_callback);
        glfwSetCursorPosCallback(this->vulkan_core.window, cursor_pos_callback);
        glfwSetScrollCallback(this->vulkan_core.window, scroll_callback);

        // One command buffer per frame slot, owned and reused by render_frame()
        this->command_buffers.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            this->command_buffers.push_back(this->vulkan_core.make_command_buffer());
        }

        // Shared scene resources: camera UBO buffers, white fallback texture, texture sampler
        this->init_scene_resources();
    }

    // The destructor body runs before member destruction, so vulkan_core (and the VkDevice it
    // holds) is still alive here: destroying cached models and pipelines in this order is
    // guaranteed safe, independent of future member reordering. Members then destruct in reverse
    // declaration order with models/pipelines already empty.
    runtime::~runtime() {
        for (auto& pipeline_models : this->models | std::views::values) {
            for (auto& model : pipeline_models) {
                model.destroy(this->vulkan_core.vma);
            }
        }
        this->models.clear();
        this->pipelines.clear();

        // Shared scene resources (views/sets/samplers are RAII and free themselves)
        for (const uint64_t handle : this->camera_buffer_handles) {
            this->vulkan_core.vma.free_buffer(handle);
        }
        for (const uint64_t handle : this->owned_texture_handles) {
            this->vulkan_core.vma.free_image(handle);
        }
        for (const uint64_t handle : this->ibl_handles) {
            this->vulkan_core.vma.free_image(handle);
        }
        if (this->material_buffer_handle != 0) {
            this->vulkan_core.vma.free_buffer(this->material_buffer_handle);
            this->material_buffer_handle = 0;
        }
    }

    void runtime::init_scene_resources() {
        // Camera UBO: one buffer per frame slot, mapped for direct writes; all models reference
        // these buffers through the shared scene set, so one memcpy per frame replaces the old
        // per-model per-frame UBO updates
        this->camera_buffer_handles.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        this->camera_mapped.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            camera_ubo initial = {};
            const uint64_t handle = this->vulkan_core.vma.create_buffer(std::span(&initial, 1), vulkan::buffer_type::uniform_coherent);
            if (handle == 0) {
                utility::panic("failed to create camera ubo buffer");
            }
            const auto* detail = this->vulkan_core.vma.get_buffer_detail(handle);
            if (detail == nullptr) {
                utility::panic("failed to get camera ubo buffer detail");
            }
            this->camera_buffer_handles.push_back(handle);
            this->camera_mapped.push_back(detail->allocation_info.pMappedData);
        }

        // 1x1 white fallback texture, always the first entry of the scene texture array; missing
        // material textures point at it
        constexpr std::array<unsigned char, 4> white_pixels = {255, 255, 255, 255};
        vulkan::image_create_info white_info = {};
        white_info.width = 1;
        white_info.height = 1;
        white_info.mip_levels = 1;
        white_info.array_layers = 1;
        white_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        const uint64_t white_handle = this->vulkan_core.vma.create_image(white_pixels.data(), white_pixels.size(), white_info, vulkan::image_type::texture_2d);
        if (white_handle == 0) {
            utility::panic("failed to create white fallback texture");
        }
        const auto* white_detail = this->vulkan_core.vma.get_image_detail(white_handle);
        if (white_detail == nullptr) {
            utility::panic("failed to get white texture detail");
        }
        this->owned_texture_handles.push_back(white_handle);
        this->owned_texture_views.push_back(this->vulkan_core.make_image_view(white_detail->image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D));
        this->white_texture_index = static_cast<uint32_t>(this->texture_array_views.size());
        this->texture_array_views.push_back(*this->owned_texture_views.back());

        // Shared sampler for the texture array entries
        this->texture_sampler = this->vulkan_core.make_sampler(VK_SAMPLER_ADDRESS_MODE_REPEAT, 0.25f);

        // GPU material table: fixed capacity, host-visible (direct mapping); records are appended
        // at registration and read-only for the GPU (set 0 binding 5)
        const std::vector<unsigned char> zeroed_materials(static_cast<size_t>(vulkan::material_capacity) * sizeof(material_record), 0);
        const uint64_t material_handle = this->vulkan_core.vma.create_buffer(zeroed_materials.data(), zeroed_materials.size(), vulkan::buffer_type::storage_coherent);
        if (material_handle == 0) {
            utility::panic("failed to create material table buffer");
        }
        const auto* material_detail = this->vulkan_core.vma.get_buffer_detail(material_handle);
        if (material_detail == nullptr) {
            utility::panic("failed to get material table buffer detail");
        }
        this->material_buffer_handle = material_handle;
        this->material_mapped = material_detail->allocation_info.pMappedData;
    }

    void runtime::ensure_scene_set() {
        if (this->scene_set_created) {
            return;
        }
        this->scene_set = this->vulkan_core.make_descriptor_set(this->vulkan_core.scene_descriptor_set_layout);
        this->scene_set_created = true;

        // binding 0: camera UBO -> buffer[0]; render_frame() rewrites it per frame with the
        // current frame slot's buffer (update-after-bind)
        const auto* detail = this->vulkan_core.vma.get_buffer_detail(this->camera_buffer_handles[0]);
        if (detail == nullptr) {
            utility::panic("failed to get camera ubo buffer detail");
        }
        const VkDescriptorBufferInfo camera_info{detail->buffer, 0, sizeof(camera_ubo)};
        VkWriteDescriptorSet camera_write = {};
        camera_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        camera_write.dstSet = *this->scene_set;
        camera_write.dstBinding = 0;
        camera_write.descriptorCount = 1;
        camera_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        camera_write.pBufferInfo = &camera_info;
        vkUpdateDescriptorSets(this->vulkan_core.device, 1, &camera_write, 0, nullptr);

        // binding 5: material table (storage buffer, written once)
        const auto* material_detail = this->vulkan_core.vma.get_buffer_detail(this->material_buffer_handle);
        if (material_detail == nullptr) {
            utility::panic("failed to get material table buffer detail");
        }
        const VkDescriptorBufferInfo material_info{material_detail->buffer, 0, static_cast<VkDeviceSize>(vulkan::material_capacity) * sizeof(material_record)};
        VkWriteDescriptorSet material_write = {};
        material_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        material_write.dstSet = *this->scene_set;
        material_write.dstBinding = 5;
        material_write.descriptorCount = 1;
        material_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        material_write.pBufferInfo = &material_info;
        vkUpdateDescriptorSets(this->vulkan_core.device, 1, &material_write, 0, nullptr);

        // bindings 2-4: IBL (or white placeholders until set_ibl() is called)
        this->write_ibl_bindings();
    }

    void runtime::write_ibl_bindings() {
        if (!this->scene_set_created) {
            return;
        }
        std::array<VkDescriptorImageInfo, 3> image_infos = {};
        std::array<VkWriteDescriptorSet, 3> writes = {};
        const VkImageView placeholder_view = *this->owned_texture_views[0]; // white
        const VkSampler placeholder_sampler = *this->texture_sampler;
        for (int i = 0; i < 3; ++i) {
            image_infos[i] = {.sampler = this->ibl_ready ? *this->env_sampler : placeholder_sampler,
                              .imageView = this->ibl_ready ? *this->ibl_views[i] : placeholder_view,
                              .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = *this->scene_set;
            writes[i].dstBinding = static_cast<uint32_t>(2 + i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo = &image_infos[i];
        }
        vkUpdateDescriptorSets(this->vulkan_core.device, 3, writes.data(), 0, nullptr);
    }

    void runtime::set_ibl(const ibl_input& info) {
        if (info.env_size == 0) {
            return;
        }
        const auto upload = [this](const std::span<const unsigned char> data, const image_create_info& create_info, const image_type type) -> uint64_t {
            const uint64_t handle = this->vulkan_core.vma.create_image(data.data(), data.size_bytes(), create_info, type);
            if (handle == 0) {
                utility::panic("failed to create IBL image");
            }
            return handle;
        };

        // prefiltered environment cubemap (mip chain)
        vulkan::image_create_info env_info = {};
        env_info.width = info.env_size;
        env_info.height = info.env_size;
        env_info.mip_levels = info.env_mip_count;
        env_info.array_layers = 6;
        env_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        uint64_t handle = upload(info.prefiltered_env, env_info, vulkan::image_type::texture_cubemap);
        const auto* env_detail = this->vulkan_core.vma.get_image_detail(handle);
        if (env_detail == nullptr) {
            utility::panic("failed to get environment image detail");
        }
        this->ibl_handles.push_back(handle);
        this->ibl_views.push_back(this->vulkan_core.make_image_view(env_detail->image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_CUBE));

        // irradiance cubemap
        vulkan::image_create_info irr_info = {};
        irr_info.width = info.irr_size;
        irr_info.height = info.irr_size;
        irr_info.mip_levels = 1;
        irr_info.array_layers = 6;
        irr_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        handle = upload(info.irradiance, irr_info, vulkan::image_type::texture_cubemap);
        const auto* irr_detail = this->vulkan_core.vma.get_image_detail(handle);
        if (irr_detail == nullptr) {
            utility::panic("failed to get irradiance image detail");
        }
        this->ibl_handles.push_back(handle);
        this->ibl_views.push_back(this->vulkan_core.make_image_view(irr_detail->image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_CUBE));

        // BRDF integration LUT
        vulkan::image_create_info lut_info = {};
        lut_info.width = info.lut_size;
        lut_info.height = info.lut_size;
        lut_info.mip_levels = 1;
        lut_info.array_layers = 1;
        lut_info.format = VK_FORMAT_R16G16_SFLOAT;
        handle = upload(info.brdf_lut, lut_info, vulkan::image_type::texture_2d);
        const auto* lut_detail = this->vulkan_core.vma.get_image_detail(handle);
        if (lut_detail == nullptr) {
            utility::panic("failed to get BRDF LUT image detail");
        }
        this->ibl_handles.push_back(handle);
        this->ibl_views.push_back(this->vulkan_core.make_image_view(lut_detail->image, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_VIEW_TYPE_2D));

        this->env_sampler = this->vulkan_core.make_sampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, static_cast<float>(info.env_mip_count - 1));
        this->ibl_ready = true;

        // if the scene set already exists, point bindings 2-4 at the real images
        this->write_ibl_bindings();
    }

    uint32_t runtime::register_material(const model_create_info& info) {
        // ---- 1. Upload the 5 texture slots into the shared array; missing ones use the white fallback ----
        if (this->texture_array_views.size() + 5 > vulkan::scene_texture_capacity) {
            utility::panic("scene texture array capacity exceeded");
        }

        const std::array<std::pair<const texture_input*, VkFormat>, 5> slots = {
            std::pair{&info.albedo, VK_FORMAT_R8G8B8A8_SRGB},
            std::pair{&info.metallic_roughness, VK_FORMAT_R8G8B8A8_UNORM},
            std::pair{&info.normal, VK_FORMAT_R8G8B8A8_UNORM},
            std::pair{&info.occlusion, VK_FORMAT_R8G8B8A8_UNORM},
            std::pair{&info.emissive, VK_FORMAT_R8G8B8A8_UNORM},
        };

        std::array<uint32_t, 5> texture_indices = {};
        std::array<VkDescriptorImageInfo, 5> image_infos = {};
        std::array<VkWriteDescriptorSet, 5> writes = {};
        uint32_t write_count = 0;
        const VkSampler sampler = *this->texture_sampler;
        for (int i = 0; i < 5; ++i) {
            VkImageView view;
            if (slots[i].first->valid && !slots[i].first->data.empty()) {
                const texture_input& tex = *slots[i].first;
                vulkan::image_create_info image_info = {};
                image_info.width = tex.width;
                image_info.height = tex.height;
                image_info.mip_levels = 1;
                image_info.array_layers = 1;
                image_info.format = tex.format;
                const uint64_t handle = this->vulkan_core.vma.create_image(tex.data.data(), tex.data.size_bytes(), image_info, vulkan::image_type::texture_2d);
                if (handle == 0) {
                    utility::panic("failed to create material texture");
                }
                const auto* detail = this->vulkan_core.vma.get_image_detail(handle);
                if (detail == nullptr) {
                    utility::panic("failed to get material texture detail");
                }
                this->owned_texture_handles.push_back(handle);
                this->owned_texture_views.push_back(this->vulkan_core.make_image_view(detail->image, tex.format, VK_IMAGE_VIEW_TYPE_2D));
                view = *this->owned_texture_views.back();
            } else {
                view = this->texture_array_views[this->white_texture_index]; // white fallback
            }

            texture_indices[i] = static_cast<uint32_t>(this->texture_array_views.size());
            this->texture_array_views.push_back(view);
            image_infos[i] = {.sampler = sampler, .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = *this->scene_set;
            writes[i].dstBinding = 1;
            writes[i].dstArrayElement = texture_indices[i];
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo = &image_infos[i];
            ++write_count;
        }
        vkUpdateDescriptorSets(this->vulkan_core.device, write_count, writes.data(), 0, nullptr);

        // ---- 2. Append one material record: texture indices + presence flags; factors keep
        //         their identity defaults (extend model_create_info to pass custom factors) ----
        if (this->material_count >= vulkan::material_capacity) {
            utility::panic("material table capacity exceeded");
        }
        material_record record = {};
        record.tex_indices = glm::uvec4(texture_indices[0], texture_indices[1], texture_indices[2], texture_indices[3]);
        record.emissive_index = texture_indices[4];
        record.flags = 0;
        if (info.normal.valid) {
            record.flags |= 1u;
        }
        if (info.occlusion.valid) {
            record.flags |= 2u;
        }
        if (info.emissive.valid) {
            record.flags |= 4u;
        }

        const uint32_t material_index = this->material_count++;
        std::memcpy(static_cast<unsigned char*>(this->material_mapped) + static_cast<size_t>(material_index) * sizeof(material_record), &record, sizeof(record));
        return material_index;
    }

    void runtime::begin_rendering(const VkCommandBuffer command_buffer, const uint32_t image_index) const {
        std::array<VkClearValue, 2> clear_values = {};
        clear_values[0].color = {{0.02f, 0.02f, 0.03f, 1.0f}};
        clear_values[1].depthStencil = {1.0f, 0};

        const core& vk = this->vulkan_core;

        if (vk.use_dynamic_rendering) {
            // Dynamic rendering (Vulkan 1.3): attachments are described inline, no render pass
            VkRenderingAttachmentInfo color_attachment = {};
            color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            color_attachment.imageView = vk.msaa_samples > VK_SAMPLE_COUNT_1_BIT
                                             ? vk.color_image_views[image_index]
                                             : vk.swap_chain_image_views[image_index];
            color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color_attachment.clearValue = clear_values[0];
            if (vk.msaa_samples > VK_SAMPLE_COUNT_1_BIT) {
                // MSAA resolve: the MSAA color attachment resolves into the swapchain image.
                // resolveImageLayout must not be PRESENT_SRC_KHR (VUID-VkRenderingAttachmentInfo-imageView-06146);
                // the swapchain image is transitioned to PRESENT_SRC_KHR after vkCmdEndRendering instead
                color_attachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
                color_attachment.resolveImageView = vk.swap_chain_image_views[image_index];
                color_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }

            VkRenderingAttachmentInfo depth_attachment = {};
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = vk.depth_image_views[image_index];
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depth_attachment.clearValue = clear_values[1];

            VkRenderingInfo rendering_info = {};
            rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering_info.renderArea = {{0, 0}, vk.swap_chain_extent};
            rendering_info.layerCount = 1;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachments = &color_attachment;
            rendering_info.pDepthAttachment = &depth_attachment;
            vkCmdBeginRendering(command_buffer, &rendering_info);
            return;
        }

        // Classic render pass fallback (devices without dynamic rendering)
        VkRenderPassBeginInfo render_pass_info = {};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.renderPass = vk.renderpass;
        render_pass_info.framebuffer = vk.swap_chain_framebuffers[image_index];
        render_pass_info.renderArea = {{0, 0}, vk.swap_chain_extent};
        render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
        render_pass_info.pClearValues = clear_values.data();
        vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    }

    frame_result runtime::render_frame() {
        core& vk = this->vulkan_core;
        GLFWwindow* window = vk.window;

        // 1. Window events first: respond to ESC / native close before any GPU work
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (glfwWindowShouldClose(window)) {
            return frame_result::closed;
        }

        // 2. Minimized: skip this frame (acquiring from an invalidated / 0-sized swapchain would
        //    fail); the restore transition below rebuilds the swapchain before the next render
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) == GLFW_TRUE) {
            this->was_minimized = true;
            return frame_result::skipped;
        }
        if (this->was_minimized) {
            this->was_minimized = false;
            utility::log("window restored, recreating swapchain");
            vk.recreate_swap_chain();
        }

        // 3. The frame slot's fence guards both the command buffer and the acquire semaphore:
        //    wait it BEFORE acquiring so the previous submission on this slot (and its semaphore
        //    wait operation) has fully completed — acquiring first would reuse a semaphore that
        //    may still have pending operations (VUID-vkAcquireNextImageKHR-semaphore-01779)
        const uint32_t frame_slot = static_cast<uint32_t>(vk.current_frame);
        vkWaitForFences(vk.device, 1, &vk.in_flight_fences[frame_slot], VK_TRUE, UINT64_MAX);

        // 4. Acquire the next swapchain image; on out-of-date (e.g. the window was resized)
        //    rebuild the swapchain and let the caller retry on the next iteration. The fence is
        //    only reset after a successful acquire, so this path never leaves a reset-but-
        //    unsubmitted fence behind (which would deadlock the next frame's wait)
        uint32_t image_index = 0;
        const VkResult acquire_result = vkAcquireNextImageKHR(vk.device,
                                                              vk.swap_chain,
                                                              UINT64_MAX,
                                                              vk.image_available_semaphores[frame_slot],
                                                              VK_NULL_HANDLE,
                                                              &image_index);
        if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
            utility::log("swapchain out of date, recreating");
            vk.recreate_swap_chain();
            return frame_result::skipped;
        }
        if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
            return frame_result::failed;
        }
        vkResetFences(vk.device, 1, &vk.in_flight_fences[frame_slot]);

        // 5. Update the shared camera UBO once: every model references these buffers through the
        //    scene set, so one memcpy (+ one update-after-bind descriptor write) replaces the old
        //    per-model per-frame UBO updates
        const float aspect = static_cast<float>(vk.swap_chain_extent.width) / static_cast<float>(vk.swap_chain_extent.height);
        const camera_ubo ubo = make_orbit_camera_ubo(this->camera.yaw, this->camera.pitch, this->camera.distance, aspect);
        if (this->camera_mapped[frame_slot] != nullptr) {
            std::memcpy(this->camera_mapped[frame_slot], &ubo, sizeof(ubo));
        }
        if (this->scene_set.get() != VK_NULL_HANDLE) {
            const auto* ubo_detail = vk.vma.get_buffer_detail(this->camera_buffer_handles[frame_slot]);
            const VkDescriptorBufferInfo camera_info{ubo_detail->buffer, 0, sizeof(camera_ubo)};
            VkWriteDescriptorSet camera_write = {};
            camera_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            camera_write.dstSet = *this->scene_set;
            camera_write.dstBinding = 0;
            camera_write.descriptorCount = 1;
            camera_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            camera_write.pBufferInfo = &camera_info;
            vkUpdateDescriptorSets(vk.device, 1, &camera_write, 0, nullptr);
        }

        // 6. Record the frame into this slot's command buffer
        vk_command_buffer& command_buffer = this->command_buffers[frame_slot];
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(*command_buffer, &begin_info) != VK_SUCCESS) {
            return frame_result::failed;
        }

        // Dynamic rendering has no automatic attachment transitions (unlike a render pass):
        // move every attachment into its render layout before vkCmdBeginRendering
        if (vk.use_dynamic_rendering) {
            std::array<VkImageMemoryBarrier2, 3> attachment_barriers = {};
            uint32_t barrier_count = 0;
            const auto add_render_barrier = [&attachment_barriers, &barrier_count](const VkImage image, const VkImageAspectFlags aspect, const VkImageLayout new_layout, const VkPipelineStageFlags2 dst_stage, const VkAccessFlags2 dst_access) {
                VkImageMemoryBarrier2& barrier = attachment_barriers[barrier_count++];
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                barrier.srcAccessMask = 0;
                barrier.dstStageMask = dst_stage;
                barrier.dstAccessMask = dst_access;
                // loadOp CLEAR discards the contents: UNDEFINED as oldLayout is valid whatever the
                // image's actual current layout is, and avoids tracking it per frame
                barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout = new_layout;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = image;
                barrier.subresourceRange = {aspect, 0, 1, 0, 1};
            };

            if (vk.msaa_samples > VK_SAMPLE_COUNT_1_BIT) {
                // MSAA color attachment and the swapchain resolve target both render in COLOR_ATTACHMENT_OPTIMAL
                add_render_barrier(vk.color_images[image_index], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                add_render_barrier(vk.swap_chain_images[image_index], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            } else {
                add_render_barrier(vk.swap_chain_images[image_index], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            }
            add_render_barrier(vk.depth_images[image_index], VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                               VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                               VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            VkDependencyInfo dependency_info = {};
            dependency_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependency_info.imageMemoryBarrierCount = barrier_count;
            dependency_info.pImageMemoryBarriers = attachment_barriers.data();
            vkCmdPipelineBarrier2(*command_buffer, &dependency_info);
        }

        this->begin_rendering(*command_buffer, image_index);

        // Bind the single scene descriptor set once: every pipeline shares the scene layout, so
        // the set stays valid across pipeline binds and only models vary per draw
        if (this->scene_set.get() != VK_NULL_HANDLE) {
            const VkDescriptorSet scene_set_handle = *this->scene_set;
            vkCmdBindDescriptorSets(*command_buffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    vk.scene_pipeline_layout,
                                    0,
                                    1,
                                    &scene_set_handle,
                                    0,
                                    nullptr);
        }

        // Pipelines cache a fullscreen viewport/scissor at creation; after a resize the swapchain
        // extent changed, so resync them from the current extent before drawing (begin_pipeline
        // applies the stored values)
        const VkViewport full_viewport = {0.0f,
                                          0.0f,
                                          static_cast<float>(vk.swap_chain_extent.width),
                                          static_cast<float>(vk.swap_chain_extent.height),
                                          0.0f,
                                          1.0f};
        const VkRect2D full_scissor = {{0, 0}, vk.swap_chain_extent};
        for (auto& [pipeline_name, pipeline] : this->pipelines) {
            pipeline.viewport = full_viewport;
            pipeline.scissor = full_scissor;
        }

        for (const auto& [pipeline_name, pipeline] : this->pipelines) {
            const auto models_it = this->models.find(pipeline_name);
            if (models_it == this->models.end() || models_it->second.empty()) {
                continue; // pipeline without models: nothing to draw
            }
            pipeline.begin_pipeline(*command_buffer);
            for (const auto& model : models_it->second) {
                model.draw(*command_buffer);
            }
        }

        if (vk.use_dynamic_rendering) {
            vkCmdEndRendering(*command_buffer);
            // Dynamic rendering has no render pass finalLayout to hand the image back to the
            // presentation engine: transition the swapchain image to PRESENT_SRC_KHR explicitly.
            // With MSAA the resolve target ends up in resolveImageLayout (COLOR_ATTACHMENT_OPTIMAL),
            // so the barrier is needed on both the direct-render and the resolve paths.
            VkImageMemoryBarrier2 present_barrier = {};
            present_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            present_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            present_barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            present_barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            present_barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            present_barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            present_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            present_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            present_barrier.image = vk.swap_chain_images[image_index];
            present_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            VkDependencyInfo dependency_info = {};
            dependency_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependency_info.imageMemoryBarrierCount = 1;
            dependency_info.pImageMemoryBarriers = &present_barrier;
            vkCmdPipelineBarrier2(*command_buffer, &dependency_info);
        } else {
            vkCmdEndRenderPass(*command_buffer);
        }
        if (vkEndCommandBuffer(*command_buffer) != VK_SUCCESS) {
            return frame_result::failed;
        }

        // 7. Submit + present; recreate the swapchain when presentation reports out of date
        if (vk.submit(*command_buffer, image_index) != VK_SUCCESS) {
            return frame_result::failed;
        }
        const VkResult present_result = vk.present(image_index);
        if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
            utility::log("present out of date, recreating swapchain");
            vk.recreate_swap_chain();
        } else if (present_result != VK_SUCCESS) {
            return frame_result::failed;
        }
        vk.to_next_frame();
        return frame_result::render_success;
    }

    std::expected<void, std::string> runtime::make_pipeline(std::string_view pipeline_name, std::span<const unsigned char> vertex_shader_code, std::span<const unsigned char> fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        auto make_result = this->vulkan_core.make_pipeline(vertex_shader_code, fragment_shader_code);
        if (!make_result) {
            return fail(make_result.error());
        }
        this->pipelines.emplace(pipeline_name, std::move(make_result).value());
        return {};
    }
    const vk_pipeline* runtime::get_pipeline(const std::string_view pipeline_name) const noexcept {
        const auto it = this->pipelines.find(pipeline_name);
        return it == this->pipelines.end() ? nullptr : &it->second;
    }

    model* runtime::make_model(const std::string_view pipeline_name, const model_create_info& info) {
        const vk_pipeline* pipeline = this->get_pipeline(pipeline_name);
        if (pipeline == nullptr) {
            return nullptr;
        }
        this->ensure_scene_set();

        model result;
        result.pipeline = pipeline;

        // ---- geometry buffers ----
        result.vertex_buffer_handle = this->vulkan_core.vma.create_buffer(info.vertex_data.data(), info.vertex_data.size_bytes(), vulkan::buffer_type::vertex);
        if (result.vertex_buffer_handle == 0) {
            utility::panic("failed to create vertex buffer");
        }
        result.vertex_detail = this->vulkan_core.vma.get_buffer_detail(result.vertex_buffer_handle);
        if (result.vertex_detail == nullptr) {
            utility::panic("failed to get vertex buffer detail");
        }

        result.index_buffer_handle = this->vulkan_core.vma.create_buffer(info.index_data.data(), info.index_data.size_bytes(), vulkan::buffer_type::index);
        if (result.index_buffer_handle == 0) {
            utility::panic("failed to create index buffer");
        }
        result.index_detail = this->vulkan_core.vma.get_buffer_detail(result.index_buffer_handle);
        if (result.index_detail == nullptr) {
            utility::panic("failed to get index buffer detail");
        }

        result.index_type = info.index_type;
        result.index_count = info.index_count;
        result.vertex_count = info.vertex_count;

        // ---- material: register textures + append a material record; the model only carries
        //         the material index (texture indices / factors / flags live in the GPU table) ----
        result.push.material_index = this->register_material(info);
        result.push.model = info.model_matrix;

        // operator[] has no heterogeneous overload (unlike find), so construct the key explicitly
        auto& pipeline_models = this->models[std::string(pipeline_name)];
        pipeline_models.push_back(std::move(result));
        return &pipeline_models.back();
    }

    const std::vector<model>* runtime::get_models(const std::string_view pipeline_name) const noexcept {
        const auto it = this->models.find(pipeline_name);
        return it == this->models.end() ? nullptr : &it->second;
    }

    void runtime::clear_models(const std::string_view pipeline_name) {
        const auto it = this->models.find(pipeline_name);
        if (it != this->models.end()) {
            for (auto& model : it->second) {
                model.destroy(this->vulkan_core.vma);
            }
            this->models.erase(it);
        }
    }
} // namespace vulkan
