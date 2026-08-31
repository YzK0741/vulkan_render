module;

#include <GLFW/glfw3.h>
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
                // MSAA resolve: the MSAA color attachment resolves into the swapchain image
                color_attachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
                color_attachment.resolveImageView = vk.swap_chain_image_views[image_index];
                color_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
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

        // 5. Update every model's camera UBO from the shared orbit camera
        //    (the model matrix is owned by the model; the aspect comes from the swapchain)
        const float aspect = static_cast<float>(vk.swap_chain_extent.width) / static_cast<float>(vk.swap_chain_extent.height);
        for (auto& pipeline_models : this->models | std::views::values) {
            for (auto& model : pipeline_models) {
                const camera_ubo ubo = make_orbit_camera_ubo(
                    this->camera.yaw, this->camera.pitch, this->camera.distance, model.model_matrix, aspect);
                model.update_camera_ubo(frame_slot, ubo);
            }
        }

        // 6. Record the frame into this slot's command buffer
        vk_command_buffer& command_buffer = this->command_buffers[frame_slot];
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(*command_buffer, &begin_info) != VK_SUCCESS) {
            return frame_result::failed;
        }

        this->begin_rendering(*command_buffer, image_index);

        for (const auto& [pipeline_name, pipeline] : this->pipelines) {
            const auto models_it = this->models.find(pipeline_name);
            if (models_it == this->models.end() || models_it->second.empty()) {
                continue; // pipeline without models: nothing to draw
            }
            pipeline.begin_pipeline(*command_buffer);
            for (const auto& model : models_it->second) {
                model.draw(*command_buffer, frame_slot);
            }
        }

        if (vk.use_dynamic_rendering) {
            vkCmdEndRendering(*command_buffer);
            // Dynamic rendering has no render pass finalLayout to hand the image back to the
            // presentation engine: transition the swapchain image to PRESENT_SRC_KHR explicitly
            // (with MSAA resolve the resolve target already ends up in PRESENT_SRC_KHR, so the
            // barrier is only needed on the direct-render path)
            if (vk.msaa_samples == VK_SAMPLE_COUNT_1_BIT) {
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
            }
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
        // operator[] has no heterogeneous overload (unlike find), so construct the key explicitly
        auto& pipeline_models = this->models[std::string(pipeline_name)];
        pipeline_models.push_back(vulkan::make_model(this->vulkan_core, *pipeline, info));
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
