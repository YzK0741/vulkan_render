module;

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

module vulkan.runtime;

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

    void runtime::begin_render_pass(const VkCommandBuffer command_buffer, const uint32_t image_index) const {
        std::array<VkClearValue, 2> clear_values = {};
        clear_values[0].color = {{0.02f, 0.02f, 0.03f, 1.0f}};
        clear_values[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo render_pass_info = {};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.renderPass = this->vulkan_core.renderpass;
        render_pass_info.framebuffer = this->vulkan_core.swap_chain_framebuffers[image_index];
        render_pass_info.renderArea = {{0, 0}, this->vulkan_core.swap_chain_extent};
        render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
        render_pass_info.pClearValues = clear_values.data();
        vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    }

    bool runtime::render_frame() {
        core& vk = this->vulkan_core;
        const uint32_t frame_slot = static_cast<uint32_t>(vk.current_frame);

        // 1. Wait until this slot's previous submission is done, then reset the fence
        vkWaitForFences(vk.device, 1, &vk.in_flight_fences[frame_slot], VK_TRUE, UINT64_MAX);
        vkResetFences(vk.device, 1, &vk.in_flight_fences[frame_slot]);

        // 2. Update every model's camera UBO from the shared orbit camera
        //    (the model matrix is owned by the model; the aspect comes from the swapchain)
        const float aspect = static_cast<float>(vk.swap_chain_extent.width) / static_cast<float>(vk.swap_chain_extent.height);
        for (auto& pipeline_models : this->models | std::views::values) {
            for (auto& model : pipeline_models) {
                const camera_ubo ubo = make_orbit_camera_ubo(
                    this->camera.yaw, this->camera.pitch, this->camera.distance, model.model_matrix, aspect);
                model.update_camera_ubo(frame_slot, ubo);
            }
        }

        // 3. Acquire the next swapchain image
        uint32_t image_index = 0;
        if (vkAcquireNextImageKHR(vk.device,
                                  vk.swap_chain,
                                  UINT64_MAX,
                                  vk.image_available_semaphores[frame_slot],
                                  VK_NULL_HANDLE,
                                  &image_index) != VK_SUCCESS) {
            return false;
        }

        // 4. Record the frame into this slot's command buffer
        vk_command_buffer& command_buffer = this->command_buffers[frame_slot];
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(*command_buffer, &begin_info) != VK_SUCCESS) {
            return false;
        }

        this->begin_render_pass(*command_buffer, image_index);

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

        vkCmdEndRenderPass(*command_buffer);
        if (vkEndCommandBuffer(*command_buffer) != VK_SUCCESS) {
            return false;
        }

        // 5. Submit + present, then advance to the next frame slot
        if (vk.submit(*command_buffer, image_index) != VK_SUCCESS) {
            return false;
        }
        if (vk.present(image_index) != VK_SUCCESS) {
            return false;
        }
        vk.to_next_frame();
        return true;
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
