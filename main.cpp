import std;
import vulkan.runtime;

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan.h>

namespace {
    // 顶点：position (location 0) + color (location 1)，与 triangle.vert 的输入布局一致
    struct vertex {
        glm::vec3 position;
        glm::vec3 color;
    };

    // 以二进制方式读取整个文件；失败返回 false
    bool read_binary_file(const std::filesystem::path& path, std::vector<unsigned char>& out) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return false;
        }
        out.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        return !file.bad();
    }

    // 从当前工作目录向上逐级查找 shaders/ 目录，
    // 兼容在项目根目录或 cmake-build-* 目录下运行的情况
    std::optional<std::filesystem::path> locate_shaders_dir() {
        std::filesystem::path current = std::filesystem::current_path();
        for (int depth = 0; depth < 4; ++depth) {
            if (std::filesystem::path candidate = current / "shaders"; std::filesystem::is_directory(candidate)) {
                return candidate;
            }
            const std::filesystem::path parent = current.parent_path();
            if (parent == current) {
                break;
            }
            current = parent;
        }
        return std::nullopt;
    }

    // 读取单个着色器 SPIR-V 文件并打印信息；失败返回 false
    bool load_shader(const std::filesystem::path& dir, std::string_view file_name, std::vector<unsigned char>& out) {
        const std::filesystem::path path = dir / file_name;
        if (!read_binary_file(path, out)) {
            std::println("ERROR: cannot open shader file '{}'", path.string());
            return false;
        }
        std::println("loaded shader: {} ({} bytes)", path.string(), out.size());
        return true;
    }

    // 加载一对 vertex/fragment SPIR-V，并通过 runtime 创建管线
    bool load_and_create_pipeline(vulkan::runtime& runtime,
                                  const std::filesystem::path& shaders_dir,
                                  std::string_view pipeline_name,
                                  std::string_view vertex_file,
                                  std::string_view fragment_file) {
        std::vector<unsigned char> vertex_code;
        std::vector<unsigned char> fragment_code;
        if (!load_shader(shaders_dir, vertex_file, vertex_code) ||
            !load_shader(shaders_dir, fragment_file, fragment_code)) {
            return false;
        }

        const std::expected<void, std::string> result = runtime.make_pipeline(pipeline_name, vertex_code, fragment_code);
        if (!result) {
            std::println("FAILED to create pipeline '{}': {}", pipeline_name, result.error());
            return false;
        }
        std::println("SUCCESS: pipeline '{}' created and cached in the runtime", pipeline_name);
        return true;
    }
} // namespace

int main() {
    // 1. 定位 shaders 目录（保存 GLSL 源码与编译好的 SPIR-V）
    const std::optional<std::filesystem::path> shaders_dir = locate_shaders_dir();
    if (!shaders_dir) {
        std::println("ERROR: cannot find shaders/ directory. run the program from the project root or a cmake-build-* directory.");
        return 1;
    }

    // 2. 构造 vulkan::runtime：默认构造会完成 window/instance/device/swapchain 等全部初始化
    vulkan::runtime runtime;

    // 3. 管线加载测试：简单三角形 + 标准 PBR
    if (!load_and_create_pipeline(runtime, *shaders_dir, "triangle", "triangle.vert.spv", "triangle.frag.spv")) {
        return 1;
    }
    if (!load_and_create_pipeline(runtime, *shaders_dir, "pbr", "pbr.vert.spv", "pbr.frag.spv")) {
        return 1;
    }

    // 4. 取回 triangle 管线，准备渲染
    const auto* pipeline = runtime.get_pipeline("triangle");
    if (pipeline == nullptr) {
        std::println("ERROR: pipeline 'triangle' not found in runtime");
        return 1;
    }

    // 5. 顶点缓冲：三个彩色顶点。
    //    注意绕序：管线的 FRONT_FACE_COUNTER_CLOCKWISE 按 framebuffer 坐标（y 向下）判定，
    //    因此 NDC 里必须是 底→左→右 的顺序（即屏幕上的逆时针），否则会被 CULL_BACK 剔除。
    std::array<vertex, 3> vertices = {
        {
            {{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}}, // 底，红
            {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}}, // 左，蓝
            {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},  // 右，绿
        },
    };
    const uint64_t vertex_buffer = runtime->vma.create_buffer(std::span(vertices), vulkan::buffer_type::vertex);
    const auto* vertex_detail = runtime->vma.get_buffer_detail(vertex_buffer);
    if (vertex_detail == nullptr) {
        std::println("ERROR: failed to create vertex buffer");
        return 1;
    }

    // 6. 每个帧槽位一个 UBO（mvp 矩阵），避免读写与上一帧 GPU 读取竞争
    constexpr size_t ubo_size = sizeof(glm::mat4); // 64 字节
    std::vector<uint64_t> ubo_buffers;
    std::vector<vulkan::vk_descriptor_set> descriptor_sets;
    ubo_buffers.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
    descriptor_sets.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
    for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
        glm::mat4 identity = glm::mat4(1.0f); // 非 const：vma.create_buffer 的 span 模板会 reinterpret_cast
        ubo_buffers.push_back(runtime->vma.create_buffer(std::span(&identity, 1), vulkan::buffer_type::uniform_coherent));
        const auto* ubo_detail = runtime->vma.get_buffer_detail(ubo_buffers.back());
        if (ubo_detail == nullptr) {
            std::println("ERROR: failed to create ubo buffer");
            return 1;
        }

        auto descriptor_set = runtime->make_descriptor_set(pipeline->get_descriptor_set_layouts()[0]);
        const VkDescriptorBufferInfo ubo_info{ubo_detail->buffer, 0, ubo_size};
        VkWriteDescriptorSet write_info{};
        write_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_info.dstSet = *descriptor_set;
        write_info.dstBinding = 0;
        write_info.descriptorCount = 1;
        write_info.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write_info.pBufferInfo = &ubo_info;
        vkUpdateDescriptorSets(runtime->device, 1, &write_info, 0, nullptr);
        descriptor_sets.push_back(std::move(descriptor_set));
    }

    // 7. 每个帧槽位一条命令缓冲（复用，begin 时自动重置）
    std::vector<vulkan::vk_command_buffer> command_buffers;
    command_buffers.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
    for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
        command_buffers.push_back(runtime->make_command_buffer());
    }

    // 8. 渲染主循环：直到关闭窗口或按 ESC
    const auto start_time = std::chrono::steady_clock::now();
    std::println("rendering... close the window or press ESC to exit");
    while (!glfwWindowShouldClose(runtime->window)) {
        glfwPollEvents();
        if (glfwGetKey(runtime->window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(runtime->window, GLFW_TRUE);
        }

        const size_t frame_slot = runtime->current_frame;

        // 等待上一帧该槽位完成并重置 fence；
        // 这里不用 core::wait_usable_image()：它会等待按图像索引记录的 fence，
        // 而该 fence 刚被重置为未触发状态，会造成死等；槽位 fence 已覆盖同样的保证。
        vkWaitForFences(runtime->device, 1, &runtime->in_flight_fences[frame_slot], VK_TRUE, UINT64_MAX);
        vkResetFences(runtime->device, 1, &runtime->in_flight_fences[frame_slot]);

        // 更新本槽位的 mvp（绕 Z 轴旋转动画）并写入 UBO
        const float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time).count();
        const glm::mat4 mvp = glm::rotate(glm::mat4(1.0f), elapsed * 0.5f, glm::vec3(0.0f, 0.0f, 1.0f));
        const auto* ubo_detail = runtime->vma.get_buffer_detail(ubo_buffers[frame_slot]);
        std::memcpy(ubo_detail->allocation_info.pMappedData, glm::value_ptr(mvp), sizeof(glm::mat4));

        // 获取可用的交换链图像（fence 传空：acquire 的 fence 必须处于未触发状态，
        // 槽位 fence 已在上面等待并重置，这里用信号量同步即可）
        uint32_t image_index = 0;
        if (vkAcquireNextImageKHR(runtime->device,
                                  runtime->swap_chain,
                                  UINT64_MAX,
                                  runtime->image_available_semaphores[frame_slot],
                                  VK_NULL_HANDLE,
                                  &image_index) != VK_SUCCESS) {
            break;
        }

        // 录制命令缓冲
        auto& command_buffer = command_buffers[frame_slot];
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(*command_buffer, &begin_info) != VK_SUCCESS) {
            break;
        }

        std::array<VkClearValue, 2> clear_values = {};
        clear_values[0].color = {{0.02f, 0.02f, 0.03f, 1.0f}};
        clear_values[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.renderPass = runtime->renderpass;
        render_pass_info.framebuffer = runtime->swap_chain_framebuffers[image_index];
        render_pass_info.renderArea = {{0, 0}, runtime->swap_chain_extent};
        render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
        render_pass_info.pClearValues = clear_values.data();
        vkCmdBeginRenderPass(*command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(*command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->get_pipeline());

        const VkViewport viewport{
            0.0f,
            0.0f,
            static_cast<float>(runtime->swap_chain_extent.width),
            static_cast<float>(runtime->swap_chain_extent.height),
            0.0f,
            1.0f,
        };
        const VkRect2D scissor{{0, 0}, runtime->swap_chain_extent};
        vkCmdSetViewport(*command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(*command_buffer, 0, 1, &scissor);

        const VkBuffer vertex_handle = vertex_detail->buffer;
        constexpr VkDeviceSize vertex_offset = 0;
        vkCmdBindVertexBuffers(*command_buffer, 0, 1, &vertex_handle, &vertex_offset);

        vkCmdBindDescriptorSets(*command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline->get_pipeline_layout(),
                                0,
                                1,
                                &*descriptor_sets[frame_slot],
                                0,
                                nullptr);
        constexpr glm::mat4 model = glm::mat4(1.0f); // push constant: mat4 model
        vkCmdPushConstants(*command_buffer, pipeline->get_pipeline_layout(), VK_SHADER_STAGE_VERTEX_BIT, 0, ubo_size, glm::value_ptr(model));
        vkCmdDraw(*command_buffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(*command_buffer);
        if (vkEndCommandBuffer(*command_buffer) != VK_SUCCESS) {
            break;
        }

        // 提交（呈现信号量按图像索引分配，由 core 创建，见 core::create_sync_objects）
        constexpr VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &runtime->image_available_semaphores[frame_slot];
        submit_info.pWaitDstStageMask = &wait_stage;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &*command_buffer;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &runtime->render_finished_semaphores[image_index];
        if (vkQueueSubmit(runtime->graphics_queue, 1, &submit_info, runtime->in_flight_fences[frame_slot]) != VK_SUCCESS) {
            break;
        }

        // 呈现
        VkPresentInfoKHR present_info{};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &runtime->render_finished_semaphores[image_index];
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &runtime->swap_chain;
        present_info.pImageIndices = &image_index;
        if (vkQueuePresentKHR(runtime->present_queue, &present_info) != VK_SUCCESS) {
            break;
        }

        runtime->to_next_frame();
    }

    // 9. 等待 GPU 完成后再释放资源（呈现信号量由 core 负责销毁）
    vkDeviceWaitIdle(runtime->device);
    for (const uint64_t ubo : ubo_buffers) {
        runtime->vma.free_buffer(ubo);
    }
    runtime->vma.free_buffer(vertex_buffer);
    std::println("render loop finished");
    return 0;
}
