#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>
import std;
import gltf_loader;
import utility;
import vulkan.runtime;

namespace {
    // 与 pbr.vert 的顶点输入一致：position(0) normal(1) uv(2) tangent(3)，stride 44
    struct vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec3 tangent;
    };

    // 与 pbr.frag 的 CameraUBO 一致
    struct camera_ubo {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
        glm::vec3 camera_pos;
        float padding = 0.0f;
    };

    // 与 pbr.frag 的 PushConstants 一致（48 字节）
    struct pbr_push_constants {
        glm::vec4 base_color_factor = glm::vec4(1.0f);
        glm::vec4 emissive_factor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        float metallic_factor = 1.0f;
        float roughness_factor = 1.0f;
        float normal_scale = 1.0f;
        uint32_t flags = 0; // bit0: normal map, bit1: occlusion map, bit2: emissive map
    };
    static_assert(sizeof(pbr_push_constants) == 48);

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
            const std::filesystem::path candidate = current / "shaders";
            if (std::filesystem::is_directory(candidate)) {
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

    // 读取单个着色器 SPIR-V 文件并打印信息；失败直接 panic
    void load_shader(const std::filesystem::path& dir, std::string_view file_name, std::vector<unsigned char>& out) {
        const std::filesystem::path path = dir / file_name;
        if (!read_binary_file(path, out)) {
            utility::panic(std::format("cannot open shader file '{}'", path.string()));
        }
        std::println("loaded shader: {} ({} bytes)", path.string(), out.size());
    }

    // 加载一对 vertex/fragment SPIR-V，并通过 runtime 创建管线；失败直接 panic
    void load_and_create_pipeline(vulkan::runtime& runtime,
                                  const std::filesystem::path& shaders_dir,
                                  std::string_view pipeline_name,
                                  std::string_view vertex_file,
                                  std::string_view fragment_file) {
        std::vector<unsigned char> vertex_code;
        std::vector<unsigned char> fragment_code;
        load_shader(shaders_dir, vertex_file, vertex_code);
        load_shader(shaders_dir, fragment_file, fragment_code);

        const std::expected<void, std::string> result = runtime.make_pipeline(pipeline_name, vertex_code, fragment_code);
        if (!result) {
            utility::panic(std::format("failed to create pipeline '{}': {}", pipeline_name, result.error()));
        }
        std::println("SUCCESS: pipeline '{}' created and cached in the runtime", pipeline_name);
    }

    // 把 stb 解码的纹理数据转成 RGBA（3 通道补 alpha，1 通道灰度扩展）
    std::vector<unsigned char> to_rgba(const gltf::texture_data& texture) {
        const size_t pixel_count = static_cast<size_t>(texture.width) * texture.height;
        std::vector<unsigned char> rgba(pixel_count * 4, 255);
        switch (texture.component) {
        case 4:
            rgba = texture.data;
            break;
        case 3:
            for (size_t i = 0; i < pixel_count; ++i) {
                rgba[i * 4 + 0] = texture.data[i * 3 + 0];
                rgba[i * 4 + 1] = texture.data[i * 3 + 1];
                rgba[i * 4 + 2] = texture.data[i * 3 + 2];
            }
            break;
        case 1:
            for (size_t i = 0; i < pixel_count; ++i) {
                rgba[i * 4 + 0] = texture.data[i];
                rgba[i * 4 + 1] = texture.data[i];
                rgba[i * 4 + 2] = texture.data[i];
            }
            break;
        default:
            rgba.clear();
            break;
        }
        return rgba;
    }

    // VMA 上传贴图 + 创建 image view
    struct texture_bundle {
        uint64_t image = 0;
        VkImageView view = VK_NULL_HANDLE;
    };

    texture_bundle create_texture(vulkan::runtime& runtime, const gltf::texture_data& texture, const VkFormat format) {
        texture_bundle bundle;
        const std::vector<unsigned char> rgba = to_rgba(texture);
        if (rgba.empty()) {
            return bundle;
        }

        vulkan::image_create_info create_info{};
        create_info.width = texture.width;
        create_info.height = texture.height;
        create_info.mip_levels = 1;
        create_info.array_layers = 1;
        create_info.format = format;
        bundle.image = runtime->vma.create_image(rgba.data(), rgba.size(), create_info, vulkan::image_type::texture_2d);
        const auto* detail = runtime->vma.get_image_detail(bundle.image);
        if (detail == nullptr) {
            return bundle;
        }

        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = detail->image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = format;
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(runtime->device, &view_info, nullptr, &bundle.view);
        return bundle;
    }

    VkSampler create_texture_sampler(const VkDevice device) {
        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.maxAnisotropy = 1.0f;
        info.minLod = 0.0f;
        info.maxLod = 0.25f; // 只有 mip 0，避免采样到未生成的层级

        VkSampler sampler = VK_NULL_HANDLE;
        vkCreateSampler(device, &info, nullptr, &sampler);
        return sampler;
    }

    // 轨道相机：左键拖拽旋转、滚轮缩放（观察模型用）
    struct orbit_camera {
        double last_x = 0.0;
        double last_y = 0.0;
        bool dragging = false;
        float yaw = 0.0f;
        float pitch = 0.35f; // 略俯视
        float distance = 2.2f;
    };

    void mouse_button_callback(GLFWwindow* window, const int button, const int action, const int) {
        auto* camera = static_cast<orbit_camera*>(glfwGetWindowUserPointer(window));
        if (button != GLFW_MOUSE_BUTTON_LEFT) {
            return;
        }
        if (action == GLFW_PRESS) {
            camera->dragging = true;
            glfwGetCursorPos(window, &camera->last_x, &camera->last_y);
        } else if (action == GLFW_RELEASE) {
            camera->dragging = false;
        }
    }

    void cursor_pos_callback(GLFWwindow* window, const double x, const double y) {
        auto* camera = static_cast<orbit_camera*>(glfwGetWindowUserPointer(window));
        if (!camera->dragging) {
            return;
        }
        constexpr float sensitivity = 0.005f;
        const float dx = static_cast<float>(x - camera->last_x);
        const float dy = static_cast<float>(y - camera->last_y);
        camera->last_x = x;
        camera->last_y = y;
        camera->yaw += dx * sensitivity; // 拖拽方向与模型旋转方向一致
        camera->pitch -= dy * sensitivity;
        camera->pitch = std::clamp(camera->pitch, -1.5f, 1.5f); // 避免翻转
    }

    void scroll_callback(GLFWwindow* window, const double, const double yoffset) {
        auto* camera = static_cast<orbit_camera*>(glfwGetWindowUserPointer(window));
        camera->distance *= std::pow(0.9f, static_cast<float>(yoffset));
        camera->distance = std::clamp(camera->distance, 0.5f, 20.0f);
    }

    // 轨道相机位置：模型已平移到原点，相机绕原点球面运动
    glm::vec3 orbit_eye(const orbit_camera& camera) {
        const float cp = std::cos(camera.pitch);
        return glm::vec3(camera.distance * cp * std::sin(camera.yaw),
                         camera.distance * std::sin(camera.pitch),
                         camera.distance * cp * std::cos(camera.yaw));
    }
} // namespace

int main(int argc, char** argv) {
    // 1. 定位 shaders 目录（保存 GLSL 源码与编译好的 SPIR-V）
    const std::optional<std::filesystem::path> shaders_dir = locate_shaders_dir();
    if (!shaders_dir) {
        utility::panic("cannot find shaders/ directory. run the program from the project root or a cmake-build-* directory.");
    }

    // 2. 构造 vulkan::runtime：默认构造会完成 window/instance/device/swapchain 等全部初始化
    vulkan::runtime runtime;

    // 3. 管线加载测试：简单三角形 + 标准 PBR
    load_and_create_pipeline(runtime, *shaders_dir, "triangle", "triangle.vert.spv", "triangle.frag.spv");
    load_and_create_pipeline(runtime, *shaders_dir, "pbr", "pbr.vert.spv", "pbr.frag.spv");

    // 4. 加载 glTF 模型（默认 DamagedHelmet，可用命令行参数指定其它 .gltf/.glb）
    std::string model_path = "C:/Users/23530/Desktop/yzk/glTF-Sample-Assets/Models/DamagedHelmet/glTF/DamagedHelmet.gltf";
    if (argc > 1) {
        model_path = argv[1];
    }
    std::println("loading model: {}", model_path);
    auto scenes = gltf::load_model(model_path);
    if (!scenes) {
        utility::panic(std::format("failed to load model '{}': error code {}", model_path, static_cast<int>(scenes.error())));
    }
    if (scenes->scene.empty() || scenes->scene[0].nodes.empty() ||
        scenes->scene[0].nodes[0].meshes.empty() || scenes->scene[0].nodes[0].meshes[0].primitives.empty()) {
        utility::panic(std::format("model '{}' has no drawable primitive", model_path));
    }
    const auto& prim = scenes->scene[0].nodes[0].meshes[0].primitives[0];
    std::println("model loaded: {} textures, {} primitives", scenes->textures.size(), scenes->scene[0].nodes[0].meshes[0].primitives.size());

    // 5. 取顶点属性（分离存储），缺少的属性用默认值
    const auto get_portion = [&prim](const std::string_view name) -> const gltf::vertex_portion* {
        const auto it = prim.vertex.find(std::string(name));
        return it == prim.vertex.end() ? nullptr : &it->second;
    };
    const auto* position_portion = get_portion("POSITION");
    const auto* normal_portion = get_portion("NORMAL");
    const auto* uv_portion = get_portion("TEXCOORD_0");
    const auto* tangent_portion = get_portion("TANGENT");
    if (position_portion == nullptr) {
        utility::panic("model has no POSITION attribute");
    }

    const glm::vec3 default_normal(0.0f, 1.0f, 0.0f);
    const glm::vec2 default_uv(0.0f, 0.0f);
    const size_t vertex_count = position_portion->data.size() / sizeof(glm::vec3);

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    positions.reserve(vertex_count);
    normals.reserve(vertex_count);
    uvs.reserve(vertex_count);
    for (size_t i = 0; i < vertex_count; ++i) {
        const auto* p = reinterpret_cast<const glm::vec3*>(position_portion->data.data()) + i;
        const auto* n = normal_portion == nullptr ? &default_normal : reinterpret_cast<const glm::vec3*>(normal_portion->data.data()) + i;
        const auto* uv = uv_portion == nullptr ? &default_uv : reinterpret_cast<const glm::vec2*>(uv_portion->data.data()) + i;
        positions.push_back(*p);
        normals.push_back(*n);
        uvs.push_back(*uv);
    }

    // 6. 索引数据与类型
    if (prim.index.empty()) {
        utility::panic("model has no index data");
    }
    VkIndexType index_type = VK_INDEX_TYPE_UINT16;
    if (prim.index_component_type == gltf::component_type::unsigned_int_t) {
        index_type = VK_INDEX_TYPE_UINT32;
    } else if (prim.index_component_type != gltf::component_type::unsigned_short_t) {
        utility::panic(std::format("unsupported index component type: {}", static_cast<int>(prim.index_component_type)));
    }
    const uint32_t index_count = static_cast<uint32_t>(prim.index.size() / (index_type == VK_INDEX_TYPE_UINT32 ? 4 : 2));
    const auto read_index = [&prim, index_type](const size_t i) -> uint32_t {
        if (index_type == VK_INDEX_TYPE_UINT32) {
            return reinterpret_cast<const uint32_t*>(prim.index.data())[i];
        }
        return reinterpret_cast<const uint16_t*>(prim.index.data())[i];
    };

    // 7. 切线：模型自带 TANGENT 则直接使用，否则按 position/uv 逐三角形计算
    //    （经典算法：按三角形累加切线，再用 Gram-Schmidt 正交化）
    std::vector<glm::vec3> tangents(vertex_count, glm::vec3(1.0f, 0.0f, 0.0f));
    if (tangent_portion != nullptr) {
        for (size_t i = 0; i < vertex_count; ++i) {
            const auto* t = reinterpret_cast<const glm::vec4*>(tangent_portion->data.data()) + i;
            tangents[i] = glm::vec3(t->x, t->y, t->z);
        }
    } else {
        std::vector<glm::vec3> tangent_accumulator(vertex_count, glm::vec3(0.0f));
        for (uint32_t i = 0; i + 2 < index_count; i += 3) {
            const uint32_t i0 = read_index(i);
            const uint32_t i1 = read_index(i + 1);
            const uint32_t i2 = read_index(i + 2);
            const glm::vec3 e1 = positions[i1] - positions[i0];
            const glm::vec3 e2 = positions[i2] - positions[i0];
            const glm::vec2 duv1 = uvs[i1] - uvs[i0];
            const glm::vec2 duv2 = uvs[i2] - uvs[i0];
            const float denom = duv1.x * duv2.y - duv2.x * duv1.y;
            if (std::abs(denom) < 1e-8f) {
                continue; // 退化的 UV 三角形
            }
            const float f = 1.0f / denom;
            const glm::vec3 tangent = f * (duv2.y * e1 - duv1.y * e2);
            tangent_accumulator[i0] += tangent;
            tangent_accumulator[i1] += tangent;
            tangent_accumulator[i2] += tangent;
        }
        for (size_t i = 0; i < vertex_count; ++i) {
            const glm::vec3 t = tangent_accumulator[i] - normals[i] * glm::dot(normals[i], tangent_accumulator[i]);
            tangents[i] = glm::length(t) > 1e-8f ? glm::normalize(t) : glm::vec3(1.0f, 0.0f, 0.0f);
        }
        std::println("TANGENT not in model, computed from position/uv");
    }

    // 8. 交错成管线要求的单绑定布局（stride 44）
    std::vector<vertex> vertices;
    vertices.reserve(vertex_count);
    for (size_t i = 0; i < vertex_count; ++i) {
        vertices.push_back(vertex{positions[i], normals[i], uvs[i], tangents[i]});
    }

    // 9. 包围盒自动适配相机
    glm::vec3 bmin = glm::vec3(std::numeric_limits<float>::infinity());
    glm::vec3 bmax = glm::vec3(-std::numeric_limits<float>::infinity());
    for (const auto& v : vertices) {
        bmin = glm::min(bmin, v.position);
        bmax = glm::max(bmax, v.position);
    }
    const glm::vec3 center = (bmin + bmax) * 0.5f;
    const glm::vec3 extent = bmax - bmin;
    const float max_extent = glm::max(extent.x, glm::max(extent.y, extent.z));
    const float fit_scale = max_extent > 0.0f ? 1.6f / max_extent : 1.0f;
    std::println("mesh: {} vertices, {} indices, center ({:.3f}, {:.3f}, {:.3f}), extent {:.3f}", vertices.size(), index_count, center.x, center.y, center.z, max_extent);

    // 10. GPU 缓冲：顶点 + 索引
    const uint64_t vertex_buffer = runtime->vma.create_buffer(std::span(vertices), vulkan::buffer_type::vertex);
    const uint64_t index_buffer = runtime->vma.create_buffer(prim.index.data(), prim.index.size(), vulkan::buffer_type::index);
    const auto* vertex_detail = runtime->vma.get_buffer_detail(vertex_buffer);
    const auto* index_detail = runtime->vma.get_buffer_detail(index_buffer);
    if (vertex_detail == nullptr || index_detail == nullptr) {
        utility::panic("failed to create mesh buffers");
    }

    // 11. 取 PBR 管线
    const auto* pipeline = runtime.get_pipeline("pbr");
    if (pipeline == nullptr) {
        utility::panic("pipeline 'pbr' not found in runtime");
    }

    // 12. 材质纹理：albedo 用 sRGB（PBR 线性空间），其余数据贴图用 UNORM
    const auto& texture_indices = prim.texture_indices;
    const auto load_material_texture = [&runtime, &scenes, &texture_indices](const std::string_view semantic, const VkFormat format) -> texture_bundle {
        const auto it = texture_indices.find(std::string(semantic));
        if (it != texture_indices.end() && it->second < scenes->textures.size()) {
            return create_texture(runtime, scenes->textures[it->second], format);
        }
        return {};
    };
    const std::array<texture_bundle, 5> material_textures = {
        load_material_texture("albedo", VK_FORMAT_R8G8B8A8_SRGB),
        load_material_texture("metallic_roughness", VK_FORMAT_R8G8B8A8_UNORM),
        load_material_texture("normal", VK_FORMAT_R8G8B8A8_UNORM),
        load_material_texture("occlusion", VK_FORMAT_R8G8B8A8_UNORM),
        load_material_texture("emissive", VK_FORMAT_R8G8B8A8_UNORM),
    };

    // 缺失贴图时的 1x1 白色占位
    gltf::texture_data white_texture_data{};
    white_texture_data.data = {255, 255, 255, 255};
    white_texture_data.width = 1;
    white_texture_data.height = 1;
    white_texture_data.component = 4;
    const texture_bundle white_texture = create_texture(runtime, white_texture_data, VK_FORMAT_R8G8B8A8_UNORM);

    const VkSampler sampler = create_texture_sampler(runtime->device);

    // set 1：5 个 combined image sampler 绑定（base_color, metallic_roughness, normal, occlusion, emissive）
    auto texture_set = runtime->make_descriptor_set(pipeline->get_descriptor_set_layouts()[1]);
    std::array<VkDescriptorImageInfo, 5> image_infos{};
    std::array<VkWriteDescriptorSet, 5> texture_writes{};
    for (int binding = 0; binding < 5; ++binding) {
        const texture_bundle& bundle = material_textures[binding].view != VK_NULL_HANDLE ? material_textures[binding] : white_texture;
        image_infos[binding] = {sampler, bundle.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        texture_writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        texture_writes[binding].dstSet = *texture_set;
        texture_writes[binding].dstBinding = static_cast<uint32_t>(binding);
        texture_writes[binding].descriptorCount = 1;
        texture_writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        texture_writes[binding].pImageInfo = &image_infos[binding];
    }
    vkUpdateDescriptorSets(runtime->device, static_cast<uint32_t>(texture_writes.size()), texture_writes.data(), 0, nullptr);

    // 13. 每帧槽位一个 CameraUBO + set 0 描述符集
    constexpr size_t ubo_size = sizeof(camera_ubo);
    std::vector<uint64_t> ubo_buffers;
    std::vector<vulkan::vk_descriptor_set> ubo_sets;
    ubo_buffers.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
    ubo_sets.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
    for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
        camera_ubo initial{};
        ubo_buffers.push_back(runtime->vma.create_buffer(std::span(&initial, 1), vulkan::buffer_type::uniform_coherent));
        const auto* ubo_detail = runtime->vma.get_buffer_detail(ubo_buffers.back());
        if (ubo_detail == nullptr) {
            utility::panic("failed to create ubo buffer");
        }

        auto ubo_set = runtime->make_descriptor_set(pipeline->get_descriptor_set_layouts()[0]);
        const VkDescriptorBufferInfo ubo_info{ubo_detail->buffer, 0, ubo_size};
        VkWriteDescriptorSet write_info{};
        write_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_info.dstSet = *ubo_set;
        write_info.dstBinding = 0;
        write_info.descriptorCount = 1;
        write_info.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write_info.pBufferInfo = &ubo_info;
        vkUpdateDescriptorSets(runtime->device, 1, &write_info, 0, nullptr);
        ubo_sets.push_back(std::move(ubo_set));
    }

    // 14. 每帧槽位一条命令缓冲
    std::vector<vulkan::vk_command_buffer> command_buffers;
    command_buffers.reserve(runtime->MAX_FRAMES_IN_FLIGHT);
    for (int slot = 0; slot < runtime->MAX_FRAMES_IN_FLIGHT; ++slot) {
        command_buffers.push_back(runtime->make_command_buffer());
    }

    // 15. 材质 push constants
    pbr_push_constants push{};
    push.flags = 0;
    if (texture_indices.contains("normal")) {
        push.flags |= 1u;
    }
    if (texture_indices.contains("occlusion")) {
        push.flags |= 2u;
    }
    if (texture_indices.contains("emissive")) {
        push.flags |= 4u;
    }

    // 16. 相机：轨道观察（左键拖拽旋转，滚轮缩放）
    const float aspect = static_cast<float>(runtime->swap_chain_extent.width) / static_cast<float>(runtime->swap_chain_extent.height);
    // RH_ZO：右手系 + 深度 [0,1]（Vulkan 约定）
    glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(45.0f), aspect, 0.1f, 100.0f);
    // glm 的投影按 OpenGL 约定（NDC y 向上），而 Vulkan framebuffer 的 y 向下：
    // 翻转投影 Y 分量，否则 glTF 的 CCW 正面绕序在帧缓冲里变成 CW，
    // 被管线的 CULL_BACK 裁掉，只会看到模型内壁。
    proj[1][1] *= -1.0f;

    orbit_camera camera{};
    glfwSetWindowUserPointer(runtime->window, &camera);
    glfwSetMouseButtonCallback(runtime->window, mouse_button_callback);
    glfwSetCursorPosCallback(runtime->window, cursor_pos_callback);
    glfwSetScrollCallback(runtime->window, scroll_callback);

    // 17. 渲染主循环：直到关闭窗口或按 ESC
    std::println("rendering '{}' with PBR... left-drag to orbit, wheel to zoom, ESC to exit", model_path);
    while (!glfwWindowShouldClose(runtime->window)) {
        glfwPollEvents();
        if (glfwGetKey(runtime->window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(runtime->window, GLFW_TRUE);
        }

        const size_t frame_slot = runtime->current_frame;

        vkWaitForFences(runtime->device, 1, &runtime->in_flight_fences[frame_slot], VK_TRUE, UINT64_MAX);
        vkResetFences(runtime->device, 1, &runtime->in_flight_fences[frame_slot]);

        // 更新本槽位的 UBO（轨道相机）
        const glm::vec3 eye = orbit_eye(camera);
        camera_ubo ubo{};
        ubo.model = glm::scale(glm::mat4(1.0f), glm::vec3(fit_scale)) * glm::translate(glm::mat4(1.0f), -center);
        ubo.view = glm::lookAt(eye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj = proj;
        ubo.camera_pos = eye;
        const auto* ubo_detail = runtime->vma.get_buffer_detail(ubo_buffers[frame_slot]);
        std::memcpy(ubo_detail->allocation_info.pMappedData, &ubo, ubo_size);

        uint32_t image_index = 0;
        if (vkAcquireNextImageKHR(runtime->device,
                                  runtime->swap_chain,
                                  UINT64_MAX,
                                  runtime->image_available_semaphores[frame_slot],
                                  VK_NULL_HANDLE,
                                  &image_index) != VK_SUCCESS) {
            break;
        }

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

        const VkViewport viewport{0.0f,
                                  0.0f,
                                  static_cast<float>(runtime->swap_chain_extent.width),
                                  static_cast<float>(runtime->swap_chain_extent.height),
                                  0.0f,
                                  1.0f};
        const VkRect2D scissor{{0, 0}, runtime->swap_chain_extent};
        vkCmdSetViewport(*command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(*command_buffer, 0, 1, &scissor);

        const VkBuffer vertex_handle = vertex_detail->buffer;
        const VkDeviceSize vertex_offset = 0;
        vkCmdBindVertexBuffers(*command_buffer, 0, 1, &vertex_handle, &vertex_offset);
        vkCmdBindIndexBuffer(*command_buffer, index_detail->buffer, 0, index_type);

        const std::array<VkDescriptorSet, 2> bound_sets = {*ubo_sets[frame_slot], *texture_set};
        vkCmdBindDescriptorSets(*command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline->get_pipeline_layout(),
                                0,
                                static_cast<uint32_t>(bound_sets.size()),
                                bound_sets.data(),
                                0,
                                nullptr);
        vkCmdPushConstants(*command_buffer,
                           pipeline->get_pipeline_layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(pbr_push_constants),
                           &push);
        vkCmdDrawIndexed(*command_buffer, index_count, 1, 0, 0, 0);

        vkCmdEndRenderPass(*command_buffer);
        if (vkEndCommandBuffer(*command_buffer) != VK_SUCCESS) {
            break;
        }

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

    // 18. 等待 GPU 完成后再释放资源
    vkDeviceWaitIdle(runtime->device);
    vkDestroySampler(runtime->device, sampler, nullptr);
    for (const auto& bundle : material_textures) {
        if (bundle.view != VK_NULL_HANDLE) {
            vkDestroyImageView(runtime->device, bundle.view, nullptr);
        }
        if (bundle.image != 0) {
            runtime->vma.free_image(bundle.image);
        }
    }
    if (white_texture.view != VK_NULL_HANDLE) {
        vkDestroyImageView(runtime->device, white_texture.view, nullptr);
    }
    if (white_texture.image != 0) {
        runtime->vma.free_image(white_texture.image);
    }
    for (const uint64_t ubo : ubo_buffers) {
        runtime->vma.free_buffer(ubo);
    }
    runtime->vma.free_buffer(index_buffer);
    runtime->vma.free_buffer(vertex_buffer);
    std::println("render loop finished");
    return 0;
}
