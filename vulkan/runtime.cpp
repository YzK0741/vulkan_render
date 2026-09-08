module;

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

module vulkan.runtime;

import utility;

// Route std::pmr allocations through mimalloc for this TU (utility.better_pmr). Idempotent:
// init_pmr() returns the same process-wide singleton no matter which TU calls it first, so
// main.cpp's keep-alive and this one coexist safely. The reference itself is never read; it
// only forces the (dynamic) initialization before any pmr container in this TU is constructed.
[[maybe_unused]] static auto& pmr = utility::init_pmr(); // NOLINT(keep-alive)

namespace {
    vulkan::runtime* runtime_from_window(GLFWwindow* window) {
        return static_cast<vulkan::runtime*>(glfwGetWindowUserPointer(window));
    }

    void mouse_button_callback(GLFWwindow* window, int const button, int const action, [[maybe_unused]] int const mods) {
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

    void cursor_pos_callback(GLFWwindow* window, double const x, double const y) {
        auto& camera = runtime_from_window(window)->camera;
        if (!camera.dragging) {
            return;
        }
        // an external (glTF/programmatic) camera owns the view: orbit dragging is ignored
        if (runtime_from_window(window)->using_external_camera()) {
            return;
        }
        constexpr float sensitivity = 0.005f;
        float const dx = static_cast<float>(x - camera.last_x);
        float const dy = static_cast<float>(y - camera.last_y);
        camera.last_x = x;
        camera.last_y = y;
        camera.yaw += dx * sensitivity; // drag direction matches the primitive rotation
        camera.pitch -= dy * sensitivity;
        camera.pitch = std::clamp(camera.pitch, -1.5f, 1.5f); // avoid flipping
    }

    void scroll_callback(GLFWwindow* window, [[maybe_unused]] double const xoffset, double const yoffset) {
        // an external (glTF/programmatic) camera owns the view: wheel zoom is ignored
        if (runtime_from_window(window)->using_external_camera()) {
            return;
        }
        auto& camera = runtime_from_window(window)->camera;
        // zoom: wheel up pulls in, wheel down pulls out. The upper bound is generous (scene
        // sizes vary from the tiny default model to e.g. the Fox rig, whose framing distance
        // is ~240); the projection far plane follows the distance (see make_orbit_camera_ubo).
        camera.distance *= std::pow(0.9f, static_cast<float>(yoffset));
        camera.distance = std::clamp(camera.distance, 0.5f, 5000.0f);
    }
} // namespace

namespace vulkan {
    // Shared worker-pool sizing: hardware_concurrency()/4 (floor 1; fall back to 2 when the
    // runtime cannot report the core count). A quarter keeps the pool off the frame thread's
    // back while still giving heavy CPU stages (animation sampling fan-out) real parallelism.
    int runtime::default_task_pool_threads() noexcept {
        unsigned const hw = std::thread::hardware_concurrency();
        return static_cast<int>(hw == 0 ? 2u : std::max(1u, hw / 4u));
    }

    // Run a batch of tasks on the shared pool and wait for exactly this stage's group: the
    // frame phases are synchronous (the paced slot is read right after animation sampling),
    // so run_tasks blocks until every task in the batch finished. The enum tier is mapped
    // onto the pool's integer priority (see task_priority in runtime.cppm).
    void runtime::run_tasks(std::span<std::function<void()>> const tasks, task_priority const priority) {
        int const pool_priority = static_cast<int>(priority);
        if (tasks.empty() || !this->task_pool.post_batch(tasks, pool_priority)) {
            return; // empty batch, or the pool is shut down (never in the running demo)
        }
        this->task_pool.wait_until_priority_done(pool_priority);
    }

    runtime::runtime()
        : runtime(core_create_info{}) {
    }

    runtime::runtime(core_create_info const& options)
        : vulkan_core{options}
        , filtered_core{vulkan_core} {
        glfwSetWindowUserPointer(this->vulkan_core.window, this);
        glfwSetMouseButtonCallback(this->vulkan_core.window, mouse_button_callback);
        glfwSetCursorPosCallback(this->vulkan_core.window, cursor_pos_callback);
        glfwSetScrollCallback(this->vulkan_core.window, scroll_callback);

        // One command buffer per frame slot, owned and reused every frame
        this->command_buffers.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            this->command_buffers.push_back(this->vulkan_core.make_command_buffer());
        }

        // Shared scene resources: camera UBO buffers, white fallback texture, texture sampler
        this->init_scene_resources();
        this->init_shadow_resources();
    }

    // The destructor body runs before member destruction, so vulkan_core (and the VkDevice it
    // holds) is still alive here: destroying cached pipelines in this order is guaranteed safe,
    // independent of future member reordering. Members then destruct in reverse declaration
    // order with pipelines already empty. The SCENE TREE is caller-owned (set_scene): the caller
    // destroys it before this runtime goes away (its leaves release GPU buffers through the vma
    // allocator while it is still alive), so no tree teardown happens here.
    runtime::~runtime() {
        // Wait for the GPU to finish BEFORE releasing anything below: the last submitted frame
        // may still be executing and destroying in-use resources would violate VUIDs (~core()
        // also waits, but that runs after this body — too late for the VMA frees here).
        this->vulkan_core.wait_idle();

        this->pipelines.clear();

        // Shared scene resources: views/sets/samplers/buffers/images are RAII and free
        // themselves as this runtime's members destruct (after this body; vulkan_core, which
        // owns the vma allocator, is declared first and destructs last, so every vk_buffer /
        // vk_image still has a live allocator when it releases).

        // Shut the debug overlay down explicitly while the VkDevice is still alive (its ImGui
        // Vulkan backend owns device resources); member destruction would also run it before
        // vulkan_core, but doing it here keeps the order obvious.
        this->debug_overlay.shutdown();
    }

    void runtime::init_scene_resources() {
        // Camera UBO: one buffer per frame slot, mapped for direct writes; all models reference
        // these buffers through the shared scene set, so one memcpy per frame replaces the old
        // per-primitive per-frame UBO updates
        this->camera_buffers.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        this->camera_mapped.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            camera_ubo initial = {};
            vk_buffer buffer = this->vulkan_core.vma.create_buffer(std::span(&initial, 1), vulkan::buffer_type::uniform_coherent);
            if (!buffer.valid()) {
                utility::panic("failed to create camera ubo buffer");
            }
            auto const* detail = this->vulkan_core.vma.get_buffer_detail(buffer.handle());
            if (detail == nullptr) {
                utility::panic("failed to get camera ubo buffer detail");
            }
            this->camera_buffers.push_back(std::move(buffer));
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
        vk_image white_image = this->vulkan_core.vma.create_image(white_pixels.data(), white_pixels.size(), white_info, vulkan::image_type::texture_2d);
        if (!white_image.valid()) {
            utility::panic("failed to create white fallback texture");
        }
        auto const* white_detail = this->vulkan_core.vma.get_image_detail(white_image.handle());
        if (white_detail == nullptr) {
            utility::panic("failed to get white texture detail");
        }
        this->owned_textures.push_back(std::move(white_image));
        this->owned_texture_views.push_back(this->vulkan_core.make_image_view(white_detail->image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D));
        this->white_texture_index = static_cast<uint32_t>(this->texture_array_views.size());
        this->texture_array_views.push_back(*this->owned_texture_views.back());

        // Shared sampler for the texture array entries
        // maxLod 12 covers mip chains up to 4096x4096 (13 levels); images with fewer mips simply
        // clamp to their last level. Sampled with a LINEAR mip filter, so far/small surfaces
        // use the pre-generated mips instead of aliasing mip0.
        this->texture_sampler = this->vulkan_core.make_sampler(VK_SAMPLER_ADDRESS_MODE_REPEAT, 12.0f);

        // GPU material table: fixed capacity, host-visible (direct mapping); records are appended
        // at registration and read-only for the GPU (set 0 binding 5)
        std::vector<unsigned char> const zeroed_materials(static_cast<size_t>(vulkan::material_capacity) * sizeof(material_record), 0);
        vk_buffer material_buf = this->vulkan_core.vma.create_buffer(zeroed_materials.data(), zeroed_materials.size(), vulkan::buffer_type::storage_coherent);
        if (!material_buf.valid()) {
            utility::panic("failed to create material table buffer");
        }
        auto const* material_detail = this->vulkan_core.vma.get_buffer_detail(material_buf.handle());
        if (material_detail == nullptr) {
            utility::panic("failed to get material table buffer detail");
        }
        this->material_buffer = std::move(material_buf);
        this->material_mapped = material_detail->allocation_info.pMappedData;

        // Per-instance transform buffer (set 0 binding 6): one mat4 per instance, host-visible;
        // filled by set_instanced_draw() for instanced stress draws (see pbr.vert)
        std::vector<unsigned char> const zeroed_instances(static_cast<size_t>(vulkan::instance_capacity) * sizeof(glm::mat4), 0);
        vk_buffer instance_buf = this->vulkan_core.vma.create_buffer(zeroed_instances.data(), zeroed_instances.size(), vulkan::buffer_type::storage_coherent);
        if (!instance_buf.valid()) {
            utility::panic("failed to create instance transform buffer");
        }
        auto const* instance_detail = this->vulkan_core.vma.get_buffer_detail(instance_buf.handle());
        if (instance_detail == nullptr) {
            utility::panic("failed to get instance transform buffer detail");
        }
        this->instance_buffer = std::move(instance_buf);
        this->instance_mapped = instance_detail->allocation_info.pMappedData;

        // Per-joint skin matrices (set 0 binding 9): one buffer PER FRAME SLOT (scene_skin_capacity
        // mat4s each, host-visible) so an in-flight frame never shares the buffer the next frame
        // rewrites. Zero-filled initially (the identity block is written by the setup upload).
        std::vector<unsigned char> const zeroed_skins(static_cast<size_t>(vulkan::scene_skin_capacity) * sizeof(glm::mat4), 0);
        this->skin_buffers.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        this->skin_mapped.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            vk_buffer skin_buf = this->vulkan_core.vma.create_buffer(zeroed_skins.data(), zeroed_skins.size(), vulkan::buffer_type::storage_coherent);
            if (!skin_buf.valid()) {
                utility::panic("failed to create skin matrix buffer");
            }
            auto const* skin_detail = this->vulkan_core.vma.get_buffer_detail(skin_buf.handle());
            if (skin_detail == nullptr) {
                utility::panic("failed to get skin matrix buffer detail");
            }
            this->skin_buffers.push_back(std::move(skin_buf));
            this->skin_mapped.push_back(skin_detail->allocation_info.pMappedData);
        }

        // Morph data (set 0 binding 10): one buffer PER FRAME SLOT (scene_morph_capacity floats
        // each, host-visible); the caller bakes per-primitive morph blocks (deltas + weights)
        // into every slot's buffer at setup, then rewrites only the active slot's weights per frame.
        // Zero-filled from one shared host vector (each create_buffer copies its own GPU buffer).
        std::vector<unsigned char> const zeroed_morphs(static_cast<size_t>(vulkan::scene_morph_capacity) * sizeof(float), 0);
        this->morph_buffers.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        this->morph_mapped.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            vk_buffer morph_buf = this->vulkan_core.vma.create_buffer(zeroed_morphs.data(), zeroed_morphs.size(), vulkan::buffer_type::storage_coherent);
            if (!morph_buf.valid()) {
                utility::panic("failed to create morph data buffer");
            }
            auto const* morph_detail = this->vulkan_core.vma.get_buffer_detail(morph_buf.handle());
            if (morph_detail == nullptr) {
                utility::panic("failed to get morph data buffer detail");
            }
            this->morph_buffers.push_back(std::move(morph_buf));
            this->morph_mapped.push_back(morph_detail->allocation_info.pMappedData);
        }
    }

    void runtime::init_shadow_resources() {
        // Shadow map: one depth image per frame slot (see the member docs). Depth-only images
        // carry no uploaded content (vma::create_image with data == nullptr skips the digest /
        // upload path), so each frame can render the scene's depth from the light's view into it.
        this->shadow_images.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        this->shadow_image_views.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            vulkan::image_create_info shadow_info = {};
            shadow_info.width = vulkan::runtime::shadow_map_size;
            shadow_info.height = vulkan::runtime::shadow_map_size;
            shadow_info.mip_levels = 1;
            shadow_info.array_layers = 1;
            shadow_info.format = this->vulkan_core.depth_format;
            shadow_info.extra_usage = VK_IMAGE_USAGE_SAMPLED_BIT; // sampled by pbr.frag
            vk_image shadow_image = this->vulkan_core.vma.create_image(nullptr, 0, shadow_info, vulkan::image_type::texture_2d_depth);
            if (!shadow_image.valid()) {
                utility::panic("failed to create shadow map image");
            }
            auto const* detail = this->vulkan_core.vma.get_image_detail(shadow_image.handle());
            if (detail == nullptr) {
                utility::panic("failed to get shadow map image detail");
            }
            this->shadow_images.push_back(std::move(shadow_image));
            this->shadow_image_views.push_back(this->vulkan_core.make_depth_image_view(detail->image, this->vulkan_core.depth_format));
        }
        this->shadow_sampler = this->vulkan_core.make_shadow_sampler();

        // Light UBO (scene set binding 7): static content, filled by enable_shadows()
        light_ubo initial = {};
        vk_buffer light_buf = this->vulkan_core.vma.create_buffer(std::span(&initial, 1), vulkan::buffer_type::uniform_coherent);
        if (!light_buf.valid()) {
            utility::panic("failed to create light ubo buffer");
        }
        auto const* light_detail = this->vulkan_core.vma.get_buffer_detail(light_buf.handle());
        if (light_detail == nullptr) {
            utility::panic("failed to get light ubo buffer detail");
        }
        this->light_buffer = std::move(light_buf);
        this->light_mapped = light_detail->allocation_info.pMappedData;
    }

    void runtime::ensure_scene_set() {
        if (this->scene_set_created) {
            return;
        }
        // One scene descriptor set per frame slot: a slot's set always points at that slot's own
        // camera / shadow / skin / morph resources, so an in-flight frame never observes the next
        // frame's descriptors and no per-frame update-after-bind writes are needed at all.
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            this->scene_sets[static_cast<std::size_t>(slot)] = this->vulkan_core.make_descriptor_set(this->vulkan_core.scene_descriptor_set_layout);
        }
        this->scene_set_created = true;

        auto const write_buffer_binding = [this](VkDescriptorSet const set, uint32_t const binding, VkBuffer const buffer, VkDeviceSize const size, VkDescriptorType const type) {
            VkDescriptorBufferInfo const info{buffer, 0, size};
            VkWriteDescriptorSet write = {};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set;
            write.dstBinding = binding;
            write.descriptorCount = 1;
            write.descriptorType = type;
            write.pBufferInfo = &info;
            vkUpdateDescriptorSets(this->vulkan_core.device, 1, &write, 0, nullptr);
        };

        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            VkDescriptorSet const set = *this->scene_sets[static_cast<std::size_t>(slot)];

            // binding 0: THIS slot's camera UBO buffer
            auto const* camera_detail = this->vulkan_core.vma.get_buffer_detail(this->camera_buffers[static_cast<std::size_t>(slot)].handle());
            if (camera_detail == nullptr) {
                utility::panic("failed to get camera ubo buffer detail");
            }
            write_buffer_binding(set, 0, camera_detail->buffer, sizeof(camera_ubo), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

            // binding 5: material table (shared, written once)
            auto const* material_detail = this->vulkan_core.vma.get_buffer_detail(this->material_buffer.handle());
            if (material_detail == nullptr) {
                utility::panic("failed to get material table buffer detail");
            }
            write_buffer_binding(set, 5, material_detail->buffer, static_cast<VkDeviceSize>(vulkan::material_capacity) * sizeof(material_record), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

            // binding 6: per-instance transforms (shared, written by set_instanced_draw)
            auto const* instance_detail = this->vulkan_core.vma.get_buffer_detail(this->instance_buffer.handle());
            if (instance_detail == nullptr) {
                utility::panic("failed to get instance transform buffer detail");
            }
            write_buffer_binding(set, 6, instance_detail->buffer, static_cast<VkDeviceSize>(vulkan::instance_capacity) * sizeof(glm::mat4), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

            // binding 9: THIS slot's skin matrix buffer
            auto const* skin_detail = this->vulkan_core.vma.get_buffer_detail(this->skin_buffers[static_cast<std::size_t>(slot)].handle());
            if (skin_detail == nullptr) {
                utility::panic("failed to get skin matrix buffer detail");
            }
            write_buffer_binding(set, 9, skin_detail->buffer, static_cast<VkDeviceSize>(vulkan::scene_skin_capacity) * sizeof(glm::mat4), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

            // binding 10: THIS slot's morph data buffer
            auto const* morph_detail = this->vulkan_core.vma.get_buffer_detail(this->morph_buffers[static_cast<std::size_t>(slot)].handle());
            if (morph_detail == nullptr) {
                utility::panic("failed to get morph data buffer detail");
            }
            write_buffer_binding(set, 10, morph_detail->buffer, static_cast<VkDeviceSize>(vulkan::scene_morph_capacity) * sizeof(float), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        }

        // binding 7 (light UBO, shared) + binding 8 (per-slot shadow map) and bindings 2-4 (IBL):
        // written on every scene set below
        this->write_light_and_shadow_bindings();
        this->write_ibl_bindings();
    }

    void runtime::update_all_scene_sets(VkWriteDescriptorSet const* writes, uint32_t const write_count) {
        if (!this->scene_set_created) {
            return;
        }
        std::vector<VkWriteDescriptorSet> per_set(writes, writes + write_count);
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            for (VkWriteDescriptorSet& write : per_set) {
                write.dstSet = *this->scene_sets[static_cast<std::size_t>(slot)];
            }
            vkUpdateDescriptorSets(this->vulkan_core.device, write_count, per_set.data(), 0, nullptr);
        }
    }

    void runtime::write_light_and_shadow_bindings() {
        if (!this->scene_set_created) {
            return;
        }
        // binding 7: light UBO (uniform buffer; light_view_proj + light_dir filled by enable_shadows)
        auto const* light_detail = this->vulkan_core.vma.get_buffer_detail(this->light_buffer.handle());
        if (light_detail == nullptr) {
            utility::panic("failed to get light ubo buffer detail");
        }
        VkDescriptorBufferInfo const light_info{light_detail->buffer, 0, sizeof(light_ubo)};

        // binding 8: THIS slot's shadow map depth texture (each slot's set points at its own
        // depth image, so no per-frame re-pointing is needed)
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            VkDescriptorSet const set = *this->scene_sets[static_cast<std::size_t>(slot)];
            auto const* shadow_detail = this->vulkan_core.vma.get_image_detail(this->shadow_images[static_cast<std::size_t>(slot)].handle());
            if (shadow_detail == nullptr) {
                utility::panic("failed to get shadow map image detail");
            }
            VkDescriptorImageInfo const shadow_info{
                .sampler = *this->shadow_sampler,
                .imageView = *this->shadow_image_views[static_cast<std::size_t>(slot)],
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            std::array<VkWriteDescriptorSet, 2> writes = {};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = set;
            writes[0].dstBinding = 7;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &light_info;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = set;
            writes[1].dstBinding = 8;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].pImageInfo = &shadow_info;
            vkUpdateDescriptorSets(this->vulkan_core.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }

    void runtime::write_ibl_bindings() const {
        if (!this->scene_set_created) {
            return;
        }
        std::array<VkDescriptorImageInfo, 3> image_infos = {};
        VkImageView const placeholder_view = *this->owned_texture_views[0]; // white
        VkSampler const placeholder_sampler = *this->texture_sampler;
        for (int i = 0; i < 3; ++i) {
            image_infos[static_cast<std::size_t>(i)] = {
                .sampler = this->ibl_ready ? *this->env_sampler : placeholder_sampler,
                .imageView = this->ibl_ready ? *this->ibl_views[static_cast<std::size_t>(i)] : placeholder_view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
        }
        // write bindings 2-4 on every scene set
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            VkDescriptorSet const set = *this->scene_sets[static_cast<std::size_t>(slot)];
            std::array<VkWriteDescriptorSet, 3> writes = {};
            for (int i = 0; i < 3; ++i) {
                writes[static_cast<std::size_t>(i)].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[static_cast<std::size_t>(i)].dstSet = set;
                writes[static_cast<std::size_t>(i)].dstBinding = static_cast<uint32_t>(2 + i);
                writes[static_cast<std::size_t>(i)].descriptorCount = 1;
                writes[static_cast<std::size_t>(i)].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[static_cast<std::size_t>(i)].pImageInfo = &image_infos[static_cast<std::size_t>(i)];
            }
            vkUpdateDescriptorSets(this->vulkan_core.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }

    void runtime::set_ibl(ibl_input const& info) {
        if (info.env_size == 0) {
            return;
        }
        auto const upload = [this](std::span<unsigned char const> const data, image_create_info const& create_info, image_type const type) -> vk_image {
            vk_image image = this->vulkan_core.vma.create_image(data.data(), data.size_bytes(), create_info, type);
            if (!image.valid()) {
                utility::panic("failed to create IBL image");
            }
            return image;
        };

        // prefiltered environment cubemap (mip chain)
        vulkan::image_create_info env_info = {};
        env_info.width = info.env_size;
        env_info.height = info.env_size;
        env_info.mip_levels = info.env_mip_count;
        env_info.array_layers = 6;
        env_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        vk_image env_image = upload(info.prefiltered_env, env_info, vulkan::image_type::texture_cubemap);
        auto const* env_detail = this->vulkan_core.vma.get_image_detail(env_image.handle());
        if (env_detail == nullptr) {
            utility::panic("failed to get environment image detail");
        }
        this->ibl_images.push_back(std::move(env_image));
        this->ibl_views.push_back(this->vulkan_core.make_image_view(env_detail->image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_CUBE));

        // irradiance cubemap
        vulkan::image_create_info irr_info = {};
        irr_info.width = info.irr_size;
        irr_info.height = info.irr_size;
        irr_info.mip_levels = 1;
        irr_info.array_layers = 6;
        irr_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        vk_image irr_image = upload(info.irradiance, irr_info, vulkan::image_type::texture_cubemap);
        auto const* irr_detail = this->vulkan_core.vma.get_image_detail(irr_image.handle());
        if (irr_detail == nullptr) {
            utility::panic("failed to get irradiance image detail");
        }
        this->ibl_images.push_back(std::move(irr_image));
        this->ibl_views.push_back(this->vulkan_core.make_image_view(irr_detail->image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_CUBE));

        // BRDF integration LUT
        vulkan::image_create_info lut_info = {};
        lut_info.width = info.lut_size;
        lut_info.height = info.lut_size;
        lut_info.mip_levels = 1;
        lut_info.array_layers = 1;
        lut_info.format = VK_FORMAT_R16G16_SFLOAT;
        vk_image lut_image = upload(info.brdf_lut, lut_info, vulkan::image_type::texture_2d);
        auto const* lut_detail = this->vulkan_core.vma.get_image_detail(lut_image.handle());
        if (lut_detail == nullptr) {
            utility::panic("failed to get BRDF LUT image detail");
        }
        this->ibl_images.push_back(std::move(lut_image));
        this->ibl_views.push_back(this->vulkan_core.make_image_view(lut_detail->image, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_VIEW_TYPE_2D));

        this->env_sampler = this->vulkan_core.make_sampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, static_cast<float>(info.env_mip_count - 1));
        this->ibl_ready = true;

        // if the scene set already exists, point bindings 2-4 at the real images
        this->write_ibl_bindings();
    }

    uint32_t runtime::register_material(primitive_create_info const& info) {
        // ---- 1. Resolve the 5 texture slots against the shared array: identical texture bytes
        //         upload once (shared glTF textures decode to one buffer, so the data pointer is
        //         a stable identity); missing slots point at the white fallback (element 0).
        std::array<std::pair<texture_input const*, VkFormat>, 5> const slots = {
            std::pair{&info.albedo, VK_FORMAT_R8G8B8A8_SRGB},
            std::pair{&info.metallic_roughness, VK_FORMAT_R8G8B8A8_UNORM},
            std::pair{&info.normal, VK_FORMAT_R8G8B8A8_UNORM},
            std::pair{&info.occlusion, VK_FORMAT_R8G8B8A8_UNORM},
            std::pair{&info.emissive, VK_FORMAT_R8G8B8A8_SRGB}, // glTF emissive textures are sRGB
        };

        std::array<uint32_t, 5> texture_indices = {};
        // one write per slot is enough when the slot is freshly added; a reused slot (cache hit)
        // was already written when it first appeared
        std::array<VkDescriptorImageInfo, 5> image_infos = {};
        std::array<VkWriteDescriptorSet, 5> writes = {};
        uint32_t write_count = 0;
        bool white_needed = false;
        VkSampler const sampler = *this->texture_sampler;
        for (int i = 0; i < 5; ++i) {
            texture_input const& tex = *slots[i].first;
            if (!tex.valid || tex.data.empty()) {
                texture_indices[i] = this->white_texture_index; // white fallback
                white_needed = true;
                continue;
            }
            auto const key = std::tuple<unsigned char const*, std::size_t, VkFormat>{tex.data.data(), tex.data.size_bytes(), slots[i].second};
            auto const cached = this->texture_slot_cache.find(key);
            if (cached != this->texture_slot_cache.end()) {
                texture_indices[i] = cached->second; // shared texture: reuse its slot
                continue;
            }
            if (this->texture_array_views.size() >= vulkan::scene_texture_capacity) {
                utility::panic("scene texture array capacity exceeded");
            }
            vulkan::image_create_info image_info = {};
            image_info.width = tex.width;
            image_info.height = tex.height;
            image_info.mip_levels = tex.mip_levels; // the caller uploads a full mip-major chain
            image_info.array_layers = 1;
            image_info.format = slots[i].second;
            vk_image tex_image = this->vulkan_core.vma.create_image(tex.data.data(), tex.data.size_bytes(), image_info, vulkan::image_type::texture_2d);
            if (!tex_image.valid()) {
                utility::panic("failed to create material texture");
            }
            auto const* detail = this->vulkan_core.vma.get_image_detail(tex_image.handle());
            if (detail == nullptr) {
                utility::panic("failed to get material texture detail");
            }
            this->owned_textures.push_back(std::move(tex_image));
            this->owned_texture_views.push_back(this->vulkan_core.make_image_view(detail->image, slots[i].second, VK_IMAGE_VIEW_TYPE_2D));
            uint32_t const index = static_cast<uint32_t>(this->texture_array_views.size());
            this->texture_array_views.push_back(*this->owned_texture_views.back());
            this->texture_slot_cache.emplace(key, index);
            texture_indices[i] = index;

            image_infos[write_count] = {.sampler = sampler, .imageView = *this->owned_texture_views.back(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            writes[write_count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[write_count].dstSet = *this->scene_sets[0]; // dstSet is replaced per set by update_all_scene_sets
            writes[write_count].dstBinding = 1;
            writes[write_count].dstArrayElement = index;
            writes[write_count].descriptorCount = 1;
            writes[write_count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[write_count].pImageInfo = &image_infos[write_count];
            ++write_count;
        }

        // material slots that fell back to white share element 0; write it once when used
        if (white_needed) {
            VkDescriptorImageInfo const white_info{
                .sampler = sampler,
                .imageView = this->texture_array_views[this->white_texture_index],
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            VkWriteDescriptorSet white_write = {};
            white_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            white_write.dstSet = *this->scene_sets[0]; // dstSet is replaced per set by update_all_scene_sets
            white_write.dstBinding = 1;
            white_write.dstArrayElement = this->white_texture_index;
            white_write.descriptorCount = 1;
            white_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            white_write.pImageInfo = &white_info;
            if (write_count < writes.size()) {
                image_infos[write_count] = white_info;
                writes[write_count] = white_write;
                ++write_count;
            } else {
                std::array<VkWriteDescriptorSet, 6> all = {};
                std::array<VkDescriptorImageInfo, 6> all_infos = {};
                for (uint32_t w = 0; w < write_count; ++w) {
                    all[w] = writes[w];
                    all_infos[w] = image_infos[w];
                }
                all_infos[write_count] = white_info;
                all[write_count] = white_write;
                this->update_all_scene_sets(all.data(), write_count + 1);
                write_count = 0; // already submitted
            }
        }
        if (write_count > 0) {
            this->update_all_scene_sets(writes.data(), write_count);
        }

        // ---- 2. Append one material record: texture indices + presence flags; factors keep
        //         their identity defaults (extend primitive_create_info to pass custom factors) ----
        if (this->material_count >= vulkan::material_capacity) {
            utility::panic("material table capacity exceeded");
        }
        material_record record = {};
        record.tex_indices = glm::uvec4(texture_indices[0], texture_indices[1], texture_indices[2], texture_indices[3]);
        record.emissive_index = texture_indices[4];
        record.base_color_factor = info.factors.base_color_factor;
        record.emissive_factor = info.factors.emissive_factor;
        record.metallic_factor = info.factors.metallic_factor;
        record.roughness_factor = info.factors.roughness_factor;
        record.normal_scale = info.factors.normal_scale;
        record.alpha_cutoff = info.factors.alpha_cutoff;
        record.occlusion_strength = info.factors.occlusion_strength;
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
        if (info.double_sided) {
            record.flags |= 8u; // bit3: back faces are rendered, fragment shader flips normals
        }
        if (info.factors.alpha_mask) {
            record.flags |= 16u; // bit4: alphaMode MASK - fragment shader discards below alpha_cutoff
        }

        uint32_t const material_index = this->material_count++;
        std::memcpy(static_cast<unsigned char*>(this->material_mapped) + static_cast<size_t>(material_index) * sizeof(material_record), &record, sizeof(record));
        return material_index;
    }

    void runtime::begin_rendering(VkCommandBuffer const command_buffer, uint32_t const image_index) const {
        std::array<VkClearValue, 2> clear_values = {};
        clear_values[0].color = {{this->clear_color.r, this->clear_color.g, this->clear_color.b, 1.0f}};
        clear_values[1].depthStencil = {1.0f, 0};

        core const& vk = this->vulkan_core;

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

    frame_status runtime::poll_events() {
        core& vk = this->vulkan_core;
        GLFWwindow* window = vk.window;

        // Window events first: respond to ESC / native close before any GPU work
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (glfwWindowShouldClose(window)) {
            return frame_status::closed;
        }

        // Minimized: skip this frame (acquiring from an invalidated / 0-sized swapchain would
        //    fail); the restore transition is handled by recreate_if_minimized()
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) == GLFW_TRUE) {
            this->was_minimized = true;
            return frame_status::skipped;
        }
        return frame_status::proceed;
    }

    void runtime::recreate_if_minimized() {
        core& vk = this->vulkan_core;
        if (this->was_minimized) {
            this->was_minimized = false;
            utility::log("window restored, recreating swapchain");
            vk.recreate_swap_chain();
            this->debug_overlay.on_swapchain_recreated();
        }
    }

    frame_status runtime::pace_and_acquire() {
        core& vk = this->vulkan_core;

        // Pace the frame slot: wait until the previous submission on this slot has completed
        //    (host-side timeline wait on the slot's last signaled value). This guards both the
        //    command buffer and the acquire semaphore — acquiring first could reuse a binary
        //    acquire semaphore with pending operations (VUID-vkAcquireNextImageKHR-semaphore-01779)
        uint32_t const frame_slot = static_cast<uint32_t>(vk.current_frame);
        vk.wait_frame_slot(frame_slot);

        // Acquire the next swapchain image; on out-of-date (e.g. the window was resized)
        //    rebuild the swapchain and let the caller retry on the next iteration.
        VkResult const acquire_result = vkAcquireNextImageKHR(vk.device,
                                                              vk.swap_chain,
                                                              UINT64_MAX,
                                                              vk.image_available_semaphores[frame_slot],
                                                              VK_NULL_HANDLE,
                                                              &this->current_image_index);
        if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
            utility::log("swapchain out of date, recreating");
            vk.recreate_swap_chain();
            this->debug_overlay.on_swapchain_recreated();
            return frame_status::skipped;
        }
        if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
            return frame_status::acquire_failed;
        }

        // Write this frame's camera UBO into the paced slot's per-slot buffer. The scene
        //    descriptor sets are static per slot (ensure_scene_set wired bindings 0/8/9/10 to
        //    each slot's own camera/shadow/skin/morph resources), so one memcpy is the whole
        //    camera update - no per-frame descriptor write exists anymore. An external
        //    (glTF/programmatic) camera, when active, supplies the matrices directly.
        this->current_aspect = static_cast<float>(vk.swap_chain_extent.width) / static_cast<float>(vk.swap_chain_extent.height);
        if (this->external_camera_active) {
            this->current_ubo.view = this->external_view;
            this->current_ubo.proj = this->external_proj;
            this->current_ubo.camera_pos = this->external_eye;
        } else {
            this->current_ubo = make_orbit_camera_ubo(this->camera.yaw, this->camera.pitch, this->camera.distance, this->camera.target, this->current_aspect);
        }
        if (this->camera_mapped[frame_slot] != nullptr) {
            std::memcpy(this->camera_mapped[frame_slot], &this->current_ubo, sizeof(camera_ubo));
        }
        // Remember the paced slot: the caller's per-frame host writes (set_skin_matrices /
        // morph_scratch) land in this slot's buffers and are safe to make now that the slot's
        // previous submission has completed.
        this->active_slot = frame_slot;
        return frame_status::proceed;
    }

    frame_status runtime::begin_recording() {
        core& vk = this->vulkan_core;
        if (this->bound_scene == nullptr) {
            utility::panic("runtime::begin_recording() called before set_scene() bound a scene");
        }
        // Record the frame into this slot's command buffer
        vk_command_buffer& command_buffer = this->command_buffers[static_cast<uint32_t>(vk.current_frame)];
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(*command_buffer, &begin_info) != VK_SUCCESS) {
            return frame_status::begin_recording_failed;
        }
        // Debug overlay: begin a fresh ImGui frame once per rendered frame (after the acquire,
        // before any UI content is built; the actual draw is recorded at the end of
        // record_main_drawcalls() while the main rendering instance is still open).
        if (this->debug_overlay.is_active()) {
            this->debug_overlay.new_frame();
        }

        // Accumulate scene-tree world transforms: every leaf's push.model = scene_transform *
        //     identity * local. With the default scene_transform (identity) this reproduces the
        //     old flat-list world matrices exactly; set_scene_transform() adds programmatic
        //     whole-scene grouping on top.
        for (scene_tree::scene_node& root : this->bound_scene->roots) {
            scene_tree::update_world(root, this->scene_transform);
        }
        // Collect the primitive leaves once (DFS over the whole scene): the shadow pass draws all
        // of them, the main pass draws the subset bound to each pipeline
        this->frame_leaves.clear();
        for (scene_tree::scene_node const& root : this->bound_scene->roots) {
            collect_leaf_primitives(root, this->frame_leaves);
        }

        // Frustum culling for the main pass: build a BVH over every leaf that has a single
        //     world AABB (normal draw primitives, whose bounds follow push.model), then keep only
        //     the leaves inside the camera frustum. Instanced primitives spread over many
        //     transforms (no single AABB) and primitives without bounds are never culled. The
        //     shadow pass below still draws the full frame_leaves set so no caster is lost.
        //
        //     Two-level reuse: the BVH is rebuilt only when the scene changed (bvh_dirty), and
        //     when the camera also did not move the culled result is reused as-is (no rebuild, no
        //     frustum_cull). update_world() above rewrites the same world matrices each frame, so
        //     a non-dirty scene keeps identical world AABBs and the cached BVH stays valid.
        // local aliases into the per-frame state filled above (keeps the cull math unchanged)
        std::pmr::vector<primitive const*> const& frame_leaves = this->frame_leaves;
        float const& aspect = this->current_aspect;
        camera_ubo const& ubo = this->current_ubo;
        std::pmr::vector<primitive const*> visible_leaves = this->frame_leaves; // fallback: no culling
        if (this->frustum_culling) {
            // camera identity: the orbit state that shapes the frustum, or the authored
            // external camera's view/projection when one is active
            std::array<float, 7> key = {}; // orbit key; unused while an external camera is active
            bool const scene_changed_this_frame = this->bvh_dirty;
            if (this->external_camera_active) {
                this->camera_moved = this->external_camera_changed;
                this->external_camera_changed = false;
            } else {
                key = {
                    this->camera.yaw,
                    this->camera.pitch,
                    this->camera.distance,
                    this->camera.target.x,
                    this->camera.target.y,
                    this->camera.target.z,
                    aspect,
                };
                this->camera_moved = key != this->camera_key;
            }

            if (scene_changed_this_frame || !this->cull_bvh.has_value()) {
                // scene changed: rebuild the BVH from current world AABBs (and drop stale leaves)
                std::vector<utility::aabb_box<primitive>> boxes;
                boxes.reserve(frame_leaves.size());
                for (primitive const* leaf : frame_leaves) {
                    if (leaf->has_bounds) {
                        auto const [wmin, wmax] = leaf->world_aabb();
                        boxes.push_back(utility::aabb_box<primitive>{.min = wmin, .max = wmax, .extra_data = const_cast<primitive*>(leaf)});
                    }
                }
                if (!boxes.empty()) {
                    this->cull_bvh.reset(); // destroy the old tree first (its leaves reference this scene)
                    auto make_result = utility::bvh<primitive>::make(boxes);
                    if (make_result) {
                        this->cull_bvh = std::move(make_result).value();
                    }
                } else {
                    this->cull_bvh = std::nullopt;
                }
                this->bvh_dirty = false;
            }

            if (this->camera_moved || scene_changed_this_frame || this->cull_visible.empty()) {
                // camera moved or the scene changed: re-run frustum cull against the current BVH
                std::pmr::vector<primitive const*> visible;
                visible.reserve(frame_leaves.size());
                // leaves without bounds (instanced etc.) are always drawn
                for (primitive const* leaf : frame_leaves) {
                    if (!leaf->has_bounds) {
                        visible.push_back(leaf);
                    }
                }
                if (this->cull_bvh.has_value()) {
                    utility::frustum const view_frustum = utility::make_frustum(ubo.proj * ubo.view);
                    auto const inside = this->cull_bvh->frustum_cull(view_frustum);
                    for (auto const* node : inside) {
                        visible.push_back(node->extra_data);
                    }
                }
                this->cull_visible = std::move(visible);
                this->camera_key = key;
            }
            visible_leaves = this->cull_visible;
        }

        // persist the cull result for the record steps below (shadow pass draws the full
        // frame_leaves set, the main pass draws this visible subset)
        this->frame_visible = std::move(visible_leaves);
        return frame_status::proceed;
    }

    void runtime::record_main_drawcalls() {
        core& vk = this->vulkan_core;
        vk_command_buffer& command_buffer = this->command_buffers[static_cast<uint32_t>(vk.current_frame)];
        uint32_t const frame_slot = static_cast<uint32_t>(vk.current_frame);

        // ---- Shadow pass: render the scene's depth from the light into this slot's shadow map.
        //      Drawn before the main pass; the depth-only pipeline shares the flat scene layout
        //      and the primitive draw() path (same vertex buffers / push constants), so the shadow
        //      pass is just "bind the shadow pipeline, then draw the same models".
        //      Dynamic rendering only: depth-only rendering needs no color attachment, which the
        //      classic render-pass fallback cannot express (make_shadow_pipeline already failed
        //      there, so this block is skipped together with shadows_enabled).
        if (vk.use_dynamic_rendering && this->shadow_pipeline && this->shadows_enabled && this->shadow_enabled) {
            auto const* shadow_detail = vk.vma.get_image_detail(this->shadow_images[frame_slot].handle());
            if (shadow_detail != nullptr) {
                // Transition the shadow image to a renderable depth attachment (loadOp CLEAR
                //     discards the previous frame's contents, so UNDEFINED as oldLayout is valid)
                VkImageMemoryBarrier2 shadow_barrier = {};
                shadow_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                shadow_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                shadow_barrier.srcAccessMask = 0;
                shadow_barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                shadow_barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                shadow_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                shadow_barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                shadow_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                shadow_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                shadow_barrier.image = shadow_detail->image;
                shadow_barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
                VkDependencyInfo shadow_dependency = {};
                shadow_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                shadow_dependency.imageMemoryBarrierCount = 1;
                shadow_dependency.pImageMemoryBarriers = &shadow_barrier;
                vkCmdPipelineBarrier2(*command_buffer, &shadow_dependency);

                // Depth-only rendering into the shadow map (no color attachment)
                VkClearValue shadow_clear = {};
                shadow_clear.depthStencil = {1.0f, 0};
                VkRenderingAttachmentInfo shadow_depth_attachment = {};
                shadow_depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                shadow_depth_attachment.imageView = *this->shadow_image_views[frame_slot];
                shadow_depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                shadow_depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                shadow_depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                shadow_depth_attachment.clearValue = shadow_clear;

                VkRenderingInfo shadow_rendering_info = {};
                shadow_rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                shadow_rendering_info.renderArea = {{0, 0}, {vulkan::runtime::shadow_map_size, vulkan::runtime::shadow_map_size}};
                shadow_rendering_info.layerCount = 1;
                shadow_rendering_info.colorAttachmentCount = 0;
                shadow_rendering_info.pDepthAttachment = &shadow_depth_attachment;
                vkCmdBeginRendering(*command_buffer, &shadow_rendering_info);

                // Draw the depth-only content into the shadow rendering instance: bind the
                // shared scene set + shadow pipeline, apply the live depth bias, draw every
                // scene-tree leaf (the whole scene casts shadows).
                this->record_shadow_content(*command_buffer);
                vkCmdEndRendering(*command_buffer);

                // Hand the shadow map back to the main pass as a sampled texture
                VkImageMemoryBarrier2 shadow_read_barrier = {};
                shadow_read_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                shadow_read_barrier.srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                shadow_read_barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                shadow_read_barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                shadow_read_barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                shadow_read_barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                shadow_read_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                shadow_read_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                shadow_read_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                shadow_read_barrier.image = shadow_detail->image;
                shadow_read_barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
                VkDependencyInfo shadow_read_dependency = {};
                shadow_read_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                shadow_read_dependency.imageMemoryBarrierCount = 1;
                shadow_read_dependency.pImageMemoryBarriers = &shadow_read_barrier;
                vkCmdPipelineBarrier2(*command_buffer, &shadow_read_dependency);
            }
        }

        // Dynamic rendering has no automatic attachment transitions (unlike a render pass):
        // move every attachment into its render layout before vkCmdBeginRendering
        if (vk.use_dynamic_rendering) {
            std::array<VkImageMemoryBarrier2, 3> attachment_barriers = {};
            uint32_t barrier_count = 0;
            auto const add_render_barrier = [&attachment_barriers, &barrier_count](VkImage const image, VkImageAspectFlags const aspect, VkImageLayout const new_layout, VkPipelineStageFlags2 const dst_stage, VkAccessFlags2 const dst_access) {
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
                add_render_barrier(vk.color_images[this->current_image_index], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                add_render_barrier(vk.swap_chain_images[this->current_image_index], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            } else {
                add_render_barrier(vk.swap_chain_images[this->current_image_index], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            }
            add_render_barrier(vk.depth_images[this->current_image_index], VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                               VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                               VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            VkDependencyInfo dependency_info = {};
            dependency_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependency_info.imageMemoryBarrierCount = barrier_count;
            dependency_info.pImageMemoryBarriers = attachment_barriers.data();
            vkCmdPipelineBarrier2(*command_buffer, &dependency_info);
        }

        this->begin_rendering(*command_buffer, this->current_image_index);

        // Pipelines cache a fullscreen viewport/scissor at creation; after a resize the swapchain
        // extent changed, so resync them from the current extent before drawing (begin_pipeline
        // applies the stored values). Done here on the primary thread (it mutates the cached
        // pipeline state), before the scene content below is recorded - inline or in secondaries.
        VkViewport const full_viewport = {
            0.0f,
            0.0f,
            static_cast<float>(vk.swap_chain_extent.width),
            static_cast<float>(vk.swap_chain_extent.height),
            0.0f,
            1.0f,
        };
        VkRect2D const full_scissor = {{0, 0}, vk.swap_chain_extent};
        for (auto& pipeline : this->pipelines | std::views::values) {
            pipeline.viewport = full_viewport;
            pipeline.scissor = full_scissor;
        }
        if (this->skybox_pipeline && this->skybox_enabled) {
            this->skybox_pipeline->viewport = full_viewport;
            this->skybox_pipeline->scissor = full_scissor;
        }

        // Scene content: bind the scene set, draw the skybox background (when enabled) then
        // every pipeline's visible leaves (see record_main_content).
        this->record_main_content(*command_buffer);

        // Debug overlay: draw the ImGui frame into the STILL OPEN main rendering instance (the
        // same MSAA color attachment the scene just rendered into, resolved together at
        // end_recording()). record() runs the registered UI builder and emits the draw data.
        if (this->debug_overlay.is_active()) {
            this->debug_overlay.record(*command_buffer);
        }
    }

    // Depth-only shadow-pass content: bind the shared scene set (the light UBO binding 7) +
    // the shadow pipeline, apply the live depth bias and draw every scene-tree leaf (the whole
    // scene casts shadows). Pure bind/push/draw commands - the caller owns the barriers and
    // the depth-only rendering instance around it. Recorded inline today; stage 2 records the
    // same content into a per-slot secondary command buffer for parallel pass recording.
    void runtime::record_shadow_content(VkCommandBuffer const command_buffer) const {
        core const& vk = this->vulkan_core;
        uint32_t const frame_slot = static_cast<uint32_t>(vk.current_frame);
        if (this->scene_set_created) {
            VkDescriptorSet const scene_set_handle = *this->scene_sets[static_cast<std::size_t>(frame_slot)];
            vkCmdBindDescriptorSets(command_buffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    vk.scene_pipeline_layout,
                                    0,
                                    1,
                                    &scene_set_handle,
                                    0,
                                    nullptr);
        }
        this->shadow_pipeline->begin_pipeline(command_buffer);
        // depth bias is dynamic state on the shadow pipeline: record the live-tunable values
        // (gui-adjustable) before the depth-only draw
        vkCmdSetDepthBias(command_buffer, this->shadow_depth_bias_constant, this->shadow_depth_bias_clamp, this->shadow_depth_bias_slope);
        for (primitive const* m : this->frame_leaves) {
            m->draw(command_buffer); // depth-only: shadow.vert transforms into light space
        }
    }

    // Main-pass scene content: bind this frame slot's scene set, draw the skybox background
    // (when enabled) then every pipeline's visible leaves. Pure bind/push/draw commands - the
    // caller owns the barriers + the color/depth rendering instance around it (and the
    // viewport/scissor resync, which updates the cached pipeline state on the CPU). Recorded
    // inline today; stage 2 records the same content into a per-slot secondary command buffer
    // for parallel pass recording. The debug overlay stays on the primary (it has its own
    // recording path), so this content is the scene only.
    void runtime::record_main_content(VkCommandBuffer const command_buffer) const {
        core const& vk = this->vulkan_core;
        // Bind this frame slot's scene descriptor set once: every pipeline shares the scene
        // layout, so the set stays valid across pipeline binds and only models vary per draw.
        // Each slot's set always points at that slot's own camera/shadow/skin/morph resources.
        if (this->scene_set_created) {
            VkDescriptorSet const scene_set_handle = *this->scene_sets[static_cast<std::size_t>(vk.current_frame)];
            vkCmdBindDescriptorSets(command_buffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    vk.scene_pipeline_layout,
                                    0,
                                    1,
                                    &scene_set_handle,
                                    0,
                                    nullptr);
        }

        // Background pass first: the skybox draws a fullscreen triangle (no vertex/index buffers)
        // with depth test/write disabled, then the models render over it. Skipped when the skybox
        // stage is toggled off — the frame then just shows the clear color behind the models.
        if (this->skybox_pipeline && this->skybox_enabled) {
            this->skybox_pipeline->begin_pipeline(command_buffer);
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
        }

        // Main pass: draw the frustum-visible leaves, grouping by their pipeline (each group
        // binds its pipeline once — same batching as the old flat primitive list)
        for (auto const& [pipeline_name, pipeline] : this->pipelines) {
            vk_pipeline const* const wanted = &pipeline;
            bool any = false;
            for (primitive const* m : this->frame_visible) {
                if (m->pipeline == wanted) {
                    if (!any) {
                        pipeline.begin_pipeline(command_buffer);
                        any = true;
                    }
                    m->draw(command_buffer); // polymorphic: normal / instanced / ...
                }
            }
        }
    }

    frame_status runtime::end_recording() {
        core& vk = this->vulkan_core;
        vk_command_buffer& command_buffer = this->command_buffers[static_cast<uint32_t>(vk.current_frame)];

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
            present_barrier.image = vk.swap_chain_images[this->current_image_index];
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
            return frame_status::end_recording_failed;
        }
        return frame_status::proceed;
    }

    frame_status runtime::submit_and_present() {
        core& vk = this->vulkan_core;
        vk_command_buffer& command_buffer = this->command_buffers[static_cast<uint32_t>(vk.current_frame)];

        // Submit + present; recreate the swapchain when presentation reports out of date
        if (vk.submit(*command_buffer, this->current_image_index) != VK_SUCCESS) {
            return frame_status::submit_failed;
        }
        VkResult const present_result = vk.present(this->current_image_index);
        if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
            utility::log("present out of date, recreating swapchain");
            vk.recreate_swap_chain();
            this->debug_overlay.on_swapchain_recreated();
        } else if (present_result != VK_SUCCESS) {
            return frame_status::present_failed;
        }
        vk.to_next_frame();
        return frame_status::proceed;
    }

    VkCommandBuffer runtime::active_command_buffer() const noexcept {
        return *this->command_buffers[static_cast<uint32_t>(this->vulkan_core.current_frame)];
    }

    frame_status runtime::render_frame() {
        // Whole frame in one call: run the frame phases directly, in order, with no caller
        // writes interleaved (callers that need per-frame host updates run the phases at fine
        // granularity themselves, writing between pace_and_acquire() and begin_recording()).
        frame_status const skip = this->poll_events();
        if (skip != frame_status::proceed) {
            return skip;
        }
        this->recreate_if_minimized();
        frame_status const paced = this->pace_and_acquire();
        if (paced != frame_status::proceed) {
            return paced;
        }
        frame_status const begin = this->begin_recording();
        if (begin != frame_status::proceed) {
            return begin;
        }
        this->record_main_drawcalls();
        frame_status const end = this->end_recording();
        if (end != frame_status::proceed) {
            return end;
        }
        return this->submit_and_present();
    }

    bool runtime::enable_debug_gui() {
        if (this->debug_overlay.is_active()) {
            return true;
        }
        // The overlay draws into the runtime's OPEN main rendering instance via dynamic
        // rendering (the backend is initialized with UseDynamicRendering=true), so it cannot
        // be enabled on the classic render-pass fallback path.
        if (!this->vulkan_core.use_dynamic_rendering) {
            utility::log("enable_debug_gui: requires dynamic rendering (classic fallback active)");
            return false;
        }
        vulkan::core const& vk = this->vulkan_core;
        gui::gui_create_info info = {};
        info.window = vk.window;
        info.instance = vk.instance;
        info.physical_device = vk.physical_device;
        info.device = vk.device;
        info.graphics_queue_family = vk.graphics_family_index;
        info.graphics_queue = vk.graphics_queue;
        info.color_format = vk.swap_chain_image_format;
        info.depth_format = vk.depth_format;
        info.msaa_samples = vk.msaa_samples;
        info.frames_in_flight = static_cast<uint32_t>(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        return this->debug_overlay.init(info);
    }

    bool runtime::debug_gui_active() const noexcept {
        return this->debug_overlay.is_active();
    }

    gui::gui_content& runtime::debug_gui() noexcept {
        return this->debug_overlay;
    }

    std::expected<void, std::string> runtime::make_pipeline(std::string_view pipeline_name, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        auto make_result = this->vulkan_core.make_pipeline(vertex_shader_code, fragment_shader_code);
        if (!make_result) {
            return fail(make_result.error());
        }
        this->pipelines.emplace(pipeline_name, std::move(make_result).value());
        return {};
    }

    std::expected<void, std::string> runtime::make_shadow_pipeline(std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        // depth-only pipeline (no color attachment, single sample); requires dynamic rendering.
        // Slope-scaled rasterization depth bias pushes the stored depths away from the light
        // proportionally to the surface's depth slope, which removes shadow acne on angled
        // surfaces (scale-free: units of depth per depth-unit of slope).
        auto make_result = this->vulkan_core.make_depth_pipeline(vertex_shader_code, fragment_shader_code, this->vulkan_core.depth_format, 0.0f, 1.5f, 0.0f);
        if (!make_result) {
            return fail(std::string(make_result.error()));
        }
        this->shadow_pipeline = std::move(make_result).value();
        // The shadow map is a fixed-size depth target: its viewport/scissor do not follow the
        // swapchain size (the frame path only re-syncs pipelines stored in the pipelines map)
        this->shadow_pipeline->viewport = {
            0.0f,
            0.0f,
            static_cast<float>(vulkan::runtime::shadow_map_size),
            static_cast<float>(vulkan::runtime::shadow_map_size),
            0.0f,
            1.0f,
        };
        this->shadow_pipeline->scissor = {{0, 0}, {vulkan::runtime::shadow_map_size, vulkan::runtime::shadow_map_size}};
        return {};
    }

    void runtime::enable_shadows(glm::vec3 const& scene_center, float const scene_radius) {
        if (!this->shadow_pipeline || this->light_mapped == nullptr) {
            utility::log("shadow mapping not enabled (no shadow pipeline / light buffer)");
            return;
        }
        // light UBO: orthographic light view-proj framing the scene + the light direction
        light_ubo const ubo = make_directional_light_ubo(scene_center, scene_radius);
        std::memcpy(this->light_mapped, &ubo, sizeof(ubo));
        this->shadows_enabled = true;
        utility::log("shadow mapping enabled: light frustum center ({:.2f}, {:.2f}, {:.2f}), radius {:.2f}",
                     scene_center.x, scene_center.y, scene_center.z, scene_radius);
    }

    void runtime::set_shadow_enabled(bool const enabled) {
        if (this->shadow_enabled == enabled) {
            return;
        }
        this->shadow_enabled = enabled;
        // The light UBO's shadow_enabled flag drives pbr.frag: when shadows are off the shader
        // skips calc_shadow entirely (fully lit), so no shadow-map clearing is needed — this
        // avoids the per-frame-slot double-buffer race that clearing once could not fix.
        if (this->light_mapped != nullptr) {
            float const flag = enabled ? 1.0f : 0.0f;
            std::memcpy(static_cast<unsigned char*>(this->light_mapped) + offsetof(light_ubo, shadow_enabled), &flag, sizeof(flag));
        }
        utility::log("shadow pass {}", enabled ? "enabled" : "disabled");
    }

    void runtime::set_scene_transform(glm::mat4 const& transform) {
        this->scene_transform = transform;
        this->bvh_dirty = true; // whole-scene transform changes every leaf's world AABB
    }

    void runtime::log_scene_tree() const noexcept {
        size_t total_nodes = 0;
        size_t leaf_count = 0;
        size_t max_depth = 0;
        std::vector<std::string> lines;
        auto const walk = [&](auto&& self, scene_tree::scene_node const& node, size_t const depth) -> void {
            ++total_nodes;
            max_depth = std::max(max_depth, depth);
            if (node.primitive_leaf != nullptr) {
                ++leaf_count;
            }
            std::string marker = node.primitive_leaf != nullptr ? " [primitive]" : "";
            lines.push_back(std::format("{}{}{}", std::string(depth * 2, ' '),
                                        node.name.empty() ? std::string("<unnamed>") : node.name, marker));
            for (scene_tree::scene_node const& child : node.children) {
                self(self, child, depth + 1);
            }
        };
        for (scene_tree::scene_node const& root : this->get_scene().roots) {
            walk(walk, root, 0);
        }
        utility::log("runtime scene tree: {} roots, {} nodes ({} leaf primitives), max depth {}", this->get_scene().roots.size(), total_nodes, leaf_count, max_depth);
        for (std::string const& line : lines) {
            utility::log("  {}", line);
        }
    }

    std::expected<void, std::string> runtime::make_skybox_pipeline(std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        // no depth test / write: the skybox is a background pass drawn before the models
        auto make_result = this->vulkan_core.make_pipeline(vertex_shader_code, fragment_shader_code, false);
        if (!make_result) {
            return fail(make_result.error());
        }
        this->skybox_pipeline = std::move(make_result).value();
        return {};
    }
    vk_pipeline const* runtime::get_pipeline(std::string_view const pipeline_name) const noexcept {
        auto const it = this->pipelines.find(pipeline_name);
        return it == this->pipelines.end() ? nullptr : &it->second;
    }

    std::unique_ptr<primitive> runtime::create_primitive(std::string_view const pipeline_name, primitive_create_info const& info) {
        vk_pipeline const* pipeline = this->get_pipeline(pipeline_name);
        if (pipeline == nullptr) {
            return nullptr;
        }
        this->ensure_scene_set();

        auto result = std::make_unique<normal_draw_primitive>();
        result->pipeline = pipeline;

        // ---- geometry buffers ----
        result->vertex_buffer = this->vulkan_core.vma.create_buffer(info.vertex_data.data(), info.vertex_data.size_bytes(), vulkan::buffer_type::vertex);
        if (!result->vertex_buffer.valid()) {
            utility::panic("failed to create vertex buffer");
        }
        result->vertex_detail = this->vulkan_core.vma.get_buffer_detail(result->vertex_buffer.handle());
        if (result->vertex_detail == nullptr) {
            utility::panic("failed to get vertex buffer detail");
        }

        result->index_buffer = this->vulkan_core.vma.create_buffer(info.index_data.data(), info.index_data.size_bytes(), vulkan::buffer_type::index);
        if (!result->index_buffer.valid()) {
            utility::panic("failed to create index buffer");
        }
        result->index_detail = this->vulkan_core.vma.get_buffer_detail(result->index_buffer.handle());
        if (result->index_detail == nullptr) {
            utility::panic("failed to get index buffer detail");
        }

        result->index_type = info.index_type;
        result->index_count = info.index_count;
        result->vertex_count = info.vertex_count;

        // ---- local-space AABB for frustum culling: the interleaved vertex layout starts every
        //      vertex with a vec3 position (see the loader's vertex struct / pbr.vert), so scan
        //      the CPU copy before it is released by the upload
        if (info.vertex_count > 0 && info.vertex_stride >= sizeof(glm::vec3) && !info.vertex_data.empty()) {
            glm::vec3 aabb_min = glm::vec3(std::numeric_limits<float>::infinity());
            glm::vec3 aabb_max = glm::vec3(-std::numeric_limits<float>::infinity());
            auto const* cursor = info.vertex_data.data();
            for (uint32_t v = 0; v < info.vertex_count; ++v) {
                glm::vec3 position;
                std::memcpy(&position, cursor, sizeof(position));
                aabb_min = glm::min(aabb_min, position);
                aabb_max = glm::max(aabb_max, position);
                cursor += info.vertex_stride;
            }
            result->local_aabb_min = aabb_min;
            result->local_aabb_max = aabb_max;
            result->has_bounds = true;
        }

        // ---- material: register textures + append a material record; the primitive only carries
        //         the material index (texture indices / factors / flags live in the GPU table) ----
        result->push.material_index = this->register_material(info);
        result->push.model = info.model_matrix;
        result->double_sided = info.double_sided;
        return result;
    }

    primitive* runtime::make_primitive(std::string_view const pipeline_name, primitive_create_info const& info) {
        std::unique_ptr<primitive> created = this->create_primitive(pipeline_name, info);
        if (created == nullptr) {
            return nullptr;
        }
        primitive* const result = created.get();

        // attach the primitive as a new root leaf of the scene tree; the node's name records the
        // pipeline it draws with (the record path groups leaves by node name / pipeline)
        scene_tree::scene_node& leaf = this->get_scene().add_root();
        leaf.name = std::string(pipeline_name);
        leaf.local = info.model_matrix;  // world = identity * local (root)
        leaf.attach(std::move(created)); // a vulkan::primitive is a scene_tree::primitive
        this->bvh_dirty = true;          // new leaf -> culling BVH must be rebuilt
        return result;
    }

    primitive* runtime::make_instanced_primitive(primitive const& source, std::span<glm::mat4 const> const transforms) {
        uint32_t const count = std::min<uint32_t>(static_cast<uint32_t>(transforms.size()), vulkan::instance_capacity - this->instance_cursor);
        if (count == 0 || this->instance_mapped == nullptr || !source.is_valid()) {
            return nullptr;
        }
        vk_pipeline const* pipeline = this->get_pipeline("pbr");
        if (pipeline == nullptr) {
            return nullptr;
        }
        this->ensure_scene_set();

        // The instance buffer is one shared region; THIS primitive gets the slice starting at
        // instance_cursor (mat4 units). Writing only its own slice keeps several instanced
        // primitives from overwriting each other - each one addresses its transforms through
        // push.instance_base in the vertex shaders.
        uint32_t const base = this->instance_cursor;
        this->instance_cursor += count;
        std::memcpy(static_cast<unsigned char*>(this->instance_mapped) + static_cast<size_t>(base) * sizeof(glm::mat4),
                    transforms.data(),
                    static_cast<size_t>(count) * sizeof(glm::mat4));

        auto result = std::make_unique<instanced_draw_primitive>();
        result->pipeline = pipeline;
        result->source = &source; // geometry owner; must stay in this runtime's scene tree
        result->instance_count = count;
        result->push.material_index = source.push.material_index;
        result->push.flags = 1u; // bit0: pbr.vert picks instances[instance_base + gl_InstanceIndex]
        result->push.instance_base = base;
        result->push.model = glm::mat4(1.0f);
        result->double_sided = source.double_sided;

        scene_tree::scene_node& leaf = this->get_scene().add_root();
        leaf.name = "pbr";
        primitive* const created = static_cast<primitive*>(leaf.attach(std::move(result)));
        this->bvh_dirty = true; // new leaf -> culling BVH must be rebuilt
        return created;
    }

    primitive* runtime::make_static_draw(static_draw_create_info const& info) {
        if (info.vertex_count == 0 || info.index_data.empty() || info.chunks.empty()) {
            return nullptr; // nothing to draw: no vertices, no indices, or no chunks
        }
        vk_pipeline const* pipeline = this->get_pipeline("pbr");
        if (pipeline == nullptr) {
            return nullptr;
        }
        this->ensure_scene_set();

        auto result = std::make_unique<static_draw_primitive>();
        result->pipeline = pipeline;

        // ---- merged geometry buffers (owned by this primitive) ----
        result->vertex_buffer = this->vulkan_core.vma.create_buffer(info.vertex_data.data(), info.vertex_data.size_bytes(), vulkan::buffer_type::vertex);
        if (!result->vertex_buffer.valid()) {
            utility::panic("failed to create static vertex buffer");
        }
        result->vertex_detail = this->vulkan_core.vma.get_buffer_detail(result->vertex_buffer.handle());
        if (result->vertex_detail == nullptr) {
            utility::panic("failed to get static vertex buffer detail");
        }
        result->index_buffer = this->vulkan_core.vma.create_buffer(info.index_data.data(), info.index_data.size_bytes(), vulkan::buffer_type::index);
        if (!result->index_buffer.valid()) {
            utility::panic("failed to create static index buffer");
        }
        result->index_detail = this->vulkan_core.vma.get_buffer_detail(result->index_buffer.handle());
        if (result->index_detail == nullptr) {
            utility::panic("failed to get static index buffer detail");
        }
        result->index_type = info.index_type;
        result->index_count = info.index_count;
        result->vertex_count = info.vertex_count;

        // ---- local AABB over the whole merged geometry (batch-level culling) ----
        if (info.vertex_count > 0 && info.vertex_stride >= sizeof(glm::vec3) && !info.vertex_data.empty()) {
            glm::vec3 aabb_min = glm::vec3(std::numeric_limits<float>::infinity());
            glm::vec3 aabb_max = glm::vec3(-std::numeric_limits<float>::infinity());
            auto const* cursor = info.vertex_data.data();
            for (uint32_t v = 0; v < info.vertex_count; ++v) {
                glm::vec3 position;
                std::memcpy(&position, cursor, sizeof(position));
                aabb_min = glm::min(aabb_min, position);
                aabb_max = glm::max(aabb_max, position);
                cursor += info.vertex_stride;
            }
            result->local_aabb_min = aabb_min;
            result->local_aabb_max = aabb_max;
            result->has_bounds = true;
        }

        // ---- chunk table: register each chunk's material, record its index range ----
        result->chunks.reserve(info.chunks.size());
        for (static_draw_chunk const& chunk : info.chunks) {
            if (chunk.index_count == 0) {
                continue; // empty chunk: skip (keeps is_valid simple)
            }
            // register this chunk's material (register_material only reads the material
            // fields of primitive_create_info, so a material-only info is enough)
            primitive_create_info material_info = {};
            material_info.albedo = chunk.albedo;
            material_info.metallic_roughness = chunk.metallic_roughness;
            material_info.normal = chunk.normal;
            material_info.occlusion = chunk.occlusion;
            material_info.emissive = chunk.emissive;
            material_info.factors = chunk.factors;
            material_info.double_sided = chunk.double_sided;
            static_draw_primitive::chunk_record record = {};
            record.first_index = chunk.first_index;
            record.index_count = chunk.index_count;
            record.vertex_offset = chunk.vertex_offset;
            record.material_index = this->register_material(material_info);
            record.double_sided = chunk.double_sided;
            result->chunks.push_back(record);
        }
        if (result->chunks.empty()) {
            return nullptr; // every chunk was empty: nothing drawable
        }
        result->push.model = info.model_matrix;

        scene_tree::scene_node& leaf = this->get_scene().add_root();
        leaf.name = "pbr";
        primitive* const created = static_cast<primitive*>(leaf.attach(std::move(result)));
        this->bvh_dirty = true; // new leaf -> culling BVH must be rebuilt
        return created;
    }

    std::vector<primitive const*> runtime::get_primitives(std::string_view const pipeline_name) const noexcept {
        vk_pipeline const* const wanted = this->get_pipeline(pipeline_name);
        if (wanted == nullptr) {
            return {};
        }
        std::vector<primitive const*> result;
        for (scene_tree::scene_node const& root : this->get_scene().roots) {
            // collect every leaf whose primitive binds the requested pipeline (models record their
            // pipeline in primitive->pipeline; the scene tree just organizes them)
            scene_tree::visit_primitives(root, glm::mat4(1.0f), [&](scene_tree::scene_node const& n, glm::mat4 const&) {
                auto const* m = static_cast<primitive const*>(n.primitive_leaf.get());
                if (m->pipeline == wanted) {
                    result.push_back(m);
                }
            });
        }
        return result;
    }

    void runtime::clear_primitives(std::string_view const pipeline_name) {
        vk_pipeline const* const unwanted = this->get_pipeline(pipeline_name);
        if (unwanted == nullptr) {
            return;
        }
        // DFS remove: erase every leaf primitive bound to the pipeline, wherever it sits in the tree
        // (imported scenes nest leaves under hierarchy nodes; make_primitive attaches them at roots).
        // A node whose leaf matches is stripped of that leaf; it (or an ancestor) is dropped only
        // when nothing remains below it, so models of other pipelines in the subtree survive.
        auto& roots = this->get_scene().roots;
        auto const matches = [unwanted](scene_tree::scene_node const& node) {
            return node.primitive_leaf != nullptr && static_cast<primitive const*>(node.primitive_leaf.get())->pipeline == unwanted;
        };
        // prune(node) -> true when the node is now empty (no leaf, no children) and should be dropped
        auto const prune = [&](auto&& self, scene_tree::scene_node& node) -> bool {
            for (auto it = node.children.begin(); it != node.children.end();) {
                if (self(self, *it)) {
                    it = node.children.erase(it);
                } else {
                    ++it;
                }
            }
            if (matches(node)) {
                static_cast<primitive*>(node.primitive_leaf.get())->destroy(this->vulkan_core.vma);
                node.primitive_leaf.reset();
            }
            return node.primitive_leaf == nullptr && node.children.empty();
        };
        for (auto it = roots.begin(); it != roots.end();) {
            if (prune(prune, *it)) {
                it = roots.erase(it);
            } else {
                ++it;
            }
        }
        // Instanced primitives are the only writers of the shared instance transform buffer;
        // when none survive the removal their slices are dead, so the cursor can recycle the
        // whole buffer (next make_instanced_primitive starts at 0 again). push.flags bit0 is the
        // instanced marker - only instanced_draw_primitive ever sets it.
        bool any_instanced_left = false;
        for (scene_tree::scene_node const& root : roots) {
            scene_tree::visit_primitives(root, glm::mat4(1.0f), [&](scene_tree::scene_node const& n, glm::mat4 const&) {
                auto const* const leaf = static_cast<primitive const*>(n.primitive_leaf.get());
                any_instanced_left = any_instanced_left || (leaf->push.flags & 1u) != 0u;
            });
        }
        if (!any_instanced_left) {
            this->instance_cursor = 0;
        }
        this->bvh_dirty = true; // leaves removed -> culling BVH must be rebuilt
    }

    void runtime::collect_leaf_primitives(scene_tree::scene_node const& node, std::pmr::vector<primitive const*>& out) {
        scene_tree::visit_primitives(node, glm::mat4(1.0f), [&out](scene_tree::scene_node const& n, glm::mat4 const&) {
            out.push_back(static_cast<primitive const*>(n.primitive_leaf.get()));
        });
    }

    void runtime::set_skin_matrices(std::span<glm::mat4 const> const matrices) {
        this->set_skin_matrices(matrices, this->active_slot);
    }

    void runtime::set_skin_matrices(std::span<glm::mat4 const> const matrices, uint32_t const slot) {
        if (slot >= this->skin_mapped.size() || this->skin_mapped[slot] == nullptr) {
            return;
        }
        std::size_t const bytes = std::min(matrices.size_bytes(), static_cast<std::size_t>(vulkan::scene_skin_capacity) * sizeof(glm::mat4));
        std::memcpy(this->skin_mapped[slot], matrices.data(), bytes);
    }

    void* runtime::morph_scratch() noexcept {
        return this->morph_scratch(this->active_slot);
    }

    void* runtime::morph_scratch(uint32_t const slot) noexcept {
        if (slot >= this->morph_mapped.size()) {
            return nullptr;
        }
        return this->morph_mapped[slot];
    }

    void runtime::set_external_camera(glm::vec3 const& eye, glm::mat4 const& view, glm::mat4 const& proj) noexcept {
        if (!this->external_camera_active || eye != this->external_eye || view != this->external_view || proj != this->external_proj) {
            this->external_eye = eye;
            this->external_view = view;
            this->external_proj = proj;
            this->external_camera_changed = true;
        }
        this->external_camera_active = true;
    }

    void runtime::clear_external_camera() noexcept {
        if (this->external_camera_active) {
            this->external_camera_active = false;
            this->external_camera_changed = true; // orbit view is different: re-cull next frame
        }
    }

    bool runtime::using_external_camera() const noexcept {
        return this->external_camera_active;
    }

    float runtime::aspect_ratio() const noexcept {
        return this->current_aspect;
    }
} // namespace vulkan
