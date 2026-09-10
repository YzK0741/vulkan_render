module;

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>

module vulkan.runtime;

import utility;
import vulkan.constant_init;
import vulkan.core.pipeline; // vulkan::make_pipeline for the post-process pipeline

// Route std::pmr allocations through mimalloc for this TU (utility.better_pmr). Idempotent:
// init_pmr() returns the same process-wide singleton no matter which TU calls it first, so
// main.cpp's keep-alive and this one coexist safely. The reference itself is never read; it
// only forces the (dynamic) initialization before any pmr container in this TU is constructed.
[[maybe_unused]] static auto& pmr = utility::init_pmr(); // NOLINT(keep-alive)

namespace {
    // A color format whose attachment write path encodes linear -> sRGB in hardware. Writing an
    // already gamma-encoded value into one of these double-encodes gamma (and leaves a UNORM target
    // under-encoded), so the post-process composite asks this before deciding who encodes.
    [[nodiscard]] constexpr bool is_srgb_format(VkFormat const format) noexcept {
        switch (format) {
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        case VK_FORMAT_R8G8B8_SRGB:
        case VK_FORMAT_B8G8R8_SRGB:
            return true;
        default:
            return false;
        }
    }

    vulkan::runtime* runtime_from_window(GLFWwindow* window) {
        return static_cast<vulkan::runtime*>(glfwGetWindowUserPointer(window));
    }

    void mouse_button_callback(GLFWwindow* window, int const button, int const action, [[maybe_unused]] int const mods) {
        auto* runtime = runtime_from_window(window);
        if (button != GLFW_MOUSE_BUTTON_LEFT) {
            return;
        }
        // the overlay owns the mouse while the cursor is over a panel / a widget is being
        // dragged: starting an orbit there would fight the ui (see gui_content::wants_mouse)
        if (runtime->debug_gui_wants_mouse()) {
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
        if (runtime_from_window(window)->debug_gui_wants_mouse()) {
            // the drag left the scene and landed on the overlay: keep the camera still but keep
            // tracking the cursor so leaving the panel does not jump the view
            glfwGetCursorPos(window, &camera.last_x, &camera.last_y);
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
        if (runtime_from_window(window)->debug_gui_wants_mouse()) {
            return; // scrolling inside an overlay panel must not zoom the camera
        }
        auto& camera = runtime_from_window(window)->camera;
        // zoom: wheel up pulls in, wheel down pulls out. The upper bound is generous (scene
        // sizes vary from the tiny default model to e.g. the Fox rig, whose framing distance
        // is ~240). The projection far plane always covers the scene (see make_orbit_camera_ubo),
        // so zooming in never clips the far side.
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
        // One shadow-pass + one gui-overlay secondary command buffer per frame slot (stage 2/3
        // of parallel recording): pre-allocated with the primaries so the GPU can read them
        // while this slot's primary executes. Stage 3 additionally gives the main pass one
        // parallel segment per task-pool worker, each as a {pool, secondary} PAIR (vma-style):
        // a VkCommandPool is not thread safe, so the workers must never begin buffers of a
        // shared pool concurrently - every worker owns its own pool + its buffer (recorded in
        // parallel; see sub_render_task).
        this->secondary_command_buffers.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        this->main_segments.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        unsigned const record_workers = static_cast<unsigned>(std::max(1, this->task_pool_threads()));
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            std::array<vk_command_buffer, static_cast<std::size_t>(secondary_pass::count)> pair = {
                this->vulkan_core.make_secondary_command_buffer(), // shadow cascade 0
                this->vulkan_core.make_secondary_command_buffer(), // shadow cascade 1
                this->vulkan_core.make_secondary_command_buffer(), // shadow cascade 2
                this->vulkan_core.make_secondary_command_buffer(), // shadow cascade 3
                this->vulkan_core.make_secondary_command_buffer(), // gui
                this->vulkan_core.make_secondary_command_buffer(), // transparent
            };
            this->secondary_command_buffers.push_back(std::move(pair));
            std::vector<std::pair<VkCommandPool, vk_command_buffer>> segments;
            segments.reserve(record_workers);
            for (unsigned s = 0; s < record_workers; ++s) {
                VkCommandPool const pool = this->vulkan_core.make_command_pool(); // one per worker
                segments.emplace_back(pool, this->vulkan_core.make_secondary_command_buffer(pool));
            }
            this->main_segments.push_back(std::move(segments));
        }

        // Shared scene resources: camera UBO buffers, white fallback texture, texture sampler
        this->init_scene_resources();
        // NOTE: the shadow resources (map layers + light UBO buffers) are created LAZILY, by
        // ensure_shadow_resources() from ensure_scene_set(). The shadow map is a layered 2D array
        // whose layer count is [render] shadow_cascades, and the app config that carries it is applied
        // after this constructor returns - creating them here would freeze the count at its default.
        // Everything between here and the first scene set works with them empty (the light-buffer
        // writes are guarded, and nothing samples the shadow map before a scene set exists).
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

        // post-process raw objects. The pipeline(s) and the sampler are RAII members; the two
        // layouts and the descriptor pool are not, so they are destroyed here - and this must stay
        // real code: an earlier edit collapsed this block onto the comment line above it, which
        // commented the three destroy calls out and leaked them (validation: "VkDevice has 18
        // leaked objects ... VkPipelineLayout, VkDescriptorSetLayout, VkDescriptorSet").
        if (this->post_descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(this->vulkan_core.device, this->post_descriptor_pool, nullptr);
            this->post_descriptor_pool = VK_NULL_HANDLE;
        }
        if (this->post_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->vulkan_core.device, this->post_pipeline_layout, nullptr);
            this->post_pipeline_layout = VK_NULL_HANDLE;
        }
        if (this->post_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(this->vulkan_core.device, this->post_set_layout, nullptr);
            this->post_set_layout = VK_NULL_HANDLE;
        }
        // the same three objects for the G-buffer debug view (its pipeline and sampler are RAII
        // members; the set layout, the pipeline layout and the pool are not)
        if (this->gbuffer_descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(this->vulkan_core.device, this->gbuffer_descriptor_pool, nullptr);
            this->gbuffer_descriptor_pool = VK_NULL_HANDLE;
        }
        if (this->gbuffer_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->vulkan_core.device, this->gbuffer_pipeline_layout, nullptr);
            this->gbuffer_pipeline_layout = VK_NULL_HANDLE;
        }
        if (this->gbuffer_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(this->vulkan_core.device, this->gbuffer_set_layout, nullptr);
            this->gbuffer_set_layout = VK_NULL_HANDLE;
        }
        // ... and the deferred lighting stage's pipeline layout (its own: two sets, scene + G-buffer)
        if (this->deferred_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->vulkan_core.device, this->deferred_pipeline_layout, nullptr);
            this->deferred_pipeline_layout = VK_NULL_HANDLE;
        }
        // ... and the TAA resolve's own objects (its set layout, layout and pool are raw handles)
        if (this->taa_descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(this->vulkan_core.device, this->taa_descriptor_pool, nullptr);
            this->taa_descriptor_pool = VK_NULL_HANDLE;
        }
        if (this->taa_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->vulkan_core.device, this->taa_pipeline_layout, nullptr);
            this->taa_pipeline_layout = VK_NULL_HANDLE;
        }
        if (this->taa_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(this->vulkan_core.device, this->taa_set_layout, nullptr);
            this->taa_set_layout = VK_NULL_HANDLE;
        }
        // Descriptor pools replaced by a later swapchain generation (see
        // retired_descriptor_pools): they outlived their generation on purpose, because the recorded
        // frame command buffers still name their sets.
        for (VkDescriptorPool const pool : this->retired_descriptor_pools) {
            vkDestroyDescriptorPool(this->vulkan_core.device, pool, nullptr);
        }
        this->retired_descriptor_pools.clear();

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

        // Reserve table index 0 as the DEFAULT material (white textures + identity factors):
        // registrations that overflow the table degrade to it (see register_material). Done
        // FIRST so it always lands at index 0 - the raw zeroed record at 0 would render black
        // (all factors zero), not white. Safe here: no scene set exists yet, so the descriptor
        // writes register_material builds are deferred (update_all_scene_sets no-ops).
        {
            primitive_create_info const default_material = {};
            material_id const default_index = this->register_material(default_material);
            if (default_index.value != 0) {
                utility::panic("default material must occupy table index 0");
            }
        }
    }

    void runtime::ensure_shadow_resources() {
        if (!this->shadow_images.empty()) {
            return; // already created (and the cascade count is frozen from here on)
        }
        // Shadow map: one layered depth image per frame slot (see the member docs), with one layer
        // per cascade. Depth-only images carry no uploaded content (vma::create_image with data ==
        // nullptr skips the digest / upload path), so each frame can render the scene's depth from
        // the light's view into every layer.
        this->shadow_cascades = std::clamp(this->shadow_cascades, 1u, vulkan::max_shadow_cascades);
        this->shadow_images.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        this->shadow_array_views.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        this->shadow_layer_views.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            vulkan::image_create_info shadow_info = {};
            shadow_info.width = vulkan::runtime::shadow_map_size;
            shadow_info.height = vulkan::runtime::shadow_map_size;
            shadow_info.mip_levels = 1;
            // ALWAYS the maximum layer count: the layer count is baked into the image at creation and
            // the image is created before [render] shadow_cascades is known (a scene set binds it), so
            // shadow_cascades below only decides how many layers are FITTED, RENDERED and SAMPLED.
            // The cost of the spare layers is memory (2048x2048x4 B each), not bandwidth.
            shadow_info.array_layers = vulkan::max_shadow_cascades;
            shadow_info.format = this->vulkan_core.depth_format;
            shadow_info.extra_usage = VK_IMAGE_USAGE_SAMPLED_BIT; // sampled by shading.glsl
            vk_image shadow_image = this->vulkan_core.vma.create_image(nullptr, 0, shadow_info, vulkan::image_type::texture_2d_depth);
            if (!shadow_image.valid()) {
                utility::panic("failed to create shadow map image");
            }
            auto const* detail = this->vulkan_core.vma.get_image_detail(shadow_image.handle());
            if (detail == nullptr) {
                utility::panic("failed to get shadow map image detail");
            }
            this->shadow_images.push_back(std::move(shadow_image));
            this->shadow_array_views.push_back(this->vulkan_core.make_depth_array_view(detail->image, this->vulkan_core.depth_format));
            std::vector<vk_image_view> layers;
            layers.reserve(this->shadow_cascades);
            for (uint32_t cascade = 0; cascade < vulkan::max_shadow_cascades; ++cascade) {
                layers.push_back(this->vulkan_core.make_depth_layer_view(detail->image, this->vulkan_core.depth_format, cascade));
            }
            this->shadow_layer_views.push_back(std::move(layers));
        }
        this->shadow_sampler = this->vulkan_core.make_shadow_sampler();

        // Light UBO (scene set binding 7): one buffer PER FRAME SLOT (host-visible, mapped), so
        // a frame being rendered never shares the buffer the next frame rewrites. CPU-side
        // content lives in light_state; the frame loop memcpys it into the paced slot's buffer
        // (pace_and_acquire) - see the member docs for the concurrency rationale.
        this->light_buffers.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        this->light_mapped.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            light_ubo initial = {};
            vk_buffer light_buf = this->vulkan_core.vma.create_buffer(std::span(&initial, 1), vulkan::buffer_type::uniform_coherent);
            if (!light_buf.valid()) {
                utility::panic("failed to create light ubo buffer");
            }
            auto const* light_detail = this->vulkan_core.vma.get_buffer_detail(light_buf.handle());
            if (light_detail == nullptr) {
                utility::panic("failed to get light ubo buffer detail");
            }
            this->light_buffers.push_back(std::move(light_buf));
            this->light_mapped.push_back(light_detail->allocation_info.pMappedData);
        }
    }

    void runtime::ensure_cluster_buffers() {
        if (!this->cluster_count_buffers.empty()) {
            return;
        }
        // Clustered light culling (M5), scene set bindings 11/12: one count per cluster and one
        // fixed-capacity index row per cluster, per frame slot (the compute pass writes them, the
        // fragment stage reads them, so a slot in flight must not be overwritten).
        //
        // Allocated for the MAXIMUM grid (max_cluster_count) once: the active grid is capped to it
        // every frame, so a resize only changes the grid dims in the light UBO - no reallocation, no
        // in-flight buffer to retire. Both are host-visible + coherent: the counts are zeroed by the
        // host each frame (that IS the pass's clear, see pace_and_acquire), and the indices only need
        // to live on the GPU between the dispatch and the shading.
        std::size_t const slots = static_cast<std::size_t>(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        std::vector<unsigned char> const zero_counts(static_cast<std::size_t>(vulkan::max_cluster_count) * sizeof(uint32_t), 0);
        std::vector<unsigned char> const zero_indices(static_cast<std::size_t>(vulkan::max_cluster_count) * vulkan::cluster_light_capacity * sizeof(uint32_t), 0);
        this->cluster_count_buffers.reserve(slots);
        this->cluster_count_mapped.reserve(slots);
        this->cluster_index_buffers.reserve(slots);
        for (std::size_t slot = 0; slot < slots; ++slot) {
            vk_buffer counts = this->vulkan_core.vma.create_buffer(zero_counts.data(), zero_counts.size(), vulkan::buffer_type::storage_coherent);
            if (!counts.valid()) {
                utility::panic("failed to create cluster count buffer");
            }
            auto const* count_detail = this->vulkan_core.vma.get_buffer_detail(counts.handle());
            if (count_detail == nullptr) {
                utility::panic("failed to get cluster count buffer detail");
            }
            this->cluster_count_buffers.push_back(std::move(counts));
            this->cluster_count_mapped.push_back(count_detail->allocation_info.pMappedData);

            vk_buffer indices = this->vulkan_core.vma.create_buffer(zero_indices.data(), zero_indices.size(), vulkan::buffer_type::storage_coherent);
            if (!indices.valid()) {
                utility::panic("failed to create cluster index buffer");
            }
            this->cluster_index_buffers.push_back(std::move(indices));
        }
    }

    void runtime::ensure_scene_set() {
        if (this->scene_set_created) {
            return;
        }
        // the scene set binds the shadow map (binding 8) and the light UBO (binding 7), so those
        // resources must exist before the writes below - and their creation is deferred to here so
        // the app config that sets the cascade count has already run (see the constructor note)
        this->ensure_shadow_resources();
        // ... and the clustered-light buffers (M5) for the same reason: binding 11/12 must point at
        // them before the writes. They are allocated for the maximum grid, so a resize never
        // rebuilds them - only the active grid dims change per frame.
        this->ensure_cluster_buffers();
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

            // bindings 11/12: THIS slot's clustered-light buffers (M5) - written by the cluster
            // compute pass, read by the fragment stage
            auto const* cluster_count_detail = this->vulkan_core.vma.get_buffer_detail(this->cluster_count_buffers[static_cast<std::size_t>(slot)].handle());
            if (cluster_count_detail == nullptr) {
                utility::panic("failed to get cluster count buffer detail");
            }
            write_buffer_binding(set, 11, cluster_count_detail->buffer, static_cast<VkDeviceSize>(vulkan::max_cluster_count) * sizeof(uint32_t), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

            auto const* cluster_index_detail = this->vulkan_core.vma.get_buffer_detail(this->cluster_index_buffers[static_cast<std::size_t>(slot)].handle());
            if (cluster_index_detail == nullptr) {
                utility::panic("failed to get cluster index buffer detail");
            }
            write_buffer_binding(set, 12, cluster_index_detail->buffer, static_cast<VkDeviceSize>(vulkan::max_cluster_count) * vulkan::cluster_light_capacity * sizeof(uint32_t), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        }

        // binding 7 (light UBO, per-slot) + binding 8 (per-slot shadow map) and bindings 2-4 (IBL):
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
        // binding 7 (light UBO) + binding 8 (shadow map): BOTH point at THIS slot's own
        // resources (per-slot light buffers like the camera UBO, per-slot shadow images), so no
        // per-frame re-pointing is needed and an in-flight frame never shares a buffer the next
        // frame rewrites.
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            VkDescriptorSet const set = *this->scene_sets[static_cast<std::size_t>(slot)];
            auto const* light_detail = this->vulkan_core.vma.get_buffer_detail(this->light_buffers[static_cast<std::size_t>(slot)].handle());
            if (light_detail == nullptr) {
                utility::panic("failed to get light ubo buffer detail");
            }
            VkDescriptorBufferInfo const light_info{light_detail->buffer, 0, sizeof(light_ubo)};
            auto const* shadow_detail = this->vulkan_core.vma.get_image_detail(this->shadow_images[static_cast<std::size_t>(slot)].handle());
            if (shadow_detail == nullptr) {
                utility::panic("failed to get shadow map image detail");
            }
            VkDescriptorImageInfo const shadow_info{
                .sampler = *this->shadow_sampler,
                .imageView = *this->shadow_array_views[static_cast<std::size_t>(slot)],
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

    material_id runtime::register_material(primitive_create_info const& info) {
        // ---- 1. Resolve the 5 texture slots against the shared array: identical texture bytes
        //         upload once, keyed by a CONTENT hash of the decoded bytes (xxh3 digest +
        //         format/dimensions - the loader hands every material its own copy of a shared
        //         glTF image, so a raw data pointer is NOT a stable identity; the digest lookup
        //         below is what actually dedups); missing slots point at the white fallback
        //         (element 0).
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
            // Content-addressed dedup: the loader hands every material its OWN byte copy of a
            // shared glTF image, so identical pixels arrive under different pointers. Hash the
            // decoded bytes (xxh3-128, the same digest vma uses for GPU-image dedup) and key the
            // slot cache on (digest, format, dimensions): N materials over one image upload
            // once and share the array element. The image itself is also vma-deduped below.
            utility::xxh3_digest const digest = utility::xxh3_128bits(std::span<unsigned char const>(tex.data.data(), tex.data.size_bytes()));
            // key on the digest data_block itself (not a raw byte array): data_block carries the
            // equality/ordering the std::map key needs
            auto const key = std::tuple<utility::xxh3_digest, VkFormat, std::uint32_t, std::uint32_t, std::uint32_t>{
                digest, slots[i].second, tex.width, tex.height, tex.mip_levels};
            auto const cached = this->texture_slot_cache.find(key);
            if (cached != this->texture_slot_cache.end()) {
                texture_indices[i] = cached->second; // shared texture: reuse its slot
                continue;
            }
            if (this->texture_array_views.size() >= vulkan::scene_texture_capacity) {
                // Array full (pathological scene with > scene_texture_capacity distinct images):
                // degrade this texture slot to the white element instead of crashing - the
                // material still renders untextured. Same policy as the material-table overflow
                // below (one-time log, then keep going); content dedup means real scenes rarely
                // get close to the cap.
                if (!this->texture_overflow_logged) {
                    this->texture_overflow_logged = true;
                    utility::log("scene texture array capacity ({}) exceeded - extra textures render as white (element {})",
                                 vulkan::scene_texture_capacity, this->white_texture_index);
                }
                texture_indices[i] = this->white_texture_index; // white fallback, like an invalid texture
                white_needed = true;                            // (re)write the white element's descriptor once
                continue;
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
        if (info.factors.alpha_blend) {
            record.flags |= 32u; // bit5: alphaMode BLEND - alpha-blended / transparent material
        }

        // ---- 3. Content-address the record, then append (or degrade on overflow) ----
        // Identical materials (same texture slots, factors and flags) share ONE table entry:
        // registration happens per primitive, so a scene with N primitives over M shared glTF
        // materials would otherwise append N records and burn the table needlessly. The key is
        // the byte-exact 80-byte record carried in a data_block - no hash collisions, because
        // the unordered lookup hashes the block only for bucketing while equality stays
        // byte-exact.
        utility::data_block<sizeof(vulkan::material_record)> material_key = {};
        std::memcpy(material_key.data.data(), &record, sizeof(record));
        if (auto const cached = this->material_slot_cache.find(material_key); cached != this->material_slot_cache.end()) {
            return cached->second; // already registered: share the existing record
        }
        if (this->material_count >= vulkan::material_capacity) {
            // Table full (pathological - 16384 unique materials): degrade to the reserved
            // default material (index 0, white + identity factors, registered at setup) instead
            // of crashing; the mesh still draws. Logged once, not per registration.
            if (!this->material_overflow_logged) {
                this->material_overflow_logged = true;
                utility::log("material table capacity ({}) exceeded - extra materials render with the default (index 0)", vulkan::material_capacity);
            }
            return {};
        }
        uint32_t const material_index = this->material_count++;
        std::memcpy(static_cast<unsigned char*>(this->material_mapped) + static_cast<size_t>(material_index) * sizeof(material_record), &record, sizeof(record));
        this->material_slot_cache.emplace(material_key, material_id{material_index});
        return material_id{material_index};
    }

    void runtime::begin_rendering(VkCommandBuffer const command_buffer, uint32_t const image_index, VkRenderingFlags const flags) const {
        core const& vk = this->vulkan_core;

        // Dynamic rendering (Vulkan 1.3 core, the only path the engine supports): attachments
        // are described inline, no render pass / framebuffer objects exist. The scene renders
        // into an HDR target: with MSAA the MSAA image resolves into the per-image HDR resolve
        // target (vk.hdr_image_views), without MSAA the HDR target is the color attachment
        // directly. The post-process pass samples that HDR target and writes the swapchain.
        bool const msaa = vk.msaa_samples > VK_SAMPLE_COUNT_1_BIT;

        // G-buffer mode: the opaque pass writes the surface instead of shading it, into three
        // single-sampled targets + their own 1x depth image, and adds its emissive into the HDR
        // target (which the skybox pass, drawn before this instance, cleared and filled). Same
        // primitives, same pipelines (the default pipeline is the G-buffer one), different
        // attachments - so the MSAA HDR path stays untouched and the lighting/debug pass writes it.
        if (this->gbuffer_pass_active()) {
            std::array<VkRenderingAttachmentInfo, vulkan::gbuffer_pass_attachment_count> gbuffer_attachments = {};
            VkClearValue clear = {}; // the surface + motion targets clear to zero: no geometry, no motion
            for (uint32_t target = 0; target < vulkan::gbuffer_target_count; ++target) {
                gbuffer_attachments[target] = make_color_attachment_info(vk.gbuffer_image_views[target][image_index], clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
            }
            gbuffer_attachments[vulkan::gbuffer_target_count] = make_color_attachment_info(vk.velocity_image_views[image_index], clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
            // the last attachment is the scene color, CLEARed to zero: it accumulates only the
            // emissive here. The deferred lighting stage then adds the lighting (and the sky, for
            // pixels no geometry wrote) on top, so a lit pixel is emissive + lighting and a background
            // pixel is sky - with no background pass anywhere in the deferred path. Under TAA the
            // scene color is the resolve's input image, and the resolve writes the HDR target the post
            // chain reads (see runtime::scene_target_view).
            gbuffer_attachments[vulkan::gbuffer_target_count + 1] = make_color_attachment_info(this->scene_target_view(image_index), clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
            // the G-buffer depth clears to the far plane (1.0), like the main depth attachment
            VkRenderingAttachmentInfo const depth_attachment = make_depth_attachment_info(vk.gbuffer_depth_image_views[image_index], VK_ATTACHMENT_STORE_OP_DONT_CARE);
            VkRenderingInfo const rendering_info = make_rendering_info(flags, {{0, 0}, vk.swap_chain_extent}, gbuffer_attachments.data(), static_cast<uint32_t>(gbuffer_attachments.size()), &depth_attachment);
            vkCmdBeginRendering(command_buffer, &rendering_info);
            return;
        }

        VkClearValue clear_color = {};
        clear_color.color = {{this->clear_color.r, this->clear_color.g, this->clear_color.b, 1.0f}};
        VkRenderingAttachmentInfo const color_attachment = make_color_attachment_info(
            msaa ? vk.color_image_views[image_index] : vk.hdr_image_views[image_index],
            clear_color,
            msaa ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
            msaa ? vk.hdr_image_views[image_index] : VK_NULL_HANDLE);

        // depth clears to the far plane (1.0): make_depth_attachment_info
        VkRenderingAttachmentInfo const depth_attachment = make_depth_attachment_info(vk.depth_image_views[image_index], VK_ATTACHMENT_STORE_OP_DONT_CARE);

        VkRenderingInfo const rendering_info = make_rendering_info(flags, {{0, 0}, vk.swap_chain_extent}, true, &color_attachment, &depth_attachment);
        vkCmdBeginRendering(command_buffer, &rendering_info);
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

        // F1 toggles the debug overlay (edge-triggered: holding the key toggles once). When the
        // overlay was never initialized ([gui] show = false at startup) the first press enables
        // it on demand, so the key also brings the ui back after a disable.
        bool const f1_down = glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS;
        if (f1_down && !this->gui_toggle_down) {
            if (!this->debug_overlay.is_active()) {
                this->debug_gui_shown = this->enable_debug_gui();
            } else {
                this->debug_gui_shown = !this->debug_gui_shown;
            }
            utility::log("gui overlay: {}", this->debug_gui_shown ? "shown (F1 to hide)" : "hidden (F1 to show)");
        }
        this->gui_toggle_down = f1_down;

        // F12 requests a screenshot (edge-triggered); the caller consumes the request and saves
        // the capture (runtime::consume_screenshot_request + acquire_current_frame_image)
        bool const f12_down = glfwGetKey(window, GLFW_KEY_F12) == GLFW_PRESS;
        if (f12_down && !this->screenshot_key_down) {
            this->screenshot_requested = true;
            utility::log("screenshot requested (F12)");
        }
        this->screenshot_key_down = f12_down;

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
            this->on_swapchain_recreated();
        }
    }

    void runtime::on_swapchain_recreated() {
        // The swapchain generation changed: every per-image target was destroyed and rebuilt, so
        // every descriptor set that pointed at the old views must be replaced before it is used
        // again. Both per-image set families (the post chain's and the G-buffer path's) are dropped
        // with their pools, and the next recorded frame allocates FRESH sets from a fresh pool.
        // That is the only safe way to rebind them here:
        //  - a pool must not be destroyed while any recorded command buffer names its sets
        //    (VUID-vkDestroyDescriptorPool-descriptorPool-00303), and the per-slot frame command
        //    buffers stay recorded between frames - so the pool is RETIRED (destroyed with the
        //    runtime) instead, and
        //  - a set must not be updated while a frame that uses it is pending
        //    (VUID-vkUpdateDescriptorSets-None-03047), and with two frames in flight the other slot's
        //    frame can still be running - a freshly allocated set is referenced by nothing, so
        //    allocating instead of updating sidesteps that entirely.
        this->debug_overlay.on_swapchain_recreated();
        this->gbuffer_bound_views = {};
        this->gbuffer_debug_sets.clear();
        if (this->gbuffer_descriptor_pool != VK_NULL_HANDLE) {
            this->retired_descriptor_pools.push_back(this->gbuffer_descriptor_pool);
            this->gbuffer_descriptor_pool = VK_NULL_HANDLE;
            this->gbuffer_pool_capacity = 0;
        }
        this->post_bound_views.clear();
        this->post_bound_blooms.clear();
        this->post_bound_ldr.clear();
        this->post_sets.clear();
        this->post_prefilter_sets.clear();
        this->post_down_sets.clear();
        if (this->post_descriptor_pool != VK_NULL_HANDLE) {
            this->retired_descriptor_pools.push_back(this->post_descriptor_pool);
            this->post_descriptor_pool = VK_NULL_HANDLE;
            this->post_pool_capacity = 0;
        }
        this->taa_bound_views = {};
        this->taa_sets.clear();
        if (this->taa_descriptor_pool != VK_NULL_HANDLE) {
            this->retired_descriptor_pools.push_back(this->taa_descriptor_pool);
            this->taa_descriptor_pool = VK_NULL_HANDLE;
            this->taa_pool_capacity = 0;
        }
        // Every swapchain image's history died with the old generation (and its size may have
        // changed): forget the matrices and mark the histories invalid, so the next frame for each
        // image starts a new accumulation instead of blending in a misaligned one.
        std::size_t const image_count = this->vulkan_core.taa_history_images.size();
        this->image_view_proj.assign(image_count, this->current_ubo.view_proj_unjittered);
        this->taa_history_valid.assign(image_count, false);
    }

    void runtime::gpu_mark(VkCommandBuffer const command_buffer, gpu_mark_id const mark, VkPipelineStageFlagBits const stage) noexcept {
        if (!this->gpu_timings_enabled) {
            return;
        }
        uint32_t const slot = static_cast<uint32_t>(this->vulkan_core.current_frame);
        // The mark's identity is positional - the interval it closes is the one opened by the mark
        // before it - so every label in gpu_timing_labels is only correct while the calls happen in
        // gpu_mark_id order. The core hands out the next index, which makes the violation visible
        // here instead of only as a mislabeled report.
        if (this->vulkan_core.gpu_timing_marks[slot] != static_cast<uint32_t>(mark)) {
            utility::log("runtime: GPU timing marks recorded out of order (mark {} at index {}) - the pass report is mislabeled",
                         static_cast<uint32_t>(mark),
                         this->vulkan_core.gpu_timing_marks[slot]);
            return;
        }
        this->vulkan_core.mark_gpu_timing(command_buffer, slot, stage);
    }

    void runtime::collect_gpu_timings(uint32_t const slot) {
        gpu_timing_result const result = this->vulkan_core.read_gpu_timings(slot);
        if (result.mark_count < 2) {
            return; // nothing measured in that submission (the first frames, or a failed recording)
        }
        // result.milliseconds[i] is the interval between mark i and mark i + 1, which is the pass
        // gpu_timing_labels[i] names: the marks are written in gpu_mark_id order on every frame, and
        // a pass that did not record left two adjacent marks behind, so its interval reads ~0.
        uint32_t const intervals = std::min(result.mark_count - 1, static_cast<uint32_t>(gpu_timing_labels.size()));
        for (uint32_t interval = 0; interval < intervals; ++interval) {
            this->gpu_timing_sum[interval] += result.milliseconds[interval];
        }
        this->gpu_timing_marks_measured = intervals;
        if (++this->gpu_timing_window_frames < GPU_TIMING_WINDOW) {
            return;
        }

        // window complete: report the means (the label keeps reading the running mean, so this only
        // snapshots and resets) and start over
        double total = 0.0;
        std::string report = std::format("gpu pass timings (avg of {} frames):", GPU_TIMING_WINDOW);
        for (uint32_t interval = 0; interval < intervals; ++interval) {
            double const mean = this->gpu_timing_sum[interval] / static_cast<double>(GPU_TIMING_WINDOW);
            report += std::format(" {} {:.2f} ms |", gpu_timing_labels[interval].name, mean);
            total += mean;
            this->gpu_timing_sum[interval] = 0.0;
        }
        this->gpu_timing_window_frames = 0;
        report += std::format(" total {:.2f} ms", total);
        utility::log("{}", report);
    }

    std::string runtime::gpu_timing_summary() const {
        if (!this->gpu_timings_enabled) {
            return "gpu timings: off";
        }
        if (!this->vulkan_core.gpu_timing_available()) {
            return "gpu timings: unavailable on this device";
        }
        if (this->gpu_timing_marks_measured == 0 || this->gpu_timing_window_frames == 0) {
            return "gpu timings: collecting...";
        }
        double const samples = static_cast<double>(this->gpu_timing_window_frames);
        std::string report = "gpu:";
        double total = 0.0;
        for (uint32_t interval = 0; interval < this->gpu_timing_marks_measured; ++interval) {
            gpu_timing_label const& label = gpu_timing_labels[interval];
            double const mean = this->gpu_timing_sum[interval] / samples;
            report += std::format("{}{} {:.2f}", label.new_line ? "\n     " : "  ", label.name, mean);
            total += mean;
        }
        return report + std::format("\n     total {:.2f} ms", total);
    }

    frame_status runtime::pace_and_acquire() {
        core& vk = this->vulkan_core;

        // A zero-sized swapchain (a window that has not been sized yet, or was restored from
        // minimized into a 0-sized client area) has no valid attachments: recording would set a
        // 0-wide viewport - a VUID - and produce nothing. Skip the frame exactly like the
        // minimized case; nothing was acquired, so no semaphore is left pending.
        if (vk.swap_chain_extent.width == 0 || vk.swap_chain_extent.height == 0) {
            return frame_status::skipped;
        }

        // Pace the frame slot: wait until the previous submission on this slot has completed
        //    (host-side timeline wait on the slot's last signaled value). This guards both the
        //    command buffer and the acquire semaphore — acquiring first could reuse a binary
        //    acquire semaphore with pending operations (VUID-vkAcquireNextImageKHR-semaphore-01779)
        uint32_t const frame_slot = static_cast<uint32_t>(vk.current_frame);
        vk.wait_frame_slot(frame_slot);
        // The slot's previous submission is complete, so its timestamps are readable: collect them
        // here, where the wait already guarantees it, and before the slot is recorded again.
        this->collect_gpu_timings(frame_slot);

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
            this->on_swapchain_recreated();
            return frame_status::skipped;
        }
        if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
            return frame_status::acquire_failed;
        }

        // Write this frame's camera UBO into the paced slot's per-slot buffer. The scene
        //    descriptor sets are static per slot (ensure_scene_set wired bindings 0/7/8/9/10 to
        //    each slot's own camera/light/shadow/skin/morph resources), so one memcpy is the whole
        //    camera update - no per-frame descriptor write exists anymore. An external
        //    (glTF/programmatic) camera, when active, supplies the matrices directly.
        this->current_aspect = static_cast<float>(vk.swap_chain_extent.width) / static_cast<float>(vk.swap_chain_extent.height);
        if (this->external_camera_active) {
            this->current_ubo.view = this->external_view;
            this->current_ubo.proj = this->external_proj;
            this->current_ubo.camera_pos = this->external_eye;
        } else {
            this->current_ubo = make_orbit_camera_ubo(this->camera.yaw, this->camera.pitch, this->camera.distance, this->camera.target, this->scene_radius, this->current_aspect);
        }
        // Motion-vector support: the G-buffer computes its vectors from the UNJITTERED pair, and the
        // previous matrix is the one THIS swapchain image's history was rendered with (not simply the
        // last frame's: several images are in rotation, so the last frame's camera is not what that
        // history shows).
        this->current_ubo.view_proj_unjittered = this->current_ubo.proj * this->current_ubo.view;
        this->current_proj_unjittered = this->current_ubo.proj; // before the jitter below
        this->current_ubo.prev_view_proj = this->current_image_index < this->image_view_proj.size() ? this->image_view_proj[this->current_image_index] : this->current_ubo.view_proj_unjittered;
        // TAA jitter: offset the projection by a sub-pixel amount so consecutive frames sample the
        // image at different positions. The offset lands in the projection's z-row (the only place a
        // perspective matrix carries an NDC translation), in NDC units derived from pixels, and it is
        // part of `proj` - so geometry AND the lighting stage's depth reconstruction agree about where
        // each sample is. Both unjittered matrices above are already taken, so the jitter cannot leak
        // into a motion vector.
        if (this->taa_active()) {
            glm::vec2 const jitter_pixels = taa_jitter_offset(this->taa_jitter_index);
            this->current_ubo.proj[2][0] += jitter_pixels.x * 2.0f / static_cast<float>(vk.swap_chain_extent.width);
            this->current_ubo.proj[2][1] += jitter_pixels.y * 2.0f / static_cast<float>(vk.swap_chain_extent.height);
            this->taa_jitter_index = (this->taa_jitter_index + 1) % taa_jitter_count;
        }
        // The deferred lighting stage reconstructs world positions from the G-buffer depth, so its
        // inverse must be the projection actually used to render that depth (jitter included).
        this->current_inv_view_proj = glm::inverse(this->current_ubo.proj * this->current_ubo.view);
        // ... and only now is the frame's camera UBO complete: upload it.
        if (this->camera_mapped[frame_slot] != nullptr) {
            std::memcpy(this->camera_mapped[frame_slot], &this->current_ubo, sizeof(camera_ubo));
        }
        // Same for the light UBO: copy the CPU-side light_state into THIS slot's own light
        // buffer. The slot was just paced (its previous submission completed) and the other
        // in-flight slot's set points at its own buffer, so this host write can never race a GPU
        // read - which is why set_shadow_enabled() / enable_shadows() only touch light_state and
        // never mapped memory directly.
        if (this->light_mapped.size() > static_cast<std::size_t>(frame_slot) && this->light_mapped[frame_slot] != nullptr) {
            // the light_count lane's y carries the exposure scale (pbr.frag / skybox.frag apply it
            // in linear space right before the tonemapper)
            // shadows follow the camera: refit the light frustum to this frame's camera before
            // the UBO upload (16 points + a few matrix multiplies per frame)
            this->update_shadow_frustum();
            this->light_state.light_count.y = this->exposure_scale;
            this->light_state.light_count.z = this->toon_steps;
            this->light_state.light_count.w = this->toon_softness;

            // ---- clustered light culling (M5): this frame's grid + the view-depth range its
            //      exponential slices span, and the host-side clear of the per-cluster counters.
            //      The grid is derived from the swapchain extent (capped to the allocated tiles) and
            //      the depth range from the same scene bound the shadow fit uses, so every visible
            //      fragment lands in a real cluster. A degenerate projection (the first frames, before
            //      the swapchain has an extent) disables the pass for that frame: shade_surface() then
            //      loop every light, which is always correct - just slower.
            {
                this->cluster_tiles_x = std::min((vk.swap_chain_extent.width + vulkan::cluster_tile_size - 1) / vulkan::cluster_tile_size, vulkan::max_cluster_tiles_x);
                this->cluster_tiles_y = std::min((vk.swap_chain_extent.height + vulkan::cluster_tile_size - 1) / vulkan::cluster_tile_size, vulkan::max_cluster_tiles_y);
                glm::mat4 const base_proj = this->current_proj_unjittered;
                bool const degenerate = !(this->current_aspect > 0.0f) || std::abs(base_proj[2][2]) < 1e-6f;
                float cluster_near = 0.1f;
                float cluster_far = 1.0f;
                if (!degenerate) {
                    float const camera_near = base_proj[3][2] / base_proj[2][2];
                    float const camera_far = base_proj[2][2] * camera_near / (1.0f + base_proj[2][2]);
                    float const scene_far = glm::distance(glm::vec3(this->current_ubo.camera_pos), this->shadow_scene_center) + this->scene_radius;
                    cluster_near = std::max(camera_near, 0.05f);
                    cluster_far = std::max(std::min(camera_far, scene_far), cluster_near * 2.0f);
                }
                bool const clustered = this->clustered_lights && this->cluster_pipeline.has_value() && !degenerate && this->cluster_tiles_x > 0 && this->cluster_tiles_y > 0;
                this->light_state.cluster_grid = glm::vec4(static_cast<float>(this->cluster_tiles_x),
                                                           static_cast<float>(this->cluster_tiles_y),
                                                           static_cast<float>(vulkan::cluster_slice_count),
                                                           clustered ? 1.0f : 0.0f);
                this->light_state.cluster_depth = glm::vec4(cluster_near,
                                                            cluster_far,
                                                            static_cast<float>(vk.swap_chain_extent.width),
                                                            static_cast<float>(vk.swap_chain_extent.height));
                // the pass only APPENDS, so the counts are cleared here - the buffer is host-coherent
                // (no flush) and this slot was just paced, so its previous GPU reads are done
                if (this->cluster_count_mapped.size() > static_cast<std::size_t>(frame_slot) && this->cluster_count_mapped[frame_slot] != nullptr) {
                    std::memset(this->cluster_count_mapped[frame_slot], 0, static_cast<std::size_t>(vulkan::max_cluster_count) * sizeof(uint32_t));
                }
            }
            std::memcpy(this->light_mapped[frame_slot], &this->light_state, sizeof(light_ubo));
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
        // Record the frame into this slot's command buffer (inline recording: no inheritance)
        vk_command_buffer& command_buffer = this->command_buffers[static_cast<uint32_t>(vk.current_frame)];
        VkCommandBufferBeginInfo const begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = 0,
            .pInheritanceInfo = nullptr,
        };
        if (vkBeginCommandBuffer(*command_buffer, &begin_info) != VK_SUCCESS) {
            return frame_status::begin_recording_failed;
        }
        // GPU pass timing: open this frame's timestamp range and take the first mark. Marks are
        // written in gpu_mark_id order from here on (see gpu_mark); opening the range outside any
        // rendering instance is required, and this is the first point of the frame where the
        // command buffer exists.
        vk.begin_gpu_timing(*command_buffer, static_cast<uint32_t>(vk.current_frame));
        this->gpu_mark(*command_buffer, gpu_mark_id::frame_begin, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
        // Debug overlay: begin a fresh ImGui frame once per rendered frame (after the acquire,
        // before any UI content is built; the actual draw is recorded at the end of
        // record_main_drawcalls() while the main rendering instance is still open).
        if (this->debug_gui_shown && this->debug_overlay.is_active()) {
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
        float const aspect = this->current_aspect;
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
                    // the cull frustum is built from the UNJITTERED view-projection: a jittered one
                    // would re-cull (and occasionally flicker a leaf in or out) on every TAA frame
                    utility::frustum const view_frustum = utility::make_frustum(ubo.view_proj_unjittered);
                    auto const inside = this->cull_bvh->frustum_cull(view_frustum);
                    for (auto const* node : inside) {
                        visible.push_back(node->extra_data);
                    }
                }
                this->cull_visible = std::move(visible);
                this->camera_key = key;

                // ---- Shadow caster set (see shadow_casters in the class docs) ----
                // Built only when the camera moved or the scene changed (reusing the cached
                // cull_visible otherwise, like the main pass does). Two regimes split on scene
                // size:
                //  - SMALL scenes: every leaf goes into the shadow map. Caster culling is only an
                //    approximation (a caster can sit arbitrarily far up-light and its PARALLEL
                //    shadow column still lands in the view) and with few leaves the full depth
                //    render is cheap - take the exact path.
                //  - HEAVY scenes (tens of thousands of leaves): the camera-visible set plus the
                //    BVH culled against the SHADOW frustum itself. That frustum is refitted every
                //    frame to the camera view AND already merges every leaf whose shadow column can
                //    reach the view (update_shadow_frustum), so it contains the off-screen casters -
                //    like the wall behind the camera - that the previous "camera frustum shifted
                //    up-light by shadow_caster_extent" heuristic dropped, while staying far smaller
                //    than the whole scene.
                constexpr std::size_t full_scene_shadow_leaf_limit = 1500;
                if (frame_leaves.size() <= full_scene_shadow_leaf_limit) {
                    this->shadow_casters = this->frame_leaves;
                } else if (!this->shadows_enabled) {
                    // no shadow frustum yet (enable_shadows() not called): the set is unused until
                    // the shadow pass records, so take the exact path instead of culling against a
                    // zeroed light matrix. enable_shadows() runs before the first frame in
                    // practice; if it does not, the camera moving refreshes this set.
                    this->shadow_casters = this->frame_leaves;
                } else {
                    if (!this->shadow_heuristic_logged) {
                        this->shadow_heuristic_logged = true;
                        utility::log("shadow caster culling: scene exceeds {} leaves - shadow casters are the camera-visible plus shadow-frustum sets",
                                     full_scene_shadow_leaf_limit);
                    }
                    this->shadow_casters = this->cull_visible;
                    if (this->cull_bvh.has_value()) {
                        // every cascade's frustum: a caster that only shadows the far range must still
                        // be drawn, so the union over the cascades is the exact caster set (a cull per
                        // cascade is cheap against the BVH)
                        for (uint32_t cascade = 0; cascade < std::clamp(this->shadow_cascades, 1u, vulkan::max_shadow_cascades); ++cascade) {
                            utility::frustum const light_frustum = utility::make_frustum(this->light_state.light_view_proj[cascade]);
                            auto const in_light = this->cull_bvh->frustum_cull(light_frustum);
                            this->shadow_casters.reserve(this->shadow_casters.size() + in_light.size());
                            for (auto const* node : in_light) {
                                this->shadow_casters.push_back(node->extra_data);
                            }
                        }
                        std::ranges::sort(this->shadow_casters);
                        this->shadow_casters.erase(std::ranges::unique(this->shadow_casters).begin(), this->shadow_casters.end());
                    }
                }
            }
            visible_leaves = this->cull_visible;
        } else {
            // culling disabled: the shadow pass draws every scene leaf (see shadow_casters)
            this->shadow_casters = this->frame_leaves;
        }

        // Split the visible set into OPAQUE leaves (frame_visible: drawn first, depth write on,
        // in the parallel segments) and TRANSPARENT leaves (frame_transparent: alpha-blended,
        // depth write off, drawn last). Transparent leaves are sorted FAR -> NEAR from the
        // camera so overlapping blends compose back-to-front. The split re-runs every frame
        // (cheap: one pass over the visible set + sort of the usually-few transparent leaves);
        // with a static camera the input and thus the result are identical, so no extra cache.
        this->frame_visible.clear();
        this->frame_transparent.clear();
        {
            glm::vec3 const eye = ubo.camera_pos;
            auto const distance_to = [&eye](primitive const* const m) -> float {
                // squared world-space distance of the leaf to the eye. Use the center of the
                // world AABB (transform of the local bounds by push.model) so large or
                // skinned/instanced-with-world leaves sort by where they actually occupy
                // space, not by an arbitrary origin point.
                glm::vec3 center;
                if (m->has_bounds) {
                    auto const [wmin, wmax] = m->world_aabb();
                    center = (wmin + wmax) * 0.5f;
                } else {
                    // no single world AABB (e.g. instanced primitive: one leaf, many world
                    // transforms, push.model is identity): fall back to the model origin.
                    // The resulting order is a no-op for instanced leaves, which is fine -
                    // instances spread over space have no meaningful per-leaf depth anyway.
                    center = glm::vec3(m->push.model[3]);
                }
                return glm::dot(center - eye, center - eye);
            };
            for (primitive const* const m : visible_leaves) {
                (m->transparent ? this->frame_transparent : this->frame_visible).push_back(m);
            }
            std::ranges::sort(this->frame_transparent,
                              [&distance_to](primitive const* const a, primitive const* const b) {
                                  return distance_to(a) > distance_to(b); // far first
                              });
        }
        return frame_status::proceed;
    }

    void runtime::record_main_drawcalls() {
        core& vk = this->vulkan_core;
        vk_command_buffer& command_buffer = this->command_buffers[static_cast<uint32_t>(vk.current_frame)];
        uint32_t const frame_slot = static_cast<uint32_t>(vk.current_frame);

        // ---- Clustered light culling (M5): one compute dispatch before anything renders. It sorts
        //      the punctual lights into the frame's cluster grid, and the shading stages (forward
        //      inside the main instance, deferred later in the frame) then loop only their own
        //      cluster's list. Runs before the shadow pass so the barrier that publishes its buffers
        //      is as early as possible; nothing before it reads the lists.
        this->record_cluster_pass(*command_buffer);

        // ---- Shadow pass: render the scene's depth from the light into this slot's shadow map.
        //      Drawn before the main pass; the depth-only pipeline shares the flat scene layout
        //      and the primitive draw() path (same vertex buffers / push constants), so the shadow
        //      pass is just "bind the shadow pipeline, then draw the same models".
        //      Depth-only rendering (no color attachment) needs dynamic rendering, which is core
        //      1.3 - the only path the engine supports.
        //
        //      Stage 2 of parallel recording: the shadow content is recorded into this slot's
        //      shadow SECONDARY command buffer first, then executed from the primary inside the
        //      shadow rendering instance (VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT).
        //      The recording is still sequential on the primary thread - rendering is identical
        //      to inline; stage 3 fans the recording out over the task pool.
        if (this->shadow_pipeline && this->shadows_enabled && this->shadow_enabled) {
            auto const* shadow_detail = vk.vma.get_image_detail(this->shadow_images[frame_slot].handle());
            if (shadow_detail != nullptr) {
                // Secondary: inherit only the depth attachment (dynamic rendering 1.3). The
                // shadow map is single-sampled; viewMask 0 = no multiview. The rendering
                // inheritance struct hangs off VkCommandBufferInheritanceInfo::pNext (NOT the
                // begin-info pNext), and a secondary buffer must always provide inheritance info.
                // One secondary PER CASCADE: the caster content is identical, but each cascade pushes
                // its own index (a secondary records that itself - state is not inherited from the
                // primary), and a buffer without SIMULTANEOUS_USE may not be executed twice in one
                // primary anyway. The content is only the leaves/pipelines; the per-cascade difference
                // is that single push.
                uint32_t const cascades = std::clamp(this->shadow_cascades, 1u, vulkan::max_shadow_cascades);
                VkCommandBufferInheritanceRenderingInfo const shadow_inheritance = make_inheritance_rendering_info(false, nullptr, vk.depth_format, VK_SAMPLE_COUNT_1_BIT);
                VkCommandBufferInheritanceInfo const shadow_sec_inherit = make_inheritance_info(&shadow_inheritance);
                VkCommandBufferBeginInfo const shadow_sec_begin = make_command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT, &shadow_sec_inherit);
                std::array<bool, vulkan::max_shadow_cascades> shadow_recorded = {};
                for (uint32_t cascade = 0; cascade < cascades; ++cascade) {
                    VkCommandBuffer const cascade_secondary = *this->secondary_command_buffers[static_cast<std::size_t>(frame_slot)][static_cast<std::size_t>(shadow_secondary(cascade))];
                    if (vkBeginCommandBuffer(cascade_secondary, &shadow_sec_begin) != VK_SUCCESS) {
                        utility::log("runtime: shadow secondary command buffer begin failed - cascade {} skipped this frame", cascade);
                        continue;
                    }
                    // which cascade these casters are projected into (the vertex stage indexes the light
                    // UBO's matrix array with it)
                    vkCmdPushConstants(cascade_secondary, vk.scene_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, vulkan::scene_cascade_push_offset, sizeof(uint32_t), &cascade);
                    this->record_shadow_content(cascade_secondary);
                    vkEndCommandBuffer(cascade_secondary);
                    shadow_recorded[cascade] = true;
                }

                // ---- one instance per cascade ----
                for (uint32_t cascade = 0; cascade < cascades; ++cascade) {
                    // Transition THIS layer to a renderable depth attachment (loadOp CLEAR discards the
                    // previous frame's contents, so UNDEFINED as the old layout is valid). One barrier
                    // per layer: the transition constant's subresource range is single-layer, and each
                    // layer is a separate attachment here.
                    VkImageMemoryBarrier2 shadow_barrier = depth_attachment_transition;
                    shadow_barrier.image = shadow_detail->image;
                    shadow_barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, cascade, 1};
                    VkDependencyInfo const shadow_dependency = make_image_dependency_info(1, &shadow_barrier);
                    vkCmdPipelineBarrier2(*command_buffer, &shadow_dependency);

                    // Depth-only rendering into this cascade (no color attachment): loadOp CLEAR
                    // (far plane) + storeOp STORE - the map must survive for the lighting pass
                    VkRenderingAttachmentInfo const shadow_depth_attachment = make_depth_attachment_info(*this->shadow_layer_views[frame_slot][cascade], VK_ATTACHMENT_STORE_OP_STORE);

                    VkRenderingInfo const shadow_rendering_info = make_rendering_info(VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT, {{0, 0}, {vulkan::runtime::shadow_map_size, vulkan::runtime::shadow_map_size}}, false, nullptr, &shadow_depth_attachment);
                    vkCmdBeginRendering(*command_buffer, &shadow_rendering_info);

                    // Run this cascade's pre-recorded secondary (the whole scene casts shadows). Never
                    // execute a secondary whose begin failed - executing an unrecorded command
                    // buffer is a VUID and can wedge the frame slot.
                    if (shadow_recorded[cascade]) {
                        VkCommandBuffer const cascade_secondary = *this->secondary_command_buffers[static_cast<std::size_t>(frame_slot)][static_cast<std::size_t>(shadow_secondary(cascade))];
                        vkCmdExecuteCommands(*command_buffer, 1, &cascade_secondary);
                    }
                    vkCmdEndRendering(*command_buffer);
                }

                // Hand the cascades back to the lighting stage as a sampled array texture. Two
                // barriers: the layers that were rendered (depth attachment -> shader read), and - when
                // fewer cascades are active than the image has layers - the spare layers, which the
                // descriptor's array view still covers, so they must be in the declared layout too.
                // UNDEFINED is the honest old layout for a layer nothing has ever written.
                std::array<VkImageMemoryBarrier2, 2> shadow_read_barriers = {};
                uint32_t read_barrier_count = 0;
                shadow_read_barriers[read_barrier_count] = shadow_map_sampling_transition;
                shadow_read_barriers[read_barrier_count].image = shadow_detail->image;
                shadow_read_barriers[read_barrier_count].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, cascades};
                ++read_barrier_count;
                if (cascades < vulkan::max_shadow_cascades) {
                    shadow_read_barriers[read_barrier_count] = vulkan::undefined_to_depth_sampling_transition;
                    shadow_read_barriers[read_barrier_count].image = shadow_detail->image;
                    shadow_read_barriers[read_barrier_count].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, cascades, vulkan::max_shadow_cascades - cascades};
                    ++read_barrier_count;
                }
                VkDependencyInfo const shadow_read_dependency = make_image_dependency_info(read_barrier_count, shadow_read_barriers.data());
                vkCmdPipelineBarrier2(*command_buffer, &shadow_read_dependency);
            }
        } else if (this->shadow_pipeline && this->shadow_images.size() > static_cast<std::size_t>(frame_slot)) {
            // The pass does not run this frame (shadows toggled off, or no light setup yet), but the
            // scene set still binds the shadow map to binding 8 - pbr.frag uses it statically and only
            // decides at runtime whether to sample it - and a sampled descriptor must point at an
            // image that is in the layout the descriptor declares. Leaving the map in UNDEFINED made
            // every shadow-off frame a VUID ("expects ... SHADER_READ_ONLY_OPTIMAL ... current layout
            // is UNDEFINED"). Contents do not matter (the shader returns "fully lit"), hence UNDEFINED
            // as the old layout.
            auto const* shadow_detail = vk.vma.get_image_detail(this->shadow_images[frame_slot].handle());
            if (shadow_detail != nullptr) {
                VkImageMemoryBarrier2 shadow_read_barrier = vulkan::undefined_to_depth_sampling_transition;
                shadow_read_barrier.image = shadow_detail->image;
                // every layer, not just layer 0: the constant's range is single-layer and all
                // cascades must be sampleable (they all are, this frame or the last one)
                shadow_read_barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, vulkan::max_shadow_cascades};
                VkDependencyInfo const shadow_read_dependency = make_image_dependency_info(1, &shadow_read_barrier);
                vkCmdPipelineBarrier2(*command_buffer, &shadow_read_dependency);
            }
        }

        // GPU timing: the shadow pass (and its hand-back barrier) ends here. The pass is optional,
        // so a frame without shadows just writes this mark next to frame_begin and reports ~0 ms.
        this->gpu_mark(*command_buffer, gpu_mark_id::shadow_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // Dynamic rendering has no automatic attachment transitions (a render pass would do
        // them implicitly): move every attachment into its render layout before
        // vkCmdBeginRendering
        // The G-buffer mode writes three single-sampled targets plus the pass's own 1x depth image
        // instead: the main HDR target is not touched by the opaque pass at all (the debug view
        // writes it afterwards), so it must not be transitioned here.
        bool const gbuffer_pass = this->gbuffer_pass_active();
        // room for every pass attachment plus the depth: three surface targets + the G-buffer's own
        // depth + the scene color the emissive goes into (the forward path uses at most three of these
        // slots). Sizing this for the old three-target G-buffer was a stack overflow the moment the
        // emissive attachment arrived - validation reported the fifth barrier as garbage.
        std::array<VkImageMemoryBarrier2, vulkan::gbuffer_pass_attachment_count + 1> attachment_barriers = {};
        uint32_t barrier_count = 0;
        auto const add_render_barrier = [&attachment_barriers, &barrier_count](VkImageMemoryBarrier2 const& transition, VkImage const image) {
            VkImageMemoryBarrier2& barrier = attachment_barriers[barrier_count++];
            barrier = transition; // copy the role default, then retarget it
            barrier.image = image;
        };

        if (gbuffer_pass) {
            for (uint32_t target = 0; target < gbuffer_target_count; ++target) {
                add_render_barrier(color_attachment_transition, vk.gbuffer_images[target][this->current_image_index]);
            }
            add_render_barrier(color_attachment_transition, vk.velocity_images[this->current_image_index]);
            add_render_barrier(depth_attachment_transition, vk.gbuffer_depth_images[this->current_image_index]);
            // The scene-color target enters the pass as an attachment too (the emissive accumulation
            // target) - the HDR image normally, scene_color when the TAA resolve owns the HDR target
            // this frame (see scene_target_image). It is cleared by the instance below, so UNDEFINED as
            // the old layout is correct whatever it held before.
            add_render_barrier(color_attachment_transition, this->scene_target_image(this->current_image_index));
        } else if (vk.msaa_samples > VK_SAMPLE_COUNT_1_BIT) {
            // the MSAA scene color and its HDR resolve target both render in COLOR_ATTACHMENT_OPTIMAL
            add_render_barrier(color_attachment_transition, vk.color_images[this->current_image_index]);
            add_render_barrier(color_attachment_transition, vk.hdr_images[this->current_image_index]);
            add_render_barrier(depth_attachment_transition, vk.depth_images[this->current_image_index]);
        } else {
            add_render_barrier(color_attachment_transition, vk.hdr_images[this->current_image_index]);
            add_render_barrier(depth_attachment_transition, vk.depth_images[this->current_image_index]);
        }

        VkDependencyInfo const dependency_info = make_image_dependency_info(barrier_count, attachment_barriers.data());
        vkCmdPipelineBarrier2(*command_buffer, &dependency_info);

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
        {
            // unique lock: mutating every cached pipeline's viewport/scissor while parallel
            // recording workers may read them through their environments
            std::unique_lock const lock(this->access_mutex);
            for (auto& pipeline : this->pipelines | std::views::values) {
                pipeline.viewport = full_viewport;
                pipeline.scissor = full_scissor;
            }
        }
        if (this->skybox_pipeline && this->skybox_enabled) {
            this->skybox_pipeline->viewport = full_viewport;
            this->skybox_pipeline->scissor = full_scissor;
        }
        // the post-process pipelines are NOT in the named cache above, and begin_pipeline() always
        // re-emits the stored viewport/scissor: leaving them at the creation-time default made every
        // post pass set a 0-wide viewport (validation: "pViewports[0].width (0.000000) is not
        // greater than zero"). The per-pass viewport is set explicitly right after the bind anyway -
        // this keeps the stored values valid. The shadow pipeline is deliberately excluded: its
        // viewport is the fixed shadow-map size (set in make_shadow_pipeline).
        if (this->post_pipeline) {
            this->post_pipeline->viewport = full_viewport;
            this->post_pipeline->scissor = full_scissor;
        }
        if (this->post_hdr_pipeline) {
            this->post_hdr_pipeline->viewport = full_viewport;
            this->post_hdr_pipeline->scissor = full_scissor;
        }
        if (this->post_fxaa_pipeline) {
            this->post_fxaa_pipeline->viewport = full_viewport;
            this->post_fxaa_pipeline->scissor = full_scissor;
        }
        // ... and the same for the G-buffer pair: gbuffer_pipeline is the opaque pass's default
        // pipeline (begin_pipeline applies the stored viewport) and gbuffer_debug_pipeline is a
        // fullscreen pass. Neither lives in the named cache above.
        if (this->gbuffer_pipeline) {
            this->gbuffer_pipeline->viewport = full_viewport;
            this->gbuffer_pipeline->scissor = full_scissor;
        }
        if (this->gbuffer_debug_pipeline) {
            this->gbuffer_debug_pipeline->viewport = full_viewport;
            this->gbuffer_debug_pipeline->scissor = full_scissor;
        }
        if (this->deferred_pipeline) {
            this->deferred_pipeline->viewport = full_viewport;
            this->deferred_pipeline->scissor = full_scissor;
        }
        // the TAA resolve is a fullscreen pass too, and begin_pipeline() re-emits the stored viewport:
        // leaving it at the creation-time zero made every resolve set a 0-wide viewport (the same VUID
        // the post pipelines hit once)
        if (this->taa_pipeline) {
            this->taa_pipeline->viewport = full_viewport;
            this->taa_pipeline->scissor = full_scissor;
        }

        // Stage 3 of parallel recording: the main-pass visible leaves are split into up-to-N
        // contiguous sub_render_tasks (N = task-pool workers), each recording its own per-slot
        // SECONDARY command buffer; the batch is posted to the task pool and the recording
        // priority group is waited on. The primary then executes the segments in order inside
        // the main rendering instance (VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT).
        // Rendering is identical to inline (same leaves, same order, same batching per
        // segment); only the recording is parallel. Shadow + attachment barriers stay on the
        // primary (see above). The gui overlay (same rendering instance) records on the last
        // worker as one more task.
        auto const& secondaries = this->secondary_command_buffers[static_cast<std::size_t>(frame_slot)];

        // one {pool, secondary} pair per task-pool worker (see the member docs): a worker never
        // shares its pool, so parallel recording cannot race on a VkCommandPool
        std::vector<std::pair<VkCommandPool, vk_command_buffer>>& main_segments = this->main_segments[static_cast<std::size_t>(frame_slot)];

        // Main secondaries inherit the color + depth attachments (dynamic rendering 1.3): same
        // formats as begin_rendering() below, rasterization samples follow MSAA in the forward pass
        // and are 1x in the G-buffer pass. The gui overlay draws into the same color+depth instance,
        // so it inherits identically. The G-buffer mode declares its four targets here (the three
        // surface targets + the HDR target it adds emissive into), which is what lets the same
        // parallel segment recording serve both passes.
        std::array<VkFormat, vulkan::gbuffer_pass_attachment_count> const pass_color_formats =
            gbuffer_pass ? std::array<VkFormat, vulkan::gbuffer_pass_attachment_count>{
                               vulkan::gbuffer_formats[0], vulkan::gbuffer_formats[1], vulkan::gbuffer_formats[2], vulkan::gbuffer_velocity_format, vulkan::hdr_format}
                         : std::array<VkFormat, vulkan::gbuffer_pass_attachment_count>{vulkan::hdr_format, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
        uint32_t const pass_color_count = gbuffer_pass ? vulkan::gbuffer_pass_attachment_count : 1u;
        VkSampleCountFlagBits const pass_samples = gbuffer_pass ? VK_SAMPLE_COUNT_1_BIT : vk.msaa_samples;
        VkCommandBufferInheritanceRenderingInfo const main_inheritance = make_inheritance_rendering_info(pass_color_formats.data(), pass_color_count, vk.depth_format, pass_samples);
        VkCommandBufferInheritanceInfo const main_sec_inherit = make_inheritance_info(&main_inheritance);
        VkCommandBufferBeginInfo const main_sec_begin = make_command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT, &main_sec_inherit);

        std::size_t const leaf_count = this->frame_visible.size();
        std::size_t const segment_count = std::min<std::size_t>(main_segments.size(), std::max<std::size_t>(1, leaf_count));

        // Transparent pass: the alpha-blended leaves, recorded on the PRIMARY thread (usually
        // few + order-sensitive, so no parallel fan-out). Its environment keeps the session
        // default but transparent draws disable depth writes via env.set_depth_write(false)
        // (each leaf's draw() requests it). Recorded only when there are transparent leaves, and
        // never in G-buffer mode: a blended surface cannot be stored in a G-buffer (the deferred
        // path keeps alpha-blended geometry as a forward pass, which is the standard hybrid).
        // recorded_* flags gate the execute below: a secondary whose begin failed must never be
        // executed (executing an unrecorded command buffer is a VUID and can wedge the slot).
        VkCommandBuffer const transparent_secondary = *secondaries[static_cast<std::size_t>(secondary_pass::transparent)];
        bool has_transparent = !gbuffer_pass && !this->frame_transparent.empty();
        auto const record_transparent_pass = [&] {
            if (!has_transparent) {
                return;
            }
            if (vkBeginCommandBuffer(transparent_secondary, &main_sec_begin) == VK_SUCCESS) {
                this->record_main_segment(transparent_secondary, this->frame_transparent, false);
                vkEndCommandBuffer(transparent_secondary);
            } else {
                utility::log("runtime: transparent secondary begin failed - transparent leaves skipped this frame");
                has_transparent = false; // do not execute the unrecorded buffer
            }
        };

        if (segment_count == 1 || leaf_count < 4) {
            // Few leaves: parallel recording would cost more than it saves - record the whole
            // main pass on one segment (identical to stage 2) on this thread.
            VkCommandBuffer const single_main = *main_segments[0].second;
            bool main_recorded = false;
            if (vkBeginCommandBuffer(single_main, &main_sec_begin) == VK_SUCCESS) {
                this->record_main_content(single_main);
                vkEndCommandBuffer(single_main);
                main_recorded = true;
            } else {
                utility::log("runtime: main secondary begin failed - scene skipped this frame");
            }

            this->begin_rendering(*command_buffer, this->current_image_index, VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT);
            if (main_recorded) {
                vkCmdExecuteCommands(*command_buffer, 1, &single_main);
            }
            record_transparent_pass(); // alpha-blended leaves compose over the opaque depth
            if (has_transparent) {
                vkCmdExecuteCommands(*command_buffer, 1, &transparent_secondary);
            }

            return;
        }

        // Parallel: slice frame_visible into segment_count contiguous spans; one sub_render_task
        // per segment records its own secondary on a pool worker (segment 0 also draws the
        // skybox). The tasks only read shared state (scene set / pipeline caches / the leaf
        // pointers) and write their own command buffer, so they run concurrently; the recording
        // priority group is waited on before the primary executes the segments in order. Each
        // task's recorded flag is set only on a successful begin+end; the primary skips a
        // segment whose flag stayed false (executing an unrecorded secondary is a VUID).
        std::vector<std::function<void()>> tasks;
        tasks.reserve(segment_count);
        std::vector<std::atomic<bool>> segment_recorded(segment_count);
        for (std::size_t s = 0; s < segment_count; ++s) {
            std::size_t const seg_first = leaf_count * s / segment_count;
            std::size_t const seg_last = leaf_count * (s + 1) / segment_count;
            sub_render_task task = {};
            task.command_buffer = *main_segments[s].second;
            task.leaves = std::span<primitive const* const>(this->frame_visible.data() + seg_first, seg_last - seg_first);
            // the skybox belongs to the first segment, and never to the G-buffer (a background is
            // not a surface: the deferred path treats "no geometry" as the sky, the debug view
            // clears those pixels)
            task.draw_skybox = s == 0 && !gbuffer_pass;
            task.color_formats = pass_color_formats;
            task.color_count = pass_color_count;
            task.depth_format = vk.depth_format;
            task.rasterization_samples = pass_samples;
            task.owner = this;
            task.recorded = &segment_recorded[s];
            tasks.emplace_back(std::move(task)); // std::function copies the value task
        }
        this->run_tasks(tasks, vulkan::task_priority::recording);

        // The gui overlay + the transparent pass record on the PRIMARY thread (the gui recorder
        // touches Dear ImGui global state via ImGui::Render()/GetDrawData; transparent leaves
        // are few + order-sensitive) - the pool workers above only recorded scene secondaries,
        // so nothing races them. Like the single-segment branch, a secondary whose begin failed
        // is never executed.
        record_transparent_pass();

        this->begin_rendering(*command_buffer, this->current_image_index, VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT);
        for (std::size_t s = 0; s < segment_count; ++s) {
            if (!segment_recorded[s].load(std::memory_order_relaxed)) {
                continue; // this segment's begin failed - never execute the unrecorded buffer
            }
            VkCommandBuffer const seg_cb = *main_segments[s].second;
            vkCmdExecuteCommands(*command_buffer, 1, &seg_cb);
        }
        // transparent leaves compose over the opaque depth, before the gui overlay
        if (has_transparent) {
            vkCmdExecuteCommands(*command_buffer, 1, &transparent_secondary);
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
        // depth bias is dynamic state on the shadow pipeline: record the live-tunable values
        // (gui-adjustable) before the depth-only draw
        vkCmdSetDepthBias(command_buffer, this->shadow_depth_bias_constant, this->shadow_depth_bias_clamp, this->shadow_depth_bias_slope);
        // Shadow pass render_environment: every leaf draws into the DEPTH-ONLY shadow map, so
        // the binder ignores the requested pipeline name and always binds the shadow pipeline -
        // whatever a custom leaf would draw in the main pass, its geometry still casts the same
        // shadow. Default-semantics leaves hit bind_default() and land here too.
        render_environment env;
        env.command_buffer = command_buffer;
        env.default_name = "shadow"; // binder ignores the name; kept for in_default_pipeline()
        env.bind = [this](VkCommandBuffer const cb, std::string_view const /*name*/) {
            this->shadow_pipeline->begin_pipeline(cb);
        };
        // The shadow pass MUST write depth for every caster: its env always records depth-write
        // ENABLED (VK_TRUE) regardless of what a leaf requests - the first leaf's
        // set_depth_write() emits the one required vkCmdSetDepthWriteEnable and later leaves
        // dedupe against it. (The pipeline declares depth-write as dynamic state, so it must be
        // set at least once even though the value matches the default.)
        env.set_depth_write_fn = [](VkCommandBuffer const cb, VkBool32 const) {
            vkCmdSetDepthWriteEnable(cb, VK_TRUE);
        };
        // ... and it draws every caster TWO-SIDED: a caster is never culled for facing away from
        // the light. A single-sided wall plane whose only face points into the room is back-facing
        // as seen from the sun, so a back-face-culled depth pass never records it and the sunlight
        // pours straight through a wall the camera sees as solid - the classic "the wall behind the
        // camera is transparent" leak. See render_environment::two_sided.
        env.two_sided = true;
        env.set_cull_mode_fn = [](VkCommandBuffer const cb, VkCullModeFlags const mode) {
            vkCmdSetCullMode(cb, mode);
        };
        env.layout = vk.scene_pipeline_layout;
        // draw only the casters that can throw a shadow into the camera frustum (see
        // shadow_casters in begin_recording); the whole scene only when culling is disabled.
        // BLEND (transparent) leaves are skipped: a depth-only pass has no sensible way to blend
        // an alpha-blended shadow, so "no shadow" stays the correct fallback. MASK leaves ARE
        // drawn - shadow.frag runs the same alpha-cutoff discard as pbr.frag, on the same texture
        // array and material record, so a cut-out caster (foliage, a curtain) casts a cut-out
        // shadow instead of a solid one or, as before, none at all.
        for (primitive const* m : this->shadow_casters) {
            if (m->transparent) {
                continue;
            }
            m->draw(env); // depth-only: shadow.vert transforms into light space
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
        // the skybox is a background, not a surface: it draws in the forward instance only (the
        // G-buffer pass leaves those pixels cleared - see the deferred path's debug/lighting pass)
        bool const draw_skybox = !this->gbuffer_pass_active() && this->skybox_pipeline && this->skybox_enabled;
        this->record_main_segment(command_buffer, this->frame_visible, draw_skybox);
    }

    // One slice of the main-pass leaves (see the declaration); when draw_skybox the skybox is
    // drawn first so the background always precedes the scene (segment 0 only). Every segment
    // binds the scene set itself (a secondary does not inherit state from the primary), then
    // walks ONLY the given leaves, drawing each through its render_environment: default-semantics
    // leaves request the runtime's default pipeline (bind_default, deduplicated), custom leaves
    // request theirs by name - so leaves of several pipelines mix freely in one segment and
    // each pipeline is bound only when the current one differs.
    void runtime::record_main_segment(VkCommandBuffer const command_buffer, std::span<primitive const* const> const leaves, bool const draw_skybox) const {
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

        // Background pass first (only the segment that carries it): the skybox draws a fullscreen
        // triangle (no vertex/index buffers) with depth test/write disabled, then the models
        // render over it. Cull mode is dynamic state: set it explicitly (previously it leaked
        // from the shadow pass's inline draws; with secondaries that leak is gone). Depth write
        // is dynamic state too (transparency): the skybox pipeline declares it, so it must be
        // set once before the draw - OFF, the skybox never writes depth (it sits at z=0.0 and
        // must not occlude the scene).
        if (draw_skybox && this->skybox_pipeline && this->skybox_enabled) {
            this->skybox_pipeline->begin_pipeline(command_buffer);
            // the skybox outputs linear HDR; exposure + tonemapping happen in the post pass
            vkCmdSetCullMode(command_buffer, VK_CULL_MODE_BACK_BIT);
            vkCmdSetDepthWriteEnable(command_buffer, VK_FALSE);
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
        }

        // Main pass: one render_environment per segment (per recording thread - never shared
        // across the parallel workers). Its binder looks the requested pipeline up in the
        // runtime cache and binds it; default-semantics leaves ask for the runtime default.
        //
        // Concurrency contract: make_pipeline() / set_default_pipeline() may be called from any
        // thread, but only OUTSIDE the frame loop (setup or while idle - same timing rule as
        // make_primitive: mid-frame creation would race the recording workers). During
        // recording the registry is therefore read-only, so the per-bind lookup below needs no
        // lock: the pipelines map is a node container (inserts never invalidate existing
        // entries), so the binder can search it directly. The default-name view below is still
        // snapshotted under a shared lock so a concurrent setup-time set_default_pipeline
        // cannot tear the std::string it points into.
        render_environment env;
        env.command_buffer = command_buffer;
        // The G-buffer pass binds its own pipeline as the pass default (see gbuffer_pipeline_name):
        // same leaves, same draw path, but the fragment stage writes the surface into three 1x
        // targets instead of shading into the HDR one.
        bool const gbuffer_pass = this->gbuffer_pass_active();
        {
            std::shared_lock const lock(this->access_mutex);
            env.default_name = gbuffer_pass ? gbuffer_pipeline_name : this->default_pipeline_name;
        }
        env.bind = [this, gbuffer_pass](VkCommandBuffer const cb, std::string_view const name) {
            if (gbuffer_pass) {
                if (name == gbuffer_pipeline_name) {
                    this->gbuffer_pipeline->begin_pipeline(cb);
                    return;
                }
                // a leaf with explicit pipeline semantics cannot draw in the G-buffer instance (the
                // named pipelines declare the single HDR attachment): say so once per leaf instead
                // of issuing a draw that would be a validation error
                utility::log("runtime: leaf requests pipeline '{}' during the G-buffer pass - draw skipped (only default-semantics leaves write the G-buffer)", name);
                return;
            }
            if (auto const it = this->pipelines.find(name); it != this->pipelines.end()) {
                it->second.begin_pipeline(cb);
            } else {
                utility::log("runtime: main pass references unknown pipeline '{}' - draw skipped", name);
            }
        };
        // transparent leaves toggle depth writes off via this (core dynamic state, 1.3)
        env.set_depth_write_fn = [](VkCommandBuffer const cb, VkBool32 const enabled) {
            vkCmdSetDepthWriteEnable(cb, enabled);
        };
        // single-sided materials keep back-face culling here (the shadow pass overrides it with
        // env.two_sided; the main pass must not, or double-sided handling would cost fill rate)
        env.set_cull_mode_fn = [](VkCommandBuffer const cb, VkCullModeFlags const mode) {
            vkCmdSetCullMode(cb, mode);
        };
        env.layout = vk.scene_pipeline_layout;
        for (primitive const* const m : leaves) {
            m->draw(env); // polymorphic: normal / instanced / static / custom
        }
    }

    // One parallel recording job (see the declaration): begin the secondary command buffer with
    // dynamic-rendering inheritance (color + depth attachments, MSAA sample count), record the
    // segment's leaves (skybox on the carrying segment) and end it. Self-contained - built
    // fresh each call so the pNext chains point at this invocation's stack structs; safe to run
    // on any pool worker.
    void runtime::sub_render_task::operator()() const {
        VkCommandBufferInheritanceRenderingInfo const rendering_inherit = make_inheritance_rendering_info(this->color_formats.data(), this->color_count, this->depth_format, this->rasterization_samples);
        VkCommandBufferInheritanceInfo const inherit = make_inheritance_info(&rendering_inherit);
        VkCommandBufferBeginInfo const begin = make_command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT, &inherit);
        if (vkBeginCommandBuffer(this->command_buffer, &begin) != VK_SUCCESS) {
            utility::log("runtime: main segment secondary begin failed - segment skipped this frame");
            if (this->recorded != nullptr) {
                this->recorded->store(false, std::memory_order_relaxed);
            }
            return;
        }
        this->owner->record_main_segment(this->command_buffer, this->leaves, this->draw_skybox);
        vkEndCommandBuffer(this->command_buffer);
        if (this->recorded != nullptr) {
            this->recorded->store(true, std::memory_order_relaxed);
        }
    }

    // ---- post-processing: HDR scene target -> exposure + ACES + gamma -> swapchain ----
    // The bloom chain lands here next; the push constants already reserve its parameters.
    std::expected<void, std::string> runtime::make_post_pipeline(std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        core& vk = this->vulkan_core;

        // sampler for the HDR scene target (linear, clamp)
        this->post_sampler = vk.make_sampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 1.0f);

        // binding 0 = the pass input (HDR for the prefilter, the previous bloom level for a
        // downsample), bindings 1..4 = the four bloom levels (only the composite pass samples
        // them; the other passes bind the same view to every binding so one layout serves all),
        // binding 5 = the gamma-encoded LDR image (only the FXAA pass samples it; post.frag does not
        // declare it, so the composite may safely write that image while the set points at it)
        std::array<VkDescriptorSetLayoutBinding, 6> bindings = {};
        for (uint32_t b = 0; b < bindings.size(); ++b) {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[b].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(vk.device, &layout_info, nullptr, &this->post_set_layout) != VK_SUCCESS) {
            return fail("post: descriptor set layout creation failed");
        }

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_range.offset = 0;
        push_range.size = sizeof(post_push_constants);

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &this->post_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &this->post_pipeline_layout) != VK_SUCCESS) {
            return fail("post: pipeline layout creation failed");
        }

        // fullscreen pipelines: post.vert synthesizes the triangle from gl_VertexIndex, so the
        // vertex input is empty; no depth attachment, 1x samples. TWO of them, one per color
        // format the pass chain renders into: the composite writes the swapchain, while the
        // bright-pass prefilter and the three downsample passes write the R16F bloom levels. A
        // pipeline's VkPipelineRenderingCreateInfo color format must match the attachment it draws
        // into, so a single swapchain-format pipeline was a validation error (and UB) for the HDR
        // passes.
        auto const make_post_variant = [&](VkFormat const color_format) -> std::expected<vk_pipeline, std::string> {
            auto pipeline_result = vulkan::make_pipeline(
                vk.device,
                this->post_pipeline_layout,
                color_format,
                VK_FORMAT_UNDEFINED,
                vertex_shader_code,
                fragment_shader_code,
                VK_SAMPLE_COUNT_1_BIT,
                false,
                true,
                0.0f,
                0.0f,
                0.0f);
            if (!pipeline_result) {
                return std::unexpected(std::string(pipeline_result.error()));
            }
            return std::move(pipeline_result).value();
        };

        auto composite_pipeline = make_post_variant(vk.swap_chain_image_format);
        if (!composite_pipeline) {
            return fail(std::move(composite_pipeline.error()));
        }
        this->post_pipeline = std::move(composite_pipeline).value();

        auto hdr_pipeline = make_post_variant(vulkan::hdr_format);
        if (!hdr_pipeline) {
            return fail(std::move(hdr_pipeline.error()));
        }
        this->post_hdr_pipeline = std::move(hdr_pipeline).value();

        return {};
    }

    std::expected<void, std::string> runtime::make_fxaa_pipeline(std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        core& vk = this->vulkan_core;
        if (this->post_pipeline_layout == VK_NULL_HANDLE) {
            return fail(std::string("fxaa: create the post-process pipeline first (it owns the set layout)"));
        }
        // Same fullscreen triangle and the same swapchain color format as the composite (FXAA writes
        // the swapchain), but its own shader module: fxaa.frag is what declares binding 5, and
        // keeping the composite's shader free of that binding is what lets the composite write the
        // LDR image while its own descriptor set points at it.
        auto pipeline_result = vulkan::make_pipeline(
            vk.device,
            this->post_pipeline_layout,
            vk.swap_chain_image_format,
            VK_FORMAT_UNDEFINED,
            vertex_shader_code,
            fragment_shader_code,
            VK_SAMPLE_COUNT_1_BIT,
            false,
            true,
            0.0f,
            0.0f,
            0.0f);
        if (!pipeline_result) {
            return fail(std::string(pipeline_result.error()));
        }
        this->post_fxaa_pipeline = std::move(pipeline_result).value();
        return {};
    }

    void runtime::ensure_post_descriptors() {
        core& vk = this->vulkan_core;
        if (this->post_pipeline == std::nullopt || this->post_hdr_pipeline == std::nullopt || this->post_set_layout == VK_NULL_HANDLE) {
            return;
        }
        std::size_t const image_count = vk.hdr_image_views.size();
        if (image_count == 0 || vk.bloom_image_views[0].size() != image_count || vk.ldr_image_views.size() != image_count) {
            return;
        }
        if (this->post_sets.size() == image_count && this->post_bound_views == vk.hdr_image_views && this->post_bound_blooms == vk.bloom_image_views[0] && this->post_bound_ldr == vk.ldr_image_views) {
            return; // already bound to the current HDR/bloom/LDR targets
        }

        // five sets per swapchain image: prefilter (HDR), three downsample inputs (level 0..2) and
        // the composite (HDR + all four levels + the LDR image for the FXAA pass)
        std::size_t const set_count = image_count * 5;
        // Recreate the pool whenever the requirement DIFFERS, not only when it grows: the sets of the
        // previous swapchain generation are still allocated from the old pool, so after a shrink
        // (fewer images) allocating the new sets on top exceeded maxSets, vkAllocateDescriptorSets
        // returned VK_ERROR_OUT_OF_POOL_MEMORY and the post pass stayed dead for the rest of the run
        // (the failure path cleared post_sets, so every frame retried and failed again). Destroying
        // the pool frees all of its sets, which is exactly what a new generation wants.
        if (this->post_descriptor_pool == VK_NULL_HANDLE || this->post_pool_capacity != set_count) {
            // Retire the old pool instead of destroying it: the frame command buffers that recorded
            // descriptor sets from it are still executable, and destroying a pool they reference is
            // VUID-vkDestroyDescriptorPool-descriptorPool-00303. The sets of the retired pool are
            // simply unused from here on; the runtime destroys the pools on teardown.
            if (this->post_descriptor_pool != VK_NULL_HANDLE) {
                this->retired_descriptor_pools.push_back(this->post_descriptor_pool);
                this->post_descriptor_pool = VK_NULL_HANDLE;
            }
            VkDescriptorPoolSize pool_size = {};
            pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            pool_size.descriptorCount = static_cast<uint32_t>(set_count * 6); // six bindings per set

            VkDescriptorPoolCreateInfo pool_info = {};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = static_cast<uint32_t>(set_count);
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            if (vkCreateDescriptorPool(vk.device, &pool_info, nullptr, &this->post_descriptor_pool) != VK_SUCCESS) {
                utility::log("runtime: post descriptor pool creation failed - post pass skipped");
                this->post_descriptor_pool = VK_NULL_HANDLE;
                this->post_sets.clear();
                this->post_prefilter_sets.clear();
                this->post_down_sets.clear();
                this->post_bound_views.clear();
                this->post_bound_blooms.clear();
                this->post_bound_ldr.clear();
                return;
            }
            this->post_pool_capacity = static_cast<uint32_t>(set_count);
            this->post_sets.clear();
            this->post_prefilter_sets.clear();
            this->post_down_sets.clear();
        }

        if (this->post_sets.size() != image_count || this->post_prefilter_sets.size() != image_count || this->post_down_sets.size() != image_count) {
            this->post_sets.assign(image_count, VK_NULL_HANDLE);
            this->post_prefilter_sets.assign(image_count, VK_NULL_HANDLE);
            this->post_down_sets.assign(image_count, {});
            std::vector<VkDescriptorSetLayout> layouts(set_count, this->post_set_layout);
            std::vector<VkDescriptorSet> allocated(set_count, VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo allocate_info = {};
            allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate_info.descriptorPool = this->post_descriptor_pool;
            allocate_info.descriptorSetCount = static_cast<uint32_t>(set_count);
            allocate_info.pSetLayouts = layouts.data();
            if (vkAllocateDescriptorSets(vk.device, &allocate_info, allocated.data()) != VK_SUCCESS) {
                utility::log("runtime: post descriptor allocation failed - post pass skipped");
                this->post_sets.clear();
                this->post_prefilter_sets.clear();
                this->post_down_sets.clear();
                this->post_bound_views.clear();
                this->post_bound_blooms.clear();
                this->post_bound_ldr.clear();
                return;
            }
            for (std::size_t i = 0; i < image_count; ++i) {
                this->post_prefilter_sets[i] = allocated[i * 5 + 0];
                this->post_down_sets[i] = {allocated[i * 5 + 1], allocated[i * 5 + 2], allocated[i * 5 + 3]};
                this->post_sets[i] = allocated[i * 5 + 4];
            }
        }

        // every set gets all six bindings; unused ones point at the same view as binding 0 (binding 5
        // is the LDR image, which only the FXAA pass reads)
        auto const write_set = [&vk](VkDescriptorSet const set, std::array<VkImageView, 6> const& views, VkSampler const sampler) {
            std::array<VkDescriptorImageInfo, 6> image_infos = {};
            for (uint32_t b = 0; b < image_infos.size(); ++b) {
                image_infos[b].sampler = sampler;
                image_infos[b].imageView = views[b];
                image_infos[b].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            std::array<VkWriteDescriptorSet, 6> writes = {};
            for (uint32_t b = 0; b < writes.size(); ++b) {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = set;
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[b].pImageInfo = &image_infos[b];
            }
            vkUpdateDescriptorSets(vk.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        };

        for (std::size_t i = 0; i < image_count; ++i) {
            VkImageView const hdr = vk.hdr_image_views[i];
            VkImageView const ldr = vk.ldr_image_views[i];
            std::array<VkImageView, 6> const hdr_set = {hdr, hdr, hdr, hdr, hdr, ldr};
            write_set(this->post_prefilter_sets[i], hdr_set, *this->post_sampler);

            for (std::size_t level = 0; level < 3; ++level) {
                VkImageView const input = vk.bloom_image_views[level][i];
                std::array<VkImageView, 6> const level_set = {input, input, input, input, input, ldr};
                write_set(this->post_down_sets[i][level], level_set, *this->post_sampler);
            }

            std::array<VkImageView, 6> const composite_set = {hdr, vk.bloom_image_views[0][i], vk.bloom_image_views[1][i], vk.bloom_image_views[2][i], vk.bloom_image_views[3][i], ldr};
            write_set(this->post_sets[i], composite_set, *this->post_sampler);
        }
        this->post_bound_views = vk.hdr_image_views;
        this->post_bound_blooms = vk.bloom_image_views[0];
        this->post_bound_ldr = vk.ldr_image_views;
    }
    // ---- G-buffer / deferred path (M1: the write pass + its debug view) ----
    // The G-buffer pass is the forward opaque pass with a different fragment stage: same vertex
    // stage, same primitives, same scene set, same instancing/skinning/morphing. What changes is
    // where the fragments go (three 1x targets + a 1x depth image instead of the MSAA HDR target)
    // and that nothing is lit - see shaders/gbuffer.frag.
    std::expected<void, std::string> runtime::make_gbuffer_pipeline(std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        auto result = this->vulkan_core.make_gbuffer_pipeline(vertex_shader_code, fragment_shader_code);
        if (!result) {
            return std::unexpected(std::string(result.error()));
        }
        this->gbuffer_pipeline = std::move(result).value();
        return {};
    }

    // The deferred lighting stage: a fullscreen pass that shades every pixel from the G-buffer with
    // the same lighting code the forward path runs per fragment (shaders/shading.glsl), added into
    // the HDR target on top of the sky and the emissive the earlier passes wrote.
    std::expected<void, std::string> runtime::make_deferred_pipeline(std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        core& vk = this->vulkan_core;
        if (this->gbuffer_set_layout == VK_NULL_HANDLE) {
            // the lighting stage reads the G-buffer through the same set layout the debug view
            // declares; the debug pipeline owns creating it, so it must come first
            return fail(std::string("deferred: create the G-buffer debug pipeline first (it owns the G-buffer set layout)"));
        }
        if (vk.scene_descriptor_set_layout == VK_NULL_HANDLE) {
            return fail(std::string("deferred: the shared scene descriptor set layout is missing"));
        }

        // Two sets: 0 = the shared scene set (camera UBO, IBL maps, light UBO, shadow map - the
        // lighting stage needs all of them), 1 = the G-buffer inputs. The scene set is exactly the
        // one every scene pipeline binds, so the lighting stage sees the same lights and shadows as
        // the forward path by construction.
        std::array<VkDescriptorSetLayout, 2> const set_layouts = {vk.scene_descriptor_set_layout, this->gbuffer_set_layout};
        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_range.offset = 0;
        push_range.size = sizeof(deferred_push_constants);
        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(set_layouts.size());
        pipeline_layout_info.pSetLayouts = set_layouts.data();
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &this->deferred_pipeline_layout) != VK_SUCCESS) {
            return fail("deferred: pipeline layout creation failed");
        }

        // fullscreen triangle (post.vert), no depth attachment, no depth test: every pixel is shaded
        // exactly once from the G-buffer, which is the point of the deferred path
        // The target is ADDED to, not overwritten: the sky (background pass) and the emissive
        // (G-buffer pass) are already in the HDR target, and a pixel with no geometry emits exactly 0.
        std::array<VkFormat, 1> const color_formats = {vulkan::hdr_format};
        std::array<VkPipelineColorBlendAttachmentState, 1> const blend = {make_color_blend_attachment_additive()};
        auto pipeline_result = vulkan::make_pipeline(
            vk.device,
            this->deferred_pipeline_layout,
            std::span<VkFormat const>(color_formats),
            VK_FORMAT_UNDEFINED,
            vertex_shader_code,
            fragment_shader_code,
            VK_SAMPLE_COUNT_1_BIT,
            false, // no depth attachment: the G-buffer depth is sampled, not tested against
            0.0f,
            0.0f,
            0.0f,
            std::span<VkPipelineColorBlendAttachmentState const>(blend));
        if (!pipeline_result) {
            return fail(std::string(pipeline_result.error()));
        }
        this->deferred_pipeline = std::move(pipeline_result).value();
        return {};
    }

    std::expected<void, std::string> runtime::make_gbuffer_debug_pipeline(std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        core& vk = this->vulkan_core;

        // its own set layout (the post chain's six bindings have nothing to do with the G-buffer):
        // binding 0..2 = the three targets, binding 3 = the pass's depth image
        std::array<VkDescriptorSetLayoutBinding, 4> bindings = {};
        for (uint32_t b = 0; b < bindings.size(); ++b) {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[b].pImmutableSamplers = nullptr;
        }
        VkDescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(vk.device, &layout_info, nullptr, &this->gbuffer_set_layout) != VK_SUCCESS) {
            return fail("gbuffer debug: descriptor set layout creation failed");
        }

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_range.offset = 0;
        push_range.size = sizeof(gbuffer_debug_push_constants);
        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &this->gbuffer_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &this->gbuffer_pipeline_layout) != VK_SUCCESS) {
            return fail("gbuffer debug: pipeline layout creation failed");
        }

        // fullscreen triangle (post.vert), no depth attachment, no depth test: the debug view writes
        // the HDR scene target, which the ordinary post chain (bloom/composite/FXAA) then consumes
        VkFormat const hdr_format_only = vulkan::hdr_format;
        auto pipeline_result = vulkan::make_pipeline(
            vk.device,
            this->gbuffer_pipeline_layout,
            hdr_format_only,
            VK_FORMAT_UNDEFINED,
            vertex_shader_code,
            fragment_shader_code,
            VK_SAMPLE_COUNT_1_BIT,
            false,
            true,
            0.0f,
            0.0f,
            0.0f);
        if (!pipeline_result) {
            return fail(std::string(pipeline_result.error()));
        }
        this->gbuffer_debug_pipeline = std::move(pipeline_result).value();

        // the sampler reads all four inputs (NEAREST: the debug view must show stored texels, not a
        // filtered average of them - the point is to inspect the data, not to make it pretty)
        VkSamplerCreateInfo sampler_info = make_texture_sampler_info(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f);
        sampler_info.magFilter = VK_FILTER_NEAREST;
        sampler_info.minFilter = VK_FILTER_NEAREST;
        VkSampler sampler = VK_NULL_HANDLE;
        if (vkCreateSampler(vk.device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
            return fail("gbuffer debug: sampler creation failed");
        }
        this->gbuffer_sampler = vk_sampler(sampler, vk.device);
        return {};
    }

    void runtime::record_deferred_lighting_pass(VkCommandBuffer const command_buffer) {
        core const& vk = this->vulkan_core;
        if (this->deferred_pipeline == std::nullopt) {
            return;
        }
        std::size_t const index = this->current_image_index;

        // Transitions, all before vkCmdBeginRendering (a pipeline barrier may not be recorded inside
        // a dynamic rendering instance): the three surface targets become shader inputs, the G-buffer
        // depth becomes a shader input, and the scene color - which the G-buffer pass already filled
        // with the emissive - stays a color attachment with its contents LOADed, because the lighting
        // is added on top of them.
        std::array<VkImageMemoryBarrier2, 4> barriers = {};
        for (uint32_t target = 0; target < vulkan::gbuffer_target_count; ++target) {
            barriers[target] = vulkan::hdr_sampling_transition; // COLOR_ATTACHMENT -> SHADER_READ
            barriers[target].image = vk.gbuffer_images[target][index];
        }
        barriers[3] = vulkan::undefined_to_depth_sampling_transition;
        barriers[3].image = vk.gbuffer_depth_images[index];
        VkDependencyInfo const dependency = make_image_dependency_info(static_cast<uint32_t>(barriers.size()), barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency);

        this->ensure_gbuffer_descriptors();
        if (this->gbuffer_debug_sets.size() <= index) {
            // No descriptor set: nothing can be shaded. Clear the HDR target so the frame is defined
            // (the post chain samples it) instead of leaving whatever the background/emissive wrote
            // mixed with garbage - and say so once per frame, because a silent black frame is worse
            // than a log line.
            utility::log("runtime: deferred lighting has no descriptor set - clearing the scene target");
            std::array<VkImageMemoryBarrier2, 1> clear_barrier = {vulkan::color_attachment_transition};
            clear_barrier[0].image = vk.scene_color_images[index];
            VkDependencyInfo const clear_dependency = make_image_dependency_info(1, clear_barrier.data());
            vkCmdPipelineBarrier2(command_buffer, &clear_dependency);
            VkClearValue clear = {};
            VkRenderingAttachmentInfo const attachment = make_color_attachment_info(vk.scene_color_image_views[index], clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
            VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, vk.swap_chain_extent}, true, &attachment, nullptr);
            vkCmdBeginRendering(command_buffer, &rendering_info);
            vkCmdEndRendering(command_buffer);
            return;
        }

        // The G-buffer pass left the scene color in COLOR_ATTACHMENT_OPTIMAL: only the layout of the
        // attachment changes state here, so a load-op LOAD instance adds to it
        VkImageView const target_view = this->scene_target_view(static_cast<uint32_t>(index));
        VkRenderingAttachmentInfo const color_attachment = make_load_color_attachment_info(target_view);
        VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, vk.swap_chain_extent}, true, &color_attachment, nullptr);
        vkCmdBeginRendering(command_buffer, &rendering_info);
        this->deferred_pipeline->begin_pipeline(command_buffer);
        VkViewport const viewport = {0.0f, 0.0f, static_cast<float>(vk.swap_chain_extent.width), static_cast<float>(vk.swap_chain_extent.height), 0.0f, 1.0f};
        VkRect2D const scissor = {{0, 0}, vk.swap_chain_extent};
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);
        vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);
        // set 0 = the shared scene set (camera / IBL / light UBO / shadow map), set 1 = the G-buffer
        std::array<VkDescriptorSet, 2> const sets = {*this->scene_sets[static_cast<std::size_t>(vk.current_frame)], this->gbuffer_debug_sets[index]};
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, this->deferred_pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);
        deferred_push_constants const push = {.inv_view_proj = this->current_inv_view_proj};
        vkCmdPushConstants(command_buffer, this->deferred_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(command_buffer, 3, 1, 0, 0);
        vkCmdEndRendering(command_buffer);
    }

    // ---- temporal anti-aliasing (M3) ----
    glm::vec2 runtime::taa_jitter_offset(uint32_t const index) noexcept {
        // Halton(2,3): the two-radical sequence has the low-discrepancy property that matters here -
        // consecutive samples fill the pixel evenly instead of clustering, so 8 frames of a static
        // scene already resolve close to a 4x4 grid. The offsets are centred on the pixel and given in
        // PIXELS; the caller converts them to a projection offset.
        auto const halton = [](uint32_t position, uint32_t base) {
            float result = 0.0f;
            float fraction = 1.0f;
            while (position > 0) {
                fraction /= static_cast<float>(base);
                result += fraction * static_cast<float>(position % base);
                position /= base;
            }
            return result;
        };
        uint32_t const i = index + 1; // Halton starts at 1 (position 0 gives 0 for every base)
        return {halton(i, 2) - 0.5f, halton(i, 3) - 0.5f};
    }

    bool runtime::taa_active() const noexcept {
        // The forward path has no motion vectors (its fragment stage does not write them), so TAA is
        // the deferred path's answer to MSAA - and only the deferred path's, for now.
        return this->taa_on && this->taa_pipeline.has_value() && this->deferred_lit_active();
    }

    VkImage runtime::scene_target_image(uint32_t const image_index) const noexcept {
        core const& vk = this->vulkan_core;
        return this->taa_active() ? vk.scene_color_images[image_index] : vk.hdr_images[image_index];
    }

    VkImageView runtime::scene_target_view(uint32_t const image_index) const noexcept {
        core const& vk = this->vulkan_core;
        return this->taa_active() ? vk.scene_color_image_views[image_index] : vk.hdr_image_views[image_index];
    }

    void runtime::set_taa(bool const enabled, float const blend_static, float const blend_min) noexcept {
        bool const was_on = this->taa_on;
        this->taa_on = enabled;
        this->taa_blend_static = std::clamp(blend_static, 0.0f, 0.99f);
        this->taa_blend_min = std::clamp(blend_min, 0.0f, this->taa_blend_static);
        if (enabled && !was_on) {
            // A fresh history - but only on the off -> on EDGE. The caller mirrors the GUI/config state
            // into the runtime every frame (see main.cpp), so resetting unconditionally here would
            // invalidate the history on every frame: the resolve would fall back to the current
            // (jittered, aliased) frame forever, which looks like TAA running while doing nothing.
            std::size_t const image_count = this->vulkan_core.taa_history_images.size();
            this->taa_history_valid.assign(image_count, false);
            this->image_view_proj.assign(image_count, this->current_ubo.view_proj_unjittered);
            this->taa_jitter_index = 0;
        }
    }

    std::expected<void, std::string> runtime::make_taa_pipeline(std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        core& vk = this->vulkan_core;

        // binding 0 = this frame's scene color, 1 = the history image, 2 = the motion vectors,
        // 3 = the G-buffer depth (the disocclusion guard)
        std::array<VkDescriptorSetLayoutBinding, 4> bindings = {};
        for (uint32_t b = 0; b < bindings.size(); ++b) {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[b].pImmutableSamplers = nullptr;
        }
        VkDescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(vk.device, &layout_info, nullptr, &this->taa_set_layout) != VK_SUCCESS) {
            return fail("taa: descriptor set layout creation failed");
        }

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_range.offset = 0;
        push_range.size = sizeof(taa_push_constants);
        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &this->taa_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &this->taa_pipeline_layout) != VK_SUCCESS) {
            return fail("taa: pipeline layout creation failed");
        }

        // fullscreen triangle (post.vert), no depth attachment, no depth test, writing the HDR target
        // which the post chain reads - the same target the scene passes would have written without TAA
        std::array<VkFormat, 1> const color_formats = {vulkan::hdr_format};
        auto pipeline_result = vulkan::make_pipeline(
            vk.device,
            this->taa_pipeline_layout,
            std::span<VkFormat const>(color_formats),
            VK_FORMAT_UNDEFINED,
            vertex_shader_code,
            fragment_shader_code,
            VK_SAMPLE_COUNT_1_BIT,
            false,
            0.0f,
            0.0f,
            0.0f);
        if (!pipeline_result) {
            return fail(std::string(pipeline_result.error()));
        }
        this->taa_pipeline = std::move(pipeline_result).value();

        // NEAREST: the history must be sampled where the motion vector says, not averaged with its
        // neighbours (that is what the resolve's own clamp is for)
        VkSamplerCreateInfo sampler_info = make_texture_sampler_info(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f);
        sampler_info.magFilter = VK_FILTER_LINEAR; // the scene color is upsampled by its own geometry...
        sampler_info.minFilter = VK_FILTER_NEAREST;
        VkSampler sampler = VK_NULL_HANDLE;
        if (vkCreateSampler(vk.device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
            return fail("taa: sampler creation failed");
        }
        this->taa_sampler = vk_sampler(sampler, vk.device);
        return {};
    }

    void runtime::ensure_taa_descriptors() {
        core& vk = this->vulkan_core;
        if (this->taa_pipeline == std::nullopt || this->taa_set_layout == VK_NULL_HANDLE) {
            return;
        }
        std::size_t const image_count = vk.scene_color_image_views.size();
        if (image_count == 0 || vk.taa_history_image_views.size() != image_count || vk.velocity_image_views.size() != image_count) {
            return;
        }
        std::array<VkImageView, 4> const signature = {vk.scene_color_image_views[0], vk.taa_history_image_views[0], vk.velocity_image_views[0], vk.gbuffer_depth_image_views[0]};
        if (this->taa_sets.size() == image_count && this->taa_bound_views == signature) {
            return; // already bound to the current targets
        }

        if (this->taa_descriptor_pool == VK_NULL_HANDLE || this->taa_pool_capacity != image_count) {
            // same retirement rule as the other per-image sets (see on_swapchain_recreated): a pool
            // whose sets a recorded command buffer still names is retired, never destroyed here
            if (this->taa_descriptor_pool != VK_NULL_HANDLE) {
                this->retired_descriptor_pools.push_back(this->taa_descriptor_pool);
                this->taa_descriptor_pool = VK_NULL_HANDLE;
            }
            VkDescriptorPoolSize pool_size = {};
            pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            pool_size.descriptorCount = static_cast<uint32_t>(image_count * signature.size());
            VkDescriptorPoolCreateInfo pool_info = {};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = static_cast<uint32_t>(image_count);
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            if (vkCreateDescriptorPool(vk.device, &pool_info, nullptr, &this->taa_descriptor_pool) != VK_SUCCESS) {
                utility::log("runtime: taa descriptor pool creation failed - TAA skipped");
                this->taa_descriptor_pool = VK_NULL_HANDLE;
                this->taa_sets.clear();
                this->taa_bound_views = {};
                return;
            }
            this->taa_pool_capacity = static_cast<uint32_t>(image_count);
            this->taa_sets.clear();
        }

        if (this->taa_sets.size() != image_count) {
            std::vector<VkDescriptorSetLayout> const layouts(image_count, this->taa_set_layout);
            this->taa_sets.assign(image_count, VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo allocate_info = {};
            allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate_info.descriptorPool = this->taa_descriptor_pool;
            allocate_info.descriptorSetCount = static_cast<uint32_t>(image_count);
            allocate_info.pSetLayouts = layouts.data();
            if (vkAllocateDescriptorSets(vk.device, &allocate_info, this->taa_sets.data()) != VK_SUCCESS) {
                utility::log("runtime: taa descriptor allocation failed - TAA skipped");
                this->taa_sets.clear();
                this->taa_bound_views = {};
                return;
            }
        }

        for (std::size_t i = 0; i < image_count; ++i) {
            std::array<VkImageView, 4> const views = {vk.scene_color_image_views[i], vk.taa_history_image_views[i], vk.velocity_image_views[i], vk.gbuffer_depth_image_views[i]};
            std::array<VkDescriptorImageInfo, 4> image_infos = {};
            std::array<VkWriteDescriptorSet, 4> writes = {};
            for (uint32_t b = 0; b < views.size(); ++b) {
                image_infos[b].sampler = *this->taa_sampler;
                image_infos[b].imageView = views[b];
                image_infos[b].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = this->taa_sets[i];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[b].pImageInfo = &image_infos[b];
            }
            vkUpdateDescriptorSets(vk.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
        this->taa_bound_views = signature;
    }

    void runtime::record_taa_pass(VkCommandBuffer const command_buffer) {
        core const& vk = this->vulkan_core;
        if (!this->taa_active()) {
            return;
        }
        std::size_t const index = this->current_image_index;
        bool const history_valid = index < this->taa_history_valid.size() && this->taa_history_valid[index];

        this->ensure_taa_descriptors();
        if (this->taa_sets.size() <= index) {
            utility::log("runtime: TAA has no descriptor set - the frame is shown unresolved");
            return;
        }

        // Layouts, all before vkCmdBeginRendering: the scene color the deferred stage wrote becomes an
        // input, the motion vectors and the depth become inputs, the history becomes an input, and the
        // HDR target - still untouched this frame - becomes the resolve's attachment.
        // The history image is left in SHADER_READ_ONLY by the previous frame's copy (and is only ever
        // read as a texture), so it needs no barrier at all once it is valid - only its very first use
        // transitions it out of UNDEFINED (its contents are then garbage, and history_valid is 0, so
        // the resolve ignores them).
        std::array<VkImageMemoryBarrier2, 4> barriers = {};
        barriers[0] = vulkan::hdr_sampling_transition; // scene_color: COLOR_ATTACHMENT -> SHADER_READ
        barriers[0].image = vk.scene_color_images[index];
        barriers[1] = vulkan::hdr_sampling_transition; // velocity: same transition, COLOR aspect
        barriers[1].image = vk.velocity_images[index];
        barriers[2] = vulkan::undefined_to_depth_sampling_transition; // G-buffer depth -> sampled
        barriers[2].image = vk.gbuffer_depth_images[index];
        uint32_t barrier_count = 3;
        if (!history_valid) {
            barriers[3] = vulkan::undefined_to_sampling_transition;
            barriers[3].image = vk.taa_history_images[index];
            barrier_count = 4;
        }
        VkDependencyInfo const dependency = make_image_dependency_info(barrier_count, barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency);

        std::array<VkImageMemoryBarrier2, 1> output_barrier = {vulkan::color_attachment_transition};
        output_barrier[0].image = vk.hdr_images[index];
        VkDependencyInfo const output_dependency = make_image_dependency_info(1, output_barrier.data());
        vkCmdPipelineBarrier2(command_buffer, &output_dependency);

        VkClearValue clear = {};
        VkRenderingAttachmentInfo const color_attachment = make_color_attachment_info(vk.hdr_image_views[index], clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
        VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, vk.swap_chain_extent}, true, &color_attachment, nullptr);
        vkCmdBeginRendering(command_buffer, &rendering_info);
        this->taa_pipeline->begin_pipeline(command_buffer);
        VkViewport const viewport = {0.0f, 0.0f, static_cast<float>(vk.swap_chain_extent.width), static_cast<float>(vk.swap_chain_extent.height), 0.0f, 1.0f};
        VkRect2D const scissor = {{0, 0}, vk.swap_chain_extent};
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);
        vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);
        VkDescriptorSet const set = this->taa_sets[index];
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, this->taa_pipeline_layout, 0, 1, &set, 0, nullptr);
        taa_push_constants const push = {
            .history_valid = history_valid ? 1.0f : 0.0f,
            .blend_static = this->taa_blend_static,
            .blend_min = this->taa_blend_min,
            .texel_size_x = 1.0f / static_cast<float>(vk.swap_chain_extent.width),
            .texel_size_y = 1.0f / static_cast<float>(vk.swap_chain_extent.height),
            .depth_scale = this->current_ubo.proj[2][2],
            .depth_offset = this->current_ubo.proj[3][2],
            .unused = 0.0f};
        vkCmdPushConstants(command_buffer, this->taa_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(command_buffer, 3, 1, 0, 0);
        vkCmdEndRendering(command_buffer);

        // ---- the resolved frame becomes the next frame's history ----
        // A copy rather than a ping-pong: the resolve necessarily writes the image the post chain
        // reads, so the history has to be a separate image, and copying into it keeps every descriptor
        // set in the frame stable (no per-frame rewrites). The barriers move the HDR target out to
        // TRANSFER_SRC and back - the post chain still finds it in COLOR_ATTACHMENT_OPTIMAL, exactly
        // where it expects it.
        std::array<VkImageMemoryBarrier2, 2> copy_barriers = {};
        copy_barriers[0] = vulkan::color_attachment_to_transfer_transition; // HDR -> TRANSFER_SRC
        copy_barriers[0].image = vk.hdr_images[index];
        copy_barriers[1] = vulkan::sampling_to_transfer_dst_transition; // history: SHADER_READ -> TRANSFER_DST
        copy_barriers[1].image = vk.taa_history_images[index];
        VkDependencyInfo const copy_dependency = make_image_dependency_info(static_cast<uint32_t>(copy_barriers.size()), copy_barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &copy_dependency);

        VkImageCopy const region = {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffset = {0, 0, 0},
            .extent = {vk.swap_chain_extent.width, vk.swap_chain_extent.height, 1},
        };
        vkCmdCopyImage(command_buffer, vk.hdr_images[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk.taa_history_images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // hand both images on: the HDR target back to the post chain, the history copy to the next
        // frame's resolve (which will find it in TRANSFER_DST and transition it from there)
        std::array<VkImageMemoryBarrier2, 2> hand_back = {};
        hand_back[0] = vulkan::transfer_to_color_attachment_transition; // HDR -> COLOR_ATTACHMENT
        hand_back[0].image = vk.hdr_images[index];
        hand_back[1] = vulkan::transfer_dst_to_sampling_transition; // history -> SHADER_READ
        hand_back[1].image = vk.taa_history_images[index];
        VkDependencyInfo const hand_back_dependency = make_image_dependency_info(static_cast<uint32_t>(hand_back.size()), hand_back.data());
        vkCmdPipelineBarrier2(command_buffer, &hand_back_dependency);

        // bookkeeping for the NEXT frame that renders this swapchain image: the matrix its history was
        // rendered with, and the fact that there is a history now. The slot must be the one this frame
        // is recorded into - the other slot is still in flight.
        if (this->image_view_proj.size() > index) {
            this->image_view_proj[index] = this->current_ubo.view_proj_unjittered;
        }
        if (this->taa_history_valid.size() > index) {
            this->taa_history_valid[index] = true;
        }
    }

    bool runtime::gbuffer_pass_active() const noexcept {
        if (!this->gbuffer_pipeline.has_value()) {
            return false;
        }
        if (this->gbuffer_debug) {
            return this->gbuffer_debug_pipeline.has_value();
        }
        return this->deferred_on && this->deferred_pipeline.has_value();
    }

    bool runtime::deferred_lit_active() const noexcept {
        // the debug view wins when both are on: looking at the stored data is an inspection, not a
        // render mode (and the two write the HDR target in incompatible ways)
        return this->gbuffer_pass_active() && this->deferred_on && !this->gbuffer_debug;
    }

    void runtime::set_gbuffer_channel(int const channel) noexcept {
        this->gbuffer_channel_index = std::clamp(channel, 0, gbuffer_channel_count - 1);
    }

    void runtime::ensure_gbuffer_descriptors() {
        core& vk = this->vulkan_core;
        if (this->gbuffer_debug_pipeline == std::nullopt || this->gbuffer_set_layout == VK_NULL_HANDLE) {
            return;
        }
        std::size_t const image_count = vk.gbuffer_image_views[0].size();
        if (image_count == 0 || vk.gbuffer_depth_image_views.size() != image_count) {
            return;
        }
        // Same rebinding rule as the post chain: the sets stay allocated for the runtime's lifetime,
        // and only their CONTENTS are rewritten when the targets change (a rebuilt swapchain
        // generation, or the first frame after setup). on_swapchain_recreated() clears the fingerprint
        // to force that rewrite.
        std::array<VkImageView, 4> const signature = {
            vk.gbuffer_image_views[0][0],
            vk.gbuffer_image_views[1][0],
            vk.gbuffer_image_views[2][0],
            vk.gbuffer_depth_image_views[0]};
        if (this->gbuffer_debug_sets.size() == image_count && this->gbuffer_bound_views == signature) {
            return; // already bound to the current targets
        }

        if (this->gbuffer_descriptor_pool == VK_NULL_HANDLE || this->gbuffer_pool_capacity != image_count) {
            // The generation's image count changed (or this is the first frame): the pool must be sized
            // for exactly this generation and its sets reallocated - a pool cannot grow, and its old
            // sets are still allocated. The old pool is RETIRED, not destroyed: its sets are still
            // named by recorded frame command buffers, and destroying it is
            // VUID-vkDestroyDescriptorPool-descriptorPool-00303 (the same reason on_swapchain_recreated
            // does not touch it). It is destroyed with the runtime.
            if (this->gbuffer_descriptor_pool != VK_NULL_HANDLE) {
                this->retired_descriptor_pools.push_back(this->gbuffer_descriptor_pool);
                this->gbuffer_descriptor_pool = VK_NULL_HANDLE;
            }
            VkDescriptorPoolSize pool_size = {};
            pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            pool_size.descriptorCount = static_cast<uint32_t>(image_count * signature.size());
            VkDescriptorPoolCreateInfo pool_info = {};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = static_cast<uint32_t>(image_count);
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            if (vkCreateDescriptorPool(vk.device, &pool_info, nullptr, &this->gbuffer_descriptor_pool) != VK_SUCCESS) {
                utility::log("runtime: gbuffer debug descriptor pool creation failed - debug view skipped");
                this->gbuffer_descriptor_pool = VK_NULL_HANDLE;
                this->gbuffer_debug_sets.clear();
                this->gbuffer_bound_views = {};
                return;
            }
            this->gbuffer_pool_capacity = static_cast<uint32_t>(image_count);
            this->gbuffer_debug_sets.clear();
        }

        if (this->gbuffer_debug_sets.size() != image_count) {
            std::vector<VkDescriptorSetLayout> const layouts(image_count, this->gbuffer_set_layout);
            this->gbuffer_debug_sets.assign(image_count, VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo allocate_info = {};
            allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate_info.descriptorPool = this->gbuffer_descriptor_pool;
            allocate_info.descriptorSetCount = static_cast<uint32_t>(image_count);
            allocate_info.pSetLayouts = layouts.data();
            if (vkAllocateDescriptorSets(vk.device, &allocate_info, this->gbuffer_debug_sets.data()) != VK_SUCCESS) {
                utility::log("runtime: gbuffer debug descriptor allocation failed - debug view skipped");
                this->gbuffer_debug_sets.clear();
                this->gbuffer_bound_views = {};
                return;
            }
        }

        for (std::size_t i = 0; i < image_count; ++i) {
            std::array<VkDescriptorImageInfo, 4> image_infos = {};
            std::array<VkImageView, 4> const views = {
                vk.gbuffer_image_views[0][i],
                vk.gbuffer_image_views[1][i],
                vk.gbuffer_image_views[2][i],
                vk.gbuffer_depth_image_views[i]};
            std::array<VkWriteDescriptorSet, 4> writes = {};
            for (uint32_t b = 0; b < views.size(); ++b) {
                image_infos[b].sampler = *this->gbuffer_sampler;
                image_infos[b].imageView = views[b];
                image_infos[b].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = this->gbuffer_debug_sets[i];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[b].pImageInfo = &image_infos[b];
            }
            vkUpdateDescriptorSets(vk.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
        this->gbuffer_bound_views = signature;
    }

    void runtime::record_gbuffer_debug_pass(VkCommandBuffer const command_buffer) {
        core const& vk = this->vulkan_core;
        if (this->gbuffer_debug_pipeline == std::nullopt) {
            return;
        }
        std::size_t const index = this->current_image_index;

        // Layout transitions FIRST, and outside the rendering instance: vkCmdPipelineBarrier2 may not
        // be recorded inside a dynamic rendering instance (VUID-vkCmdPipelineBarrier2-None-09553,
        // unless dynamic rendering local read is enabled, which the engine does not need). So all
        // three steps happen before vkCmdBeginRendering:
        //   1. the HDR target the debug view writes enters COLOR_ATTACHMENT_OPTIMAL (the G-buffer
        //      pass wrote its own targets, so it was never an attachment this frame). This happens
        //      even when the descriptor set below is missing, because the post chain that follows
        //      samples that image: an undefined layout would be a lie, a cleared image is a valid
        //      black frame.
        //   2. the three G-buffer targets the pass just wrote become shader inputs.
        //   3. the G-buffer depth image becomes a shader input too (UNDEFINED -> SHADER_READ: its
        //      contents are new, so discarding the old layout is correct; the aspect must be DEPTH).
        std::array<VkImageMemoryBarrier2, 5> barriers = {};
        barriers[0] = vulkan::color_attachment_transition;
        barriers[0].image = vk.hdr_images[index];
        for (uint32_t target = 0; target < vulkan::gbuffer_target_count; ++target) {
            barriers[target + 1] = vulkan::hdr_sampling_transition; // COLOR_ATTACHMENT -> SHADER_READ
            barriers[target + 1].image = vk.gbuffer_images[target][index];
        }
        barriers[4] = vulkan::undefined_to_depth_sampling_transition;
        barriers[4].image = vk.gbuffer_depth_images[index];
        VkDependencyInfo const dependency = make_image_dependency_info(static_cast<uint32_t>(barriers.size()), barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency);

        this->ensure_gbuffer_descriptors();

        VkClearValue clear = {};
        VkRenderingAttachmentInfo const color_attachment = make_color_attachment_info(vk.hdr_image_views[index], clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
        VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, vk.swap_chain_extent}, true, &color_attachment, nullptr);
        vkCmdBeginRendering(command_buffer, &rendering_info);
        bool const can_draw = this->gbuffer_debug_sets.size() > index;
        if (can_draw) {
            this->gbuffer_debug_pipeline->begin_pipeline(command_buffer);
            VkViewport const viewport = {0.0f, 0.0f, static_cast<float>(vk.swap_chain_extent.width), static_cast<float>(vk.swap_chain_extent.height), 0.0f, 1.0f};
            VkRect2D const scissor = {{0, 0}, vk.swap_chain_extent};
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);
            vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);
            VkDescriptorSet const set = this->gbuffer_debug_sets[index];
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, this->gbuffer_pipeline_layout, 0, 1, &set, 0, nullptr);
            gbuffer_debug_push_constants const push = {
                .channel = static_cast<float>(this->gbuffer_channel_index),
                .proj_22 = this->current_ubo.proj[2][2],
                .proj_32 = this->current_ubo.proj[3][2],
                .unused = 0.0f};
            vkCmdPushConstants(command_buffer, this->gbuffer_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
        } else {
            utility::log("runtime: gbuffer debug pass has no descriptor set - showing a cleared frame");
        }
        vkCmdEndRendering(command_buffer);
    }

    bool runtime::record_post_process(VkCommandBuffer const command_buffer) {
        core const& vk = this->vulkan_core;
        vkCmdEndRendering(command_buffer);
        // GPU timing: the geometry instance ends with the instance close above (the forward main
        // pass, or the G-buffer write pass in the deferred path).
        this->gpu_mark(command_buffer, gpu_mark_id::scene_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // Deferred mode: the surface is in the G-buffer and the sky + emissive are in the scene color
        // target; this stage shades every pixel from the G-buffer and adds the result on top.
        if (this->deferred_lit_active()) {
            this->record_deferred_lighting_pass(command_buffer);
        }
        this->gpu_mark(command_buffer, gpu_mark_id::lighting_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // TAA resolve: blend the scene color with the reprojected history into the HDR target the post
        // chain reads, then copy the result into the history image for the next frame that renders this
        // swapchain image (see record_taa_pass).
        this->record_taa_pass(command_buffer);
        this->gpu_mark(command_buffer, gpu_mark_id::taa_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // G-buffer debug mode (an inspection of the stored data, never combined with the lighting
        // stage or TAA): turn one channel into a visible image in the HDR target.
        if (this->gbuffer_pass_active() && !this->deferred_lit_active()) {
            this->record_gbuffer_debug_pass(command_buffer);
        }
        this->gpu_mark(command_buffer, gpu_mark_id::main_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        this->ensure_post_descriptors();
        if (this->post_pipeline == std::nullopt || this->post_hdr_pipeline == std::nullopt || this->post_sets.size() <= this->current_image_index) {
            return false; // no post pipeline (creation failed): the HDR frame cannot be presented correctly
        }

        std::size_t const index = this->current_image_index;

        // HDR scene target -> fragment-shader read (the prefilter and the composite both read it)
        {
            std::array<VkImageMemoryBarrier2, 1> barriers = {vulkan::hdr_sampling_transition};
            barriers[0].image = vk.hdr_images[index];
            VkDependencyInfo const dependency_info = make_image_dependency_info(1, barriers.data());
            vkCmdPipelineBarrier2(command_buffer, &dependency_info);
        }

        VkViewport const full_viewport = {0.0f, 0.0f, static_cast<float>(vk.swap_chain_extent.width), static_cast<float>(vk.swap_chain_extent.height), 0.0f, 1.0f};
        VkRect2D const full_scissor = {{0, 0}, vk.swap_chain_extent};

        // level L target size: half the swapchain extent per level (min 1x1, matches core)
        auto const level_size = [&vk](uint32_t const level) {
            return VkExtent2D{std::max(1u, vk.swap_chain_extent.width >> (level + 1u)), std::max(1u, vk.swap_chain_extent.height >> (level + 1u))};
        };
        auto const barrier_to_read = [&command_buffer](VkImage const image) {
            std::array<VkImageMemoryBarrier2, 1> barriers = {vulkan::hdr_sampling_transition};
            barriers[0].image = image;
            VkDependencyInfo const dependency_info = make_image_dependency_info(1, barriers.data());
            vkCmdPipelineBarrier2(command_buffer, &dependency_info);
        };
        auto const barrier_to_color = [&command_buffer](VkImage const image) {
            std::array<VkImageMemoryBarrier2, 1> barriers = {vulkan::color_attachment_transition};
            barriers[0].image = image;
            VkDependencyInfo const dependency_info = make_image_dependency_info(1, barriers.data());
            vkCmdPipelineBarrier2(command_buffer, &dependency_info);
        };
        auto const run_fullscreen = [&](vk_pipeline const& pipeline, VkImageView const target, VkExtent2D const extent, VkDescriptorSet const set, float const mode) {
            VkClearValue clear = {};
            VkRenderingAttachmentInfo const attachment = make_color_attachment_info(target, clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
            VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, extent}, true, &attachment, nullptr);
            vkCmdBeginRendering(command_buffer, &rendering_info);
            pipeline.begin_pipeline(command_buffer);
            VkViewport const viewport = {0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
            VkRect2D const scissor = {{0, 0}, extent};
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);
            vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, this->post_pipeline_layout, 0, 1, &set, 0, nullptr);
            post_push_constants const push = {
                .exposure = this->exposure_scale,
                .bloom_intensity = this->bloom_intensity,
                .bloom_threshold = this->bloom_threshold,
                .mode = mode};
            vkCmdPushConstants(command_buffer, this->post_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
            vkCmdEndRendering(command_buffer);
        };

        // ---- bloom chain: bright-pass prefilter into level 0, then downsample level by level ----
        // every stage here renders into an R16F bloom level, so it needs the HDR-format pipeline.
        // Skipped entirely when the composite multiplies the bloom sum by 0 (intensity 0): the four
        // fullscreen passes would be pure cost. The levels are still moved to SHADER_READ_ONLY
        // (from UNDEFINED - their contents are dead and the composite's static use of those bindings
        // still requires a valid layout), because the composite samples them and multiplies by 0.
        // The G-buffer debug view forces the bloom weight to 0 as well: bloom is a display effect,
        // and a glow smeared over the channel being inspected is the opposite of a debug view (it
        // would also invent colors that are not in the G-buffer at all).
        // The G-buffer pass forces the bloom weight to 0 as well when its data is being displayed:
        // a display effect smeared over the channel being inspected is the opposite of a debug view
        // (it would also invent colors that are not in the G-buffer at all). The deferred LIT image
        // is a real image, so bloom stays on for it.
        bool const debug_view = this->gbuffer_pass_active() && !this->deferred_lit_active();
        float const bloom_intensity = debug_view ? 0.0f : this->bloom_intensity;
        bool const bloom_enabled = bloom_intensity > 0.0f;
        if (bloom_enabled) {
            barrier_to_color(vk.bloom_images[0][index]);
            run_fullscreen(*this->post_hdr_pipeline, vk.bloom_image_views[0][index], level_size(0), this->post_prefilter_sets[index], 0.0f);
            for (std::size_t level = 0; level < 3; ++level) {
                barrier_to_read(vk.bloom_images[level][index]);
                barrier_to_color(vk.bloom_images[level + 1][index]);
                run_fullscreen(*this->post_hdr_pipeline, vk.bloom_image_views[level + 1][index], level_size(static_cast<uint32_t>(level) + 1u), this->post_down_sets[index][level], 1.0f);
            }
            barrier_to_read(vk.bloom_images[3][index]);
        } else {
            for (uint32_t level = 0; level < vulkan::core::bloom_level_count; ++level) {
                std::array<VkImageMemoryBarrier2, 1> barriers = {vulkan::undefined_to_sampling_transition};
                barriers[0].image = vk.bloom_images[level][index];
                VkDependencyInfo const dependency_info = make_image_dependency_info(1, barriers.data());
                vkCmdPipelineBarrier2(command_buffer, &dependency_info);
            }
        }

        // GPU timing: the bloom chain ends here (a disabled bloom chain is just the layout fixups
        // above, so its interval reads ~0).
        this->gpu_mark(command_buffer, gpu_mark_id::bloom_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // ---- composite: HDR + weighted bloom levels -> exposure -> ACES -> display ----
        // With FXAA enabled the composite cannot write the swapchain (the FXAA pass has to read what
        // it produced, and a pass may not read the image it renders into), so it renders into the
        // LDR image instead and the FXAA pass finishes the frame. It also has to leave that image
        // GAMMA-ENCODED, because FXAA's luma thresholds are defined on display-referred data - which
        // is why encode_gamma is forced on for this pass even though the target is R16F (nothing
        // decodes it again on read).
        bool const fxaa = this->fxaa_on && this->post_fxaa_pipeline.has_value();
        VkImage const composite_image = fxaa ? vk.ldr_images[index] : vk.swap_chain_images[index];
        VkImageView const composite_view = fxaa ? vk.ldr_image_views[index] : vk.swap_chain_image_views[index];
        // ... and therefore also the HDR-format pipeline variant: a pipeline's declared color format
        // has to match the attachment it renders into, and the LDR image is R16F like the bloom
        // levels (the shader/descriptor side is identical - only mode and encode_gamma differ).
        vk_pipeline const& composite_pipeline = fxaa ? *this->post_hdr_pipeline : *this->post_pipeline;

        barrier_to_color(composite_image);
        VkClearValue clear = {};
        VkRenderingAttachmentInfo const color_attachment = make_color_attachment_info(composite_view, clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
        VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, vk.swap_chain_extent}, true, &color_attachment, nullptr);
        vkCmdBeginRendering(command_buffer, &rendering_info);
        composite_pipeline.begin_pipeline(command_buffer);
        vkCmdSetViewport(command_buffer, 0, 1, &full_viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &full_scissor);
        vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE); // the fullscreen triangle has no facing
        VkDescriptorSet const composite_set = this->post_sets[index];
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, this->post_pipeline_layout, 0, 1, &composite_set, 0, nullptr);
        post_push_constants const composite_push = {
            .exposure = this->exposure_scale,
            .bloom_intensity = bloom_intensity, // 0 while the G-buffer debug view is up (see above)
            .bloom_threshold = this->bloom_threshold,
            .mode = 2.0f,
            // Without FXAA the composite writes a LINEAR tonemapped image into an sRGB swapchain
            // attachment, which encodes it to display values in hardware, so the shader must NOT
            // apply gamma as well; only a non-sRGB (UNORM) swapchain needs the manual transfer
            // function. With FXAA the target is the R16F LDR image and the shader must encode.
            .encode_gamma = fxaa ? 1.0f : (is_srgb_format(vk.swap_chain_image_format) ? 0.0f : 1.0f),
            .fxaa_subpixel = this->fxaa_subpixel,
            .fxaa_edge_threshold = this->fxaa_edge_threshold};
        vkCmdPushConstants(command_buffer, this->post_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(composite_push), &composite_push);
        vkCmdDraw(command_buffer, 3, 1, 0, 0);
        // the debug overlay draws on the final 1x swapchain image (initialized with msaa = 1 and
        // no depth attachment, see enable_debug_gui). With FXAA it must wait for the FXAA pass, or
        // the UI text would be anti-aliased into mush.
        if (!fxaa && this->debug_gui_shown && this->debug_overlay.is_active()) {
            this->debug_overlay.record(command_buffer);
        }
        vkCmdEndRendering(command_buffer);

        // GPU timing: the composite (and the debug overlay, when it draws here) is done.
        this->gpu_mark(command_buffer, gpu_mark_id::composite_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        if (fxaa) {
            // ---- FXAA: gamma-encoded LDR image -> anti-aliased swapchain (+ overlay on top) ----
            barrier_to_read(vk.ldr_images[index]);
            barrier_to_color(vk.swap_chain_images[index]);
            VkRenderingAttachmentInfo const fxaa_attachment = make_color_attachment_info(vk.swap_chain_image_views[index], clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
            VkRenderingInfo const fxaa_rendering_info = make_rendering_info(0, {{0, 0}, vk.swap_chain_extent}, true, &fxaa_attachment, nullptr);
            vkCmdBeginRendering(command_buffer, &fxaa_rendering_info);
            this->post_fxaa_pipeline->begin_pipeline(command_buffer);
            vkCmdSetViewport(command_buffer, 0, 1, &full_viewport);
            vkCmdSetScissor(command_buffer, 0, 1, &full_scissor);
            vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, this->post_pipeline_layout, 0, 1, &composite_set, 0, nullptr);
            post_push_constants const fxaa_push = {
                .exposure = this->exposure_scale,
                .bloom_intensity = this->bloom_intensity,
                .bloom_threshold = this->bloom_threshold,
                .mode = 3.0f,
                // Same meaning as in the composite: 0 = the swapchain attachment encodes to display
                // values in hardware, so FXAA must hand it LINEAR values; 1 = the target is a UNORM
                // format and FXAA's own display-encoded result is what should be stored.
                .encode_gamma = is_srgb_format(vk.swap_chain_image_format) ? 0.0f : 1.0f,
                .fxaa_subpixel = this->fxaa_subpixel,
                .fxaa_edge_threshold = this->fxaa_edge_threshold};
            vkCmdPushConstants(command_buffer, this->post_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(fxaa_push), &fxaa_push);
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
            if (this->debug_gui_shown && this->debug_overlay.is_active()) {
                this->debug_overlay.record(command_buffer);
            }
            vkCmdEndRendering(command_buffer);
        }
        // GPU timing: the FXAA pass (and the overlay it carries when it is the last writer) is done.
        // Without FXAA the composite already ended the frame's display work, so this interval is ~0.
        this->gpu_mark(command_buffer, gpu_mark_id::fxaa_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // true in both paths: the FXAA pass (or the composite, when FXAA is off) wrote the swapchain
        return true;
    }
    frame_status runtime::end_recording() {
        core& vk = this->vulkan_core;
        vk_command_buffer& command_buffer = this->command_buffers[static_cast<uint32_t>(vk.current_frame)];

        // close the scene rendering instance and run the post-process pass (exposure/tonemap).
        // The return value says whether a fullscreen pass actually wrote the swapchain image: only
        // then is it in COLOR_ATTACHMENT_OPTIMAL and only then does it hold this frame's result.
        bool const post_wrote_swapchain = this->record_post_process(*command_buffer);
        // Screenshot: while the post pass wrote the swapchain image it is still in
        // COLOR_ATTACHMENT_OPTIMAL and still owned by this frame - the only point where a read-back
        // copy is legal. Doing it here (rather than after the present, as the old path did) also
        // means the capture needs no extra submit, no re-acquire and no layout hand-back to the WSI.
        if (post_wrote_swapchain && this->screenshot_requested) {
            this->record_screenshot_copy(*command_buffer);
            if (this->screenshot_pending) {
                this->screenshot_requested = false; // served; a failed copy stays pending for a retry
            }
        }
        // Dynamic rendering has no render pass finalLayout to hand the image back to the
        // presentation engine: transition the swapchain image to PRESENT_SRC_KHR explicitly.
        // With MSAA the resolve target ends up in resolveImageLayout (COLOR_ATTACHMENT_OPTIMAL),
        // so the barrier is needed on both the direct-render and the resolve paths. When the post
        // pass was skipped the image never entered COLOR_ATTACHMENT_OPTIMAL, and claiming that old
        // layout would be a lie (validation: "oldLayout is not matching with the current layout"):
        // transition from UNDEFINED instead - the frame has no content to preserve anyway.
        VkImageMemoryBarrier2 present_barrier = post_wrote_swapchain ? present_transition : vulkan::undefined_to_present_transition;
        present_barrier.image = vk.swap_chain_images[this->current_image_index];

        VkDependencyInfo const dependency_info = make_image_dependency_info(1, &present_barrier);
        vkCmdPipelineBarrier2(*command_buffer, &dependency_info);
        // GPU timing: last mark of the frame. The interval it closes is everything after the FXAA
        // (or composite) pass - the screenshot read-back copy and the present barrier.
        this->gpu_mark(*command_buffer, gpu_mark_id::frame_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
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
            this->on_swapchain_recreated();
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
        // rendering (the backend is initialized with UseDynamicRendering=true).
        vulkan::core const& vk = this->vulkan_core;
        gui::gui_create_info info = {};
        info.window = vk.window;
        info.instance = vk.instance;
        info.physical_device = vk.physical_device;
        info.device = vk.device;
        info.graphics_queue_family = vk.graphics_family_index;
        info.graphics_queue = vk.graphics_queue;
        info.color_format = vk.swap_chain_image_format;
        info.depth_format = VK_FORMAT_UNDEFINED;   // the post/gui pass has no depth attachment
        info.msaa_samples = VK_SAMPLE_COUNT_1_BIT; // the overlay draws on the 1x swapchain after the post pass
        info.frames_in_flight = static_cast<uint32_t>(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        return this->debug_overlay.init(info);
    }

    bool runtime::debug_gui_active() const noexcept {
        return this->debug_overlay.is_active();
    }

    bool runtime::debug_gui_visible() const noexcept {
        return this->debug_overlay.is_active() && this->debug_gui_shown;
    }

    bool runtime::debug_gui_wants_mouse() const noexcept {
        return this->debug_gui_shown && this->debug_overlay.is_active() && this->debug_overlay.wants_mouse();
    }

    void runtime::set_debug_gui_visible(bool const visible) noexcept {
        this->debug_gui_shown = visible;
    }

    gui::gui_content& runtime::debug_gui() noexcept {
        return this->debug_overlay;
    }

    std::expected<void, std::string> runtime::make_pipeline(std::string_view pipeline_name, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        {
            // unique lock around the duplicate check + registry append: a concurrent reader
            // (recording worker) must never observe a half-inserted map / name table
            std::unique_lock const lock(this->access_mutex);
            if (this->pipelines.contains(pipeline_name)) {
                return fail(std::string("pipeline '") + std::string(pipeline_name) + "' already exists");
            }
        }
        // GPU pipeline creation is expensive and touches no shared registry state: build it
        // OUTSIDE the lock so a reader is never blocked by shader compilation.
        auto make_result = this->vulkan_core.make_pipeline(vertex_shader_code, fragment_shader_code);
        if (!make_result) {
            return fail(make_result.error());
        }
        {
            std::unique_lock const lock(this->access_mutex);
            this->pipelines.emplace(pipeline_name, std::move(make_result).value());
            this->pipeline_names.emplace_back(pipeline_name); // stable name table (see runtime.cppm)
            if (this->default_pipeline_name.empty()) {
                this->default_pipeline_name = pipeline_name; // first pipeline is the implicit default
            }
        }
        return {};
    }

    void runtime::set_default_pipeline(std::string_view const pipeline_name) {
        std::unique_lock const lock(this->access_mutex);
        if (this->pipelines.contains(pipeline_name)) {
            this->default_pipeline_name = pipeline_name;
        }
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

    std::expected<void, std::string> runtime::make_cluster_pipeline(std::span<unsigned char const> const compute_shader_code) {
        auto result = this->vulkan_core.make_cluster_pipeline(compute_shader_code);
        if (!result) {
            return std::unexpected(std::string(result.error()));
        }
        // no viewport/scissor: a compute dispatch binds no graphics state, so the frame path's
        // viewport resync (which walks the named pipeline cache) never touches this pipeline
        this->cluster_pipeline = std::move(result).value();
        return {};
    }

    void runtime::set_clustered_lights(bool const enabled) noexcept {
        // CPU-side only, like set_brdf_model: the flag rides light_state's cluster_grid.w lane and
        // pace_and_acquire() copies light_state into the paced slot's buffer, so the next frame's
        // cluster dispatch and shading both see it (no in-flight buffer is touched).
        this->clustered_lights = enabled;
    }

    void runtime::record_cluster_pass(VkCommandBuffer const command_buffer) {
        core& vk = this->vulkan_core;
        if (!this->cluster_pipeline.has_value() || !this->clustered_lights || !this->scene_set_created) {
            return;
        }
        uint32_t const tiles_x = this->cluster_tiles_x;
        uint32_t const tiles_y = this->cluster_tiles_y;
        uint32_t const cluster_count = tiles_x * tiles_y * vulkan::cluster_slice_count;
        if (cluster_count == 0) {
            return; // no paced frame yet (the grid comes from the swapchain extent)
        }
        uint32_t const frame_slot = static_cast<uint32_t>(vk.current_frame);
        if (frame_slot >= this->cluster_count_buffers.size()) {
            return;
        }
        // The dispatch reads the frame's OWN scene set (the paced slot's camera/light UBOs) and
        // writes the same slot's cluster buffers: a compute stage is not part of a rendering
        // instance, so this records before vkCmdBeginRendering.
        VkDescriptorSet const scene_set_handle = *this->scene_sets[static_cast<std::size_t>(frame_slot)];
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk.scene_pipeline_layout, 0, 1, &scene_set_handle, 0, nullptr);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->cluster_pipeline->get_pipeline());
        constexpr uint32_t group_size = 64; // matches local_size_x in shaders/light_cluster.comp
        vkCmdDispatch(command_buffer, (cluster_count + group_size - 1) / group_size, 1, 1);

        // Hand the two buffers to the fragment stages that read them later in this submission
        // (forward shading inside the main instance, and the deferred lighting pass): a compute
        // SHADER_WRITE is not visible to a later SHADER_READ without this barrier. One barrier per
        // buffer (VkBufferMemoryBarrier2 covers a single buffer).
        std::array<VkBufferMemoryBarrier2, 2> barriers = {};
        auto const* count_detail = vk.vma.get_buffer_detail(this->cluster_count_buffers[static_cast<std::size_t>(frame_slot)].handle());
        auto const* index_detail = vk.vma.get_buffer_detail(this->cluster_index_buffers[static_cast<std::size_t>(frame_slot)].handle());
        if (count_detail == nullptr || index_detail == nullptr) {
            return;
        }
        barriers[0].buffer = count_detail->buffer;
        barriers[1].buffer = index_detail->buffer;
        for (VkBufferMemoryBarrier2& barrier : barriers) {
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            // the shader reads counts/indices as storage buffers, not as sampled images
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.offset = 0;
            barrier.size = VK_WHOLE_SIZE;
        }
        VkDependencyInfo dependency = {};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
        dependency.pBufferMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(command_buffer, &dependency);
    }

    // Tighten the directional shadow frustum to the camera's own view frustum every frame. One
    // 2048^2 map cannot cover a whole scene and still resolve a thin caster: the orthographic box
    // therefore follows the camera - its xy footprint is the camera frustum's, and the depth range
    // covers the frustum corners plus every caster whose shadow column can reach the view (a wall
    // behind the camera included). The box center is snapped to the texel grid, which is what keeps
    // the shadow edges from crawling while the camera moves.
    bool runtime::instanced_world_aabb(primitive const& leaf, glm::vec3& wmin, glm::vec3& wmax) const {
        // Instanced draws have no single world AABB (primitive::has_bounds is false for them), but
        // their bounds are perfectly computable: pbr.vert uses instances[instance_base + i] as the
        // whole world matrix (push flag bit0), so the union over the instance slice of the SOURCE
        // geometry's local AABB is exact. Without this the fit had to assume "anywhere in the scene"
        // and a single instanced draw (the grid stress mode places instances outside the scene
        // bounds) both lost its own shadows and coarsened everyone else's.
        if ((leaf.push.flags & 1u) == 0u || this->instance_mapped == nullptr) {
            return false;
        }
        auto const& instanced = static_cast<instanced_draw_primitive const&>(leaf);
        primitive const* const source = instanced.source;
        if (source == nullptr || !source->has_bounds || instanced.instance_count == 0) {
            return false;
        }
        std::size_t const base = leaf.push.instance_base;
        if (base + instanced.instance_count > vulkan::instance_capacity) {
            return false; // slice outside the shared buffer: cannot read it
        }
        auto const* matrices = static_cast<glm::mat4 const*>(this->instance_mapped);
        glm::vec3 const lo = source->local_aabb_min;
        glm::vec3 const hi = source->local_aabb_max;
        wmin = glm::vec3(std::numeric_limits<float>::max());
        wmax = glm::vec3(std::numeric_limits<float>::lowest());
        for (uint32_t i = 0; i < instanced.instance_count; ++i) {
            glm::mat4 const& model = matrices[base + i];
            for (int corner = 0; corner < 8; ++corner) {
                glm::vec3 const p((corner & 1) != 0 ? hi.x : lo.x,
                                  (corner & 2) != 0 ? hi.y : lo.y,
                                  (corner & 4) != 0 ? hi.z : lo.z);
                glm::vec3 const world = glm::vec3(model * glm::vec4(p, 1.0f));
                wmin = glm::min(wmin, world);
                wmax = glm::max(wmax, world);
            }
        }
        return true;
    }

    void runtime::update_shadow_frustum() {
        if (!this->shadows_enabled) {
            return;
        }
        // Refit only when the result can change: this walks every leaf and transforms 8 corners per
        // caster (tens of thousands of transforms on a heavy scene), which is exactly the cost the
        // BVH caster cull exists to avoid. The camera matrices are the fit's only inputs besides the
        // scene, so comparing them (plus the scene-changed flag) is the complete condition.
        //
        // The UNJITTERED view-projection is what this must key on and fit to: the TAA jitter is a
        // sub-pixel rendering offset, and letting it into the fit (a) misses the cache on every single
        // frame and (b) - the real bug - re-quantizes the light-space box to whole texels every frame,
        // so the shadow map's texel grid alternates between two alignments. A surface at a grazing
        // light angle then reads as shadowed in one alignment and lit in the other, and the TAA
        // history averages the flicker into a dark band across it.
        if (this->shadow_frustum_valid && !this->bvh_dirty &&
            this->shadow_fit_view_proj == this->current_ubo.view_proj_unjittered) {
            return;
        }
        this->shadow_fit_view_proj = this->current_ubo.view_proj_unjittered;
        this->shadow_frustum_valid = true;
        if (!(this->current_aspect > 0.0f) || this->current_ubo.proj[2][2] == 0.0f) {
            // Degenerate camera (the very first frames, before the swapchain has an extent): the
            // frustum corners would come out of an inverse of a singular matrix and the fit would be
            // garbage (measured: a light-space z span of 1631 for two unrelated scenes). Keep the
            // enable_shadows() default this frame and try again next frame.
            this->shadow_frustum_valid = false;
            return;
        }

        float const map_size = static_cast<float>(vulkan::runtime::shadow_map_size);
        glm::vec3 const light_dir = glm::normalize(glm::vec3(this->light_state.light_dir));
        glm::vec3 const up(0.0f, 1.0f, 0.0f);
        // Light space is a pure ROTATION here (each cascade's translation comes from its own box
        // center), so one rotation serves every cascade - and with it one caster AABB pass per frame
        // instead of one per cascade.
        glm::mat4 const light_rotation = glm::lookAt(glm::vec3(0.0f), -light_dir, up);

        // ---- the camera's usable depth range, then the cascade splits over it ----
        // The camera's own projection carries a deliberately generous far plane (see
        // make_orbit_camera_ubo: max(100, distance + 2r, 8 * distance)) so that zooming in never clips
        // the scene - but fitting the SHADOW volume to that whole range throws the map away on empty
        // space AND (the actual bug this originally fixed) pushes the up-light end of the fit behind
        // the light camera's near plane, so the roof and upper walls were clipped out of the depth map
        // entirely and the sun poured straight through them. Fit only the part of the view volume that
        // can hold the scene: everything lies within |eye - scene centre| + scene_radius.
        // The PROJECTION, not the view*proj product: near/far live in its z row - for a RH_ZO
        // perspective matrix proj[2][2] = far / (near - far) and proj[3][2] = -(far * near) /
        // (far - near) - and the sub-frustum corners below come from its inverse. Extracting them from
        // the product yields nonsense, because the view rotation mixes the rows: measured, it turned
        // the near plane into -22 and made every cascade split NaN.
        glm::mat4 const base_proj = this->current_proj_unjittered;
        float const camera_near = base_proj[3][2] / base_proj[2][2];
        float const camera_far = base_proj[2][2] * camera_near / (1.0f + base_proj[2][2]);
        float const scene_far = glm::distance(glm::vec3(this->current_ubo.camera_pos), this->shadow_scene_center) + this->scene_radius;
        float const split_near = camera_near;
        float const split_far = std::max(std::min(camera_far, scene_far), camera_near * 2.0f);
        uint32_t const cascades = std::clamp(this->shadow_cascades, 1u, vulkan::max_shadow_cascades);
        // Practical split scheme (Zhang et al.): a logarithmic and a uniform split blended by lambda.
        // Pure logarithmic puts almost all the resolution in the first few metres (the camera near
        // plane is 0.1), pure uniform wastes the near range - the blend is what real-time shadows use.
        // lambda is fixed at 0.75: higher favors the near field, lower the far one.
        constexpr float split_lambda = 0.75f;
        auto const split_distance = [&](float const t) {
            float const logarithmic = split_near * std::pow(split_far / split_near, t);
            float const uniform = split_near + (split_far - split_near) * t;
            return split_lambda * logarithmic + (1.0f - split_lambda) * uniform;
        };

        // ---- every caster's light-space AABB, computed ONCE for all cascades ----
        // Casters outside the view still cast into it (a wall behind the camera): fitting only the
        // frustum corners clipped them and sunlight leaked through. Every scene leaf whose light-space
        // xy overlaps a cascade's footprint therefore contributes its light-space DEPTH range to that
        // cascade's fit - under the orthographic light a caster's shadow lands at the caster's own
        // light-space xy, so that overlap test is exact for "can this shadow land inside the view".
        // Geometry whose shadow cannot reach the view (e.g. the floor slab behind the camera) is left
        // out entirely.
        std::vector<std::pair<glm::vec3, glm::vec3>> caster_boxes; // light-space min/max per leaf
        bool unbounded_caster = false;
        if (this->bound_scene != nullptr) {
            this->shadow_caster_scratch.clear();
            for (scene_tree::scene_node const& root : this->bound_scene->roots) {
                collect_leaf_primitives(root, this->shadow_caster_scratch);
            }
            caster_boxes.reserve(this->shadow_caster_scratch.size());
            for (primitive const* const leaf : this->shadow_caster_scratch) {
                if (leaf == nullptr) {
                    continue;
                }
                glm::vec3 wmin = {};
                glm::vec3 wmax = {};
                if (leaf->has_bounds) {
                    std::tie(wmin, wmax) = leaf->world_aabb();
                } else if (instanced_world_aabb(*leaf, wmin, wmax)) {
                    // exact bounds, computed from this leaf's own instance matrices
                } else {
                    // a caster whose geometry we genuinely cannot bound: fall back to the scene
                    // sphere below, which is sound but costs resolution
                    unbounded_caster = true;
                    continue;
                }
                glm::vec3 caster_min(std::numeric_limits<float>::max());
                glm::vec3 caster_max(std::numeric_limits<float>::lowest());
                for (int corner = 0; corner < 8; ++corner) {
                    glm::vec3 const p((corner & 1) != 0 ? wmax.x : wmin.x,
                                      (corner & 2) != 0 ? wmax.y : wmin.y,
                                      (corner & 4) != 0 ? wmax.z : wmin.z);
                    glm::vec3 const ls = glm::vec3(light_rotation * glm::vec4(p, 1.0f));
                    caster_min = glm::min(caster_min, ls);
                    caster_max = glm::max(caster_max, ls);
                }
                caster_boxes.emplace_back(caster_min, caster_max);
            }
        }
        // the scene sphere's light-space box, for the unbounded-caster fallback (used by every cascade)
        std::pair<glm::vec3, glm::vec3> scene_sphere_ls = {glm::vec3(std::numeric_limits<float>::max()), glm::vec3(std::numeric_limits<float>::lowest())};
        for (int corner = 0; corner < 8; ++corner) {
            glm::vec3 const p = this->shadow_scene_center +
                                glm::vec3((corner & 1) != 0 ? this->scene_radius : -this->scene_radius,
                                          (corner & 2) != 0 ? this->scene_radius : -this->scene_radius,
                                          (corner & 4) != 0 ? this->scene_radius : -this->scene_radius);
            glm::vec3 const ls = glm::vec3(light_rotation * glm::vec4(p, 1.0f));
            scene_sphere_ls.first = glm::min(scene_sphere_ls.first, ls);
            scene_sphere_ls.second = glm::max(scene_sphere_ls.second, ls);
        }

        // ---- one fit per cascade ----
        float last_texel_world = 0.0f;
        for (uint32_t cascade = 0; cascade < cascades; ++cascade) {
            float const cascade_near = cascade == 0 ? split_near : split_distance(static_cast<float>(cascade) / static_cast<float>(cascades));
            float const cascade_far = split_distance(static_cast<float>(cascade + 1) / static_cast<float>(cascades));

            // the camera projection restricted to THIS cascade's depth range (same x/y scaling)
            glm::mat4 fit_proj = base_proj;
            fit_proj[2][2] = cascade_far / (cascade_near - cascade_far);
            fit_proj[3][2] = -(cascade_far * cascade_near) / (cascade_far - cascade_near);
            glm::mat4 const inverse_view_proj = glm::inverse(fit_proj * this->current_ubo.view);

            // the sub-frustum's corners, each extended up-light by the caster reach: a caster that far
            // up-sun can still throw its shadow into this cascade
            std::array<glm::vec3, 16> points = {};
            std::size_t count = 0;
            for (int zi = 0; zi < 2; ++zi) {
                for (int yi = 0; yi < 2; ++yi) {
                    for (int xi = 0; xi < 2; ++xi) {
                        glm::vec4 const clip(xi == 0 ? -1.0f : 1.0f, yi == 0 ? -1.0f : 1.0f, zi == 0 ? 0.0f : 1.0f, 1.0f);
                        glm::vec4 const world = inverse_view_proj * clip;
                        glm::vec3 const corner = glm::vec3(world) / world.w;
                        points[count++] = corner;
                        points[count++] = corner + light_dir * this->shadow_caster_extent;
                    }
                }
            }
            glm::vec3 min_ls(std::numeric_limits<float>::max());
            glm::vec3 max_ls(std::numeric_limits<float>::lowest());
            for (glm::vec3 const& point : points) {
                glm::vec3 const ls = glm::vec3(light_rotation * glm::vec4(point, 1.0f));
                min_ls = glm::min(min_ls, ls);
                max_ls = glm::max(max_ls, ls);
            }

            // merge the casters that can shadow this cascade (see the note where they were collected)
            float const cone_min_x = min_ls.x;
            float const cone_max_x = max_ls.x;
            float const cone_min_y = min_ls.y;
            float const cone_max_y = max_ls.y;
            for (auto const& [caster_min, caster_max] : caster_boxes) {
                bool const overlaps_view = caster_max.x >= cone_min_x && caster_min.x <= cone_max_x && caster_max.y >= cone_min_y && caster_min.y <= cone_max_y;
                if (!overlaps_view) {
                    continue; // this caster's shadow cannot land inside this cascade
                }
                // Only the merged DEPTH (light-space z) may grow past the cascade footprint: an
                // orthographic sun drops a caster's shadow at the caster's own light-space xy, and only
                // xy inside the view is ever sampled - so clamping the merged xy to the cone keeps the
                // box (and thus the texel density) as tight as the cascade fit, while the full z range
                // makes sure the caster's depth actually reaches the map. Merging the raw AABB instead
                // let one big floor slab inflate the box to the whole scene; leaving z alone clipped the
                // caster and leaked the sun through it.
                min_ls = glm::min(min_ls, glm::vec3(std::clamp(caster_min.x, cone_min_x, cone_max_x),
                                                    std::clamp(caster_min.y, cone_min_y, cone_max_y),
                                                    caster_min.z));
                max_ls = glm::max(max_ls, glm::vec3(std::clamp(caster_max.x, cone_min_x, cone_max_x),
                                                    std::clamp(caster_max.y, cone_min_y, cone_max_y),
                                                    caster_max.z));
            }
            if (unbounded_caster) {
                // A caster we cannot bound can be anywhere in the scene, so the only sound bound is
                // the whole scene sphere - but only its DEPTH may enter the fit: clamping the sphere's
                // xy to the cascade footprint keeps the box tight (the same argument as above), while
                // merging the raw sphere restored a box of scene size and made every thin caster's
                // shadow dissolve in the coarser texels.
                min_ls = glm::min(min_ls, glm::vec3(std::clamp(scene_sphere_ls.first.x, cone_min_x, cone_max_x),
                                                    std::clamp(scene_sphere_ls.first.y, cone_min_y, cone_max_y),
                                                    scene_sphere_ls.first.z));
                max_ls = glm::max(max_ls, glm::vec3(std::clamp(scene_sphere_ls.second.x, cone_min_x, cone_max_x),
                                                    std::clamp(scene_sphere_ls.second.y, cone_min_y, cone_max_y),
                                                    scene_sphere_ls.second.z));
            }

            // square the box (isotropic resolution), quantize its size to whole texels and snap its
            // center to the texel grid. Snapping the CENTER alone keeps the grid aligned while the
            // camera translates, but the box SIZE follows the fitted footprint continuously, so every
            // camera move rescaled the whole map and the shadow edges crawled anyway; rounding the size
            // up to a texel multiple means small moves keep the same texel size. The extra texel is
            // margin: without it the visible footprint touched the very edge of the map, where the
            // outermost half texel used to fall outside the box and casters there had no shadow edge.
            float const extent = std::max(max_ls.x - min_ls.x, max_ls.y - min_ls.y);
            float const half_raw = std::max(extent * 0.5f, 0.001f);
            float const texel_raw = (2.0f * half_raw) / map_size;
            float const half = std::max(std::ceil(half_raw / texel_raw) * texel_raw + texel_raw, 0.001f);
            float const texel = (2.0f * half) / map_size;
            glm::vec3 center_ls = (min_ls + max_ls) * 0.5f;
            center_ls.x = std::floor(center_ls.x / texel) * texel;
            center_ls.y = std::floor(center_ls.y / texel) * texel;
            glm::vec3 const center_ws = glm::vec3(glm::inverse(light_rotation) * glm::vec4(center_ls, 1.0f));

            // light camera far enough up-sun to see every fitted point, then fit near/far to the SAME
            // fitted range. Deriving near/far from the camera frustum corners alone clipped away every
            // caster merged above that sat further up-light than the corners: its depth never reached
            // the map, so the sun leaked straight through it. light_rotation is a pure rotation and
            // lookAt(.., -light_dir, ..) makes light-space z = dot(light_dir, p), so along the light
            // view (eye = center_ws + light_dir * distance) a point's depth is
            // center_ls.z + distance - ls_z - the extremes therefore come straight from min_ls/max_ls.z.
            //
            // distance comes from the fitted DEPTH SPAN, not from the scene radius: the eye has to sit
            // half a span up-light of the box centre for the box centre's plane to be in front of it,
            // and a scene-radius estimate can be smaller than that span (a deep fit pushed the up-light
            // end behind the eye, the near plane clamped to its 0.05 minimum and clipped those casters).
            float const half_span_z = 0.5f * (max_ls.z - min_ls.z);
            float const distance = half_span_z + std::max(1.0f, this->shadow_caster_extent);
            glm::vec3 const eye = center_ws + light_dir * distance;
            glm::mat4 const view = glm::lookAt(eye, center_ws, up);
            float const near_plane = std::max(distance + center_ls.z - max_ls.z - 1.0f, 0.05f);
            float const far_plane = distance + center_ls.z - min_ls.z + 1.0f;

            glm::mat4 proj = glm::orthoRH_ZO(-half, half, -half, half, near_plane, far_plane);
            proj[1][1] *= -1.0f; // same y-flip convention as the camera projection

            this->light_state.light_view_proj[cascade] = proj * view;
            this->light_state.cascade_texel_world[cascade] = (2.0f * half) / map_size;
            this->light_state.cascade_splits[cascade] = cascade_far;
            last_texel_world = (2.0f * half) / map_size;
        }

        // Unused cascade lanes must not hold garbage: a shader that samples a lane beyond the active
        // count (it does not - every path checks cascade_count first) would otherwise read whatever the
        // previous frame's fit left there. Repeating the last fitted matrix is the honest value.
        for (uint32_t cascade = cascades; cascade < vulkan::max_shadow_cascades; ++cascade) {
            this->light_state.light_view_proj[cascade] = this->light_state.light_view_proj[cascades - 1];
            this->light_state.cascade_texel_world[cascade] = last_texel_world;
            this->light_state.cascade_splits[cascade] = this->light_state.cascade_splits[cascades - 1];
        }
        this->light_state.cascade_count = static_cast<float>(cascades);
        this->light_state.light_dir = glm::vec4(light_dir, 1.0f / map_size);

        if (!this->shadow_cascade_logged) {
            this->shadow_cascade_logged = true;
            utility::log("shadow cascades: {} over the view range [{:.2f}, {:.2f}] (splits {:.2f}/{:.2f}/{:.2f}), texel world sizes {:.4f}/{:.4f}/{:.4f}/{:.4f}",
                         cascades,
                         split_near,
                         split_far,
                         this->light_state.cascade_splits[0],
                         this->light_state.cascade_splits[1],
                         this->light_state.cascade_splits[2],
                         this->light_state.cascade_texel_world[0],
                         this->light_state.cascade_texel_world[1],
                         this->light_state.cascade_texel_world[2],
                         this->light_state.cascade_texel_world[3]);
        }
    }
    void runtime::set_shadow_cascades(uint32_t const cascades) noexcept {
        uint32_t const clamped = std::clamp(cascades, 1u, vulkan::max_shadow_cascades);
        if (clamped == this->shadow_cascades) {
            return;
        }
        this->shadow_cascades = clamped;
        // a different cascade layout invalidates the cached fit (and the one-time density log)
        this->shadow_frustum_valid = false;
        this->shadow_cascade_logged = false;
        utility::log("shadow cascades set to {}", clamped);
    }

    void runtime::set_shadow_cascade_blend(float const blend) noexcept {
        // a band wider than half a cascade would reach back into the previous one
        this->shadow_cascade_blend = std::clamp(blend, 0.0f, 0.5f);
        this->light_state.cascade_blend = this->shadow_cascade_blend;
    }
    void runtime::enable_shadows(glm::vec3 const& scene_center, float const scene_radius) {
        // Remember the scene extent even if shadow setup below fails: the camera far plane
        // (make_orbit_camera_ubo) needs it to keep the whole scene visible when zooming in.
        this->scene_radius = scene_radius;
        // shadow caster culling (begin_recording): casters up to ~1/8 of the scene radius
        // up-light of the camera frustum can still throw a shadow into the view
        this->shadow_caster_extent = std::max(1.0f, scene_radius * 0.125f);
        // remembered for update_shadow_frustum's fallback fit (a caster without its own world AABB)
        this->shadow_scene_center = scene_center;
        // a new light setup invalidates the cached fit (see update_shadow_frustum)
        this->shadow_frustum_valid = false;
        if (!this->shadow_pipeline || this->light_mapped.empty()) {
            utility::log("shadow mapping not enabled (no shadow pipeline / light buffer)");
            return;
        }
        // light UBO: orthographic light view-proj framing the scene + the light direction.
        // Fill the CPU-side mirror only - pace_and_acquire copies it into every slot's own
        // light buffer as each slot is paced (nothing here touches mapped memory directly).
        this->light_state = make_directional_light_ubo(scene_center, scene_radius, static_cast<float>(vulkan::runtime::shadow_map_size));
        // the cascade settings are the runtime's, not the UBO builder's: re-apply them over the defaults
        this->light_state.cascade_count = static_cast<float>(std::clamp(this->shadow_cascades, 1u, vulkan::max_shadow_cascades));
        this->light_state.cascade_blend = this->shadow_cascade_blend;
        // respect the current GUI toggle: the flag in the slot's buffer tells pbr.frag whether
        // the depth map was rendered this frame
        this->light_state.shadow_enabled = this->shadow_enabled ? 1.0f : 0.0f;
        this->shadows_enabled = true;
        utility::log("shadow mapping enabled: light frustum center ({:.2f}, {:.2f}, {:.2f}), radius {:.2f}",
                     scene_center.x, scene_center.y, scene_center.z, scene_radius);
    }

    void runtime::set_shadow_enabled(bool const enabled) {
        if (this->shadow_enabled == enabled) {
            return;
        }
        this->shadow_enabled = enabled;
        // Only the CPU-side flag changes here: the frame loop copies light_state into the paced
        // slot's OWN light buffer (pace_and_acquire), so this call is safe at ANY time (GUI
        // callbacks included) - it never touches memory a frame in flight may be reading. The
        // light UBO's shadow_enabled flag drives pbr.frag: when shadows are off the shader
        // skips calc_shadow entirely (fully lit), so no shadow-map clearing is needed - this
        // avoids the per-frame-slot double-buffer race that clearing once could not fix.
        this->light_state.shadow_enabled = (enabled && this->shadows_enabled) ? 1.0f : 0.0f;
        utility::log("shadow pass {}", enabled ? "enabled" : "disabled");
    }

    void runtime::set_brdf_model(int const model) noexcept {
        // CPU-side only, like set_shadow_enabled: pace_and_acquire() copies light_state (which
        // carries the selected models in the UBO's std140 padding) into the paced slot's light
        // buffer every frame, so flipping the model mid-run never races an in-flight frame.
        this->light_state.brdf_model = static_cast<float>(std::clamp(model, 0, 3));
    }

    void runtime::set_diffuse_model(int const model) noexcept {
        this->light_state.diffuse_model = static_cast<float>(std::clamp(model, 0, 1));
    }

    void runtime::set_exposure(float const exposure) noexcept {
        // CPU-side only (same rule as set_brdf_model): remembered here, written into the light
        // UBO's light_count.y lane by pace_and_acquire() and pushed to the skybox pass
        this->exposure_scale = std::clamp(exposure, 0.05f, 20.0f);
    }

    float runtime::exposure() const noexcept {
        return this->exposure_scale;
    }

    void runtime::set_toon_shading(float const steps, float const softness) noexcept {
        // 0 disables the cel path (plain PBR); the shader rounds to whole bands
        this->toon_steps = steps < 1.5f ? 0.0f : std::round(std::clamp(steps, 2.0f, 8.0f));
        this->toon_softness = std::clamp(softness, 0.01f, 0.5f);
    }

    void runtime::set_bloom(float const intensity, float const threshold) noexcept {
        this->bloom_intensity = std::clamp(intensity, 0.0f, 4.0f);
        // above ~0.75 the scene has almost no pixel brighter than the threshold, so nothing
        // would glow; the gui slider is limited to the same visible range
        this->bloom_threshold = std::clamp(threshold, 0.0f, 0.75f);
    }

    void runtime::set_fxaa(bool const enabled, float const subpixel, float const edge_threshold) noexcept {
        // no pipeline = the shader was never loaded: keep the flag off rather than silently
        // rendering the composite into an LDR image nothing will ever read back
        this->fxaa_on = enabled && this->post_fxaa_pipeline.has_value();
        if (enabled && !this->fxaa_on) {
            utility::log("fxaa: requested but no fxaa pipeline exists (is fxaa.frag.spv present?)");
        }
        this->fxaa_subpixel = std::clamp(subpixel, 0.0f, 1.0f);
        // below ~0.05 every shaded gradient counts as an edge (the whole image gets softened),
        // above ~0.5 almost nothing does; the gui slider uses the same range
        this->fxaa_edge_threshold = std::clamp(edge_threshold, 0.05f, 0.5f);
    }

    std::expected<runtime::frame_image, std::string> runtime::acquire_current_frame_image() {
        core& vk = this->vulkan_core;
        if (vk.swap_chain == VK_NULL_HANDLE || vk.swap_chain_images.empty()) {
            return std::unexpected(std::string("screenshot: no swapchain image available"));
        }

        VkFormat const format = vk.swap_chain_image_format;
        bool const bgra = format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_B8G8R8A8_UNORM;
        bool const rgba = format == VK_FORMAT_R8G8B8A8_SRGB || format == VK_FORMAT_R8G8B8A8_UNORM;
        if (!bgra && !rgba) {
            return std::unexpected(std::string("screenshot: unsupported swapchain format (need 8-bit RGBA/BGRA)"));
        }

        // The pixels come from the persistent read-back buffer that record_screenshot_copy() filled
        // while the frame was being recorded (see the class docs): by the time the caller asks, the
        // frame has been submitted, so one wait for the GPU is all that is left to do.
        if (!this->screenshot_readback.valid() || this->screenshot_readback_mapped == nullptr) {
            return std::unexpected(std::string("screenshot: no captured frame (the read-back copy was never recorded)"));
        }
        VkExtent2D const extent = this->screenshot_readback_extent;
        VkDeviceSize const buffer_size = static_cast<VkDeviceSize>(extent.width) * static_cast<VkDeviceSize>(extent.height) * 4u;
        vk.wait_idle();

        frame_image result = {};
        result.width = extent.width;
        result.height = extent.height;
        result.rgba.resize(static_cast<std::size_t>(buffer_size));
        auto const* source = static_cast<unsigned char const*>(this->screenshot_readback_mapped);
        if (bgra) {
            // the swapchain is BGRA (sRGB); the PNG writer wants RGBA
            for (std::size_t i = 0; i < result.rgba.size(); i += 4) {
                result.rgba[i + 0] = source[i + 2]; // R
                result.rgba[i + 1] = source[i + 1]; // G
                result.rgba[i + 2] = source[i + 0]; // B
                result.rgba[i + 3] = source[i + 3]; // A
            }
        } else {
            std::memcpy(result.rgba.data(), source, static_cast<std::size_t>(buffer_size));
        }
        return result;
    }

    void* runtime::ensure_screenshot_readback(VkExtent2D const extent) {
        if (extent.width == 0 || extent.height == 0) {
            return nullptr;
        }
        VkDeviceSize const needed = static_cast<VkDeviceSize>(extent.width) * static_cast<VkDeviceSize>(extent.height) * 4u;
        if (this->screenshot_readback.valid() && this->screenshot_readback_size >= needed && this->screenshot_readback_mapped != nullptr) {
            this->screenshot_readback_extent = extent; // same size, new swapchain generation: reuse
            return this->screenshot_readback_mapped;
        }
        // (re)allocate - the first capture, or the swapchain was resized. Assigning the new owner
        // releases the previous buffer (RAII), and nothing GPU-side references it: the copy that
        // used it was recorded in an earlier frame that has already been submitted and waited on.
        this->screenshot_readback = this->vulkan_core.vma.create_buffer(nullptr, needed, vulkan::buffer_type::readback_coherent);
        this->screenshot_readback_mapped = nullptr;
        this->screenshot_readback_buffer = VK_NULL_HANDLE;
        this->screenshot_readback_size = 0;
        this->screenshot_readback_extent = {0, 0};
        if (!this->screenshot_readback.valid()) {
            utility::log("screenshot: read-back buffer creation failed ({} bytes)", needed);
            return nullptr;
        }
        auto const* detail = this->vulkan_core.vma.get_buffer_detail(this->screenshot_readback.handle());
        if (detail == nullptr) {
            this->screenshot_readback.reset();
            return nullptr;
        }
        this->screenshot_readback_buffer = detail->buffer;
        this->screenshot_readback_mapped = detail->allocation_info.pMappedData;
        this->screenshot_readback_size = needed;
        this->screenshot_readback_extent = extent;
        return this->screenshot_readback_mapped;
    }

    void runtime::record_screenshot_copy(VkCommandBuffer const command_buffer) {
        core& vk = this->vulkan_core;
        VkExtent2D const extent = vk.swap_chain_extent;
        if (extent.width == 0 || extent.height == 0 || this->current_image_index >= vk.swap_chain_images.size()) {
            return; // nothing sensible to copy (the frame will be skipped anyway)
        }
        if (!vk.swapchain_transfer_src_supported) {
            // The swapchain images lack VK_IMAGE_USAGE_TRANSFER_SRC_BIT, so vkCmdCopyImageToBuffer
            // from one of them would violate VUID-vkCmdCopyImageToBuffer-srcImage-00186. The surface
            // cannot do screenshots at all: say it once, drop the request (a permanent condition -
            // retrying every frame would only spam), and let main see "nothing captured".
            if (!this->screenshot_unsupported_logged) {
                this->screenshot_unsupported_logged = true;
                utility::log("screenshot: unsupported (swapchain has no TRANSFER_SRC usage) - F12 disabled");
            }
            this->screenshot_requested = false;
            return;
        }
        if (this->ensure_screenshot_readback(extent) == nullptr) {
            return;
        }
        // The swapchain image is in COLOR_ATTACHMENT_OPTIMAL here (the composite pass just wrote
        // it, and the overlay with it): COLOR_ATTACHMENT -> TRANSFER_SRC -> copy -> back to
        // COLOR_ATTACHMENT, so end_recording's present_transition still sees the layout it expects.
        std::array<VkImageMemoryBarrier2, 1> barriers = {vulkan::color_attachment_to_transfer_transition};
        barriers[0].image = vk.swap_chain_images[this->current_image_index];
        VkDependencyInfo dependency_info = make_image_dependency_info(1, barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency_info);

        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(command_buffer, vk.swap_chain_images[this->current_image_index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, this->screenshot_readback_buffer, 1, &region);

        barriers[0] = vulkan::transfer_to_color_attachment_transition;
        barriers[0].image = vk.swap_chain_images[this->current_image_index];
        dependency_info = make_image_dependency_info(1, barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency_info);

        this->screenshot_pending = true;
    }

    bool runtime::consume_screenshot_request() noexcept {
        // Could the requested frame be captured? Single-shot by design: the flag is cleared HERE, on
        // success and on failure alike. A failing read-back (unsupported swapchain format, missing
        // read-back buffer) used to leave the flag set, so main's loop called
        // acquire_current_frame_image() - which begins with vkDeviceWaitIdle - and logged an error
        // every single frame until exit. A dropped capture is the correct outcome; one F12 is one
        // attempt.
        bool const captured = this->screenshot_pending;
        this->screenshot_pending = false;
        return captured;
    }

    void runtime::set_point_lights(std::span<punctual_light const> lights) noexcept {
        // CPU-side only (same rule as set_brdf_model): encode into light_state's GPU-layout
        // array; pace_and_acquire() copies the whole light UBO into the paced slot's buffer.
        uint32_t count = 0;
        for (punctual_light const& light : lights) {
            if (count >= max_punctual_lights) {
                break;
            }
            point_light& slot = this->light_state.punctual_lights[count];
            slot.position = glm::vec4(light.position, 0.0f);
            // color already scaled by intensity (radiance units, as the shader expects)
            slot.color = glm::vec4(light.color * light.intensity, 0.0f);
            // a zero-length spot axis would encode NaNs (normalize(0) divides by 0): fall back
            // to the default downward axis so the shader's normalize() stays finite
            float const axis_length = glm::length(light.spot_direction);
            glm::vec3 const axis = axis_length > 1e-6f ? light.spot_direction / axis_length : glm::vec3(0.0f, -1.0f, 0.0f);
            slot.spot_dir = glm::vec4(axis, 0.0f);
            // inner cone: cos of the inner half-angle (glTF KHR innerConeAngle when set), else the
            // legacy soft-inner derivation mix(outer, 1, 0.6) == 0.6 + 0.4 * outer
            float const inner_cos = light.spot_inner_cos.value_or(0.6f + 0.4f * light.spot_outer_cos);
            slot.params = glm::vec4(light.range, light.spot ? 1.0f : 0.0f, light.spot_outer_cos, light.spot ? inner_cos : 0.0f);
            ++count;
        }
        this->light_state.light_count.x = static_cast<float>(count);
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
        std::shared_lock const lock(this->access_mutex);
        auto const it = this->pipelines.find(pipeline_name);
        return it == this->pipelines.end() ? nullptr : &it->second;
    }

    std::unique_ptr<primitive> runtime::create_primitive(std::string_view const pipeline_name, primitive_create_info const& info) {
        // The pipeline must exist (the caller names the pipeline this geometry is for), but a
        // normal_draw_primitive has DEFAULT semantics: it does not store the name, it draws with
        // whatever pipeline the recording pass binds as default (render_environment). A custom
        // draw strategy that needs a specific pipeline stores its own name and requests it.
        {
            std::shared_lock const lock(this->access_mutex);
            if (!this->pipelines.contains(pipeline_name)) {
                return nullptr;
            }
        }
        this->ensure_scene_set();

        auto result = std::make_unique<normal_draw_primitive>();

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
        result->transparent = info.factors.alpha_blend;
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
        // same pipeline semantics as the source geometry it draws (empty = default semantics)
        result->pipeline_name = source.pipeline_name;
        result->source = &source; // geometry owner; must stay in this runtime's scene tree
        result->instance_count = count;
        result->push.material_index = source.push.material_index;
        result->push.flags = 1u; // bit0: pbr.vert picks instances[instance_base + gl_InstanceIndex]
        result->push.instance_base = base;
        result->push.model = glm::mat4(1.0f);
        result->double_sided = source.double_sided;
        result->transparent = source.transparent; // same material semantics as the source geometry

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
        // Default-semantics draw (like normal_draw_primitive): static geometry renders with the
        // runtime's default pipeline, which must exist by now (the first make_pipeline() set it).
        {
            std::shared_lock const lock(this->access_mutex);
            if (this->default_pipeline_name.empty() || !this->pipelines.contains(this->default_pipeline_name)) {
                return nullptr;
            }
        }
        this->ensure_scene_set();

        auto result = std::make_unique<static_draw_primitive>();

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
        // The merged index element count is the hard bound every chunk must stay inside.
        uint32_t const index_element_size = info.index_type == VK_INDEX_TYPE_UINT16 ? 2u : 4u;
        if (info.index_data.size_bytes() % index_element_size != 0) {
            utility::error("static draw: index data size is not a multiple of the index element size");
            return nullptr;
        }
        uint64_t const index_element_count = info.index_data.size_bytes() / index_element_size;
        auto const* const index_bytes = info.index_data.data();

        result->chunks.reserve(info.chunks.size());
        for (static_draw_chunk const& chunk : info.chunks) {
            if (chunk.index_count == 0) {
                continue; // empty chunk: skip (keeps is_valid simple)
            }
            // Bounds check against the merged buffers BEFORE anything is registered: a bad chunk
            // table (out-of-range index window or a vertex_offset pushing past vertex_count)
            // would otherwise read out of bounds in vkCmdDrawIndexed - only visible under
            // validation/debug. Offending chunks are logged and skipped (the batch keeps the
            // valid remainder, like the material-table overflow degradation).
            bool const index_window_ok = static_cast<uint64_t>(chunk.first_index) + chunk.index_count <= index_element_count;
            bool vertex_reference_ok = chunk.vertex_offset < info.vertex_count;
            if (index_window_ok && vertex_reference_ok) {
                // walk the chunk's indices to verify the referenced vertices exist (the packer
                // may have left per-chunk offsets instead of remapping indices)
                for (uint32_t k = 0; k < chunk.index_count; ++k) {
                    uint32_t index = 0;
                    std::memcpy(&index, index_bytes + (static_cast<size_t>(chunk.first_index) + k) * index_element_size, index_element_size);
                    if (static_cast<uint64_t>(index) + chunk.vertex_offset >= info.vertex_count) {
                        vertex_reference_ok = false;
                        break;
                    }
                }
            }
            if (!index_window_ok || !vertex_reference_ok) {
                utility::error("static draw: chunk [first_index {}, count {}, vertex_offset {}] out of the merged buffer range ({} indices, {} vertices) - chunk skipped",
                               chunk.first_index, chunk.index_count, chunk.vertex_offset, index_element_count, info.vertex_count);
                continue;
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
            // A merged batch with any transparent chunk is drawn as ONE transparent unit in the
            // transparent pass (depth-write off). Within the batch the chunk order is the draw
            // order - the caller packs back-to-front chunks itself. Masked chunks need no batch
            // flag: the alpha test lives in the material record, which shadow.frag reads too.
            result->transparent = result->transparent || chunk.factors.alpha_blend;
        }
        if (result->chunks.empty()) {
            return nullptr; // every chunk was empty or out of range: nothing drawable
        }

        scene_tree::scene_node& leaf = this->get_scene().add_root();
        leaf.name = "static";
        leaf.local = info.model_matrix; // whole-batch placement, like make_primitive (update_world fills push.model from it)
        primitive* const created = static_cast<primitive*>(leaf.attach(std::move(result)));
        this->bvh_dirty = true; // new leaf -> culling BVH must be rebuilt
        return created;
    }

    std::vector<primitive const*> runtime::get_primitives(std::string_view const pipeline_name) const noexcept {
        // match by the pipeline a leaf effectively draws with: an explicit pipeline_name, or the
        // runtime default for default-semantics leaves (empty pipeline_name). Snapshot the
        // default under a shared lock (it can change via set_default_pipeline on another thread).
        std::string_view default_name;
        {
            std::shared_lock const lock(this->access_mutex);
            default_name = this->default_pipeline_name;
        }
        auto const effective = [default_name](primitive const& m) -> std::string_view {
            return m.pipeline_name.empty() ? default_name : m.pipeline_name;
        };
        std::vector<primitive const*> result;
        for (scene_tree::scene_node const& root : this->get_scene().roots) {
            // collect every leaf whose primitive draws with the requested pipeline (leaves name
            // their pipeline, or fall back to the runtime default; the tree just organizes them)
            scene_tree::visit_primitives(root, glm::mat4(1.0f), [&](scene_tree::scene_node const& n, glm::mat4 const&) {
                auto const* m = static_cast<primitive const*>(n.primitive_leaf.get());
                if (effective(*m) == pipeline_name) {
                    result.push_back(m);
                }
            });
        }
        return result;
    }

    void runtime::clear_primitives(std::string_view const pipeline_name) {
        // snapshot the default (a concurrent set_default_pipeline must not tear the comparison)
        std::string_view default_name;
        {
            std::shared_lock const lock(this->access_mutex);
            default_name = this->default_pipeline_name;
        }
        auto const matches = [default_name, pipeline_name](scene_tree::scene_node const& node) {
            if (node.primitive_leaf == nullptr) {
                return false;
            }
            auto const* const m = static_cast<primitive const*>(node.primitive_leaf.get());
            std::string_view const effective = m->pipeline_name.empty() ? default_name : m->pipeline_name;
            return effective == pipeline_name;
        };
        // DFS remove: erase every leaf primitive drawing with the pipeline, wherever it sits in
        // the tree (imported scenes nest leaves under hierarchy nodes; make_primitive attaches
        // them at roots). A node whose leaf matches is stripped of that leaf; it (or an ancestor)
        // is dropped only when nothing remains below it, so models of other pipelines survive.
        auto& roots = this->get_scene().roots;
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
