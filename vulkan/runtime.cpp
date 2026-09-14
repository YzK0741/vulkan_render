module;

#include <GLFW/glfw3.h>
#include <bit> // std::bit_cast for the caster world-matrix hash
#include <chrono>
#include <cstring> // std::memcpy, for composing a pass's push block
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thread> // std::this_thread::yield in the frame limiter
#include <vulkan/vulkan.h>

module vulkan.runtime;

import vulkan.profiling;
import vulkan.pipelines;
import vulkan.bindings;
import vulkan.render_resource;
import vulkan.render_resource.shared;

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
        // A QUARTER of the hardware threads, not a half: measured on a 16-thread machine, moving this
        // to hw/2 cost 11-12% fps (822 -> 735 forward, 1706 -> 1497 unlit) and lengthened the shadow
        // sub-phase (0.61 -> 0.65 ms) - the recording stages are not worker-starved at hw/4, and more
        // workers only add wake/join, cache and driver-side recording contention. Kept as a documented
        // negative result so the experiment is not repeated.
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
        // readback owns GPU resources and is deliberately neither copyable nor movable (two owners of
        // one staging buffer is the bug its deletion prevents), so it must be constructed here - which
        // is why its member declaration sits ABOVE filtered_core's, matching this order. Both only need
        // the core, so the order between them is otherwise free.
        , readback_staging{vulkan_core}
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
            // one entry: the alpha-blended pass's secondary (see secondary_pass). The shadow cascades
            // and the main-pass segments own their buffers elsewhere, because they record concurrently.
            std::array<vk_command_buffer, static_cast<std::size_t>(secondary_pass::count)> pair = {
                this->vulkan_core.make_secondary_command_buffer(), // transparent
            };
            this->secondary_command_buffers.push_back(std::move(pair));

            // Shadow cascades record on the task pool, so each cascade gets its OWN {pool, buffer}: a
            // VkCommandPool is not thread safe and concurrent recording must not share one (M9).
            std::vector<std::pair<VkCommandPool, vk_command_buffer>> cascade_recording;
            cascade_recording.reserve(vulkan::max_shadow_cascades);
            for (uint32_t cascade = 0; cascade < vulkan::max_shadow_cascades; ++cascade) {
                VkCommandPool const cascade_pool = this->vulkan_core.make_command_pool();
                cascade_recording.emplace_back(cascade_pool, this->vulkan_core.make_secondary_command_buffer(cascade_pool));
            }
            this->shadow_recording.push_back(std::move(cascade_recording));
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
        // The G-buffer depth layout flags are one per swapchain image, and the core has already
        // built this generation's G-buffer targets (core::create_hdr_resolve_resources runs in the
        // core constructor), so they can be sized here - before any frame records. Every flag
        // starts clear, which is what a freshly created depth image is in (UNDEFINED);
        // on_swapchain_recreated() re-sizes them for every later generation.
        this->gbuffer_depth_written.assign(this->vulkan_core.gbuffer_depth_images.size(), false);
        this->velocity_written.assign(this->vulkan_core.velocity_images.size(), false);
        this->gbuffer_targets_written.assign(this->vulkan_core.gbuffer_images[0].size(), false);
        // The GI accumulation starts empty for the same reason (see gi_history_valid): the first
        // frame of a generation has nothing to blend with, and sized HERE rather than only on the
        // off -> on edge in set_ssgi, because a run that starts with GI enabled never sees that edge
        // (an empty vector reads as "no history" for every frame, which silently turns the temporal
        // resolve into a pass-through of the raw trace).
        this->gi_history_valid.assign(this->vulkan_core.gi_history_images.size(), false);
        this->gi_spec_seen.assign(this->vulkan_core.gi_spec_images.size(), false); // new targets: the lobe's outputs need their first-use transition again

        // The probe cache's remaining flag starts where the images do: nothing has transitioned the grid out
        // of UNDEFINED, so the tracer's gain is 0 until the TRACE pass - the frame's first reader of the grid
        // - has taken its first-use transition once. (What the grid HOLDS is the pass's own state now:
        // gi_probe_pass::cache_valid, which starts false with the pass.)
        this->gi_probe_grid_seen = false;
        // ... and the furnace cube is a new image too, so its level has to be written again.
        this->furnace_cube_ready = false;
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
        this->vulkan_core.vma.log_statistics();
        // Wait for the GPU to finish BEFORE releasing anything below: the last submitted frame
        // may still be executing and destroying in-use resources would violate VUIDs (~core()
        // also waits, but that runs after this body — too late for the VMA frees here).
        this->vulkan_core.wait_idle();

        this->pipelines.clear();

        // post-process raw objects. The pipeline(s) and the sampler are RAII members; the two layouts
        // are not, so they are destroyed here - and this must stay real code: an earlier edit collapsed
        // this block onto the comment line above it, which commented the destroy calls out and leaked
        // them (validation: "VkDevice has 18 leaked objects ... VkPipelineLayout,
        // VkDescriptorSetLayout, VkDescriptorSet"). The pool belongs to post_family now.
        if (this->post_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->vulkan_core.device, this->post_pipeline_layout, nullptr);
            this->post_pipeline_layout = VK_NULL_HANDLE;
        }
        if (this->post_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(this->vulkan_core.device, this->post_set_layout, nullptr);
            this->post_set_layout = VK_NULL_HANDLE;
        }
        // the same objects for the G-buffer debug view, minus the pool: that one belongs to
        // gbuffer_family, whose destructor destroys it (and the generations it retired)
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
        // ... and the GI tracer's (same two sets, which is why it needs a layout of its own)
        if (this->ssgi_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->vulkan_core.device, this->ssgi_pipeline_layout, nullptr);
            this->ssgi_pipeline_layout = VK_NULL_HANDLE;
        }
        if (this->ssgi_temporal_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(this->vulkan_core.device, this->ssgi_temporal_set_layout, nullptr);
            this->ssgi_temporal_set_layout = VK_NULL_HANDLE;
        }
        if (this->ssgi_temporal_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->vulkan_core.device, this->ssgi_temporal_pipeline_layout, nullptr);
            this->ssgi_temporal_pipeline_layout = VK_NULL_HANDLE;
        }
        if (this->rt_shadow_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->vulkan_core.device, this->rt_shadow_pipeline_layout, nullptr);
            this->rt_shadow_pipeline_layout = VK_NULL_HANDLE;
        }
        if (this->mask_bake_pipeline_layout != VK_NULL_HANDLE) {
            // It owns no set layout either (it is created from the scene one), so only the layout. Its
            // descriptor set is a vk_descriptor_set member, which frees itself with the pool still alive.
            vkDestroyPipelineLayout(this->vulkan_core.device, this->mask_bake_pipeline_layout, nullptr);
            this->mask_bake_pipeline_layout = VK_NULL_HANDLE;
        }
        if (this->compute_skin_pipeline_layout != VK_NULL_HANDLE) {
            // The same shape as the mask bake's: it is created from the scene set layout and owns none of
            // its own, so only the VkPipelineLayout is ours to destroy. Found the same way, too - as a
            // "[ERROR] vkDestroyDevice(): ... has 1 leaked objects" on the FIRST capture of the L2.2b A/B,
            // which is why the off arm of that measurement was taken again afterwards.
            vkDestroyPipelineLayout(this->vulkan_core.device, this->compute_skin_pipeline_layout, nullptr);
            this->compute_skin_pipeline_layout = VK_NULL_HANDLE;
        }
        if (this->ssgi_spatial_pipeline_layout != VK_NULL_HANDLE) {
            // it owns no set layout (it binds the shared scene and G-buffer sets), so only the layout
            vkDestroyPipelineLayout(this->vulkan_core.device, this->ssgi_spatial_pipeline_layout, nullptr);
            this->ssgi_spatial_pipeline_layout = VK_NULL_HANDLE;
        }
        if (this->ssgi_spec_pipeline_layout != VK_NULL_HANDLE) {
            // the same shape as the spatial filter's: the shared two sets, so only the layout is ours
            vkDestroyPipelineLayout(this->vulkan_core.device, this->ssgi_spec_pipeline_layout, nullptr);
            this->ssgi_spec_pipeline_layout = VK_NULL_HANDLE;
        }
        // The probe cache's pipeline and its layout are NOT destroyed here any more: they are the pass's
        // (vulkan.pass.gi_probe builds and destroys them), which is the whole point of the extraction - a
        // handle only that pass names is that pass's to release. The TAA resolve's set layout, pipeline
        // layout, pipeline AND descriptor family left the same way (vulkan.pass.taa), so nothing about it is
        // torn down here either.

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

        // Per-motion-slot previous world matrices (set 0 binding 13): ONE buffer per frame slot, like
        // the skin and morph buffers below, so a frame in flight never shares the buffer the next
        // frame rewrites. Zero-filled: a leaf's first frame reports "no motion", which is right -
        // nothing was there to move from. motion_previous is the CPU-side copy of what is currently
        // in it, advanced by advance_motion_transforms().
        std::vector<unsigned char> const zeroed_motion(static_cast<size_t>(vulkan::scene_motion_capacity) * sizeof(glm::mat4), 0);
        this->motion_buffers.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        this->motion_mapped.reserve(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            vk_buffer motion_buf = this->vulkan_core.vma.create_buffer(zeroed_motion.data(), zeroed_motion.size(), vulkan::buffer_type::storage_coherent);
            if (!motion_buf.valid()) {
                utility::panic("failed to create motion transform buffer");
            }
            auto const* motion_detail = this->vulkan_core.vma.get_buffer_detail(motion_buf.handle());
            if (motion_detail == nullptr) {
                utility::panic("failed to get motion transform buffer detail");
            }
            this->motion_buffers.push_back(std::move(motion_buf));
            this->motion_mapped.push_back(motion_detail->allocation_info.pMappedData);
        }
        this->motion_previous.assign(vulkan::scene_motion_capacity, glm::mat4(1.0f));

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
        if (!this->shadow_images.empty() && this->shadow_allocated_layers == this->shadow_cascades) {
            return; // already created for this cascade count
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
            shadow_info.width = this->shadow_map_size;
            shadow_info.height = this->shadow_map_size;
            shadow_info.mip_levels = 1;
            // One layer per ACTIVE cascade, NOT max_shadow_cascades: the spare layers the old code
            // always allocated were 2048x2048x4 B each per frame slot (33.5 MB with the default three
            // cascades) that nothing ever fitted, rendered or sampled. Growing the count rebuilds
            // these images (see set_shadow_cascades) and SHRINKING keeps the layers already owned,
            // which is why shadow_allocated_layers - not shadow_cascades - is the image's real layer
            // count and the value every subresource range over the whole array has to use.
            shadow_info.array_layers = this->shadow_cascades;
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
            for (uint32_t cascade = 0; cascade < this->shadow_cascades; ++cascade) {
                layers.push_back(this->vulkan_core.make_depth_layer_view(detail->image, this->vulkan_core.depth_format, cascade));
            }
            this->shadow_layer_views.push_back(std::move(layers));
        }
        this->shadow_allocated_layers = this->shadow_cascades;
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
        if (this->scene_sets.created()) {
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
        this->scene_sets.create(this->vulkan_core, this->vulkan_core.scene_descriptor_set_layout);

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
            VkDescriptorSet const set = this->scene_sets.set(static_cast<uint32_t>(slot));

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

            // binding 13: THIS slot's previous world matrices (advanced once per frame)
            auto const* motion_detail = this->vulkan_core.vma.get_buffer_detail(this->motion_buffers[static_cast<std::size_t>(slot)].handle());
            if (motion_detail == nullptr) {
                utility::panic("failed to get motion transform buffer detail");
            }
            write_buffer_binding(set, 13, motion_detail->buffer, static_cast<VkDeviceSize>(vulkan::scene_motion_capacity) * sizeof(glm::mat4), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

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

            // bindings 14/15: THIS slot's ray-traced sun visibility, as the sampler the lighting stage
            // reads and as the storage image the compute pass writes. Two descriptors for one image on
            // purpose: the layouts differ (SHADER_READ for the reader, GENERAL for the writer), and each
            // is only ever used while the image really is in that layout (see record_scene_tail). Both
            // are written unconditionally - the images exist on every device - so a device without ray
            // tracing never has to know these bindings exist.
            {
                std::array<VkDescriptorImageInfo, 2> visibility_infos = {};
                // LINEAR clamp, and deliberately NOT the shadow sampler: that one has compareEnable set
                // (it is a sampler2DShadow sampler), and a compare sampler paired with a plain
                // sampler2D read is not what the descriptor declares. post_sampler exists by the time any
                // primitive is created (main.cpp loads the scene after setup_pipeline), which is what
                // every caller of this function is.
                visibility_infos[0].sampler = *this->post_sampler;
                visibility_infos[0].imageView = this->vulkan_core.rt_shadow_image_views[static_cast<std::size_t>(slot)];
                visibility_infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                visibility_infos[1].sampler = VK_NULL_HANDLE; // a storage image has no sampler
                visibility_infos[1].imageView = this->vulkan_core.rt_shadow_image_views[static_cast<std::size_t>(slot)];
                visibility_infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                std::array<VkWriteDescriptorSet, 2> visibility_writes = {};
                visibility_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                visibility_writes[0].dstSet = set;
                visibility_writes[0].dstBinding = 14;
                visibility_writes[0].descriptorCount = 1;
                visibility_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                visibility_writes[0].pImageInfo = &visibility_infos[0];
                visibility_writes[1] = visibility_writes[0];
                visibility_writes[1].dstBinding = 15;
                visibility_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                visibility_writes[1].pImageInfo = &visibility_infos[1];
                vkUpdateDescriptorSets(this->vulkan_core.device, static_cast<uint32_t>(visibility_writes.size()), visibility_writes.data(), 0, nullptr);
            }
            // binding 16: THIS slot's top level structure, but ONLY once one exists. A null
            // acceleration-structure descriptor is not legal without the nullDescriptor feature
            // (VUID-VkWriteDescriptorSetAccelerationStructureKHR-pAccelerations-03580), and the structure
            // is created by the first frame that asks for ray-traced shadows - so the write happens there
            // (record_top_level_structure) for the slot that just got one. Nothing reads the binding
            // before that: the pass that uses it is gated on the same handle.
            if (this->vulkan_core.ray_query_available && this->rt_top_levels.has_value()) {
                VkAccelerationStructureKHR const tlas = this->rt_top_levels->handle(static_cast<uint32_t>(slot));
                if (tlas != VK_NULL_HANDLE) {
                    this->write_rt_structure_binding(set, tlas);
                }
            }
        }

        // binding 7 (light UBO, per-slot) + binding 8 (per-slot shadow map) and bindings 2-4 (IBL):
        // written on every scene set below
        this->write_light_and_shadow_bindings();
        this->write_ibl_bindings();
    }

    void runtime::write_rt_structure_binding(VkDescriptorSet const set, VkAccelerationStructureKHR const tlas) {
        VkWriteDescriptorSetAccelerationStructureKHR structure_info = {};
        structure_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        structure_info.accelerationStructureCount = 1;
        structure_info.pAccelerationStructures = &tlas;
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.pNext = &structure_info;
        write.dstSet = set;
        write.dstBinding = 16;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        vkUpdateDescriptorSets(this->vulkan_core.device, 1, &write, 0, nullptr);
    }
    void runtime::update_all_scene_sets(VkWriteDescriptorSet const* const writes, uint32_t const write_count) {
        // the sets, the per-slot dstSet substitution and the "not created yet" case belong to the
        // bindings, because a material registered before setup finished has nothing to write to
        this->scene_sets.update_all(this->vulkan_core, writes, write_count);
    }

    void runtime::write_light_and_shadow_bindings() {
        if (!this->scene_sets.created()) {
            return;
        }
        // binding 7 (light UBO) + binding 8 (shadow map): BOTH point at THIS slot's own
        // resources (per-slot light buffers like the camera UBO, per-slot shadow images), so no
        // per-frame re-pointing is needed and an in-flight frame never shares a buffer the next
        // frame rewrites.
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            VkDescriptorSet const set = this->scene_sets.set(static_cast<uint32_t>(slot));
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
        if (!this->scene_sets.created()) {
            return;
        }
        // The IBL bindings need nothing but plain handles - the three environment views, the
        // environment sampler and the white placeholder - so they are written by the bindings; the
        // buffer bindings below and in ensure_scene_set() stay here, because they need this runtime's
        // buffer details and capacities.
        //
        // The view count is NOT guaranteed here: set_ibl() is what fills ibl_views, and it early-
        // returns when env_size == 0 (the documented way to run without IBL), while this runs from
        // ensure_scene_set() - i.e. as soon as the first primitive exists. So a caller that builds
        // its scene before calling set_ibl() reaches this with the vector still empty, and indexing
        // it would be an out-of-bounds read. An empty span makes write_ibl bind the white
        // placeholder to all three slots instead, which is the state an unloaded IBL is meant to be
        // in; set_ibl() rewrites the bindings once the environment is actually there.
        std::array<VkImageView, 3> views = {};
        bool const have_ibl_views = this->ibl_views.size() >= views.size() && *this->ibl_views[0] != VK_NULL_HANDLE && *this->ibl_views[1] != VK_NULL_HANDLE && *this->ibl_views[2] != VK_NULL_HANDLE;
        if (have_ibl_views) {
            for (std::size_t i = 0; i < views.size(); ++i) {
                views[i] = *this->ibl_views[i]; // the wrappers unwrap to the raw VkImageView here
            }
        }
        // The furnace verification mode: slots 0 and 1 - the prefiltered environment and the irradiance cube
        // - point at the CONSTANT cube instead of the real environment, while slot 2 keeps its own BRDF LUT.
        // The two are not interchangeable: the cube slots are samplerCube, and the 2D white placeholder this
        // function falls back to for an unloaded IBL cannot be bound there at all - doing so is a viewType/Dim
        // validation error, which is how this was found rather than assumed.
        if (this->furnace && !this->vulkan_core.furnace_cube_views.empty() && this->vulkan_core.furnace_cube_views[0] != VK_NULL_HANDLE) {
            views[0] = this->vulkan_core.furnace_cube_views[0];
            views[1] = this->vulkan_core.furnace_cube_views[0];
        }
        std::span<VkImageView const> const view_span = have_ibl_views ? std::span<VkImageView const>(views) : std::span<VkImageView const>{};
        this->scene_sets.write_ibl(this->vulkan_core, this->ibl_ready, view_span, *this->env_sampler, *this->owned_texture_views[0], *this->texture_sampler);
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
            writes[write_count].dstSet = this->scene_sets.set(0u); // dstSet is replaced per set by update_all_scene_sets
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
            white_write.dstSet = this->scene_sets.set(0u); // dstSet is replaced per set by update_all_scene_sets
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

        // Dynamic rendering (Vulkan 1.3 core, the only path the engine supports): attachments are
        // described inline, no render pass / framebuffer objects exist. The scene instance is always
        // the G-buffer pass: the opaque pass writes the surface instead of shading it, into three
        // single-sampled targets + the motion-vector target + the G-buffer's own 1x depth image, and
        // adds the emissive into the scene color target. The lighting stage then shades it into the
        // image the post chain reads (scene_target_view).
        if (this->gbuffer_pass_active()) {
            std::array<VkRenderingAttachmentInfo, vulkan::gbuffer_pass_attachment_count> gbuffer_attachments = {};
            VkClearValue clear = {}; // the surface + motion targets clear to zero: no geometry, no motion
            for (uint32_t target = 0; target < vulkan::gbuffer_target_count; ++target) {
                gbuffer_attachments[target] = make_color_attachment_info(vk.gbuffer_image_views[target][image_index], clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
            }
            gbuffer_attachments[vulkan::gbuffer_target_count] = make_color_attachment_info(vk.velocity_image_views[image_index], clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
            // the last attachment is the scene color, CLEARed to zero: it accumulates only the
            // emissive here. The lighting stage then adds the lighting (and the sky, for pixels no
            // geometry wrote) on top, so a lit pixel is emissive + lighting and a background pixel is
            // sky - with no background pass anywhere. Under TAA the scene color is the resolve's input
            // image, and the resolve writes the HDR target the post chain reads (see
            // runtime::scene_target_view).
            gbuffer_attachments[vulkan::gbuffer_target_count + 1] = make_color_attachment_info(this->scene_target_view(image_index), clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
            // the G-buffer depth clears to the far plane (1.0) - but unlike the old forward path's
            // main depth, its contents must SURVIVE the instance: the lighting stage, the transparent
            // pass, the TAA resolve and the debug view all read this image later in the same
            // submission (binding 3 of the G-buffer set). STORE_OP_DONT_CARE would leave the contents
            // undefined, which is exactly what those four read.
            VkRenderingAttachmentInfo const depth_attachment = make_depth_attachment_info(vk.gbuffer_depth_image_views[image_index], VK_ATTACHMENT_STORE_OP_STORE);
            VkRenderingInfo const rendering_info = make_rendering_info(flags, {{0, 0}, vk.swap_chain_extent}, gbuffer_attachments.data(), static_cast<uint32_t>(gbuffer_attachments.size()), &depth_attachment);
            vkCmdBeginRendering(command_buffer, &rendering_info);
            return;
        }

        // No G-buffer pipeline: no scene can be drawn. Open an empty instance anyway, so every frame
        // still has a matching vkCmdEndRendering and the post chain samples a defined target.
        VkClearValue clear_color = {};
        clear_color.color = {{this->clear_color.r, this->clear_color.g, this->clear_color.b, 1.0f}};
        VkRenderingAttachmentInfo const color_attachment = make_color_attachment_info(
            vk.hdr_image_views[image_index],
            clear_color,
            VK_RESOLVE_MODE_NONE,
            VK_NULL_HANDLE);

        // depth clears to the far plane (1.0): make_depth_attachment_info. DONT_CARE is correct here -
        // nothing samples this depth (the G-buffer depth above is the one that is read back).
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
        // The debug view's family does all three: it forgets the views its sets were bound to and the
        // sets themselves (so the next recorded frame allocates fresh ones) and retires its pool.
        this->gbuffer_family.retire_all();
        // The post chain's family forgets its sets and retires its pool (five sets per image, so the
        // largest of the three); the next frame allocates fresh ones from a fresh pool.
        this->post_family.retire_all();
        // The TAA resolve's family is NOT retired here any more: the pass owns it, and the call below tells
        // the pass (vulkan.pass.taa::on_swapchain_recreated retires its own family and forgets the
        // generation it fingerprinted).
        // ... and the GI denoiser's, which binds five per-image views (trace, history, motion, depth,
        // resolve) and is therefore the one family here with the most stale pointers in it.
        this->ssgi_temporal_family.retire_all();
        // AND THE PASSES ARE TOLD, by the runner rather than by this list. That is the hazard this layer was
        // built to remove: the retires above are a hand-kept list and it covered four of the six families -
        // the probe cache's and the reflection denoiser's survived only because `ensure` re-detects changed
        // views. A pass that owns a family now cannot be missed here, because it is not this function that
        // remembers: `recreate_stage` calls every pass in the stage, and a pass added to it is covered.
        {
            pass::stage const probe_stage = {.name = "gi_probe", .passes = this->gi_probe_stage, .marks = false};
            [[maybe_unused]] pass::run_report const probe_recreated = pass::recreate_stage(probe_stage, this->make_pass_host());
            pass::stage const taa_stage = {.name = "taa", .passes = this->taa_stage, .marks = false};
            [[maybe_unused]] pass::run_report const taa_recreated = pass::recreate_stage(taa_stage, this->make_pass_host());
        }
        // Every swapchain image's history died with the old generation (and its size may have
        // changed): forget the matrices, so the next frame for each image starts a new accumulation
        // instead of blending in a misaligned one. (WHETHER a history holds anything is the TAA pass's own
        // state, and the call above is what cleared it.)
        std::size_t const image_count = this->vulkan_core.taa_history_images.size();
        this->image_view_proj.assign(image_count, this->current_ubo.view_proj_unjittered);
        // The GI accumulation is per image for the same reason (see gi_history_valid): a new
        // generation has no history to blend with, and the resolve would otherwise reproject into an
        // image that holds a different resolution's data.
        this->gi_history_valid.assign(this->vulkan_core.gi_history_images.size(), false);
        this->gi_spec_seen.assign(this->vulkan_core.gi_spec_images.size(), false); // new targets: the lobe's outputs need their first-use transition again
        // The probe cache's generation flag, for the same reason (see gi_probe_grid_seen). Whether the grid
        // HOLDS anything is the pass's own state, and the pass was told above (recreate_stage).
        this->gi_probe_grid_seen = false;
        // ... and the furnace cube is a new image too, so its level has to be written again.
        this->furnace_cube_ready = false;
        // The motion-vector images died with the generation as well, and a brand new one is in
        // UNDEFINED until this frame's G-buffer instance renders into it: clear the layout flag so the
        // first frame of the new generation takes the attachment -> sampled transition (see
        // ensure_velocity_sampled).
        this->velocity_written.assign(this->vulkan_core.velocity_images.size(), false);
        this->gbuffer_targets_written.assign(this->vulkan_core.gbuffer_images[0].size(), false);
        // The G-buffer depth images died with the generation too: the same flag, the same reason
        // (see ensure_gbuffer_depth_sampled).
        this->gbuffer_depth_written.assign(this->vulkan_core.gbuffer_depth_images.size(), false);
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
        // The OVERLAY line is built here as well, and deliberately not every frame: every value is a
        // fixed-width field, because a label whose text width changes re-wraps against the panel edge
        // and the whole overlay appears to twitch (the numbers of a live running mean change in width
        // frame by frame: 0.25 -> 10.25). Fixed fields + a report that changes once per window make
        // the panel layout stable between updates.
        std::string label = std::format("gpu ({}f):", GPU_TIMING_WINDOW);
        for (uint32_t interval = 0; interval < intervals; ++interval) {
            double const mean = this->gpu_timing_sum[interval] / static_cast<double>(GPU_TIMING_WINDOW);
            report += std::format(" {} {:.2f} ms |", gpu_timing_labels[interval].name, mean);
            label += std::format("{} {:>5.2f}", gpu_timing_labels[interval].new_line ? "\n     " : " |", mean);
            total += mean; // GPU intervals have no sub-phases: every mark interval counts once
            this->gpu_timing_sum[interval] = 0.0;
        }
        this->gpu_timing_window_frames = 0;
        report += std::format(" total {:.2f} ms", total);
        this->gpu_timing_report_label = label + std::format("\n     total {:>5.2f} ms", total);
        utility::log("{}", report);
    }

    std::string runtime::gpu_timing_summary() const {
        if (!this->gpu_timings_enabled) {
            return "gpu timings: off";
        }
        if (!this->vulkan_core.gpu_timing_available()) {
            return "gpu timings: unavailable on this device";
        }
        if (this->gpu_timing_report_label.empty()) {
            return "gpu timings: collecting...";
        }
        // The last COMPLETED window's report, not a per-frame running mean: the overlay line is a
        // text layout, and a text layout that changes every frame is a UI that twitches (see the
        // report construction above). It refreshes once per GPU_TIMING_WINDOW frames, which is also
        // what the log line reports - so the overlay and the log always agree.
        return this->gpu_timing_report_label;
    }

    frame_status runtime::pace_and_acquire() {
        vulkan::profiling::cpu_phase_timer const phase_timer{this->cpu_timings, vulkan::profiling::cpu_phase::pace};
        core& vk = this->vulkan_core;

        // A zero-sized swapchain (a window that has not been sized yet, or was restored from
        // minimized into a 0-sized client area) has no valid attachments: recording would set a
        // 0-wide viewport - a VUID - and produce nothing. Skip the frame exactly like the
        // minimized case; nothing was acquired, so no semaphore is left pending.
        if (vk.swap_chain_extent.width == 0 || vk.swap_chain_extent.height == 0) {
            return frame_status::skipped;
        }

        // Frame limit (set_max_fps): hold the frame until its deadline before touching the swapchain, so
        // the wait can never happen with an image acquired. yield() and not a timer: on a fast scene the
        // remaining slack is a few milliseconds at most, and an OS sleep at Windows' default 15.6 ms
        // granularity overshoots by more than the target period (a 144 Hz target would sag towards 60).
        // The deadline moves by exactly one period, so overrunning a frame resyncs to now instead of
        // accumulating debt that would be paid back as a burst.
        if (this->max_fps > 0.0) {
            auto const period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / this->max_fps));
            auto const now = std::chrono::steady_clock::now();
            this->next_frame_deadline += period;
            if (this->next_frame_deadline > now) {
                // Sleep to a millisecond before the deadline, then spin that last millisecond: sleeping
                // alone cannot hit a sub-millisecond target, and spinning alone reaches it but burns a
                // whole core doing nothing. Splitting it costs about a millisecond of spin per frame
                // (6% of a core at 60 fps) and still lands on the deadline to within microseconds.
                utility::sleep_for_nanoseconds(std::chrono::duration_cast<std::chrono::nanoseconds>(this->next_frame_deadline - std::chrono::milliseconds(1) - std::chrono::steady_clock::now()).count()); // portable: Win32 waitable timer / POSIX nanosleep
                while (std::chrono::steady_clock::now() < this->next_frame_deadline) {
                    std::this_thread::yield();
                }
            } else {
                this->next_frame_deadline = now; // fell behind: resync, do not bank debt
            }
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
            // Ray-traced sun shadows: composed HERE rather than in set_rt_shadows, because the light UBO
            // is rebuilt from light_state every frame and a later enable_shadows() (main.cpp calls it
            // after the settings are applied, which is where this flag was first lost) resets fields of
            // this struct. Recomposing it makes the flag authoritative - the lighting stage reads exactly
            // what the passes below will do this frame.
            this->light_state.rt_shadows = (this->rt_shadows && this->rt_shadow_pipeline.has_value() && this->vulkan_core.ray_query_available) ? 1.0f : 0.0f;
            this->light_state.sun_intensity = this->furnace ? 0.0f : 1.0f;
            this->light_state.furnace_level = this->furnace ? 1.0f : 0.0f;

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
                // Gated on the SAME feature flag the dispatch uses: without an active punctual light,
                // or in the flat render mode, nothing will ever read the counts (49 KB per frame).
                bool const cluster_counts_used = this->active_features().clustered;
                if (cluster_counts_used && this->cluster_count_mapped.size() > static_cast<std::size_t>(frame_slot) && this->cluster_count_mapped[frame_slot] != nullptr) {
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

    void runtime::advance_motion_transforms() {
        if (this->bound_scene == nullptr || this->motion_mapped.empty()) {
            return;
        }
        uint32_t const slot = static_cast<uint32_t>(this->vulkan_core.current_frame);
        if (slot >= this->motion_mapped.size()) {
            return; // no buffer for this slot: nothing to publish, and the shader reads slot 0's set
        }
        auto* const published = static_cast<glm::mat4*>(this->motion_mapped[slot]);
        if (published == nullptr) {
            return;
        }
        for (scene_tree::scene_node const& root : this->bound_scene->roots) {
            scene_tree::visit_primitives(root, this->scene_transform, [this, published](scene_tree::scene_node const& node, glm::mat4 const& world) {
                uint32_t const index = node.primitive_leaf->motion_slot();
                if (index == scene_tree::no_motion_slot || index >= vulkan::scene_motion_capacity) {
                    return; // instanced (filled at setup) or out of range: nothing to advance
                }
                // Publish what this leaf looked like one frame ago, THEN remember this frame's world
                // matrix - so the next frame publishes the matrix this frame is about to draw with.
                published[index] = this->motion_previous[index];
                this->motion_previous[index] = world;
            });
        }
    }

    frame_status runtime::begin_recording() {
        vulkan::profiling::cpu_phase_timer const phase_timer{this->cpu_timings, vulkan::profiling::cpu_phase::begin};
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

        // The furnace verification mode's constant environment, written here and ONCE per target generation:
        // this is the frame's first command buffer, and the environment is sampled by the SKYBOX - which runs
        // long before the lighting stage and the GI chain - so any later point would leave the background of
        // every frame reading the previous contents. The level is the same 1.0 the light UBO's furnace lane
        // carries, so the analytic answer and the environment agree by construction; the IBL bindings point
        // at this cube only while the mode is on.
        if (this->furnace && !this->furnace_cube_ready && !vk.furnace_cube_images.empty() && vk.furnace_cube_images[0] != VK_NULL_HANDLE) {
            VkImageMemoryBarrier2 to_transfer = vulkan::undefined_to_transfer_dst_transition;
            to_transfer.image = vk.furnace_cube_images[0];
            to_transfer.subresourceRange.layerCount = 6; // all six faces, not the one the constant defaults to
            VkDependencyInfo const to_transfer_dependency = make_image_dependency_info(1, &to_transfer);
            vkCmdPipelineBarrier2(*command_buffer, &to_transfer_dependency);

            VkClearColorValue const level = {{1.0f, 1.0f, 1.0f, 1.0f}};
            VkImageSubresourceRange const faces = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
            vkCmdClearColorImage(*command_buffer, vk.furnace_cube_images[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &level, 1, &faces);

            VkImageMemoryBarrier2 to_sampling = vulkan::transfer_dst_to_sampling_transition;
            to_sampling.image = vk.furnace_cube_images[0];
            to_sampling.subresourceRange.layerCount = 6;
            VkDependencyInfo const to_sampling_dependency = make_image_dependency_info(1, &to_sampling);
            vkCmdPipelineBarrier2(*command_buffer, &to_sampling_dependency);

            this->furnace_cube_ready = true;
        }
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
        // ... and, in the same walk's shadow, the previous-frame world matrices that give TAA its
        // object motion: see the declaration for why this has to happen here and not at draw time.
        this->advance_motion_transforms();
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

    // Everything that decides whether a slot's shadow maps are still valid, folded into one 64-bit
    // fingerprint: the caster count (the set itself can change - heavy scenes BVH-cull the casters
    // per frame), each caster's world matrix, the uploaded skin matrices (a skinned mesh keeps a
    // constant push.model; its pose lives only in those) and the morph-scratch revision. It is XXH3
    // rather than a byte loop because it runs on every frame, the reused ones included: 0.99 us
    // against 15.5 us for Sponza's 300 casters. The fold is boost::hash_combine's mix - the values
    // are hashed rather than compared, so the mix carries the collision resistance.
    uint64_t runtime::shadow_geometry_signature() const {
        auto const fold = [](uint64_t const accumulator, uint64_t const value) {
            return accumulator ^ (value + 0x9e3779b97f4a7c15ull + (accumulator << 6) + (accumulator >> 2));
        };
        uint64_t signature = static_cast<uint64_t>(this->shadow_casters.size());
        for (primitive const* caster : this->shadow_casters) {
            uint64_t const matrix = utility::xxh3_64bits({reinterpret_cast<unsigned char const*>(&caster->push.model), sizeof(glm::mat4)});
            signature = fold(signature, matrix);
        }
        return fold(fold(signature, this->skin_matrix_hash), this->morph_revision.load(std::memory_order_relaxed));
    }

    void runtime::record_main_drawcalls() {
        vulkan::profiling::cpu_phase_timer const phase_timer{this->cpu_timings, vulkan::profiling::cpu_phase::scene};
        core& vk = this->vulkan_core;
        vk_command_buffer& command_buffer = this->command_buffers[static_cast<uint32_t>(vk.current_frame)];
        uint32_t const frame_slot = static_cast<uint32_t>(vk.current_frame);

        // ---- Clustered light culling (M5): one compute dispatch before anything renders. It sorts
        //      the punctual lights into the frame's cluster grid, and the shading stages (forward
        //      inside the main instance, deferred later in the frame) then loop only their own
        //      cluster's list. Runs before the shadow pass so the barrier that publishes its buffers
        //      is as early as possible; nothing before it reads the lists.
        {
            vulkan::profiling::cpu_phase_timer const cluster_timer{this->cpu_timings, vulkan::profiling::cpu_phase::cluster};
            this->record_cluster_pass(*command_buffer);
        }

        // ---- Ray-traced shadows: build the acceleration structures once, before the passes that will
        //      trace against them. It happens HERE (inside the frame's command buffer, before any
        //      rendering instance opens) because a build is a transfer/compute-class command that must
        //      not be recorded inside vkCmdBeginRendering, and because the caster set is only complete
        //      now that the scene is loaded and culled. Nothing reads the structures yet, so a frame
        //      with the flag on renders exactly like one with it off - what this records is the input
        //      the ray-traced pass will need, not a change to the image.
        this->record_acceleration_structures(*command_buffer);
        // ... and the top level structure, which is rebuilt EVERY frame: the instance set is culled per
        // frame and a caster's world matrix can change (animation, a moved node), so the instance list
        // is frame data like any other. On the frame that builds the bottom levels it is a no-op for
        // the reasons above (nothing to trace yet); from the next frame on it is the structure a shadow
        // ray will traverse.
        this->record_top_level_structure(*command_buffer);
        // GPU timing: the structures' builds end here. Written UNCONDITIONALLY, like every other mark -
        // a frame that skips a pass still writes its mark next to the previous one (0 ms interval), and
        // the report's labels are positional: leaving a gap here relabeled the whole frame ("marks
        // recorded out of order", which the harness caught on all seven scenarios at once).
        this->gpu_mark(*command_buffer, gpu_mark_id::rt_build_end, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

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
        // Feature registry (runtime::active_features): the pass runs only when a shading stage will
        // actually read the map - the flat render mode skips it entirely, which is worth ~45% of the
        // frame (see the measured numbers in the struct's documentation).
        render_features const features = this->active_features();
        // Geometry half of the reuse test: one fingerprint of every input the pass reads - see
        // shadow_geometry_signature().
        uint64_t const geometry_signature = this->shadow_geometry_signature();
        // Skip the whole pass when this slot's maps are still valid: unchanged fit (content version)
        // AND unchanged caster geometry (hash). The maps stay in SHADER_READ_ONLY and the descriptor
        // set still points at them, so there is nothing to record - not even the read barriers.
        bool const shadow_reuse = this->shadow_rendered_version[frame_slot] == this->shadow_content_version && this->shadow_rendered_models[frame_slot] == geometry_signature;
        if (features.shadow && !shadow_reuse) {
            vulkan::profiling::cpu_phase_timer const shadow_timer{this->cpu_timings, vulkan::profiling::cpu_phase::shadow}; // the sub-phase of scene that records every cascade
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
                // One task per cascade on the task pool (M9): each records into its OWN {pool, buffer}
                // pair (see shadow_recording), because a VkCommandPool is not thread safe - the same
                // rule the main-pass workers follow. Only the CONTENT recording moves off the primary
                // thread; the barriers, the per-cascade rendering instances and the executions below
                // stay here, in the layer order the attachments require, so the recorded commands are
                // identical to the sequential version.
                {
                    std::vector<std::function<void()>> cascade_tasks;
                    cascade_tasks.reserve(cascades);
                    for (uint32_t cascade = 0; cascade < cascades; ++cascade) {
                        cascade_tasks.emplace_back([this, frame_slot, cascade, &shadow_sec_begin, &shadow_recorded] {
                            VkCommandBuffer const cascade_secondary = *this->shadow_recording[frame_slot][cascade].second;
                            if (vkBeginCommandBuffer(cascade_secondary, &shadow_sec_begin) != VK_SUCCESS) {
                                utility::log("runtime: shadow secondary command buffer begin failed - cascade {} skipped this frame", cascade);
                                return;
                            }
                            // which cascade these casters are projected into (the vertex stage indexes
                            // the light UBO's matrix array with it)
                            vkCmdPushConstants(cascade_secondary, this->vulkan_core.scene_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, vulkan::scene_cascade_push_offset, sizeof(uint32_t), &cascade);
                            this->record_shadow_content(cascade_secondary);
                            vkEndCommandBuffer(cascade_secondary);
                            shadow_recorded[cascade] = true;
                        });
                    }
                    this->run_tasks(cascade_tasks, vulkan::task_priority::recording);
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

                    VkRenderingInfo const shadow_rendering_info = make_rendering_info(VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT, {{0, 0}, {this->shadow_map_size, this->shadow_map_size}}, false, nullptr, &shadow_depth_attachment);
                    vkCmdBeginRendering(*command_buffer, &shadow_rendering_info);

                    // Run this cascade's pre-recorded secondary (the whole scene casts shadows). Never
                    // execute a secondary whose begin failed - executing an unrecorded command
                    // buffer is a VUID and can wedge the frame slot.
                    if (shadow_recorded[cascade]) {
                        VkCommandBuffer const cascade_secondary = *this->shadow_recording[frame_slot][cascade].second;
                        vkCmdExecuteCommands(*command_buffer, 1, &cascade_secondary);
                    }
                    vkCmdEndRendering(*command_buffer);
                }

                // Hand the cascades back to the lighting stage as a sampled array texture: the layers
                // just rendered go depth-attachment -> shader-read (the src masks publish the
                // attachment write, so the sampled contents are the ones the pass produced).
                //
                // The spare layers - present only after set_shadow_cascades() SHRANK the count, which
                // deliberately keeps the layers it already owns - are covered here too, by ONE range
                // over every allocated layer. That is deliberate rather than a second barrier: the
                // descriptor's array view spans all of them, so a spare layer left in the attachment
                // layout would be a layout mismatch the moment the shader sampled it, and this range
                // is what makes "one barrier, whole array" the invariant. It is a no-op for the
                // rendered layers, whose range this already covers.
                std::array<VkImageMemoryBarrier2, 1> shadow_read_barrier = {shadow_map_sampling_transition};
                shadow_read_barrier[0].image = shadow_detail->image;
                shadow_read_barrier[0].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, this->shadow_allocated_layers};
                VkDependencyInfo const shadow_read_dependency = make_image_dependency_info(1, shadow_read_barrier.data());
                vkCmdPipelineBarrier2(*command_buffer, &shadow_read_dependency);
                this->shadow_rendered_version[frame_slot] = this->shadow_content_version;
                this->shadow_rendered_models[frame_slot] = geometry_signature;
            }
        } else if (features.shadow) {
            // Reuse: the maps were rendered into this slot by an earlier frame and nothing that feeds
            // them has changed since, so the sampling layout they are already in is the one the
            // lighting pass needs. Deliberately does nothing.
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
                // Every layer the image OWNS, not just layer 0: the constant's range is single-layer
                // and the whole array view the descriptor covers must be sampleable. The count is
                // shadow_allocated_layers - the layer count the image was actually created with - and
                // NOT max_shadow_cascades: the two differ whenever fewer cascades are active than
                // were ever allocated (the default is three), and a subresource range reaching past
                // the image's own layer count is a VUID on every shadow-off frame.
                shadow_read_barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, this->shadow_allocated_layers};
                VkDependencyInfo const shadow_read_dependency = make_image_dependency_info(1, &shadow_read_barrier);
                vkCmdPipelineBarrier2(*command_buffer, &shadow_read_dependency);
            }
        }

        // GPU timing: the shadow pass (and its hand-back barrier) ends here. The pass is optional,
        // so a frame without shadows just writes this mark next to frame_begin and reports ~0 ms.
        this->gpu_mark(*command_buffer, gpu_mark_id::shadow_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // The scene pass: the opaque leaves write the G-buffer, and everything else follows from it
        // in record_scene_tail (lighting, the transparent pass over the shaded image, TAA).
        this->record_scene(*command_buffer);
    }

    void runtime::record_scene(VkCommandBuffer const command_buffer) {
        this->record_scene_attachments(command_buffer);
        this->update_pass_geometry();
        // THE SCENE PASS records the surface write: the instance over its six declared targets, the segmented
        // draw of the visible leaves, and closing the instance - all inside one function now (see
        // vulkan.pass.scene for why that is the point of this extraction).
        if (this->gbuffer_pass_active()) {
            this->scene.set_frame(this->make_scene_frame());
            pass::stage const scene_stage = {.name = "scene", .passes = this->scene_stage, .marks = false};
            [[maybe_unused]] pass::run_report const scene_report = pass::record_stage(scene_stage, this->make_pass_host());
            return;
        }
        // THE DEGENERATE CASE, which is all the old begin_rendering() is still needed for: no surface pipeline,
        // so there is no pass to run. An EMPTY instance is opened and closed anyway, so the frame keeps a
        // matching pair and the scene target ends in a layout the post chain can sample. (The old code also
        // recorded the scene segments here - with a pipeline that does not exist, which validation would have
        // refused; drawing nothing is both shorter and true.)
        this->begin_rendering(command_buffer, this->current_image_index, 0);
        vkCmdEndRendering(command_buffer);
    }

    // Move the scene pass's attachments into their render layouts; see the declaration for why this
    // cannot be left to a render pass.
    void runtime::record_scene_attachments(VkCommandBuffer const command_buffer) {
        core& vk = this->vulkan_core;
        // Dynamic rendering has no automatic attachment transitions (a render pass would do them
        // implicitly): move every attachment into its render layout before vkCmdBeginRendering. The
        // set is the G-buffer mode's three single-sampled surface targets, the motion-vector target
        // and the scene color the emissive goes into, plus the pass's own 1x depth image. The main
        // HDR target is not touched by the opaque G-buffer pass at all (the debug view writes it
        // afterwards), so it must not be transitioned here.
        //
        // room for every pass attachment plus the depth. Sizing this for the old three-target
        // G-buffer was a stack overflow the moment the emissive attachment arrived - validation
        // reported the fifth barrier as garbage.
        std::array<VkImageMemoryBarrier2, vulkan::gbuffer_pass_attachment_count + 1> attachment_barriers = {};
        uint32_t barrier_count = 0;
        auto const add_render_barrier = [&attachment_barriers, &barrier_count](VkImageMemoryBarrier2 const& transition, VkImage const image) {
            VkImageMemoryBarrier2& barrier = attachment_barriers[barrier_count++];
            barrier = transition; // copy the role default, then retarget it
            barrier.image = image;
        };

        {
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
            // this instance writes the G-buffer depth, so it is in the attachment layout from here on:
            // the stage that samples it later calls ensure_gbuffer_depth_sampled() (see the accessor).
            // The flag vector is per swapchain image and sized with the generation; the guard is a
            // no-op outside that range, so a stale size can never set a flag for an image that does
            // not exist (and on_swapchain_recreated re-sizes it on every generation change).
            if (this->current_image_index < this->gbuffer_depth_written.size()) {
                this->gbuffer_depth_written[this->current_image_index] = true;
            }
            // ... and the motion-vector target it just wrote as a color attachment. Two stages sample
            // it (TAA's resolve and, since the GI denoiser, a compute resolve), so the "has this been
            // handed to a sampler yet" question is asked through a flag rather than assumed - see
            // ensure_velocity_sampled.
            if (this->current_image_index < this->velocity_written.size()) {
                this->velocity_written[this->current_image_index] = true;
            }
            // ... and the three stored surface targets, written as color attachments by the same
            // instance: the ray-traced shadow pass samples them before the lighting stage does, so which
            // stage publishes them is a question for a flag too (see ensure_gbuffer_targets_sampled).
            if (this->current_image_index < this->gbuffer_targets_written.size()) {
                this->gbuffer_targets_written[this->current_image_index] = true;
            }
        }

        VkDependencyInfo const dependency_info = make_image_dependency_info(barrier_count, attachment_barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency_info);
    }

    // Resync the cached viewport/scissor of every pipeline that draws this frame.
    void runtime::update_pass_geometry() {
        core& vk = this->vulkan_core;
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
        // The TAA resolve's pipeline and viewport are NOT resynced here: the runner sets a fullscreen pass's
        // viewport and scissor from the extent its declaration produced, which is what `resync_viewport`
        // means once a pass states it (see vulkan.pass's behaviour). The hazard this list used to guard - a
        // fullscreen pass setting a zero-width viewport because it was left out - is gone by construction.
    }

    // Depth-only shadow-pass content: bind the shared scene set (the light UBO binding 7) +
    // the shadow pipeline, apply the live depth bias and draw every scene-tree leaf (the whole
    // scene casts shadows). Pure bind/push/draw commands - the caller owns the barriers and
    // the depth-only rendering instance around it. Recorded inline today; stage 2 records the
    // same content into a per-slot secondary command buffer for parallel pass recording.
    void runtime::record_shadow_content(VkCommandBuffer const command_buffer) const {
        core const& vk = this->vulkan_core;
        uint32_t const frame_slot = static_cast<uint32_t>(vk.current_frame);
        if (this->scene_sets.created()) {
            VkDescriptorSet const scene_set_handle = this->scene_sets.set(static_cast<uint32_t>(frame_slot));
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

    // ---- post-processing: HDR scene target -> exposure + ACES + gamma -> swapchain ----
    // The bloom chain lands here next; the push constants already reserve its parameters.
    std::expected<void, std::string> runtime::make_post_pipeline(std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        core& vk = this->vulkan_core;
        // sampler for the HDR scene target (linear, clamp) - the descriptor sets use it
        this->post_sampler = vk.make_sampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 1.0f);
        // ... and the nearest one the composite's GI upsample needs (see post_nearest_sampler)
        {
            VkSamplerCreateInfo nearest_info = make_texture_sampler_info(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f);
            nearest_info.magFilter = VK_FILTER_NEAREST;
            nearest_info.minFilter = VK_FILTER_NEAREST;
            VkSampler nearest = VK_NULL_HANDLE;
            if (vkCreateSampler(vk.device, &nearest_info, nullptr, &nearest) != VK_SUCCESS) {
                return std::unexpected(std::string("post: nearest sampler creation failed"));
            }
            this->post_nearest_sampler = vk_sampler(nearest, vk.device);
        }

        // the layout and both composites come from vulkan.pipelines; the push constant block stays here
        // (it must match post.frag, so it lives next to the code that fills it)
        auto built = pipelines::build_post(vk, sizeof(post_push_constants), vertex_shader_code, fragment_shader_code);
        if (!built) {
            return std::unexpected(std::move(built.error()));
        }
        this->post_set_layout = built->set_layout;
        this->post_pipeline_layout = built->pipeline_layout;
        this->post_pipeline = std::move(built->composite);
        this->post_hdr_pipeline = std::move(built->hdr);
        return {};
    }

    std::expected<void, std::string> runtime::make_fxaa_pipeline(std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        core& vk = this->vulkan_core;
        if (this->post_pipeline_layout == VK_NULL_HANDLE) {
            return fail(std::string("fxaa: create the post-process pipeline first (it owns the set layout)"));
        }
        auto built = pipelines::build_fxaa(vk, this->post_pipeline_layout, vertex_shader_code, fragment_shader_code);
        if (!built) {
            return std::unexpected(std::move(built.error()));
        }
        this->post_fxaa_pipeline = std::move(*built);
        return {};
    }

    void runtime::ensure_post_descriptors() {
        core& vk = this->vulkan_core;
        if (this->post_pipeline == std::nullopt || this->post_hdr_pipeline == std::nullopt || this->post_set_layout == VK_NULL_HANDLE) {
            return;
        }
        std::size_t const image_count = vk.hdr_image_views.size();
        if (image_count == 0 || vk.bloom_image_views[0].size() != image_count || vk.ldr_image_views.size() != image_count ||
            vk.gi_spatial_image_views.size() != image_count || vk.gbuffer_depth_image_views.size() != image_count ||
            vk.gbuffer_image_views[1].size() != image_count) {
            return;
        }
        // The family owns the rebinding rule, the pool sizing (five sets per image, six descriptors
        // each) and the retirement (see vulkan.bindings); what stays here is what is specific to the
        // post chain: six fingerprints - HDR, bloom, LDR, GI, depth and normal views - and how one
        // image's five sets are written.
        std::array<std::span<VkImageView const>, 6> const fingerprints = {
            vk.hdr_image_views, vk.bloom_image_views[0], vk.ldr_image_views, vk.gi_spatial_image_views, vk.gbuffer_depth_image_views, vk.gbuffer_image_views[1]};
        auto const write_sets = [this](uint32_t const image_index, std::span<VkDescriptorSet const> const sets) {
            // every set gets all nine bindings; the unused ones point at the same view as binding 0
            // (binding 5 is the LDR image, which only the FXAA pass reads, and 7/8 are the G-buffer
            // depth and normal, which only the composite's GI upsample reads)
            auto const write_set = [this](VkDescriptorSet const set, std::array<VkImageView, 9> const& views) {
                std::array<VkDescriptorImageInfo, 9> image_infos = {};
                for (uint32_t b = 0; b < image_infos.size(); ++b) {
                    // The composite's GI upsample taps the depth and the normal AT texel centres, and
                    // for those two an interpolated value is not a rounding error but a different
                    // surface - so they get the nearest sampler and the edge test sees the stored
                    // values. The GI image itself keeps the LINEAR one, for a reason that is about
                    // measurement rather than quality: its texels are still read at centres (a centre
                    // fetch of a linear sampler returns that texel), but the upsample's off switch is
                    // the plain bilinear fetch, and that has to be the same fetch the chain used before
                    // the upsample existed - with a nearest sampler it would be a much blurrier
                    // comparison and the A/B would be measuring two differences at once.
                    bool const nearest = b == 7u || b == 8u;
                    image_infos[b].sampler = nearest ? *this->post_nearest_sampler : *this->post_sampler;
                    image_infos[b].imageView = views[b];
                    image_infos[b].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                }
                std::array<VkWriteDescriptorSet, 9> writes = {};
                for (uint32_t b = 0; b < writes.size(); ++b) {
                    writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    writes[b].dstSet = set;
                    writes[b].dstBinding = b;
                    writes[b].descriptorCount = 1;
                    writes[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    writes[b].pImageInfo = &image_infos[b];
                }
                vkUpdateDescriptorSets(this->vulkan_core.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
            };

            VkImageView const hdr = this->vulkan_core.hdr_image_views[image_index];
            VkImageView const ldr = this->vulkan_core.ldr_image_views[image_index];
            // The FILTERED GI (the last pass of the GI chain), not the raw trace or the temporal
            // accumulation: the composite is the consumer of the denoiser's output, and the earlier
            // images are bound in the G-buffer set or in the denoiser's own set instead.
            VkImageView const gi = this->vulkan_core.gi_spatial_image_views[image_index];
            VkImageView const depth = this->vulkan_core.gbuffer_depth_image_views[image_index];
            VkImageView const normal = this->vulkan_core.gbuffer_image_views[1][image_index];
            std::array<VkImageView, 9> const hdr_set = {hdr, hdr, hdr, hdr, hdr, ldr, gi, depth, normal};
            write_set(sets[0], hdr_set);

            for (std::size_t level = 0; level < 3; ++level) {
                VkImageView const input = this->vulkan_core.bloom_image_views[level][image_index];
                std::array<VkImageView, 9> const level_set = {input, input, input, input, input, ldr, gi, depth, normal};
                write_set(sets[1 + level], level_set);
            }

            std::array<VkImageView, 9> const composite_set = {hdr,
                                                              this->vulkan_core.bloom_image_views[0][image_index],
                                                              this->vulkan_core.bloom_image_views[1][image_index],
                                                              this->vulkan_core.bloom_image_views[2][image_index],
                                                              this->vulkan_core.bloom_image_views[3][image_index],
                                                              ldr,
                                                              gi,
                                                              depth,
                                                              normal};
            write_set(sets[4], composite_set);
        };
        if (!this->post_family.ensure_all(vk.device, this->post_set_layout, static_cast<uint32_t>(image_count), 5u, 9u, fingerprints, write_sets)) {
            utility::log("runtime: post descriptor sets unavailable - post pass skipped");
        }
    }
    // ---- G-buffer / deferred path (M1: the write pass + its debug view) ----
    // The G-buffer pass is the forward opaque pass with a different fragment stage: same vertex
    // stage, same primitives, same scene set, same instancing/skinning/morphing. What changes is
    // where the fragments go (three 1x targets + a 1x depth image instead of the scene color)
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
            return fail(std::string("deferred: create the G-buffer debug pipeline first (it owns the G-buffer set layout)"));
        }
        if (vk.scene_descriptor_set_layout == VK_NULL_HANDLE) {
            return fail(std::string("deferred: the shared scene descriptor set layout is missing"));
        }
        std::array<VkPipelineColorBlendAttachmentState, 1> const blend = {make_color_blend_attachment_additive()};
        auto built = pipelines::build_deferred(vk, vk.scene_descriptor_set_layout, this->gbuffer_set_layout, sizeof(deferred_push_constants), std::span<VkPipelineColorBlendAttachmentState const>(blend), vertex_shader_code, fragment_shader_code);
        if (!built) {
            return std::unexpected(std::move(built.error()));
        }
        this->deferred_pipeline_layout = built->pipeline_layout;
        this->deferred_pipeline = std::move(built->lighting);
        return {};
    }

    std::expected<void, std::string> runtime::make_gbuffer_debug_pipeline(std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        core& vk = this->vulkan_core;
        // the G-buffer set layout is owned here (vulkan.pipelines) because deferred reuses it
        auto built = pipelines::build_gbuffer_debug(vk, sizeof(gbuffer_debug_push_constants), vertex_shader_code, fragment_shader_code);
        if (!built) {
            return std::unexpected(std::move(built.error()));
        }
        this->gbuffer_set_layout = built->set_layout;
        this->gbuffer_pipeline_layout = built->pipeline_layout;
        this->gbuffer_debug_pipeline = std::move(built->debug);

        // nearest, clamp: the debug view reads the G-buffer at exact texel centers
        VkSamplerCreateInfo sampler_info = make_texture_sampler_info(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f);
        sampler_info.magFilter = VK_FILTER_NEAREST;
        sampler_info.minFilter = VK_FILTER_NEAREST;
        VkSampler sampler = VK_NULL_HANDLE;
        if (vkCreateSampler(vk.device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
            return std::unexpected(std::string("gbuffer debug: sampler creation failed"));
        }
        this->gbuffer_sampler = vk_sampler(sampler, vk.device);

        // ... and the probe cache's, which is the same thing with LINEAR filtering: the grid is sampled
        // to interpolate between cells (see the member's comment). Created HERE rather than with the
        // probe pipeline because the tracer's descriptor set writes it whether or not that optional
        // pipeline exists - a null sampler in a set is a validation error, not a skipped fetch.
        VkSamplerCreateInfo probe_info = make_texture_sampler_info(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f);
        VkSampler probe_sampler = VK_NULL_HANDLE;
        if (vkCreateSampler(vk.device, &probe_info, nullptr, &probe_sampler) != VK_SUCCESS) {
            return std::unexpected(std::string("gbuffer debug: probe sampler creation failed"));
        }
        this->gi_probe_sampler = vk_sampler(probe_sampler, vk.device);
        return {};
    }

    void runtime::record_lighting_pass(VkCommandBuffer const command_buffer) {
        core const& vk = this->vulkan_core;
        if (this->deferred_pipeline == std::nullopt) {
            return;
        }
        std::size_t const index = this->current_image_index;

        // Transitions, all before vkCmdBeginRendering (a pipeline barrier may not be recorded inside
        // a dynamic rendering instance): the three surface targets and the G-buffer depth become
        // shader inputs, and the scene color - which the G-buffer pass already filled with the
        // emissive - stays a color attachment with its contents LOADed, because the lighting is
        // added on top of them.
        //
        // The depth uses ensure_gbuffer_depth_sampled(), NOT a bare transition: the G-buffer pass
        // rendered that depth earlier in THIS command buffer, so its old layout is known to be
        // DEPTH_STENCIL_ATTACHMENT_OPTIMAL and the src masks have to publish the attachment write.
        // Declaring UNDEFINED would let the implementation discard precisely the contents the
        // lighting stage reconstructs world positions from (see the accessor).
        //
        // The scene-color dependency is separate and cannot be folded into those three: it does not
        // change layout, and its consumer is the second instance's LOAD of the attachment, which is
        // a color-attachment access rather than the FRAGMENT_SHADER read the sampling transitions
        // publish. Dynamic rendering inserts no dependency of its own between two instances, so
        // without it the load is not ordered after (nor made visible from) the G-buffer pass's store.
        // The image comes from scene_target_image(), the same accessor the attachment below uses, so
        // the barrier always names the image this instance actually LOADs.
        VkImage const scene_target = this->scene_target_image(static_cast<uint32_t>(index));
        std::array<VkImageMemoryBarrier2, 1> barriers = {};
        barriers[0] = vulkan::color_attachment_dependency; // G-buffer store -> this instance's LOAD
        barriers[0].image = scene_target;
        VkDependencyInfo const dependency = make_image_dependency_info(static_cast<uint32_t>(barriers.size()), barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency);
        // The three stored targets become samples HERE, but only if an earlier stage (the ray-traced
        // shadow pass, which runs between the G-buffer instance and this one) has not already published
        // them - the flag is what keeps the second transition from claiming a layout they are not in.
        this->ensure_gbuffer_targets_sampled(command_buffer, static_cast<uint32_t>(index));
        this->ensure_gbuffer_depth_sampled(command_buffer, static_cast<uint32_t>(index));

        this->ensure_gbuffer_descriptors();
        if (this->gbuffer_family.set(static_cast<uint32_t>(index), 0) == VK_NULL_HANDLE) {
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

        // The G-buffer pass left the scene color in COLOR_ATTACHMENT_OPTIMAL, and the dependency
        // barrier above ordered its store before this instance's LOAD, so the attachment needs no
        // layout change of its own: a load-op LOAD instance adds the lighting on top of the emissive
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
        std::array<VkDescriptorSet, 2> const sets = {this->scene_sets.set(static_cast<uint32_t>(vk.current_frame)), this->gbuffer_family.set(static_cast<uint32_t>(index), 0)};
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, this->deferred_pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);
        // SSAO (M6) rides the same block: an intensity of 0 when the feature is off, which makes
        // ssao_occlusion() return exactly 1.0 - the shaded result is then the pre-M6 value bit for bit
        deferred_push_constants const push = {
            .inv_view_proj = this->current_inv_view_proj,
            .ssao = glm::vec4(this->ssao_radius,
                              this->ssao_enabled ? this->ssao_intensity : 0.0f,
                              static_cast<float>(this->ssao_samples),
                              this->ssao_bias),
            // render mode: the flat "unlit" default pipeline becomes "write the stored albedo" here
            .unlit = this->unlit_active ? 1.0f : 0.0f,
            // ... and whether the traced chain is replacing the ambient this frame, which the lighting stage
            // needs so it does not scale a term that is about to be taken back out (see the field's comment).
            // The SAME predicate the spatial filter's subtraction uses, so the two cannot disagree about
            // whether the ambient the chain replaces is the occluded one or the plain one.
            .gi_replaces_ambient = this->ssgi_traced_active() ? 1.0f : 0.0f,
        };
        vkCmdPushConstants(command_buffer, this->deferred_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(command_buffer, 3, 1, 0, 0);
        vkCmdEndRendering(command_buffer);
    }

    // The deferred path's transparent pass. Everything about its position is load-bearing:
    //  - after the lighting stage, because a blended surface composites over SHADED pixels, and the
    //    G-buffer instance has no shaded image to composite over;
    //  - not INSIDE the lighting instance, because that one samples the G-buffer depth as a texture
    //    while this one needs the same image as its depth ATTACHMENT, and an image cannot be both in
    //    one instance;
    //  - before the TAA resolve, so the resolve sees the composited frame.
    void runtime::record_transparent_pass(VkCommandBuffer const command_buffer) {
        static_cast<void>(command_buffer); // the pass records into the frame's command buffer it is resolved with
        // THE TRANSPARENT PASS records the blended geometry: the two hand-off barriers, the LOAD instance over
        // its two declared targets, one secondary and the depth hand-back - all in one function now (see
        // vulkan.pass.transparent). Like the scene pass it is SKIPPED without resolving anything on a frame
        // whose culling left nothing blended, which is what keeps a frame with no blended leaves byte-exact.
        this->transparent.set_frame(this->make_transparent_frame());
        pass::stage const transparent_stage = {.name = "transparent", .passes = this->transparent_stage, .marks = false};
        [[maybe_unused]] pass::run_report const transparent_report = pass::record_stage(transparent_stage, this->make_pass_host());
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
        // the engine's answer to aliasing, now that there is no MSAA to fall back on.
        return this->taa_on && this->taa_resolve.pipeline_ready() && this->deferred_lit_active();
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
        if (enabled) {
            if (!this->taa_resolve.pipeline_ready()) {
                this->warn_missing_feature("taa", "TAA has no effect: the taa pipeline was not created (see the startup log)");
            } else if (!this->deferred_lit_active()) {
                this->warn_missing_feature("taa", "TAA has no effect: the G-buffer pass or its lighting stage was not created (see the startup log)");
            }
        }
        this->taa_blend_static = std::clamp(blend_static, 0.0f, 0.99f);
        this->taa_blend_min = std::clamp(blend_min, 0.0f, this->taa_blend_static);
        if (enabled && !was_on) {
            // A fresh history - but only on the off -> on EDGE. The caller mirrors the GUI/config state
            // into the runtime every frame (see main.cpp), so resetting unconditionally here would
            // invalidate the history on every frame: the resolve would fall back to the current
            // (jittered, aliased) frame forever, which looks like TAA running while doing nothing.
            std::size_t const image_count = this->vulkan_core.taa_history_images.size();
            this->taa_resolve.reset_history(); // the pass owns whether each image's history holds anything
            this->image_view_proj.assign(image_count, this->current_ubo.view_proj_unjittered);
            this->taa_jitter_index = 0;
        }
    }

    // The TAA resolve's factory is gone: `vulkan.pass.taa` builds its own set layout, pipeline layout and
    // pipeline in its create step, from its own declaration and its own shaders (the app registers those).

    bool runtime::ensure_gbuffer_depth_sampled(VkCommandBuffer const command_buffer, uint32_t const image_index) {
        // Nothing to do when no G-buffer instance ran for this image: the depth is already in the
        // layout the sampling descriptors declare (it keeps whatever the last frame for this image
        // left it in), so re-transitioning would only claim a layout the image is not in.
        if (image_index >= this->gbuffer_depth_written.size() || !this->gbuffer_depth_written[image_index]) {
            return false;
        }
        // The G-buffer pass left it in DEPTH_STENCIL_ATTACHMENT_OPTIMAL: publish the attachment
        // write and flip it to the layout the sampling descriptors declare. One barrier per frame,
        // whichever of the three sampling stages gets here first.
        VkImageMemoryBarrier2 barrier = vulkan::shadow_map_sampling_transition;
        barrier.image = this->vulkan_core.gbuffer_depth_images[image_index];
        VkDependencyInfo const dependency = make_image_dependency_info(1, &barrier);
        vkCmdPipelineBarrier2(command_buffer, &dependency);
        this->gbuffer_depth_written[image_index] = false;
        return true;
    }

    bool runtime::ensure_gbuffer_targets_sampled(VkCommandBuffer const command_buffer, uint32_t const image_index) {
        // Nothing to do when no G-buffer instance ran for this image, or when an earlier stage already
        // took the transition (the ray-traced shadow pass runs first when it runs): re-transitioning
        // would claim a COLOR_ATTACHMENT old layout the image is not in.
        if (image_index >= this->gbuffer_targets_written.size() || !this->gbuffer_targets_written[image_index]) {
            return false;
        }
        std::array<VkImageMemoryBarrier2, vulkan::gbuffer_target_count> barriers = {};
        for (uint32_t target = 0; target < vulkan::gbuffer_target_count; ++target) {
            barriers[target] = vulkan::hdr_sampling_transition; // COLOR_ATTACHMENT -> SHADER_READ
            barriers[target].image = this->vulkan_core.gbuffer_images[target][image_index];
        }
        VkDependencyInfo const dependency = make_image_dependency_info(static_cast<uint32_t>(barriers.size()), barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency);
        this->gbuffer_targets_written[image_index] = false;
        return true;
    }
    bool runtime::ensure_velocity_sampled(VkCommandBuffer const command_buffer, uint32_t const image_index) {
        // Nothing to do when no G-buffer instance ran for this image, or when an earlier sampler
        // already took the transition (the TAA resolve, which runs before the GI denoiser in the same
        // frame): re-transitioning would claim a COLOR_ATTACHMENT old layout the image is not in.
        if (image_index >= this->velocity_written.size() || !this->velocity_written[image_index]) {
            return false;
        }
        VkImageMemoryBarrier2 barrier = vulkan::hdr_sampling_transition; // COLOR_ATTACHMENT -> SHADER_READ
        barrier.image = this->vulkan_core.velocity_images[image_index];
        VkDependencyInfo const dependency = make_image_dependency_info(1, &barrier);
        vkCmdPipelineBarrier2(command_buffer, &dependency);
        this->velocity_written[image_index] = false;
        return true;
    }

    bool runtime::gbuffer_pass_active() const noexcept {
        if (!this->gbuffer_pipeline.has_value()) {
            return false;
        }
        if (this->gbuffer_debug) {
            return this->gbuffer_debug_pipeline.has_value();
        }
        return this->deferred_pipeline.has_value();
    }

    bool runtime::deferred_lit_active() const noexcept {
        // the debug view wins when both are available: looking at the stored data is an inspection,
        // not a render mode (and the two write the HDR target in incompatible ways)
        return this->gbuffer_pass_active() && !this->gbuffer_debug;
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
        if (image_count == 0 || vk.gbuffer_depth_image_views.size() != image_count || vk.hdr_image_views.size() != image_count || vk.gi_image_views.size() != image_count ||
            vk.gi_spec_image_views.size() != image_count || vk.gi_spec_reproject_image_views.size() != image_count ||
            vk.gi_spec_resolve_image_views.size() != image_count ||
            vk.gi_probe_image_views.empty() || this->gi_probe_sampler.get() == VK_NULL_HANDLE) {
            return;
        }
        // The family owns the rebinding rule now (see vulkan.bindings): the sets stay allocated, their
        // contents are rewritten only when the targets below change, and a pool replaced by a later
        // generation is retired rather than destroyed, because recorded frame command buffers still
        // name its sets. on_swapchain_recreated() retires the family, which is what forces the rewrite.
        // The signature is ALSO this family's descriptors-per-set (it sizes the pool - see vulkan.bindings),
        // so it has to list every binding the layout declares, not just the ones that can move together.
        std::array<VkImageView, 16> const signature = {
            vk.gbuffer_image_views[0][0],
            vk.gbuffer_image_views[1][0],
            vk.gbuffer_image_views[2][0],
            vk.gbuffer_depth_image_views[0],
            vk.velocity_image_views[0],
            vk.hdr_image_views[0],
            vk.gi_image_views[0],
            vk.gi_resolve_image_views[0],
            vk.gi_spatial_image_views[0],
            // Bindings 9..12 are the world-space probe cache's four SH-2 coefficients: ONE set of images for
            // the whole device (the cache is anchored to the world, not to a swapchain image), so every set
            // fingerprints the same views - and a recreated grid still invalidates them all, which is what
            // these entries are for.
            vk.gi_probe_image_views[0],
            vk.gi_probe_image_views[1],
            vk.gi_probe_image_views[2],
            vk.gi_probe_image_views[3],
            // 13 and 14 are the glossy lobe's own outputs. Per swapchain image, like the trace they shadow:
            // a reflection's correction and its reprojection belong to the frame that produced them.
            vk.gi_spec_image_views[0],
            vk.gi_spec_reproject_image_views[0],
            vk.gi_spec_resolve_image_views[0]};
        // One set per image with one descriptor per binding: the three stored targets, the depth, the
        // motion-vector target, the direct-radiance image the tracer samples at a hit, the raw trace it
        // writes, the accumulated image the spatial filter reads, the filtered image it writes, the four
        // probe coefficients it reads for a hit the screen cannot answer, and the glossy lobe's two outputs -
        // the same fifteen the signature above fingerprints.
        // image_count is the generation's, signature is only the fingerprint of image 0 above - the two
        // are different things and the family needs both (see vulkan.bindings).
        auto const write_sets = [this](uint32_t const image_index, std::span<VkDescriptorSet const> const sets) {
            std::array<VkDescriptorImageInfo, 16> image_infos = {};
            std::array<VkImageView, 16> const views = {
                this->vulkan_core.gbuffer_image_views[0][image_index],
                this->vulkan_core.gbuffer_image_views[1][image_index],
                this->vulkan_core.gbuffer_image_views[2][image_index],
                this->vulkan_core.gbuffer_depth_image_views[image_index],
                this->vulkan_core.velocity_image_views[image_index],
                this->vulkan_core.hdr_image_views[image_index],        // 5: direct radiance, what a hit returns
                this->vulkan_core.gi_image_views[image_index],         // 6: the RAW trace the tracer writes
                this->vulkan_core.gi_resolve_image_views[image_index], // 7: the accumulation the filter reads
                this->vulkan_core.gi_spatial_image_views[image_index], // 8: the filtered GI the composite reads
                // 9..12: the world-space probe cache's four SH-2 coefficients (one copy for the whole
                // device, so index 0 rather than this image's - see the ping-pong in vulkan.pass.gi_probe:
                // the cache side is always the one the tracer reads).
                this->vulkan_core.gi_probe_image_views[0],
                this->vulkan_core.gi_probe_image_views[1],
                this->vulkan_core.gi_probe_image_views[2],
                this->vulkan_core.gi_probe_image_views[3],
                // 13 and 14: the glossy lobe's own two outputs (see core.cppm's gi_spec_*).
                this->vulkan_core.gi_spec_image_views[image_index],
                this->vulkan_core.gi_spec_reproject_image_views[image_index],
                // 15: the reflection's own accumulation, which the spatial filter samples and sums the
                // diffuse one into (see shaders/ssgi_spatial.comp and ssgi_temporal.comp's mode 1).
                this->vulkan_core.gi_spec_resolve_image_views[image_index]};
            std::array<VkWriteDescriptorSet, 16> writes = {};
            for (uint32_t b = 0; b < views.size(); ++b) {
                // 6, 8, 13 and 14 are STORAGE images (a compute pass writes each) and therefore have no
                // sampler and live in GENERAL; the twelve sampler bindings are all SHADER_READ, including 15,
                // which the spatial filter reads rather than writes.
                bool const storage = b == 6u || b == 8u || b == 13u || b == 14u;
                // The probe cache is a 3D texture read with LINEAR filtering: the whole point of sampling
                // it is interpolating between cells, so it cannot borrow the G-buffer's NEAREST sampler.
                image_infos[b].sampler = storage ? VK_NULL_HANDLE : (b >= 9u && b <= 12u ? *this->gi_probe_sampler : *this->gbuffer_sampler);
                image_infos[b].imageView = views[b];
                image_infos[b].imageLayout = storage ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = sets[0];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[b].pImageInfo = &image_infos[b];
            }
            vkUpdateDescriptorSets(this->vulkan_core.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        };
        if (!this->gbuffer_family.ensure(vk.device, this->gbuffer_set_layout, static_cast<uint32_t>(image_count), 1u, static_cast<uint32_t>(signature.size()), signature, write_sets)) {
            utility::log("runtime: gbuffer debug descriptor sets unavailable - debug view skipped");
        }
    }

    bool runtime::ssgi_active() const noexcept {
        // The tracer reads the direct radiance the lighting stage produced and the G-buffer depth it
        // wrote, so that stage has to have run. The debug view replaces it, so there is no GI there.
        // The FLAT render mode is excluded for the same reason and it is not a style choice: in that
        // mode the lighting stage returns the stored albedo, so the image the tracer averages is not
        // radiance, and the GI it would add is a product of two albedos rather than a transport term.
        // It became reachable when `ssgi` defaulted to true, which is why the `unlit` reference frame is
        // the check for it: with this exclusion in place that frame comes out byte-identical to what it
        // was before the default moved.
        // The whole denoise chain is required as well, and not merely as a quality step: what the
        // composite samples is the SPATIAL filter's output, so a build with any one of the three
        // passes missing has nothing to composite. Treating that as "GI off" keeps the composite's
        // weight at 0 - the alternative is a full-resolution frame of whatever the last image happens
        // to contain.
        return this->ssgi_on && this->ssgi_pipeline.has_value() && this->ssgi_temporal_pipeline.has_value() &&
               this->ssgi_spatial_pipeline.has_value() && this->deferred_lit_active() && !this->unlit_active;
    }

    std::expected<void, std::string> runtime::make_ssgi_pipeline(std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        if (!this->deferred_pipeline.has_value()) {
            return fail(std::string("ssgi: create the deferred lighting pipeline first (it owns the G-buffer set layout)"));
        }
        auto built = pipelines::build_ssgi(this->vulkan_core, this->vulkan_core.scene_descriptor_set_layout, this->gbuffer_set_layout, sizeof(ssgi_push_constants), compute_shader_code);
        if (!built) {
            return fail(built.error());
        }
        this->ssgi_pipeline_layout = built->pipeline_layout;
        this->ssgi_pipeline = std::move(built->trace);
        return {};
    }

    void runtime::set_ssgi(bool const enabled, float const intensity, float const radius, uint32_t const rays, uint32_t const steps) noexcept {
        bool const was_on = this->ssgi_on;
        this->ssgi_on = enabled;
        this->ssgi_intensity = intensity;
        this->ssgi_radius = radius;
        this->ssgi_rays = std::clamp(rays, 0u, 16u);
        this->ssgi_steps = std::clamp(steps, 0u, 64u);
        if (enabled && !this->ssgi_pipeline.has_value()) {
            this->warn_missing_feature("ssgi", "screen-space GI has no effect: its compute pipeline was not created (see the startup log)");
        } else if (enabled && !this->ssgi_temporal_pipeline.has_value()) {
            this->warn_missing_feature("ssgi", "screen-space GI has no effect: its temporal resolve was not created (see the startup log)");
        } else if (enabled && !this->ssgi_spatial_pipeline.has_value()) {
            this->warn_missing_feature("ssgi", "screen-space GI has no effect: its spatial filter was not created (see the startup log)");
        }
        if (enabled && !was_on) {
            // A fresh accumulation, on the off -> on EDGE only. Today the one caller is startup
            // (main.cpp applies the config once), so this is mostly a guard for the shape of the
            // setter: it mirrors set_taa, and a future overlay control that calls it every frame must
            // not have the history thrown away on each of those calls - the resolve would show the raw
            // trace forever, which looks like a denoiser running while doing nothing.
            this->gi_history_valid.assign(this->vulkan_core.gi_history_images.size(), false);
            this->gi_spec_seen.assign(this->vulkan_core.gi_spec_images.size(), false); // new targets: the lobe's outputs need their first-use transition again
        }
    }

    void runtime::record_ssgi_pass(VkCommandBuffer const command_buffer) {
        core& vk = this->vulkan_core;
        std::size_t const index = this->current_image_index;
        if (index >= vk.gi_images.size() || vk.gi_images[index] == VK_NULL_HANDLE) {
            return;
        }
        this->ensure_gbuffer_descriptors();
        VkDescriptorSet const gbuffer_set = this->gbuffer_family.set(static_cast<uint32_t>(index), 0);
        if (gbuffer_set == VK_NULL_HANDLE) {
            return;
        }

        // The G-buffer depth, albedo and normal are already SHADER_READ (the lighting stage put them
        // there) and so is the HDR scene target (record_post_process transitioned it just before this
        // runs). Only the GI image needs anything: it is written as a storage image, so UNDEFINED ->
        // GENERAL here and GENERAL -> SHADER_READ below, for the composite that samples it.
        //
        // The resolve image needs one transition too, and only on its FIRST use for this image: the
        // traced path samples the PREVIOUS frame's resolve at a hit (the multi-bounce feedback), which
        // is a read of a storage image the temporal pass has not rewritten yet this frame. Every later
        // frame finds it in SHADER_READ (the denoise pass's hand-back leaves it there) and needs no
        // barrier at all - and must not get one claiming UNDEFINED, which would discard the very image
        // the feedback reads. The same per-image first-use flag the history image uses, for the same
        // reason (see record_ssgi_denoise_pass and gi_history_valid).
        bool const history_valid = index < this->gi_history_valid.size() && this->gi_history_valid[index];
        // The probe cache needs the same treatment for the same class of reason, and it is needed even
        // when the cache is OFF: the tracer declares the sampler3D unconditionally (the descriptor is
        // always written with a real view, because a null one is illegal), so the image behind it has to
        // be in a legal layout on every frame the tracer dispatches. UNDEFINED is the honest old layout
        // ONCE per target generation; after that the update pass's hand-back leaves the cache in
        // SHADER_READ (which is what the tracer wants) and the scratch in GENERAL (which is all it ever
        // is).
        bool const probe_first_use = !this->gi_probe_grid_seen && vk.gi_probe_images.size() == 8;
        std::array<VkImageMemoryBarrier2, 12> start_barriers = {};
        start_barriers[0] = vulkan::undefined_to_general_transition;
        start_barriers[0].image = vk.gi_images[index];
        uint32_t start_count = 1;
        if (!history_valid && index < vk.gi_spec_resolve_images.size()) {
            // The REFLECTION's own accumulation needs the same treatment, one step further out: the G-buffer
            // set names it at binding 15, so it has to be in the layout THAT set declares before the first
            // pass that binds the set - which is this one, not the pass that writes it. Leaving it undefined
            // until its own resolve ran is the validation error this line was written for: "expects VkImage
            // ... to be in layout SHADER_READ_ONLY_OPTIMAL--instead, current layout is VK_IMAGE_LAYOUT_UNDEFINED".
            start_barriers[start_count] = vulkan::undefined_to_sampling_transition;
            start_barriers[start_count].image = vk.gi_spec_resolve_images[index];
            ++start_count;
        }
        if (!history_valid && index < vk.gi_resolve_images.size()) {
            // SHADER_READ, not GENERAL, even though this frame's temporal pass will WRITE it: it is read
            // here FIRST (the tracer's bounce feedback samples it), so the layout its descriptor declares is
            // the one it has to start in - the temporal pass transitions it to GENERAL itself before writing
            // and back afterwards. A first-use transition is for the layout the frame's first READER needs,
            // not the writer. Getting this wrong is what the validation layer caught here: "Cannot use
            // VkImage ... with specific layout SHADER_READ_ONLY_OPTIMAL ... that doesn't match the previous
            // known layout VK_IMAGE_LAYOUT_GENERAL".
            start_barriers[start_count] = vulkan::undefined_to_sampling_transition;
            start_barriers[start_count].image = vk.gi_resolve_images[index];
            ++start_count;
        }
        if (probe_first_use) {
            // The cache's FOUR coefficients are sampled by this dispatch, so they start in SHADER_READ; the
            // scratch's four are only ever written by the propagation; and the per-cell geometry is written
            // by the injection. Nine images, which is why the array above is sized 12 (with the GI trace, the
            // diffuse resolve on its first use, and the reflection's accumulation on its first use).
            for (uint32_t c = 0; c < 4; ++c) {
                start_barriers[start_count] = vulkan::undefined_to_sampling_transition;
                start_barriers[start_count].image = vk.gi_probe_images[c];
                ++start_count;
            }
            for (uint32_t c = 0; c < 4; ++c) {
                start_barriers[start_count] = vulkan::undefined_to_general_transition;
                start_barriers[start_count].image = vk.gi_probe_images[4 + c];
                ++start_count;
            }
            start_barriers[start_count] = vulkan::undefined_to_general_transition; // the per-cell geometry: written by the injection
            start_barriers[start_count].image = vk.gi_probe_surface_images[0];
            ++start_count;
            this->gi_probe_grid_seen = true;
        }
        VkDependencyInfo const general_dependency = make_image_dependency_info(start_count, start_barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &general_dependency);

        // Same half-resolution rule as the images themselves (create_render_targets).
        uint32_t const gi_width = std::max(1u, vk.swap_chain_extent.width / 2u);
        uint32_t const gi_height = std::max(1u, vk.swap_chain_extent.height / 2u);

        std::array<VkDescriptorSet, 2> const sets = {this->scene_sets.set(static_cast<uint32_t>(vk.current_frame)), gbuffer_set};
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->ssgi_pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->ssgi_pipeline->get_pipeline());

        // ssgi_radius is a FRACTION of the scene radius, so one value means the same thing on a 1.6
        // unit model and on Sponza's 18.5 (the same reason shadow_fit works in scene units).
        //
        // The probe cache's two numbers are the same scene-relative ones: its cube is the scene's bounds
        // (what the shadow fit uses) and its cell size falls out of that cube and the fixed extent.
        bool const probe_ready = this->gi_probe_active() && this->gi_probe.cache_valid();
        float const probe_cell_size = (2.0f * this->scene_radius) / static_cast<float>(vulkan::gi_probe_grid_extent);
        // The instance table's device address, split across the two free lanes (see the shader): it is how
        // a hit learns which triangle it landed on, and it is also the switch - 0 keeps the screen-sampling
        // path, which is the A/B and is what a frame whose structures are not built yet gets.
        uint64_t instance_table = 0;
        if (this->ssgi_hit_shading && this->rt_top_levels.has_value()) {
            VkBuffer const table = this->rt_top_levels->instance_table(static_cast<uint32_t>(vk.current_frame));
            if (table != VK_NULL_HANDLE) {
                VkBufferDeviceAddressInfo const table_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = table};
                instance_table = vkGetBufferDeviceAddress(vk.device, &table_info);
            }
        }
        float const table_low = std::bit_cast<float>(static_cast<uint32_t>(instance_table & 0xFFFFFFFFu));
        float const table_high = std::bit_cast<float>(static_cast<uint32_t>(instance_table >> 32u));
        // Which of the two oracles this frame uses, evaluated ONCE and pushed in one lane
        // (frame_info.y): it also decides what params.w means, and two readings of the same predicate in
        // the same push would be two places to drift apart.
        bool const traced = this->ssgi_ray_tracing && this->vulkan_core.ray_query_available && this->rt_top_levels.has_value();
        ssgi_push_constants const push = {
            .inv_view_proj = this->current_inv_view_proj,
            // w is steps on a marched frame and the ray-origin bias on a traced one (see the shader's
            // push comment). The bias is a WORLD distance - a fraction of the scene radius, the same
            // meaning on a 1.6-unit model and on Sponza's 87.8 - rather than a fraction of the ray length,
            // which would make it scale with the reach knob: at the default settings that put every
            // traced ray's origin 0.21 world units above the surface inside Sponza.
            .params = glm::vec4(this->ssgi_radius * this->scene_radius,
                                this->ssgi_intensity,
                                static_cast<float>(this->ssgi_rays),
                                traced ? this->scene_radius * 0.0002f : static_cast<float>(this->ssgi_steps)),
            // The GI extent is NOT pushed: the shader asks the image it writes for its own size
            // (imageSize), which is the same number and one less lane to keep in sync. z/w carry the
            // instance table's address instead: a push constant is raw bytes, so a float lane holds an
            // address's half exactly as written and the shader reinterprets it (see the shader).
            .proj_terms = glm::vec4(this->current_ubo.proj[2][2], this->current_ubo.proj[3][2], table_low, table_high),
            .frame_info = glm::vec4(static_cast<float>(this->ssgi_frame),
                                    // ... and y = 1.0 only when the rays are actually traced: the device has ray queries,
                                    // the tracer ran and the structures exist. Resolved HERE rather than in the shader so
                                    // the shader never has to know why it is marching instead.
                                    traced ? 1.0f : 0.0f,
                                    // ... and z = the multi-bounce gain. Pushed on BOTH paths (a marched hit is
                                    // confirmed against the depth buffer too, so it has an indirect to re-emit), and
                                    // pushed every frame so that turning the knob off is a byte-exact no-op. Zero on
                                    // the frames before this image has a resolve: there is no previous frame to
                                    // re-emit, and the image the feedback would read is not defined yet.
                                    history_valid ? this->ssgi_bounce : 0.0f,
                                    // ... and w = the probe cache's gain. Zero unless the cache is active
                                    // AND has been written at least once: a grid nothing has deposited
                                    // into holds undefined texels (the first-use transition below makes
                                    // its layout legal, not its contents), and the shader branches on this
                                    // rather than multiplying by it, so a zero gain reads nothing at all.
                                    probe_ready ? this->gi_probe_gain : 0.0f),
            .probe_grid = glm::vec4(this->shadow_scene_center - glm::vec3(this->scene_radius), probe_cell_size)};
        vkCmdPushConstants(command_buffer, this->ssgi_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        constexpr uint32_t group_size = 8; // shaders/ssgi.comp's local_size_x/y
        vkCmdDispatch(command_buffer, (gi_width + group_size - 1) / group_size, (gi_height + group_size - 1) / group_size, 1);

        // A compute SHADER_WRITE is not visible to a later read without this. Both possible readers
        // are covered: the denoiser's resolve (a COMPUTE dispatch, next) and nothing else until the
        // image is rewritten - the composite samples the RESOLVED image, not this one.
        //
        // ... UNLESS the glossy pass runs next, in which case this barrier would be wrong twice over: it
        // would hand the image to a stage whose descriptor says SHADER_READ while that pass both reads and
        // writes it (a validation error, and a dependency the resolve does not need yet). So the hand-off
        // moves to the END of that pass, after the LAST writer of the raw trace. The predicate is the same
        // pure function record_ssgi_spec_pass decides on, so the two cannot disagree about which of them
        // owes the resolve its barrier.
        if (this->ssgi_specular_active()) {
            return;
        }
        VkImageMemoryBarrier2 to_sampling = vulkan::general_to_sampling_transition;
        to_sampling.image = vk.gi_images[index];
        VkDependencyInfo const sampling_dependency = make_image_dependency_info(1, &to_sampling);
        vkCmdPipelineBarrier2(command_buffer, &sampling_dependency);
    }

    std::expected<void, std::string> runtime::make_ssgi_temporal_pipeline(std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        auto built = pipelines::build_ssgi_temporal(this->vulkan_core, sizeof(ssgi_temporal_push_constants), compute_shader_code);
        if (!built) {
            return fail(built.error());
        }
        this->ssgi_temporal_set_layout = built->set_layout;
        this->ssgi_temporal_pipeline_layout = built->pipeline_layout;
        this->ssgi_temporal_pipeline = std::move(built->resolve);
        return {};
    }

    void runtime::ensure_ssgi_denoise_descriptors() {
        core& vk = this->vulkan_core;
        if (this->ssgi_temporal_pipeline == std::nullopt || this->ssgi_temporal_set_layout == VK_NULL_HANDLE) {
            return;
        }
        std::size_t const image_count = vk.gi_images.size();
        if (image_count == 0 || vk.gi_history_image_views.size() != image_count || vk.gi_resolve_image_views.size() != image_count ||
            vk.velocity_image_views.size() != image_count || vk.gbuffer_depth_image_views.size() != image_count ||
            vk.gbuffer_image_views[1].size() != image_count ||
            vk.gi_spec_image_views.size() != image_count || vk.gi_spec_reproject_image_views.size() != image_count ||
            vk.gi_spec_history_image_views.size() != image_count || vk.gi_spec_resolve_image_views.size() != image_count) {
            return;
        }
        // Six fingerprints, because six images feed one set - and the count has to match the LAYOUT,
        // not only the images that change independently, because it is also what sizes this family's
        // descriptor pool (see vulkan.bindings). This array held four, omitting the resolve image that
        // binding 4 points at, so the pool was built for four descriptors per set while the allocation
        // asked for five. The validation layer reported it - "Trying to allocate 15 of
        // VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER descriptors from VkDescriptorPool ..., but this pool
        // only has a total of 12 descriptors for this type" (3 images x 5 against 3 x 4) - and the
        // message is right that a stricter driver answers VK_ERROR_OUT_OF_POOL_MEMORY, which this
        // function's own failure path would turn into "this session has no GI" rather than a frame that
        // is merely missing a descriptor.
        std::array<VkImageView, 7> const signature = {vk.gi_image_views[0], vk.gi_history_image_views[0], vk.velocity_image_views[0],
                                                      vk.gbuffer_depth_image_views[0], vk.gi_resolve_image_views[0], vk.gbuffer_image_views[1][0],
                                                      vk.gbuffer_depth_image_views[0]};
        auto const write_sets = [this](uint32_t const image_index, std::span<VkDescriptorSet const> const sets) {
            std::array<VkDescriptorImageInfo, 7> image_infos = {};
            std::array<VkImageView, 7> const views = {
                this->vulkan_core.gi_image_views[image_index],
                this->vulkan_core.gi_history_image_views[image_index],
                this->vulkan_core.velocity_image_views[image_index],
                this->vulkan_core.gbuffer_depth_image_views[image_index],
                this->vulkan_core.gi_resolve_image_views[image_index],
                this->vulkan_core.gbuffer_image_views[1][image_index],
                // Binding 6 is the REFLECTION's reprojection, which only mode 1 reads. The diffuse dispatch
                // still has to name a valid view there (a shader that samples it in a branch leaves the
                // access in the SPIR-V, so validation checks the descriptor whether or not the branch is
                // taken), and binding the lobe's image would make the diffuse resolve require a layout that
                // only the lobe maintains - which fails on exactly the frames the lobe is OFF. The depth
                // target is always readable on a frame that resolves anything, and mode 0 ignores the value.
                this->vulkan_core.gbuffer_depth_image_views[image_index]};
            std::array<VkWriteDescriptorSet, 7> writes = {};
            for (uint32_t b = 0; b < views.size(); ++b) {
                // 4 is the STORAGE image the resolve writes: no sampler, and GENERAL rather than
                // SHADER_READ (a compute stage writes it, it does not sample it).
                bool const storage = b == 4u;
                image_infos[b].sampler = storage ? VK_NULL_HANDLE : *this->gbuffer_sampler;
                image_infos[b].imageView = views[b];
                image_infos[b].imageLayout = storage ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = sets[0];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[b].pImageInfo = &image_infos[b];
            }
            vkUpdateDescriptorSets(this->vulkan_core.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        };
        if (!this->ssgi_temporal_family.ensure(vk.device, this->ssgi_temporal_set_layout, static_cast<uint32_t>(image_count), 1u, static_cast<uint32_t>(signature.size()), signature, write_sets)) {
            utility::log("runtime: GI denoiser descriptor sets unavailable - this frame has no GI (its weight stays 0)");
        }
        // ... and the reflection's own resolve: the SAME layout with a different list of images, which is what
        // makes it a second family. Bindings 2 and 3 (the surface's motion vectors and depth) are unused in
        // mode 1 - its reprojection carries its own depth - but every binding of the layout has to name a real
        // view, so they are filled with the same ones the diffuse set uses. Binding 6 is the lobe's
        // reprojection, which is what mode 1 actually reprojects by.
        std::array<VkImageView, 7> const spec_signature = {vk.gi_spec_image_views[0], vk.gi_spec_history_image_views[0], vk.velocity_image_views[0],
                                                           vk.gbuffer_depth_image_views[0], vk.gi_spec_resolve_image_views[0], vk.gbuffer_image_views[1][0],
                                                           vk.gi_spec_reproject_image_views[0]};
        auto const write_spec_sets = [this](uint32_t const image_index, std::span<VkDescriptorSet const> const sets) {
            std::array<VkDescriptorImageInfo, 7> image_infos = {};
            std::array<VkImageView, 7> const views = {
                this->vulkan_core.gi_spec_image_views[image_index],
                this->vulkan_core.gi_spec_history_image_views[image_index],
                this->vulkan_core.velocity_image_views[image_index],
                this->vulkan_core.gbuffer_depth_image_views[image_index],
                this->vulkan_core.gi_spec_resolve_image_views[image_index],
                this->vulkan_core.gbuffer_image_views[1][image_index],
                this->vulkan_core.gi_spec_reproject_image_views[image_index]};
            std::array<VkWriteDescriptorSet, 7> writes = {};
            for (uint32_t b = 0; b < views.size(); ++b) {
                bool const storage = b == 4u;
                image_infos[b].sampler = storage ? VK_NULL_HANDLE : *this->gbuffer_sampler;
                image_infos[b].imageView = views[b];
                image_infos[b].imageLayout = storage ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = sets[0];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[b].pImageInfo = &image_infos[b];
            }
            vkUpdateDescriptorSets(this->vulkan_core.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        };
        if (!this->ssgi_spec_temporal_family.ensure(vk.device, this->ssgi_temporal_set_layout, static_cast<uint32_t>(image_count), 1u, static_cast<uint32_t>(spec_signature.size()), spec_signature, write_spec_sets)) {
            utility::log("runtime: GI reflection descriptor sets unavailable - this frame's reflection is not resolved");
        }
    }

    bool runtime::record_ssgi_denoise_pass(VkCommandBuffer const command_buffer) {
        core& vk = this->vulkan_core;
        std::size_t const index = this->current_image_index;
        if (index >= vk.gi_resolve_images.size() || vk.gi_history_images.size() != vk.gi_resolve_images.size()) {
            return false;
        }
        this->ensure_ssgi_denoise_descriptors();
        VkDescriptorSet const set = this->ssgi_temporal_family.set(static_cast<uint32_t>(index), 0);
        if (set == VK_NULL_HANDLE) {
            return false; // no set: the composite's GI weight stays 0 for this frame (see gi_resolved)
        }
        // The history-validity flag is read ONCE and set ONCE, around both dispatches: the two accumulations
        // share it, and a flag read after the first dispatch would tell the reflection its history exists on
        // the very frame that created it - which is the one frame it must not blend with.
        bool const history_valid = index < this->gi_history_valid.size() && this->gi_history_valid[index];
        bool const resolved = this->record_ssgi_resolve_pass(command_buffer, set, vk.gi_resolve_images[index], vk.gi_history_images[index], history_valid, 0.0f);
        // ... and the reflection's own accumulation, when this frame's lobe produced one. Its reprojection
        // is the one the lobe published, its history is its own, and the spatial filter's spec_weight lane is
        // what keeps a stale accumulation out of a frame whose lobe did not run.
        bool spec_resolved = false;
        if (this->ssgi_specular_active() && index < vk.gi_spec_resolve_images.size() &&
            vk.gi_spec_history_images.size() == vk.gi_spec_resolve_images.size()) {
            VkDescriptorSet const spec_set = this->ssgi_spec_temporal_family.set(static_cast<uint32_t>(index), 0);
            if (spec_set != VK_NULL_HANDLE) {
                spec_resolved = this->record_ssgi_resolve_pass(command_buffer, spec_set, vk.gi_spec_resolve_images[index], vk.gi_spec_history_images[index], history_valid, 1.0f);
            }
        }
        this->gi_spec_resolved = spec_resolved;
        if (resolved && this->gi_history_valid.size() > index) {
            this->gi_history_valid[index] = true;
        }
        return resolved;
    }

    bool runtime::record_ssgi_resolve_pass(VkCommandBuffer const command_buffer, VkDescriptorSet const set, VkImage const resolve_image, VkImage const history_image,
                                           bool const history_valid, float const mode) {
        core& vk = this->vulkan_core;
        std::size_t const index = this->current_image_index;
        bool const reflection = mode > 0.5f;

        uint32_t const gi_width = std::max(1u, vk.swap_chain_extent.width / 2u);
        uint32_t const gi_height = std::max(1u, vk.swap_chain_extent.height / 2u);

        // Layouts, all before the dispatch (a compute pass may barrier anywhere, but keeping them
        // together is what makes the set of states one image passes through readable):
        //   resolve -> GENERAL (storage write). The old layout is SHADER_READ once the image has been
        //   resolved before, and UNDEFINED on its first frame: the resolve is READ across frames (the
        //   tracer samples the previous frame's copy at a hit, the multi-bounce feedback), so this write
        //   has to keep its contents - claiming UNDEFINED every frame would discard exactly what the
        //   feedback reads, which is why record_ssgi_pass's first-use barrier leaves it in SHADER_READ.
        //   history -> SHADER_READ, and only on its FIRST use for this image: the previous frame's
        //   copy left it readable (see the hand-back below), so a later frame needs no barrier at all
        //   - claiming TRANSFER_DST as the old layout would be a layout the image is not in. Exactly
        //   the TAA resolve's arrangement, for exactly the same reason.
        //   The raw trace needs no barrier either: record_ssgi_pass handed it to SHADER_READ with a
        //   barrier that names COMPUTE as well as FRAGMENT (see general_to_sampling_transition),
        //   which is the read this dispatch does.
        std::array<VkImageMemoryBarrier2, 2> barriers = {};
        uint32_t barrier_count = 0;
        barriers[barrier_count] = history_valid ? vulkan::sampling_to_general_transition : vulkan::undefined_to_general_transition;
        barriers[barrier_count].image = resolve_image;
        ++barrier_count;
        if (!history_valid) {
            barriers[barrier_count] = vulkan::undefined_to_sampling_transition;
            barriers[barrier_count].image = history_image;
            ++barrier_count;
        }
        VkDependencyInfo const dependency = make_image_dependency_info(barrier_count, barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency);

        // The depth guard samples the G-buffer depth: its own barrier, written only if the G-buffer
        // pass actually rendered this frame (the lighting stage normally got here first). The
        // motion-vector target the reprojection reads is the same story, through its own accessor:
        // the G-buffer instance wrote it as a color attachment and the TAA resolve - the only other
        // sampler of it - runs before this pass, so whether it still needs the transition depends on
        // which of the two stages is the frame's first sampler.
        if (!reflection) {
            // Both are mode 0's inputs only: the reflection's own reprojection carries the depth its guard
            // needs, so mode 1 samples neither of these. The diffuse dispatch runs first in every frame that
            // resolves both, which is what leaves them readable here.
            this->ensure_gbuffer_depth_sampled(command_buffer, static_cast<uint32_t>(index));
            this->ensure_velocity_sampled(command_buffer, static_cast<uint32_t>(index));
        }

        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->ssgi_temporal_pipeline_layout, 0, 1, &set, 0, nullptr);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->ssgi_temporal_pipeline->get_pipeline());

        ssgi_temporal_push_constants const push = {
            .history_valid = history_valid ? 1.0f : 0.0f,
            .blend_static = this->gi_blend_static,
            .blend_min = this->gi_blend_min,
            .depth_scale = this->current_ubo.proj[2][2],
            .depth_offset = this->current_ubo.proj[3][2],
            // Which signal this dispatch resolves: 0.0 = the diffuse bounce, 1.0 = the reflection (see the
            // shader's `glossy`). One pipeline serves both, each with a set and a history of its own.
            .mode = mode,
            .unused1 = 0.0f,
            .unused2 = 0.0f,
            .gi_size = glm::vec4(static_cast<float>(gi_width), static_cast<float>(gi_height),
                                 static_cast<float>(vk.swap_chain_extent.width), static_cast<float>(vk.swap_chain_extent.height))};
        vkCmdPushConstants(command_buffer, this->ssgi_temporal_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        constexpr uint32_t group_size = 8; // shaders/ssgi_temporal.comp's local_size_x/y
        vkCmdDispatch(command_buffer, (gi_width + group_size - 1) / group_size, (gi_height + group_size - 1) / group_size, 1);

        // ---- the resolved image becomes the next frame's history ----
        // A copy rather than a ping-pong, exactly like the TAA resolve: the resolve writes the image
        // the composite reads, so the history has to be separate, and copying into it keeps every
        // descriptor set in the frame stable. The resolve is a storage image (GENERAL), so it goes out
        // through TRANSFER_SRC and comes back as a sample - the post chain still finds it in
        // SHADER_READ, exactly where it expects it.
        std::array<VkImageMemoryBarrier2, 2> copy_barriers = {};
        copy_barriers[0] = vulkan::general_to_transfer_src_transition; // resolve: GENERAL -> TRANSFER_SRC
        copy_barriers[0].image = resolve_image;
        copy_barriers[1] = vulkan::sampling_to_transfer_dst_transition;
        copy_barriers[1].image = history_image;
        VkDependencyInfo const copy_dependency = make_image_dependency_info(static_cast<uint32_t>(copy_barriers.size()), copy_barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &copy_dependency);

        VkImageCopy const region = {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffset = {0, 0, 0},
            .extent = {gi_width, gi_height, 1},
        };
        vkCmdCopyImage(command_buffer, resolve_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, history_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // Hand both on: the resolve to the composite, the history copy to the next frame's resolve
        // (which will find it in TRANSFER_DST and transition it from there). The raw trace has been
        // readable since record_ssgi_pass left it that way and nothing here touches it.
        std::array<VkImageMemoryBarrier2, 2> hand_back = {};
        hand_back[0] = vulkan::transfer_src_to_sampling_transition; // resolve -> SHADER_READ
        hand_back[0].image = resolve_image;
        hand_back[1] = vulkan::transfer_dst_to_sampling_transition; // history -> SHADER_READ
        hand_back[1].image = history_image;
        VkDependencyInfo const hand_back_dependency = make_image_dependency_info(static_cast<uint32_t>(hand_back.size()), hand_back.data());
        vkCmdPipelineBarrier2(command_buffer, &hand_back_dependency);

        // gi_history_valid is NOT touched here: record_ssgi_denoise_pass owns it, because it has to be set
        // once for both signals (see the note there).
        return true;
    }

    std::expected<void, std::string> runtime::make_ssgi_spatial_pipeline(std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        if (!this->deferred_pipeline.has_value()) {
            return fail(std::string("ssgi spatial: create the deferred lighting pipeline first (it owns the G-buffer set layout)"));
        }
        auto built = pipelines::build_ssgi_spatial(this->vulkan_core, this->vulkan_core.scene_descriptor_set_layout, this->gbuffer_set_layout, sizeof(ssgi_spatial_push_constants), compute_shader_code);
        if (!built) {
            return fail(built.error());
        }
        this->ssgi_spatial_pipeline_layout = built->pipeline_layout;
        this->ssgi_spatial_pipeline = std::move(built->trace);
        return {};
    }

    bool runtime::ssgi_traced_active() const noexcept {
        return this->ssgi_active() && this->ssgi_ray_tracing && this->vulkan_core.ray_query_available && this->rt_top_levels.has_value();
    }

    std::expected<void, std::string> runtime::make_ssgi_spec_pipeline(std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        if (!this->deferred_pipeline.has_value()) {
            return fail(std::string("ssgi spec: create the deferred lighting pipeline first (it owns the G-buffer set layout)"));
        }
        auto built = pipelines::build_ssgi_spec(this->vulkan_core, this->vulkan_core.scene_descriptor_set_layout, this->gbuffer_set_layout, sizeof(ssgi_spec_push_constants), compute_shader_code);
        if (!built) {
            return fail(built.error());
        }
        this->ssgi_spec_pipeline_layout = built->pipeline_layout;
        this->ssgi_spec_pipeline = std::move(built->trace);
        return {};
    }

    bool runtime::ssgi_specular_active() const noexcept {
        // Hit shading is part of the predicate, and not as a quality preference: without the instance
        // table a glossy ray that LANDS on geometry cannot be shaded, so the pass would have nothing to
        // add for the only samples that make it more than the lighting stage's own term. The table is the
        // same switch the tracer uses (record_ssgi_pass publishes it), so the two agree by construction.
        return this->ssgi_specular && this->ssgi_hit_shading && this->ssgi_traced_active() && this->ssgi_spec_pipeline.has_value();
    }

    void runtime::set_ssgi_specular(bool const enabled, uint32_t const rays, float const radius) noexcept {
        this->ssgi_specular = enabled;
        // One is the feature's definition and the low end of the noise/cost trade; the upper bound is a
        // cost guard, not a quality claim - the pass shades every hit it finds, which is the expensive
        // part of this chain.
        this->ssgi_specular_rays = std::clamp(rays, 1u, 8u);
        // A reach, not a quality knob: below it the reflection falls back to the sky, and past the knee the
        // curve is flat in BOTH effect and cost (measured: -0.907 / -2.126 / -2.350 at 0.12 / 0.5 / 2.0, and
        // the `gi` interval 1.28 / 2.22 / 2.21 ms). The FLOOR is deliberately low enough to be useless: a
        // reach whose ray length lands at or below the ray query's own tmin (a fixed 0.01 world units) makes
        // every sample answer with the environment - the identity configuration the L2.3 acceptance is read
        // in - and a floor above that would not be expressible on a small scene (0.01 scene radii of the
        // helmet's 1.64 is 0.0164 units, i.e. just past tmin, which is why the floor is 0.001 and not 0.01).
        this->ssgi_specular_radius = std::clamp(radius, 0.001f, 8.0f);
        if (enabled && !this->ssgi_hit_shading) {
            this->warn_missing_feature("ssgi", "glossy reflections have no effect: without hit shading a reflection ray cannot be shaded where it lands");
        } else if (enabled && !this->ssgi_ray_tracing) {
            this->warn_missing_feature("ssgi", "glossy reflections have no effect: they need the traced GI path, not the marched one");
        }
    }

    void runtime::set_ssgi_ray_tracing(bool const enabled) noexcept {
        this->ssgi_ray_tracing = enabled;
        if (enabled && !this->vulkan_core.ray_query_available) {
            this->warn_missing_feature("ssgi", "GI rays are marched, not traced: this device has no ray queries");
        } else if (enabled && !this->rt_shadow_pipeline.has_value()) {
            this->warn_missing_feature("ssgi", "GI rays are marched, not traced: the ray-traced pipelines were not created");
        }
    }

    void runtime::set_ssgi_bounce(float const gain) noexcept {
        // The knob stops at one: above it the geometric series a diffuse loop forms is not guaranteed to
        // converge (the surfaces' albedos approach one), and the failure mode is a frame that gets
        // brighter every frame rather than a visibly wrong one.
        this->ssgi_bounce = std::clamp(gain, 0.0f, 1.0f);
    }

    void runtime::set_furnace(bool const enabled) noexcept {
        this->furnace = enabled;
    }

    void runtime::set_ssgi_hit_shading(bool const enabled) noexcept {
        this->ssgi_hit_shading = enabled;
        if (enabled && !this->vulkan_core.ray_query_available) {
            this->warn_missing_feature("ssgi", "hits are read from the screen, not shaded: this device has no ray queries");
        } else if (enabled && !this->ssgi_ray_tracing) {
            this->warn_missing_feature("ssgi", "hit shading has no effect: only the traced GI path lands on a surface to shade");
        }
    }

    void runtime::set_ssgi_spatial(float const sigma) noexcept {
        this->gi_spatial_sigma = std::clamp(sigma, 0.0f, 8.0f);
    }

    void runtime::set_ssgi_upsample(bool const enabled) noexcept {
        this->gi_upsample = enabled;
    }

    // ---- the world-space probe cache (see shaders/gi_probe.comp) ----
    void runtime::set_ssgi_probes(bool const enabled, float const rate, uint32_t const rounds, float const gain) noexcept {
        this->gi_probe_enabled = enabled;
        // The rate is the loop gain of the grid's own cycle (tracer -> resolve -> grid -> tracer): a rate
        // of 1 would make the grid an immediate echo of the frame that read it, so it stops below that.
        this->gi_probe_rate = std::clamp(rate, 0.0f, 0.5f);
        this->gi_probe_rounds = std::clamp(rounds, 0u, 4u);
        // The dispatch count is the pass's: it owns the update sequence, so the renderer hands it the one
        // number of that sequence that is configuration rather than structure.
        this->gi_probe.set_rounds(this->gi_probe_rounds);
        // The sign is the cache's DIRECTION A/B rather than a mistake: |gain| is the gain, and a negative
        // value looks the cache up along the opposite direction of the ray - the same cell, the other side.
        // It is clamped rather than rejected because it is a documented measurement setting.
        this->gi_probe_gain = std::clamp(gain, -4.0f, 4.0f);
        if (enabled && !this->ssgi_on) {
            this->warn_missing_feature("ssgi", "the probe cache has no effect: it is injected from the screen-space GI chain, which is off");
        } else if (enabled && !this->gi_probe.pipeline_ready()) {
            this->warn_missing_feature("ssgi", "the probe cache has no effect: it has not built its pipeline (see the startup log)");
        }
    }

    bool runtime::gi_probe_active() const noexcept {
        // The cache is deposited from the screen-space chain's resolved image and lives on the deferred
        // path's G-buffer, so it needs both of those, plus the pass having built what it records with: a
        // build without the cache keeps the tracer's environment-probe fallback and changes nothing else.
        return this->gi_probe_enabled && this->gi_probe.pipeline_ready() && this->ssgi_active();
    }

    void runtime::register_shader(std::string_view const name, std::span<unsigned char const> const bytecode) {
        // The app loads shaders (it knows the directory and the file names) and hands them over here; a pass
        // asks for its own by name at create time. A COPY, because the caller's buffer is a startup local.
        for (auto& [registered_name, registered_bytes] : this->registered_shaders) {
            if (registered_name == name) {
                registered_bytes.assign(bytecode.begin(), bytecode.end());
                return;
            }
        }
        this->registered_shaders.emplace_back(std::string(name), std::vector<unsigned char>(bytecode.begin(), bytecode.end()));
    }

    void runtime::create_passes() {
        // The runner's create step for every stage this runtime wires. A pass builds what it owns from its own
        // declaration and its own shader, so this is also where a pass that could not build itself says so -
        // and a pass that says so stays INACTIVE (its feature predicate is false), which is what makes a
        // startup failure here a log line rather than a broken frame.
        //
        // IT IS HANDED A CONTEXT, NOT A HOST: building a pass needs a device and the shared lookups, and it
        // needs no frame. ANY owner can fill this struct - that is what makes a pass usable outside this
        // renderer - and this runtime is one such owner, filling the device from the core it owns.
        //
        // The SHARED samplers must exist before the context is filled, because a pass caches the six it may
        // choose between at create time (a declaration picks one by hint, and a null sampler in a set is a
        // validation error rather than a skipped fetch). Two of the six are still created inside the pipeline
        // builders that first needed them - `make_gbuffer_debug_pipeline` makes the G-buffer pair's and the
        // probe grid's - which is the naming accident `docs/runtime_split.md` records; the TAA resolve's is
        // made here because the pass that declares it is what needs it now.
        this->ensure_taa_sampler();
        pass::pass_context const build = {
            .device = this->vulkan_core.device,
            .samplers = this->shared_samplers(),
            .shared_set_layout = [](void* owner, uint32_t const set) {
                // The scene set is the only shared set a pass's OWN pipeline layout ever needs today: it is
                // set 0, and it is what every compute pass's tracing/shading reads. A pass asking for any
                // other set gets "none", which makes it build nothing and say so.
                return set == 0u ? static_cast<runtime*>(owner)->vulkan_core.scene_descriptor_set_layout : VkDescriptorSetLayout{VK_NULL_HANDLE}; },
            .shader = [](void* owner, std::string_view const name) { return static_cast<runtime*>(owner)->registered_shader(name); },
            .owner = this,
        };
        pass::stage const scene_stage = {.name = "scene", .passes = this->scene_stage, .marks = false};
        pass::run_report const scene_created = pass::create_stage(scene_stage, build);
        if (!scene_created.rejected.empty()) {
            utility::log("pass '{}': its declaration was refused by the validator, so it does not run", scene_created.rejected);
        }
        pass::stage const probe_stage = {.name = "gi_probe", .passes = this->gi_probe_stage, .marks = false};
        pass::run_report const created = pass::create_stage(probe_stage, build);
        if (!created.rejected.empty()) {
            utility::log("pass '{}': its declaration was refused by the validator, so it does not run", created.rejected);
        }
        pass::stage const taa_stage = {.name = "taa", .passes = this->taa_stage, .marks = false};
        pass::run_report const taa_created = pass::create_stage(taa_stage, build);
        if (!taa_created.rejected.empty()) {
            utility::log("pass '{}': its declaration was refused by the validator, so it does not run", taa_created.rejected);
        }
    }

    void runtime::ensure_taa_sampler() {
        // The resolve upsamples the scene colour but must not average neighbouring history texels: linear
        // magnification, nearest minification. It was created inside `make_taa_pipeline` before this pass owned
        // that pipeline; the sampler is a SHARED handle (a declaration chooses it by hint) so it stays the
        // renderer's, and it has to exist before the pass caches the six it may choose between.
        if (this->taa_sampler.get() != VK_NULL_HANDLE) {
            return;
        }
        VkSamplerCreateInfo sampler_info = make_texture_sampler_info(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f);
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_NEAREST;
        VkSampler sampler = VK_NULL_HANDLE;
        if (vkCreateSampler(this->vulkan_core.device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
            utility::log("runtime: the TAA resolve's sampler could not be created - the resolve will have no descriptor to write");
            return;
        }
        this->taa_sampler = vk_sampler(sampler, this->vulkan_core.device);
    }

    render_resource::shared::sampler_set runtime::shared_samplers() const noexcept {
        // The six samplers a declaration chooses between, as handles. One place, so that two passes cannot end
        // up with two different ideas of "the post sampler".
        return {.gbuffer = *this->gbuffer_sampler,
                .probe_grid = *this->gi_probe_sampler,
                .taa = *this->taa_sampler,
                .post = *this->post_sampler,
                .nearest = *this->post_nearest_sampler,
                .shadow = *this->shadow_sampler};
    }

    std::span<unsigned char const> runtime::registered_shader(std::string_view const name) const noexcept {
        for (auto const& [registered_name, registered_bytes] : this->registered_shaders) {
            if (registered_name == name) {
                return registered_bytes;
            }
        }
        return {}; // a pass whose shader was never registered builds nothing and says so
    }

    /// the scene pass's per-frame input: the leaves, the segments, and the three things only the renderer can
    /// answer (see scene_frame). Built here rather than stored, because every field is this frame's.
    pass::scene_frame runtime::make_scene_frame() noexcept {
        core const& vk = this->vulkan_core;
        uint32_t const frame_slot = static_cast<uint32_t>(vk.current_frame);
        // the pass's view of the per-slot secondary buffers (members, so the span it holds outlives the stage)
        auto const& segments = this->main_segments[static_cast<std::size_t>(frame_slot)];
        this->scene_segment_view.clear();
        this->scene_segment_view.reserve(segments.size());
        for (auto const& [pool, buffer] : segments) {
            this->scene_segment_view.push_back(pass::segment_buffer{.pool = pool, .buffer = *buffer});
        }
        // the secondaries inherit the instance's attachments: the three surface targets in order, the velocity
        // target, and the scene colour - the same order the pass's declaration lists them in
        this->scene_color_formats = {vulkan::gbuffer_formats[0], vulkan::gbuffer_formats[1], vulkan::gbuffer_formats[2], vulkan::gbuffer_velocity_format, vulkan::hdr_format};
        return pass::scene_frame{
            .leaves = this->frame_visible,
            .segments = this->scene_segment_view,
            .make_environment = &runtime::make_scene_environment,
            .run_tasks = &runtime::run_scene_tasks,
            .owner = this,
            .color_formats = this->scene_color_formats,
            .depth_format = vk.depth_format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .gbuffer = true,
            .extent = vk.swap_chain_extent,
        };
    }

    /// build ONE segment's draw state; a fresh one per segment, because the pipeline-dedup state is per
    /// recording session (see scene_frame::make_environment)
    render_environment runtime::make_scene_environment(void* owner, VkCommandBuffer const command_buffer, bool const gbuffer) {
        runtime& self = *static_cast<runtime*>(owner);
        render_environment env;
        env.command_buffer = command_buffer;
        // The G-buffer pass binds its own pipeline as the pass default (see gbuffer_pipeline_name): same
        // leaves, same draw path, but the fragment stage writes the surface instead of shading.
        {
            std::shared_lock const lock(self.access_mutex);
            env.default_name = gbuffer ? gbuffer_pipeline_name : self.default_pipeline_name;
        }
        env.bind = [&self, gbuffer](VkCommandBuffer const cb, std::string_view const name) {
            if (gbuffer) {
                if (name == gbuffer_pipeline_name) {
                    self.gbuffer_pipeline->begin_pipeline(cb);
                    return;
                }
                // a leaf with explicit pipeline semantics cannot draw in the G-buffer instance (the named
                // pipelines declare the single HDR attachment): say so once per leaf instead of issuing a draw
                // that would be a validation error
                utility::log("runtime: leaf requests pipeline '{}' during the G-buffer pass - draw skipped (only default-semantics leaves write the G-buffer)", name);
                return;
            }
            if (auto const it = self.pipelines.find(name); it != self.pipelines.end()) {
                it->second.begin_pipeline(cb);
            } else {
                utility::log("runtime: main pass references unknown pipeline '{}' - draw skipped", name);
            }
        };
        // transparent leaves toggle depth writes off via this (core dynamic state, 1.3)
        env.set_depth_write_fn = [](VkCommandBuffer const cb, VkBool32 const enabled) { vkCmdSetDepthWriteEnable(cb, enabled); };
        // single-sided materials keep back-face culling here (the shadow pass overrides it with env.two_sided;
        // the main pass must not, or double-sided handling would cost fill rate)
        env.set_cull_mode_fn = [](VkCommandBuffer const cb, VkCullModeFlags const mode) { vkCmdSetCullMode(cb, mode); };
        env.layout = self.vulkan_core.scene_pipeline_layout;
        return env;
    }

    /// the renderer's scheduler, handed to the pass so the segment fan-out stays the frame loop's policy
    void runtime::run_scene_tasks(void* owner, std::span<std::function<void()>> const tasks) {
        static_cast<runtime*>(owner)->run_tasks(tasks, vulkan::task_priority::recording);
    }

    bool runtime::resolve_scene_pass(pass::resolved_io& out) {
        core const& vk = this->vulkan_core;
        std::size_t const index = this->current_image_index;
        std::size_t const image_count = vk.gbuffer_image_views.empty() ? 0 : vk.gbuffer_image_views[0].size();
        if (!this->gbuffer_pass_active() || image_count == 0 || index >= image_count || vk.velocity_image_views.size() != image_count ||
            vk.gbuffer_depth_image_views.size() != image_count) {
            return false; // no surface pipeline, or no target generation to draw into
        }
        out.frame = this->pass_frame();
        out.cmd = *this->command_buffers[static_cast<uint32_t>(vk.current_frame)];
        out.own = {};
        out.own_set = VK_NULL_HANDLE;
        // the shared scene set (the pass's declaration names the SET, not its bindings: its owner decides them)
        out.shared.scene = this->scene_sets.set(static_cast<uint32_t>(vk.current_frame));
        // the six declared targets, in declaration order: the three stored surface targets, the motion-vector
        // target, the scene colour target (an ALIAS - see scene_io: the TAA input while the resolve runs, the
        // HDR target otherwise) and the surface depth
        out.target_storage[0] = {.view = vk.gbuffer_image_views[0][index], .buffer = VK_NULL_HANDLE, .image = vk.gbuffer_images[0][index]};
        out.target_storage[1] = {.view = vk.gbuffer_image_views[1][index], .buffer = VK_NULL_HANDLE, .image = vk.gbuffer_images[1][index]};
        out.target_storage[2] = {.view = vk.gbuffer_image_views[2][index], .buffer = VK_NULL_HANDLE, .image = vk.gbuffer_images[2][index]};
        out.target_storage[3] = {.view = vk.velocity_image_views[index], .buffer = VK_NULL_HANDLE, .image = vk.velocity_images[index]};
        out.target_storage[4] = {.view = this->scene_target_view(index), .buffer = VK_NULL_HANDLE, .image = this->scene_target_image(index)};
        out.target_storage[5] = {.view = vk.gbuffer_depth_image_views[index], .buffer = VK_NULL_HANDLE, .image = vk.gbuffer_depth_images[index]};
        out.targets = std::span<pass::resolved_binding const>(out.target_storage.data(), 6);
        // NO pipelines: a leaf names the pipeline it wants and the renderer's registry resolves it through the
        // environment the pass is handed (see scene_frame::make_environment)
        out.pipelines = {};
        out.pipeline_layout = VK_NULL_HANDLE;
        out.push = {};
        out.extent = vk.swap_chain_extent;
        return true;
    }

    pass::transparent_frame runtime::make_transparent_frame() noexcept {
        core const& vk = this->vulkan_core;
        auto const& secondaries = this->secondary_command_buffers[static_cast<std::size_t>(vk.current_frame)];
        return pass::transparent_frame{
            .leaves = this->frame_transparent,
            .secondary = *secondaries[static_cast<std::size_t>(secondary_pass::transparent)],
            .make_environment = &runtime::make_scene_environment,
            .owner = this,
            .color_format = vulkan::hdr_format,
            .depth_format = vk.depth_format,
            .extent = vk.swap_chain_extent,
        };
    }

    bool runtime::resolve_transparent_pass(pass::resolved_io& out) {
        core const& vk = this->vulkan_core;
        std::size_t const index = this->current_image_index;
        std::size_t const image_count = vk.scene_color_image_views.size();
        if (this->frame_transparent.empty() || image_count == 0 || index >= image_count || vk.gbuffer_depth_image_views.size() != image_count) {
            return false; // nothing blended this frame: no instance and no barriers to pay for
        }
        out.frame = this->pass_frame();
        out.cmd = *this->command_buffers[static_cast<uint32_t>(vk.current_frame)];
        out.own = {};
        out.own_set = VK_NULL_HANDLE;
        out.shared.scene = this->scene_sets.set(static_cast<uint32_t>(vk.current_frame));
        // the two declared targets, in declaration order: the scene colour it composites over (an ALIAS - the
        // same image the scene pass writes) and the surface depth it depth-tests against
        out.target_storage[0] = {.view = this->scene_target_view(index), .buffer = VK_NULL_HANDLE, .image = this->scene_target_image(index)};
        out.target_storage[1] = {.view = vk.gbuffer_depth_image_views[index], .buffer = VK_NULL_HANDLE, .image = vk.gbuffer_depth_images[index]};
        out.targets = std::span<pass::resolved_binding const>(out.target_storage.data(), 2);
        out.pipelines = {}; // a leaf names its pipeline; see the scene pass
        out.pipeline_layout = VK_NULL_HANDLE;
        out.push = {};
        out.extent = vk.swap_chain_extent;
        return true;
    }

    pass::frame_identity runtime::pass_frame() const noexcept {
        core const& vk = this->vulkan_core;
        return pass::frame_identity{
            .image_index = this->current_image_index,
            .slot = static_cast<uint32_t>(vk.current_frame),
            // the generation's image count, which is what a pass that owns a per-image family sizes it from -
            // and NOT the same number as the image index above
            .image_count = static_cast<uint32_t>(vk.gi_resolve_images.size()),
            .extent = vk.swap_chain_extent,
        };
    }

    // ---- the pass host: the runner's callbacks, answered by the renderer ---------------------------------
    //
    // It is rebuilt per call because it is a struct of function pointers (it holds no state of its own); the
    // context is this runtime, which is what each callback casts back to. This is the RUNNER's half of the
    // interface and a pass never sees it: at create time a pass is given a `pass_context`, and while recording
    // it is handed `resolved_io`, so it cannot reach a resource its declaration did not name.
    pass::pass_host runtime::make_pass_host() noexcept {
        return pass::pass_host{
            .context = this,
            .frame = [](void* context) { return static_cast<runtime*>(context)->pass_frame(); },
            .feature_active = [](void* context, std::string_view const feature) { return static_cast<runtime*>(context)->feature_active(feature); },
            .resolve = [](void* context, pass::frame_pass const& pass, pass::resolved_io& out) { return static_cast<runtime*>(context)->resolve_pass(pass, out); },
            .apply_behaviour = [](void* context, pass::frame_pass const& pass, pass::resolved_io const& io) { static_cast<runtime*>(context)->apply_pass_behaviour(pass, io); },
            // No mark pair for the probe's stage yet, and deliberately: the frame has an END mark for the
            // cache and no begin mark (its interval is measured from the GI chain's end), so adding a query
            // pair here would change the timing report for a pass whose cost is ~0. The runtime keeps writing
            // the end mark itself, right after the stage.
            .mark_begin = nullptr,
            .mark_end = nullptr,
        };
    }

    bool runtime::resolve_pass(pass::frame_pass const& pass, pass::resolved_io& out) {
        core const& vk = this->vulkan_core;
        if (&pass == static_cast<pass::frame_pass const*>(&this->transparent)) {
            return this->resolve_transparent_pass(out);
        }
        if (&pass == static_cast<pass::frame_pass const*>(&this->scene)) {
            return this->resolve_scene_pass(out);
        }
        if (&pass == static_cast<pass::frame_pass const*>(&this->taa_resolve)) {
            return this->resolve_taa_pass(out);
        }
        if (&pass != static_cast<pass::frame_pass const*>(&this->gi_probe)) {
            return false; // no other pass is wired into a stage yet
        }
        // A frame whose grid images are not there cannot run this pass at all: the images are created and
        // destroyed with the target generation (see core::create_render_targets).
        if (vk.gi_probe_image_views.size() != 8 || vk.gi_probe_images.size() != 8 || vk.gi_probe_surface_image_views.empty() || vk.gi_probe_surface_images.empty() ||
            !this->gi_probe.pipeline_ready()) {
            return false;
        }
        out.frame = this->pass_frame();
        out.cmd = *this->command_buffers[static_cast<uint32_t>(vk.current_frame)];
        // The nine own bindings, resolved BY DECLARATION ELEMENT: the declaration says binding k is element k
        // of `probe_grid` (0..7) and binding 8 element 0 of `probe_surface`, and mapping an element onto the
        // renderer's image is exactly what a resolver is for. Which half of the ping-pong each of the pass's
        // two sets binds is the PASS's fact, not this function's, so nothing here decides it.
        for (uint32_t k = 0; k < 8; ++k) {
            out.own_storage[k] = {.view = vk.gi_probe_image_views[k], .buffer = VK_NULL_HANDLE, .image = vk.gi_probe_images[k]};
        }
        out.own_storage[8] = {.view = vk.gi_probe_surface_image_views[0], .buffer = VK_NULL_HANDLE, .image = vk.gi_probe_surface_images[0]};
        out.own = std::span<pass::resolved_binding const>(out.own_storage.data(), 9);
        // The pass owns its family, so it resolves its own sets; what the host resolves is the SHARED set the
        // declaration uses (a cell's ray needs the top level structure, the material table, the light UBO).
        out.own_set = VK_NULL_HANDLE;
        out.shared.scene = this->scene_sets.set(static_cast<uint32_t>(vk.current_frame));
        // The pipeline and its layout are the PASS's objects now (it built them in its create step), and the
        // host relays them to the runner the same way it would relay its own: the runner's guarantee - bind
        // before record, through the layout the pass pushes and binds with - does not depend on who owns them.
        out.pipeline_storage[0] = this->gi_probe.pipeline();
        out.pipelines = std::span<VkPipeline const>(out.pipeline_storage.data(), 1);
        out.pipeline_layout = this->gi_probe.pipeline_layout();
        // The push block, composed HERE because its values are the renderer's: the grid is anchored to the
        // scene's bounds (the same cube the shadow fit uses, so one set of numbers means the same thing on a
        // 1.6-unit model and on Sponza's 18.5), the rate is the config's, the table address is the tracing
        // structures', and the light direction is what the cache's own invalidation compares against.
        pass::gi_probe_pass::push_constants push = {};
        float const cell_size = (2.0f * this->scene_radius) / static_cast<float>(vulkan::gi_probe_grid_extent);
        push.grid_min_cell = glm::vec4(this->shadow_scene_center - glm::vec3(this->scene_radius), cell_size);
        push.params = glm::vec4(this->gi_probe_rate, 0.0f, 0.0f, 0.0f);
        uint64_t probe_table = 0;
        if (this->rt_top_levels.has_value()) {
            VkBuffer const table = this->rt_top_levels->instance_table(static_cast<uint32_t>(vk.current_frame));
            if (table != VK_NULL_HANDLE) {
                VkBufferDeviceAddressInfo const table_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = table};
                probe_table = vkGetBufferDeviceAddress(vk.device, &table_info);
            }
        }
        push.instance_table = glm::uvec2(static_cast<uint32_t>(probe_table & 0xFFFFFFFFu), static_cast<uint32_t>(probe_table >> 32u));
        push.light_dir = glm::vec4(glm::normalize(glm::vec3(this->light_state.light_dir)), 0.0f);
        static_assert(sizeof(push) <= pass::max_push_bytes, "the probe cache's push block must fit the guaranteed minimum");
        std::memcpy(out.push_storage.data(), &push, sizeof(push));
        out.push = std::span<std::byte const>(out.push_storage.data(), sizeof(push));
        // The extent, from the declaration's rule: `resource` and the probe grid is 32 cells on a side. The
        // rule is applied HERE because it is a mapping from the declaration to a number the renderer owns.
        VkExtent2D extent = {};
        switch (pass.behaviour().extent) {
        case pass::extent_rule::full:
            extent = vk.swap_chain_extent;
            break;
        case pass::extent_rule::half:
            extent = VkExtent2D{std::max(1u, vk.swap_chain_extent.width / 2u), std::max(1u, vk.swap_chain_extent.height / 2u)};
            break;
        case pass::extent_rule::resource:
            if (pass.behaviour().extent_of == pass::resource_id::probe_grid) {
                extent = VkExtent2D{vulkan::gi_probe_grid_extent, vulkan::gi_probe_grid_extent};
            }
            break;
        }
        out.extent = extent;
        return true;
    }

    bool runtime::resolve_taa_pass(pass::resolved_io& out) {
        core const& vk = this->vulkan_core;
        std::size_t const image_count = vk.scene_color_image_views.size();
        std::size_t const index = this->current_image_index;
        // A frame whose targets are not there cannot run this pass at all: they are created and destroyed with
        // the target generation (see core::create_render_targets).
        if (image_count == 0 || index >= image_count || vk.taa_history_image_views.size() != image_count || vk.velocity_image_views.size() != image_count ||
            vk.gbuffer_depth_image_views.size() != image_count || vk.hdr_image_views.size() != image_count || !this->taa_resolve.pipeline_ready()) {
            return false;
        }
        out.frame = this->pass_frame();
        out.cmd = *this->command_buffers[static_cast<uint32_t>(vk.current_frame)];
        // The four own bindings, resolved BY DECLARATION ELEMENT: the declaration names four per-swapchain-image
        // resources, so the element selects within the family (all four are element 0) and the FRAME selects the
        // image. The views are what the descriptor takes and the images are what the pass's barriers take.
        out.own_storage[0] = {.view = vk.scene_color_image_views[index], .buffer = VK_NULL_HANDLE, .image = vk.scene_color_images[index]};
        out.own_storage[1] = {.view = vk.taa_history_image_views[index], .buffer = VK_NULL_HANDLE, .image = vk.taa_history_images[index]};
        out.own_storage[2] = {.view = vk.velocity_image_views[index], .buffer = VK_NULL_HANDLE, .image = vk.velocity_images[index]};
        out.own_storage[3] = {.view = vk.gbuffer_depth_image_views[index], .buffer = VK_NULL_HANDLE, .image = vk.gbuffer_depth_images[index]};
        out.own = std::span<pass::resolved_binding const>(out.own_storage.data(), 4);
        // The declared render TARGET: the frame's HDR image, which the resolve writes as its colour attachment
        // and then copies out of. Resolved the same way an own binding is, so the pass reaches nothing it did
        // not declare - and the pass owns the rendering instance over it.
        out.target_storage[0] = {.view = vk.hdr_image_views[index], .buffer = VK_NULL_HANDLE, .image = vk.hdr_images[index]};
        out.targets = std::span<pass::resolved_binding const>(out.target_storage.data(), 1);
        out.own_set = VK_NULL_HANDLE;      // the pass owns its family and therefore its sets
        out.shared.scene = VK_NULL_HANDLE; // the resolve reads nothing shared: its four inputs are its own
        out.pipeline_storage[0] = this->taa_resolve.pipeline();
        out.pipelines = std::span<VkPipeline const>(out.pipeline_storage.data(), 1);
        out.pipeline_layout = this->taa_resolve.pipeline_layout();
        // The push block, composed HERE because its values are the renderer's: the two blend weights are the
        // config's, the texel size is the target's, and the two depth terms come from this frame's projection.
        // The lane that says whether the history may be trusted is the PASS's, and it writes that one itself.
        pass::taa_pass::push_constants push = {};
        push.blend_static = this->taa_blend_static;
        push.blend_min = this->taa_blend_min;
        push.texel_size_x = 1.0f / static_cast<float>(vk.swap_chain_extent.width);
        push.texel_size_y = 1.0f / static_cast<float>(vk.swap_chain_extent.height);
        push.depth_scale = this->current_ubo.proj[2][2];
        push.depth_offset = this->current_ubo.proj[3][2];
        static_assert(sizeof(push) <= pass::max_push_bytes, "the TAA resolve's push block must fit the guaranteed minimum");
        std::memcpy(out.push_storage.data(), &push, sizeof(push));
        out.push = std::span<std::byte const>(out.push_storage.data(), sizeof(push));
        out.extent = vk.swap_chain_extent; // the declaration's rule is `full`
        return true;
    }

    void runtime::apply_pass_behaviour(pass::frame_pass const& pass, pass::resolved_io const& io) {
        // The mechanical part of "how this pass is called", done by the runner so that a pass cannot forget
        // it: the pipeline is bound HERE, and the viewport/scissor are set HERE for a pass that asked for them
        // (which is what replaces the hand-kept pipeline list in update_pass_geometry - a pass cannot drop
        // itself from a list it does not maintain).
        //
        // What is deliberately NOT here: the rendering instance. EVERY graphics pass opens its own, over the
        // targets it declared, because the load op and the clear value are the PASS's knowledge. That is why
        // this function no longer has a "not a compute pass" branch: the three graphics kinds differ in how
        // many draws they issue, and only the pass knows that.
        pass::behaviour const& behaviour = pass.behaviour();
        if (behaviour.resync_viewport) {
            // io.extent is the extent the declaration's rule produced (the frame's, half of it, or a
            // resource's), so a fullscreen pass gets a viewport that matches the target it declared.
            VkViewport const viewport = {0.0f, 0.0f, static_cast<float>(io.extent.width), static_cast<float>(io.extent.height), 0.0f, 1.0f};
            VkRect2D const scissor = {{0, 0}, io.extent};
            vkCmdSetViewport(io.cmd, 0, 1, &viewport);
            vkCmdSetScissor(io.cmd, 0, 1, &scissor);
        }
        VkPipelineBindPoint const bind_point = behaviour.kind == pass::behaviour_kind::compute ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS;
        for (VkPipeline const pipeline : io.pipelines) {
            if (pipeline != VK_NULL_HANDLE) {
                vkCmdBindPipeline(io.cmd, bind_point, pipeline);
            }
        }
    }

    std::expected<void, std::string> runtime::make_mask_bake_pipeline(std::span<unsigned char const> const compute_shader_code) {
        if (!this->vulkan_core.ray_query_available) {
            return std::unexpected(std::string("mask bake: this device has no ray queries (VK_KHR_acceleration_structure + VK_KHR_ray_query)"));
        }
        // The scene set ALONE, because everything the bake reads is in it: the material table (the alpha
        // texture's index, the base colour factor's alpha, the cutoff) and the bindless texture array.
        auto built = pipelines::build_mask_bake(this->vulkan_core, this->vulkan_core.scene_descriptor_set_layout, sizeof(mask_bake_push_constants), compute_shader_code);
        if (!built) {
            return std::unexpected(std::move(built.error()));
        }
        this->mask_bake_pipeline_layout = built->pipeline_layout;
        this->mask_bake_pipeline = std::move(built->trace);

        // The bake's own set (see the member's comment for why it is not the scene set): the layout is the
        // scene one, so binding 1 is the bindless texture array and binding 5 the material table, exactly as
        // the raster path declares them. Written once, here, when both already exist.
        auto const* const material_detail = this->vulkan_core.vma.get_buffer_detail(this->material_buffer.handle());
        if (material_detail == nullptr || this->owned_texture_views.empty() || this->texture_sampler.get() == VK_NULL_HANDLE) {
            return std::unexpected(std::string("mask bake: the material table or the texture array is not ready"));
        }
        this->mask_bake_set = this->vulkan_core.make_descriptor_set(this->vulkan_core.scene_descriptor_set_layout);
        if (this->mask_bake_set.get() == VK_NULL_HANDLE) {
            return std::unexpected(std::string("mask bake: descriptor set allocation failed"));
        }
        VkDescriptorImageInfo const textures_info = {
            .sampler = *this->texture_sampler, .imageView = *this->owned_texture_views[0], .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorBufferInfo const materials_info = {.buffer = material_detail->buffer, .offset = 0, .range = VK_WHOLE_SIZE};
        std::array<VkWriteDescriptorSet, 2> writes = {};
        for (uint32_t b = 0; b < writes.size(); ++b) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = this->mask_bake_set.get();
            writes[b].dstBinding = b == 0u ? 1u : 5u; // the texture array, then the material table
            writes[b].dstArrayElement = 0;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = b == 0u ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pImageInfo = b == 0u ? &textures_info : nullptr;
            writes[b].pBufferInfo = b == 0u ? nullptr : &materials_info;
        }
        vkUpdateDescriptorSets(this->vulkan_core.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return {};
    }

    std::expected<void, std::string> runtime::make_compute_skin_pipeline(std::span<unsigned char const> const compute_shader_code) {
        if (!this->vulkan_core.ray_query_available) {
            return std::unexpected(std::string("compute skin: this device has no ray queries (VK_KHR_acceleration_structure + VK_KHR_ray_query)"));
        }
        // The scene set alone, because the pass reads exactly one thing from it: the per-joint matrices at
        // binding 9. The vertices come through push-constant device addresses, like every other traced pass.
        auto built = pipelines::build_compute_skin(this->vulkan_core, this->vulkan_core.scene_descriptor_set_layout, sizeof(compute_skin_push_constants), compute_shader_code);
        if (!built) {
            return std::unexpected(std::move(built.error()));
        }
        this->compute_skin_pipeline_layout = built->pipeline_layout;
        this->compute_skin_pipeline = std::move(built->trace);

        // One set per frame slot, from the SCENE layout, with only binding 9 written: the slot's OWN
        // per-joint matrices. The animation rewrites that BUFFER every frame, not the descriptor, so the
        // sets are written once here and stay valid - which matters twice over, because a set updated while
        // a recording command buffer holds it invalidates that buffer (the trap the mask bake's own set
        // documents) and one set would point at the wrong slot's matrices for half the frames.
        if (this->skin_buffers.size() != this->compute_skin_sets.size()) {
            return std::unexpected(std::string("compute skin: the per-slot skin matrix buffers are not created"));
        }
        for (std::size_t slot = 0; slot < this->compute_skin_sets.size(); ++slot) {
            auto const* const detail = this->vulkan_core.vma.get_buffer_detail(this->skin_buffers[slot].handle());
            if (detail == nullptr) {
                return std::unexpected(std::string("compute skin: a skin matrix buffer has no VMA detail"));
            }
            this->compute_skin_sets[slot] = this->vulkan_core.make_descriptor_set(this->vulkan_core.scene_descriptor_set_layout);
            if (this->compute_skin_sets[slot].get() == VK_NULL_HANDLE) {
                return std::unexpected(std::string("compute skin: descriptor set allocation failed"));
            }
            VkDescriptorBufferInfo const skins_info = {.buffer = detail->buffer, .offset = 0, .range = VK_WHOLE_SIZE};
            VkWriteDescriptorSet write = {};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = this->compute_skin_sets[slot].get();
            write.dstBinding = 9; // SkinMatrices, the same binding shaders/pbr.vert reads
            write.dstArrayElement = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &skins_info;
            vkUpdateDescriptorSets(this->vulkan_core.device, 1, &write, 0, nullptr);
        }
        return {};
    }

    bool runtime::record_compute_skin_pass(VkCommandBuffer const command_buffer) {
        if (!this->rt_skin_bake || !this->compute_skin_pipeline.has_value() || this->rt_skin_levels.empty()) {
            return false;
        }
        uint32_t const slot = static_cast<uint32_t>(this->vulkan_core.current_frame);
        if (slot >= this->compute_skin_sets.size() || this->compute_skin_sets[slot].get() == VK_NULL_HANDLE) {
            return false;
        }
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->compute_skin_pipeline->get_pipeline());
        VkDescriptorSet const set = this->compute_skin_sets[slot].get();
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->compute_skin_pipeline_layout, 0, 1, &set, 0, nullptr);

        auto const halves = [](VkDeviceAddress const address) {
            return glm::uvec2(static_cast<uint32_t>(address & 0xFFFFFFFFu), static_cast<uint32_t>(address >> 32u));
        };
        constexpr uint32_t group_size = 64; // shaders/compute_skin.comp's local_size_x
        bool recorded = false;
        for (auto const& built : this->rt_caster_levels) {
            if (built.skin_destination_address == 0) {
                continue; // not a skinned caster: its geometry is what the build read, unchanged
            }
            compute_skin_push_constants push = {};
            push.source_vertices = halves(built.skin_source_address);
            push.destination = halves(built.skin_destination_address);
            push.source_stride = built.skin_source_stride;
            push.destination_stride = built.skin_destination_stride;
            push.vertex_count = built.skin_vertex_count;
            push.skin_base = built.skin_base;
            vkCmdPushConstants(command_buffer, this->compute_skin_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(command_buffer, (push.vertex_count + group_size - 1u) / group_size, 1, 1);
            recorded = true;
        }
        if (!recorded) {
            return false;
        }

        // What follows reads what this dispatch wrote: the BUILD on the frame the structures are created, and
        // the REFIT on every frame after. A compute write is not visible to the acceleration structure build
        // stage without this barrier, and the symptom would be a structure built or refitted against the
        // previous frame's vertices - a shadow one frame behind, which reads as animation lag.
        VkMemoryBarrier2 skin_order = {};
        skin_order.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        skin_order.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        skin_order.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        skin_order.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        skin_order.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo const skin_dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                  .pNext = nullptr,
                                                  .dependencyFlags = 0,
                                                  .memoryBarrierCount = 1,
                                                  .pMemoryBarriers = &skin_order,
                                                  .bufferMemoryBarrierCount = 0,
                                                  .pBufferMemoryBarriers = nullptr,
                                                  .imageMemoryBarrierCount = 0,
                                                  .pImageMemoryBarriers = nullptr};
        vkCmdPipelineBarrier2(command_buffer, &skin_dependency);
        return true;
    }

    std::expected<void, std::string> runtime::make_rt_shadow_pipeline(std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        if (!this->vulkan_core.ray_query_available) {
            return fail(std::string("rt shadow: this device has no ray queries (VK_KHR_acceleration_structure + VK_KHR_ray_query)"));
        }
        if (!this->deferred_pipeline.has_value()) {
            return fail(std::string("rt shadow: create the deferred lighting pipeline first (it owns the G-buffer set layout)"));
        }
        auto built = pipelines::build_rt_shadow(this->vulkan_core, this->vulkan_core.scene_descriptor_set_layout, this->gbuffer_set_layout, sizeof(rt_shadow_push_constants), compute_shader_code);
        if (!built) {
            return fail(built.error());
        }
        this->rt_shadow_pipeline_layout = built->pipeline_layout;
        this->rt_shadow_pipeline = std::move(built->trace);
        return {};
    }

    void runtime::record_rt_shadow_pass(VkCommandBuffer const command_buffer) {
        core& vk = this->vulkan_core;
        uint32_t const frame_slot = static_cast<uint32_t>(vk.current_frame);
        auto const& visibility_images = vk.rt_shadow_images;
        if (frame_slot >= visibility_images.size() || visibility_images[frame_slot] == VK_NULL_HANDLE) {
            return;
        }
        // Nothing to trace against, or this slot's structure is not built yet: the caller's off path
        // leaves the image readable and the light UBO's flag is 0, so the frame shades from the cascaded
        // maps. Gating on the SAME handle the binding-16 write is gated on is what keeps a dispatch from
        // ever reading an unwritten descriptor.
        if (!this->rt_top_levels.has_value() || this->rt_top_levels->handle(frame_slot) == VK_NULL_HANDLE) {
            return;
        }
        // The G-buffer set is written by the accessor the GI passes and the debug view share; this pass
        // can be the first to need it on a frame where none of them ran.
        this->ensure_gbuffer_descriptors();
        // ... and this pass is the FIRST sampler of the stored surface when it runs, so it is the one
        // that has to publish the G-buffer instance's attachment writes (the lighting stage's identical
        // call then finds the flags clear).
        this->ensure_gbuffer_targets_sampled(command_buffer, static_cast<uint32_t>(this->current_image_index));
        this->ensure_gbuffer_depth_sampled(command_buffer, static_cast<uint32_t>(this->current_image_index));

        // The image is written as a storage image (GENERAL) and read by the lighting stage as a sampler
        // (SHADER_READ). Both transitions happen here, around the dispatch, because this is the only
        // place that knows the image is being rewritten - the lighting stage's descriptor declares
        // SHADER_READ whether or not this pass ran (see the off path at the caller).
        VkImageMemoryBarrier2 to_general = vulkan::undefined_to_general_transition;
        to_general.image = visibility_images[frame_slot];
        VkDependencyInfo const general_dependency = make_image_dependency_info(1, &to_general);
        vkCmdPipelineBarrier2(command_buffer, &general_dependency);

        std::array<VkDescriptorSet, 2> const sets = {this->scene_sets.set(frame_slot), this->gbuffer_family.set(this->current_image_index, 0)};
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->rt_shadow_pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->rt_shadow_pipeline->get_pipeline());

        rt_shadow_push_constants const push = {
            .inv_view_proj = this->current_inv_view_proj,
            .params = glm::vec4(0.01f, 0.002f, 0.0015f, 0.0f),
        };
        vkCmdPushConstants(command_buffer, this->rt_shadow_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        constexpr uint32_t group_size = 8; // shaders/rt_shadow.comp's local_size_x/y
        vkCmdDispatch(command_buffer, (vk.swap_chain_extent.width + group_size - 1) / group_size, (vk.swap_chain_extent.height + group_size - 1) / group_size, 1);

        VkImageMemoryBarrier2 to_sampling = vulkan::general_to_sampling_transition;
        to_sampling.image = visibility_images[frame_slot];
        VkDependencyInfo const sampling_dependency = make_image_dependency_info(1, &to_sampling);
        vkCmdPipelineBarrier2(command_buffer, &sampling_dependency);

        if (!this->rt_shadow_logged) {
            this->rt_shadow_logged = true;
            utility::log("ray-traced shadows: tracing {}x{} rays per frame (one per pixel, terminated on the first hit)",
                         vk.swap_chain_extent.width,
                         vk.swap_chain_extent.height);
        }
    }

    bool runtime::record_ssgi_spatial_pass(VkCommandBuffer const command_buffer) {
        core& vk = this->vulkan_core;
        std::size_t const index = this->current_image_index;
        if (index >= vk.gi_spatial_images.size() || vk.gi_resolve_images.size() != vk.gi_spatial_images.size()) {
            return false;
        }
        if (this->ssgi_spatial_pipeline == std::nullopt) {
            return false;
        }
        // The G-buffer set carries every binding this pass uses (the normal, the depth, the image it
        // reads and the one it writes), so it is written by the same accessor the tracer uses.
        this->ensure_gbuffer_descriptors();
        VkDescriptorSet const gbuffer_set = this->gbuffer_family.set(static_cast<uint32_t>(index), 0);
        if (gbuffer_set == VK_NULL_HANDLE) {
            return false; // no set: the composite's GI weight stays 0 for this frame (see gi_resolved)
        }

        uint32_t const gi_width = std::max(1u, vk.swap_chain_extent.width / 2u);
        uint32_t const gi_height = std::max(1u, vk.swap_chain_extent.height / 2u);

        // The output is a storage image: UNDEFINED -> GENERAL here (its contents are fully overwritten)
        // and GENERAL -> SHADER_READ below, for the composite. The input needs no barrier: the temporal
        // resolve handed it to SHADER_READ through a transition that names COMPUTE as well as FRAGMENT.
        VkImageMemoryBarrier2 to_general = vulkan::undefined_to_general_transition;
        to_general.image = vk.gi_spatial_images[index];
        VkDependencyInfo const general_dependency = make_image_dependency_info(1, &to_general);
        vkCmdPipelineBarrier2(command_buffer, &general_dependency);

        std::array<VkDescriptorSet, 2> const sets = {this->scene_sets.set(static_cast<uint32_t>(vk.current_frame)), gbuffer_set};
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->ssgi_spatial_pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->ssgi_spatial_pipeline->get_pipeline());

        ssgi_spatial_push_constants const push = {
            .depth_scale = this->current_ubo.proj[2][2],
            .depth_offset = this->current_ubo.proj[3][2],
            .sigma_spatial = this->gi_spatial_sigma,
            .sigma_depth = this->gi_spatial_depth_sigma,
            .normal_power = this->gi_spatial_normal_power,
            // The subtraction belongs to the TRACED path only: the marched one is an ADDITION to the probe
            // ambient, so it must not remove anything. Same predicate the tracer's push uses, evaluated in
            // the same frame, so the two cannot disagree about which path ran.
            .subtract_ambient = this->ssgi_traced_active() ? 1.0f : 0.0f,
            // ... and the REFLECTION's own accumulation is summed in by the filter at binding 15. The
            // SPECULAR AMBIENT is still not subtracted here: the glossy pass removes the lighting stage's
            // specular term at its own texel, which is exact where a subtraction in this filter could only
            // approximate (see shaders/ssgi_spatial.comp's binding comment and docs/gi_hit_shading.md's L2.3
            // section for the measurement that chose it). This lane is only about how much of the
            // reflection's own accumulation to include, and zero is what keeps a stale one out of a frame
            // whose lobe did not run. The predicate is whether the reflection was actually RESOLVED this
            // frame rather than whether the lobe is enabled: if its descriptor set could not be had, the
            // accumulation holds an older frame and must not be summed in. The denoise pass runs before this
            // one, so the flag is this frame's.
            .spec_weight = this->gi_spec_resolved ? 1.0f : 0.0f,
            .unused2 = 0.0f,
            .gi_size = glm::vec4(static_cast<float>(gi_width), static_cast<float>(gi_height),
                                 static_cast<float>(vk.swap_chain_extent.width), static_cast<float>(vk.swap_chain_extent.height))};
        vkCmdPushConstants(command_buffer, this->ssgi_spatial_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        constexpr uint32_t group_size = 8; // shaders/ssgi_spatial.comp's local_size_x/y
        vkCmdDispatch(command_buffer, (gi_width + group_size - 1) / group_size, (gi_height + group_size - 1) / group_size, 1);

        // Hand the filtered image to the composite. This is also where the frame's GI becomes usable:
        // gi_resolved is what the composite's weight is read from, and only this pass writes the image
        // that weight applies to.
        VkImageMemoryBarrier2 to_sampling = vulkan::general_to_sampling_transition;
        to_sampling.image = vk.gi_spatial_images[index];
        VkDependencyInfo const sampling_dependency = make_image_dependency_info(1, &to_sampling);
        vkCmdPipelineBarrier2(command_buffer, &sampling_dependency);

        this->gi_resolved = true;
        return true;
    }

    bool runtime::record_ssgi_spec_pass(VkCommandBuffer const command_buffer) {
        core& vk = this->vulkan_core;
        std::size_t const index = this->current_image_index;
        if (!this->ssgi_specular_active() || index >= vk.gi_images.size() || vk.gi_images[index] == VK_NULL_HANDLE) {
            return false;
        }
        // The tracer's own sets, unchanged: set 0 is the shared scene set (the camera, the environment, the
        // BRDF LUT, the material table and the top level structure) and set 1 is the G-buffer set, whose
        // binding 6 is the raw trace this pass reads, adds to and writes back. No descriptor work at all -
        // the image was already written as a read-write storage image in GENERAL for the tracer.
        this->ensure_gbuffer_descriptors();
        VkDescriptorSet const gbuffer_set = this->gbuffer_family.set(static_cast<uint32_t>(index), 0);
        if (gbuffer_set == VK_NULL_HANDLE) {
            return false;
        }

        // The instance table's address, exactly as the tracer's push carries it, and the switch with it: a
        // zero would leave the pass with nothing to shade a hit from, which ssgi_specular_active() already
        // refused above - so a table that turns out to be missing here is a frame whose structures went away
        // between the two calls, and doing nothing is the right answer rather than replacing the lighting
        // stage's specular ambient with the environment alone (which would be a no-op anyway).
        uint64_t instance_table = 0;
        if (this->ssgi_hit_shading && this->rt_top_levels.has_value()) {
            VkBuffer const table = this->rt_top_levels->instance_table(static_cast<uint32_t>(vk.current_frame));
            if (table != VK_NULL_HANDLE) {
                VkBufferDeviceAddressInfo const table_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = table};
                instance_table = vkGetBufferDeviceAddress(vk.device, &table_info);
            }
        }
        if (instance_table == 0) {
            return false;
        }

        // The raw trace is about to be READ as well as written, by a dispatch that is not the one that
        // wrote it: consecutive dispatches in one command buffer have no memory dependency between them, so
        // without this the glossy pass can read a texel the tracer has not finished writing. Same layout on
        // both sides (GENERAL), which is why this is the compute-storage barrier rather than a transition -
        // and why record_ssgi_pass left the image in GENERAL instead of handing it to the denoiser itself.
        //
        // The lobe's own two outputs are written by the same dispatch, so they need whatever layout a
        // storage image must be in before its FIRST write: UNDEFINED -> GENERAL, once per target generation
        // (see runtime::gi_spec_seen). After that they are already in GENERAL - this dispatch is their only
        // writer, so no later frame needs a barrier for them - and claiming UNDEFINED again would discard
        // the image for no reason. A missing first-use transition is the trap this project has paid for
        // twice already; a new image is not in a legal layout because its neighbours are.
        std::array<VkImageMemoryBarrier2, 3> lobe_barriers = {};
        lobe_barriers[0] = vulkan::compute_storage_transition;
        lobe_barriers[0].image = vk.gi_images[index];
        uint32_t lobe_barrier_count = 1;
        if (index < vk.gi_spec_images.size() && index < vk.gi_spec_reproject_images.size()) {
            // EVERY frame, not only the first: this pass is the storage writer of both images and the resolve
            // reads them back as samplers, so the frame's LAST transition of each is to SHADER_READ (see the
            // hand-back at the end of this function). The first frame of a target generation comes from
            // UNDEFINED, and later ones from that readable state. Claiming UNDEFINED every frame would also
            // work - the pass rewrites every non-background texel - but it would throw the images away for no
            // reason; claiming a layout an image is not in is the thing that is actually illegal.
            VkImageMemoryBarrier2 const from = this->gi_spec_seen[index] ? vulkan::sampling_to_general_transition : vulkan::undefined_to_general_transition;
            lobe_barriers[lobe_barrier_count] = from;
            lobe_barriers[lobe_barrier_count].image = vk.gi_spec_images[index];
            ++lobe_barrier_count;
            lobe_barriers[lobe_barrier_count] = from;
            lobe_barriers[lobe_barrier_count].image = vk.gi_spec_reproject_images[index];
            ++lobe_barrier_count;
            this->gi_spec_seen[index] = true;
        }
        VkDependencyInfo const order_dependency = make_image_dependency_info(lobe_barrier_count, lobe_barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &order_dependency);

        uint32_t const gi_width = std::max(1u, vk.swap_chain_extent.width / 2u);
        uint32_t const gi_height = std::max(1u, vk.swap_chain_extent.height / 2u);
        std::array<VkDescriptorSet, 2> const sets = {this->scene_sets.set(static_cast<uint32_t>(vk.current_frame)), gbuffer_set};
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->ssgi_spec_pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->ssgi_spec_pipeline->get_pipeline());

        // The two halves of the instance table's address, bit-cast into float lanes - a push constant is raw
        // bytes, so half an address survives the trip exactly (the same trick the tracer's proj_terms uses).
        float const table_low = std::bit_cast<float>(static_cast<uint32_t>(instance_table & 0xFFFFFFFFu));
        float const table_high = std::bit_cast<float>(static_cast<uint32_t>(instance_table >> 32u));
        ssgi_spec_push_constants const push = {
            .inv_view_proj = this->current_inv_view_proj,
            // The ray length is the lobe's OWN reach (`ssgi_specular_radius`), not the diffuse bounce's
            // `ssgi_radius` - see the setter for why the two are different questions and what the curve
            // between them costs (measured: 39% of the available signal at the marched path's 0.12, 90% at
            // 0.5, flat past it). z is the self-intersection bias as an explicit WORLD length rather than a
            // fraction of x - so that raising the reach does not also lift every ray's origin further off its
            // surface (see the shader's push comment).
            .params = glm::vec4(this->ssgi_specular_radius * this->scene_radius,
                                static_cast<float>(this->ssgi_specular_rays),
                                this->scene_radius * 0.0002f,
                                static_cast<float>(this->ssgi_frame)),
            .table = glm::vec4(0.0f, 0.0f, table_low, table_high)};
        vkCmdPushConstants(command_buffer, this->ssgi_spec_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        constexpr uint32_t group_size = 8; // shaders/ssgi_spec.comp's local_size_x/y
        vkCmdDispatch(command_buffer, (gi_width + group_size - 1) / group_size, (gi_height + group_size - 1) / group_size, 1);

        // ... and hand the completed raw trace to the denoiser. This is the transition record_ssgi_pass
        // would have written when this pass does not run, moved here because the image has a second writer
        // now: the barrier has to come after the LAST one, or the resolve could sample a half-written trace.
        //
        // The lobe's own two outputs come along: they were written as storage images by the same dispatch and
        // the reflection's resolve reads BOTH as samplers (its trace, and the reprojection it reprojects by),
        // so this is where they become readable. Their descriptors in the G-buffer set declare GENERAL, which
        // is what they were written in; the temporal set declares SHADER_READ, which is what this leaves them
        // in - the two sets describe the same image at different points of the frame's pass order.
        VkImageMemoryBarrier2 const to_sampling = vulkan::general_to_sampling_transition;
        std::array<VkImageMemoryBarrier2, 3> to_sampling_barriers = {to_sampling, to_sampling, to_sampling};
        to_sampling_barriers[0].image = vk.gi_images[index];
        to_sampling_barriers[1].image = index < vk.gi_spec_images.size() ? vk.gi_spec_images[index] : vk.gi_images[index];
        to_sampling_barriers[2].image = index < vk.gi_spec_reproject_images.size() ? vk.gi_spec_reproject_images[index] : vk.gi_images[index];
        VkDependencyInfo const sampling_dependency = make_image_dependency_info(static_cast<uint32_t>(to_sampling_barriers.size()), to_sampling_barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &sampling_dependency);
        return true;
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
        //   3. the motion-vector target becomes a shader input as well. It needs its own transition
        //      here because the TAA resolve is the only other stage that samples it, and the debug
        //      view runs INSTEAD of the lighting stage - which is what the TAA resolve hangs off -
        //      so without this the read happens against COLOR_ATTACHMENT_OPTIMAL, which is a
        //      validation error and, on a driver that believes it, garbage.
        //   4. the G-buffer depth image becomes a shader input too - through the same accessor the
        //      other two sampling stages use, because its old layout depends on whether the G-buffer
        //      instance rendered this frame (the aspect must be DEPTH; see the accessor).
        std::array<VkImageMemoryBarrier2, 5> barriers = {};
        barriers[0] = vulkan::color_attachment_transition;
        barriers[0].image = vk.hdr_images[index];
        for (uint32_t target = 0; target < vulkan::gbuffer_target_count; ++target) {
            barriers[target + 1] = vulkan::hdr_sampling_transition; // COLOR_ATTACHMENT -> SHADER_READ
            barriers[target + 1].image = vk.gbuffer_images[target][index];
        }
        barriers[4] = vulkan::hdr_sampling_transition; // same COLOR_ATTACHMENT -> SHADER_READ, color aspect
        barriers[4].image = vk.velocity_images[index];
        VkDependencyInfo const dependency = make_image_dependency_info(static_cast<uint32_t>(barriers.size()), barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency);
        // the debug view ran instead of the lighting stage, so it is the stage that hands the
        // motion-vector target to a sampler this frame (see ensure_velocity_sampled)
        if (index < this->velocity_written.size()) {
            this->velocity_written[index] = false;
        }
        this->ensure_gbuffer_depth_sampled(command_buffer, static_cast<uint32_t>(index));

        this->ensure_gbuffer_descriptors();

        VkClearValue clear = {};
        VkRenderingAttachmentInfo const color_attachment = make_color_attachment_info(vk.hdr_image_views[index], clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
        VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, vk.swap_chain_extent}, true, &color_attachment, nullptr);
        vkCmdBeginRendering(command_buffer, &rendering_info);
        bool const can_draw = this->gbuffer_family.set(static_cast<uint32_t>(index), 0) != VK_NULL_HANDLE;
        if (can_draw) {
            this->gbuffer_debug_pipeline->begin_pipeline(command_buffer);
            VkViewport const viewport = {0.0f, 0.0f, static_cast<float>(vk.swap_chain_extent.width), static_cast<float>(vk.swap_chain_extent.height), 0.0f, 1.0f};
            VkRect2D const scissor = {{0, 0}, vk.swap_chain_extent};
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);
            vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);
            VkDescriptorSet const set = this->gbuffer_family.set(static_cast<uint32_t>(index), 0);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, this->gbuffer_pipeline_layout, 0, 1, &set, 0, nullptr);
            gbuffer_debug_push_constants const push = {
                .channel = static_cast<float>(this->gbuffer_channel_index),
                .proj_22 = this->current_ubo.proj[2][2],
                .proj_32 = this->current_ubo.proj[3][2],
                // Four pixels of motion saturate the motion channel (see the field's docs): derived
                // from the width so it means the same thing at any resolution.
                .motion_gain = static_cast<float>(vk.swap_chain_extent.width) * 0.25f};
            vkCmdPushConstants(command_buffer, this->gbuffer_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
        } else {
            utility::log("runtime: gbuffer debug pass has no descriptor set - showing a cleared frame");
        }
        vkCmdEndRendering(command_buffer);
    }

    void runtime::record_scene_tail(VkCommandBuffer const command_buffer) {
        // NO vkCmdEndRendering HERE ANY MORE: the scene PASS owns its instance and closes it at the end of its
        // own record (see vulkan.pass.scene). That line used to be the far half of a pair whose near half was
        // three functions away - the coupling this extraction removed.
        // GPU timing: the geometry instance ended where the scene pass closed it (the surface write).
        this->gpu_mark(command_buffer, gpu_mark_id::scene_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        core const& vk = this->vulkan_core;

        // Deferred mode: the surface is in the G-buffer and the sky + emissive are in the scene color
        // target; this stage shades every pixel from the G-buffer and adds the result on top, and the
        // alpha-blended leaves then composite over the shaded image (their own instance - see
        // record_transparent_pass).
        if (this->deferred_lit_active()) {
            // Ray-traced sun shadows run HERE: after the G-buffer pass (whose depth and normal the rays
            // start from) and before the lighting stage (which multiplies the sun term by the result).
            // Running it before the G-buffer pass would mean starting rays from the PREVIOUS frame's
            // surface, so the position is not a detail - it is the ordering constraint.
            if (this->rt_shadow_pipeline.has_value() && this->rt_shadows_active()) {
                this->record_rt_shadow_pass(command_buffer);
            } else if (static_cast<std::size_t>(vk.current_frame) < vk.rt_shadow_images.size() && vk.rt_shadow_images[vk.current_frame] != VK_NULL_HANDLE) {
                // The pass did not run, but the lighting stage's descriptor still declares the image as
                // a shader input: its shader samples the binding only under a flag, and Vulkan requires
                // a statically-used binding's image to be in the layout the descriptor declares whether
                // or not the value is used. UNDEFINED as the old layout asserts nothing - the same
                // answer the GI image's off path gives.
                VkImageMemoryBarrier2 to_sampling = vulkan::undefined_to_sampling_transition;
                to_sampling.image = vk.rt_shadow_images[vk.current_frame];
                VkDependencyInfo const sampling_dependency = make_image_dependency_info(1, &to_sampling);
                vkCmdPipelineBarrier2(command_buffer, &sampling_dependency);
            }
            // GPU timing: the ray-traced shadow pass ends here (before the lighting stage reads its
            // output). It gets its own interval because it sits between the G-buffer pass and the
            // lighting stage - without it the traversals were reported as lighting time, which made the
            // lighting interval look four times more expensive with rays on (measured 0.32 -> 1.18 ms
            // while the rays themselves were ~0.85 of that).
            this->gpu_mark(command_buffer, gpu_mark_id::rt_shadow_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
            this->record_lighting_pass(command_buffer);
            this->record_transparent_pass(command_buffer);
        } else {
            // The pass does not run (the debug view replaces the lighting stage, and the forward path has
            // no G-buffer to start rays from), but every mark is written in order on every frame - the
            // report's labels are positional. Written next to scene_end, so the interval is 0 ms.
            this->gpu_mark(command_buffer, gpu_mark_id::rt_shadow_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        }
        this->gpu_mark(command_buffer, gpu_mark_id::lighting_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // TAA resolve: blend the scene color with the reprojected history into the HDR target the post
        // chain reads, then copy the result into the history image for the next frame that renders this
        // swapchain image. The PASS owns all of that now (vulkan.pass.taa), and the two lines below are the
        // part it cannot own yet:
        //
        //  * the G-buffer depth's transition to a sampled layout, whose "was it written this frame" flag
        //    belongs to the G-buffer pass, and
        //  * clearing the motion-vector flag, which is what stops the GI chain (later in the same frame) from
        //    transitioning the velocity image a second time.
        //
        // Both are shared per-image bookkeeping - the barrier/order stage's job in the long run - and both are
        // gated on THE SAME predicate the runner gates the stage on, so the host never touches them on a frame
        // the pass does not run (clearing the velocity flag for a frame with no resolve would make the GI
        // tracer sample an image still in ATTACHMENT layout).
        if (this->active_features().taa) {
            if (this->current_image_index < this->velocity_written.size()) {
                this->velocity_written[this->current_image_index] = false;
            }
            static_cast<void>(this->ensure_gbuffer_depth_sampled(command_buffer, static_cast<uint32_t>(this->current_image_index)));
        }
        {
            pass::stage const taa_stage = {.name = "taa", .passes = this->taa_stage, .marks = false};
            [[maybe_unused]] pass::run_report const taa_report = pass::record_stage(taa_stage, this->make_pass_host());
        }
        // The matrix the NEXT frame's motion vectors are computed against is this frame's, and it is only
        // recorded when the resolve actually wrote a history: a resolve that bailed out (no descriptor set)
        // must not claim one. `image_view_proj` stays the renderer's because the camera UBO - not TAA - reads
        // it as `prev_view_proj`.
        if (this->taa_resolve.wrote_history() && this->current_image_index < this->image_view_proj.size()) {
            this->image_view_proj[this->current_image_index] = this->current_ubo.view_proj_unjittered;
        }
        this->gpu_mark(command_buffer, gpu_mark_id::taa_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // G-buffer debug mode (an inspection of the stored data, never combined with the lighting
        // stage or TAA): turn one channel into a visible image in the HDR target.
        if (this->gbuffer_pass_active() && !this->deferred_lit_active()) {
            this->record_gbuffer_debug_pass(command_buffer);
        }
        this->gpu_mark(command_buffer, gpu_mark_id::main_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    }

    void runtime::barrier_image_to_sampling(VkCommandBuffer const command_buffer, VkImage const image) {
        // A pipeline barrier may not be recorded inside a dynamic rendering instance
        // (VUID-vkCmdPipelineBarrier2-None-09553), which is why every caller of this runs BEFORE its
        // vkCmdBeginRendering.
        std::array<VkImageMemoryBarrier2, 1> barriers = {vulkan::hdr_sampling_transition};
        barriers[0].image = image;
        VkDependencyInfo const dependency_info = make_image_dependency_info(1, barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency_info);
    }

    void runtime::barrier_image_to_color_attachment(VkCommandBuffer const command_buffer, VkImage const image) {
        // UNDEFINED as the old layout: every caller renders into the image with loadOp CLEAR, so the
        // previous contents are irrelevant whatever layout they were in (see the transition constants).
        std::array<VkImageMemoryBarrier2, 1> barriers = {vulkan::color_attachment_transition};
        barriers[0].image = image;
        VkDependencyInfo const dependency_info = make_image_dependency_info(1, barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency_info);
    }

    void runtime::record_overlay_if_enabled(VkCommandBuffer const command_buffer) {
        if (this->debug_gui_shown && this->debug_overlay.is_active()) {
            this->debug_overlay.record(command_buffer);
        }
    }

    void runtime::record_fullscreen_triangle(VkCommandBuffer const command_buffer, vk_pipeline const& pipeline, VkImageView const target_view, VkExtent2D const extent, VkDescriptorSet const set, post_push_constants const& push, bool const overlay_after) {
        VkClearValue clear = {};
        VkRenderingAttachmentInfo const attachment = make_color_attachment_info(target_view, clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
        VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, extent}, true, &attachment, nullptr);
        vkCmdBeginRendering(command_buffer, &rendering_info);
        pipeline.begin_pipeline(command_buffer);
        VkViewport const viewport = {0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
        VkRect2D const scissor = {{0, 0}, extent};
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);
        vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE); // the fullscreen triangle has no facing
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, this->post_pipeline_layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(command_buffer, this->post_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(command_buffer, 3, 1, 0, 0);
        // Inside THIS instance, before it closes: the overlay is not a fullscreen pass and has no
        // loadOp of its own, so a pass of its own would CLEAR the image the triangle just wrote.
        if (overlay_after) {
            this->record_overlay_if_enabled(command_buffer);
        }
        vkCmdEndRendering(command_buffer);
    }

    void runtime::record_bloom_chain(VkCommandBuffer const command_buffer, uint32_t const image_index, float const bloom_intensity) {
        core const& vk = this->vulkan_core;
        std::size_t const index = image_index;

        // level L target size: half the swapchain extent per level (min 1x1, matches core)
        auto const level_size = [&vk](uint32_t const level) {
            return VkExtent2D{std::max(1u, vk.swap_chain_extent.width >> (level + 1u)), std::max(1u, vk.swap_chain_extent.height >> (level + 1u))};
        };
        auto const bloom_push = [this](float const mode) {
            return post_push_constants{
                .exposure = this->exposure_scale,
                .bloom_intensity = this->bloom_intensity,
                .bloom_threshold = this->bloom_threshold,
                .mode = mode};
        };

        // ---- bloom chain: bright-pass prefilter into level 0, then downsample level by level ----
        // Every stage here renders into an R16F bloom level, so it uses the HDR-format pipeline
        // variant. Skipped entirely when the composite multiplies the bloom sum by 0: the four
        // fullscreen passes would be pure cost. The levels are still moved to SHADER_READ_ONLY (from
        // UNDEFINED - their contents are dead and the composite's static use of those bindings still
        // requires a valid layout), because the composite samples them and multiplies by 0.
        if (bloom_intensity > 0.0f) {
            this->barrier_image_to_color_attachment(command_buffer, vk.bloom_images[0][index]);
            this->record_fullscreen_triangle(command_buffer, *this->post_hdr_pipeline, vk.bloom_image_views[0][index], level_size(0), this->post_family.set(image_index, 0), bloom_push(0.0f));
            for (std::size_t level = 0; level < 3; ++level) {
                this->barrier_image_to_sampling(command_buffer, vk.bloom_images[level][index]);
                this->barrier_image_to_color_attachment(command_buffer, vk.bloom_images[level + 1][index]);
                this->record_fullscreen_triangle(command_buffer, *this->post_hdr_pipeline, vk.bloom_image_views[level + 1][index], level_size(static_cast<uint32_t>(level) + 1u), this->post_family.set(image_index, static_cast<uint32_t>(level) + 1u), bloom_push(1.0f));
            }
            this->barrier_image_to_sampling(command_buffer, vk.bloom_images[3][index]);
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
    }

    bool runtime::record_composite(VkCommandBuffer const command_buffer, uint32_t const image_index, float const bloom_intensity) {
        core const& vk = this->vulkan_core;
        std::size_t const index = image_index;
        VkExtent2D const full_extent = vk.swap_chain_extent;

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
        // Without FXAA the composite writes a LINEAR tonemapped image into an sRGB swapchain
        // attachment, which encodes it to display values in hardware, so the shader must NOT apply
        // gamma as well; only a non-sRGB (UNORM) swapchain needs the manual transfer function. With
        // FXAA the target is the R16F LDR image and the shader must encode.
        float const composite_encode_gamma = fxaa ? 1.0f : (is_srgb_format(vk.swap_chain_image_format) ? 0.0f : 1.0f);

        // The overlay draws on the final 1x swapchain image (no depth attachment,
        // attachment, see enable_debug_gui) and must be the LAST writer, so it goes inside the same
        // instance as the pass it sits on top of. With FXAA that is the FXAA pass, not the composite:
        // drawing it here would let the edge filter blur the UI text into mush.
        this->barrier_image_to_color_attachment(command_buffer, composite_image);
        post_push_constants const composite_push = {
            .exposure = this->exposure_scale,
            .bloom_intensity = bloom_intensity,
            .bloom_threshold = this->bloom_threshold,
            .mode = 2.0f,
            .encode_gamma = composite_encode_gamma,
            // 0 unless THIS frame's GI resolve ran, which makes the composite's added term exactly
            // zero on every frame that has no GI to add (see gi_resolved)
            .gi_intensity = this->gi_resolved ? 1.0f : 0.0f,
            .gi_depth_scale = this->current_ubo.proj[2][2],
            .gi_depth_offset = this->current_ubo.proj[3][2],
            // The SAME edge criterion the spatial filter uses: one silhouette test for the whole chain,
            // so what survives the filter is not undone by the upsample.
            .gi_depth_sigma = this->gi_spatial_depth_sigma,
            .gi_normal_power = this->gi_spatial_normal_power,
            .gi_upsample = this->gi_upsample ? 1.0f : 0.0f,
            .fxaa_subpixel = this->fxaa_subpixel,
            .fxaa_edge_threshold = this->fxaa_edge_threshold};
        this->record_fullscreen_triangle(command_buffer, composite_pipeline, composite_view, full_extent, this->post_family.set(image_index, 4), composite_push, /*overlay_after=*/!fxaa);

        // GPU timing: the composite (and the debug overlay, when it draws here) is done.
        this->gpu_mark(command_buffer, gpu_mark_id::composite_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        if (fxaa) {
            // ---- FXAA: gamma-encoded LDR image -> anti-aliased swapchain (+ overlay on top) ----
            // Same meaning of encode_gamma as in the composite: 0 = the swapchain attachment encodes
            // to display values in hardware, so FXAA must hand it LINEAR values; 1 = the target is a
            // UNORM format and FXAA's own display-encoded result is what should be stored.
            post_push_constants const fxaa_push = {
                .exposure = this->exposure_scale,
                .bloom_intensity = this->bloom_intensity,
                .bloom_threshold = this->bloom_threshold,
                .mode = 3.0f,
                .encode_gamma = is_srgb_format(vk.swap_chain_image_format) ? 0.0f : 1.0f,
                .fxaa_subpixel = this->fxaa_subpixel,
                .fxaa_edge_threshold = this->fxaa_edge_threshold};
            this->barrier_image_to_sampling(command_buffer, vk.ldr_images[index]);
            this->barrier_image_to_color_attachment(command_buffer, vk.swap_chain_images[index]);
            this->record_fullscreen_triangle(command_buffer, *this->post_fxaa_pipeline, vk.swap_chain_image_views[index], full_extent, this->post_family.set(image_index, 4), fxaa_push, /*overlay_after=*/true);
        }
        // GPU timing: the FXAA pass (and the overlay it carries when it is the last writer) is done.
        // Without FXAA the composite already ended the frame's display work, so this interval is ~0.
        this->gpu_mark(command_buffer, gpu_mark_id::fxaa_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // true in both paths: the FXAA pass (or the composite, when FXAA is off) wrote the swapchain
        return true;
    }

    bool runtime::record_post_process(VkCommandBuffer const command_buffer) {
        core const& vk = this->vulkan_core;

        // ---- the scene side: close the geometry instance, then the stages that consume the G-buffer
        this->record_scene_tail(command_buffer);

        this->ensure_post_descriptors();
        if (this->post_pipeline == std::nullopt || this->post_hdr_pipeline == std::nullopt || this->post_family.set(static_cast<uint32_t>(this->current_image_index), 4) == VK_NULL_HANDLE) {
            return false; // no post pipeline (creation failed): the HDR frame cannot be presented correctly
        }

        std::size_t const index = this->current_image_index;

        // HDR scene target -> fragment-shader read (the prefilter and the composite both read it)
        this->barrier_image_to_sampling(command_buffer, vk.hdr_images[index]);

        // The composite's GI upsample also samples the G-buffer depth and normal, and the stage that
        // normally publishes those two is the deferred lighting stage (or the debug view) - both part
        // of the G-buffer path. A frame that never ran it (the deferred pipeline missing, so
        // gbuffer_pass_active() is false and nothing wrote the targets) still reaches the composite,
        // whose descriptor declares SHADER_READ for both bindings either way. UNDEFINED as the old
        // layout is honest here - there is no content to preserve - and the GI weight is 0 on such a
        // frame, so the taps' values cannot influence the image.
        if (!this->gbuffer_pass_active()) {
            std::array<VkImageMemoryBarrier2, 2> gbuffer_barriers = {};
            gbuffer_barriers[0] = vulkan::undefined_to_depth_sampling_transition; // DEPTH aspect
            gbuffer_barriers[0].image = vk.gbuffer_depth_images[index];
            gbuffer_barriers[1] = vulkan::undefined_to_sampling_transition;
            gbuffer_barriers[1].image = vk.gbuffer_images[1][index]; // the world normal
            VkDependencyInfo const gbuffer_dependency = make_image_dependency_info(static_cast<uint32_t>(gbuffer_barriers.size()), gbuffer_barriers.data());
            vkCmdPipelineBarrier2(command_buffer, &gbuffer_dependency);
        }

        // Screen-space GI runs HERE, and the position is the whole reason it cannot feed back: `hdr`
        // was rewritten earlier in this frame (by the TAA resolve, or by the G-buffer pass's clear
        // when TAA is off) and the composite that ADDS this pass's output is downstream, so what the
        // tracer samples at a hit is direct radiance and never its own previous result. Sampling an
        // image that already contained GI would make the loop gain > 1 and accumulate energy.
        //
        // gi_resolved says whether the composite may actually use this frame's GI, and it is the LAST
        // pass of the chain - the spatial filter - that sets it: the composite samples that filter's
        // output, so a frame whose filter did not run (no descriptor set, no images) has nothing to add
        // and must weigh 0 rather than show whatever that image happens to hold. The filter in turn
        // only runs when the temporal resolve ran, because filtering a stale accumulation would just
        // make the staleness smoother.
        this->gi_resolved = false;
        if (this->ssgi_active()) {
            this->record_ssgi_pass(command_buffer);
            // The glossy lobe, between the tracer and the denoiser: it adds to the tracer's own image (see
            // shaders/ssgi_spec.comp), so it has to run BEFORE the temporal resolve reads that image, and
            // the tracer skipped its hand-off barrier for exactly this case.
            this->record_ssgi_spec_pass(command_buffer);
            if (this->record_ssgi_denoise_pass(command_buffer)) {
                this->record_ssgi_spatial_pass(command_buffer);
            }
            ++this->ssgi_frame; // the next frame's ray sequence must differ (see ssgi_frame)
        }
        if (!this->gi_resolved && index < vk.gi_spatial_images.size() && vk.gi_spatial_images[index] != VK_NULL_HANDLE) {
            // Nothing wrote the GI image this frame, but the composite's descriptor set still declares
            // it as a shader input - its shader uses that binding and multiplies it by the 0 pushed
            // above, and Vulkan requires a statically-used binding's descriptor to be in the layout the
            // write declared, whether or not the value ends up mattering. Nothing else touches the
            // image in this case, so it would sit in UNDEFINED and every frame would be a layout error.
            // This is the same situation the shadow map's spare layers are in, and the same answer: an
            // UNDEFINED old layout asserts nothing (it discards the contents rather than claiming a
            // layout), so the transition is valid whether the image is untouched or already readable.
            VkImageMemoryBarrier2 to_sampling = vulkan::undefined_to_sampling_transition;
            to_sampling.image = vk.gi_spatial_images[index];
            VkDependencyInfo const sampling_dependency = make_image_dependency_info(1, &to_sampling);
            vkCmdPipelineBarrier2(command_buffer, &sampling_dependency);
        }

        // GPU timing: the GI chain ends here (trace, temporal resolve, spatial filter; the composite's
        // bilateral upsample is part of the composite). Written unconditionally like every mark, so a
        // frame with GI off reports 0 ms and the positional labels stay aligned.
        this->gpu_mark(command_buffer, gpu_mark_id::gi_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // The world-space probe cache, last in the GI stretch: it deposits THIS frame's resolved GI into
        // the cells the frame can see, so it has to run after the spatial filter (the image it reads is
        // this frame's) and before the composite (nothing about it is needed for this frame's image -
        // what it produces is read by the NEXT frame's tracer, which is what a cache costs). The RUNNER
        // decides whether it runs at all: the pass declares the feature `ssgi_probes`, and an inactive
        // feature is skipped WITHOUT being resolved - which is what keeps a cache that is off bit for bit
        // what the frame was before it existed. Its stage writes no mark pair (the interval is measured
        // from the GI chain's end), so `marks = false` and the end mark below is the runtime's.
        {
            pass::stage const probe_stage = {.name = "gi_probe", .passes = this->gi_probe_stage, .marks = false};
            [[maybe_unused]] pass::run_report const probe_report = pass::record_stage(probe_stage, this->make_pass_host());
        }
        this->gpu_mark(command_buffer, gpu_mark_id::gi_probe_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        // The G-buffer debug view forces the bloom weight to 0: bloom is a display effect, and a glow
        // smeared over the channel being inspected is the opposite of a debug view (it would also
        // invent colors that are not in the G-buffer at all). The deferred LIT image is a real image,
        // so bloom stays on for it - which is why this is not simply `this->bloom_intensity`.
        bool const debug_view = this->gbuffer_pass_active() && !this->deferred_lit_active();
        float const bloom_intensity = debug_view ? 0.0f : this->bloom_intensity;

        this->record_bloom_chain(command_buffer, static_cast<uint32_t>(index), bloom_intensity);
        return this->record_composite(command_buffer, static_cast<uint32_t>(index), bloom_intensity);
    }
    frame_status runtime::end_recording() {
        vulkan::profiling::cpu_phase_timer const phase_timer{this->cpu_timings, vulkan::profiling::cpu_phase::post};
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
        // presentation engine: transition the swapchain image to PRESENT_SRC_KHR explicitly. The
        // post pass leaves the image in COLOR_ATTACHMENT_OPTIMAL, so the barrier is needed whenever
        // it ran. When it was skipped the image never entered COLOR_ATTACHMENT_OPTIMAL, and claiming
        // that old layout would be a lie (validation: "oldLayout is not matching with the current
        // layout"): transition from UNDEFINED instead - the frame has no content to preserve anyway.
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
        vulkan::profiling::cpu_phase_timer const phase_timer{this->cpu_timings, vulkan::profiling::cpu_phase::submit};
        core& vk = this->vulkan_core;
        vk_command_buffer& command_buffer = this->command_buffers[static_cast<uint32_t>(vk.current_frame)];

        // Submit + present; recreate the swapchain when presentation reports out of date
        auto const submit_started = std::chrono::steady_clock::now(); // sub-phase marks: what of submit is the
        // queue submission (with its timeline signal) and what is the present call below
        if (vk.submit(*command_buffer, this->current_image_index) != VK_SUCCESS) {
            return frame_status::submit_failed;
        }
        this->cpu_timings.add(vulkan::profiling::cpu_phase::submit_queue, std::chrono::steady_clock::now() - submit_started);
        auto const present_started = std::chrono::steady_clock::now();
        VkResult const present_result = vk.present(this->current_image_index);
        this->cpu_timings.add(vulkan::profiling::cpu_phase::present, std::chrono::steady_clock::now() - present_started);
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
        info.depth_format = VK_FORMAT_UNDEFINED; // the post/gui pass has no depth attachment
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
            static_cast<float>(this->shadow_map_size),
            static_cast<float>(this->shadow_map_size),
            0.0f,
            1.0f,
        };
        this->shadow_pipeline->scissor = {{0, 0}, {this->shadow_map_size, this->shadow_map_size}};
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

    void runtime::set_shadow_map_size(uint32_t const size) noexcept {
        // Startup-only: everything that consumes the size (the layered image + its views + the
        // descriptor, the depth pass rendering instance, the pipeline viewport, the light UBO texel
        // size and the fit) is built from it when the scene set is first created, so a change after
        // that cannot take effect - say so instead of pretending otherwise.
        if (this->shadow_pipeline.has_value() || !this->shadow_images.empty()) {
            utility::log("runtime: set_shadow_map_size({}) ignored - the shadow resources already exist (set it before the scene import)", size);
            return;
        }
        uint32_t clamped = std::clamp(size, 256u, 8192u);
        uint32_t rounded = 256u;
        while (rounded * 2u <= clamped) {
            rounded *= 2u;
        }
        if (rounded != size) {
            utility::log("runtime: shadow map size {} -> {} (clamped to 256..8192 and rounded to a power of two)", size, rounded);
        }
        this->shadow_map_size = rounded;
        ++this->shadow_content_version; // a new map size reallocates the images: every slot must render again
    }

    runtime::render_features runtime::active_features() const noexcept {
        // The single derivation of "what runs this frame" (see the struct's docs): pass recording,
        // the overlay's visibility predicates and the log all read THIS, so they cannot drift apart.
        render_features f;
        f.unlit = this->unlit_active;
        f.gbuffer_debug = this->gbuffer_debug && this->gbuffer_pipeline.has_value() && this->gbuffer_debug_pipeline.has_value();
        f.ssgi = this->ssgi_active();
        // The probe cache is a pass of its own, so its activity is a feature of its own: the tracer asks
        // whether the cache is READY (probe_ready below), and the runner asks whether the pass RUNS.
        f.ssgi_probes = this->gi_probe_active();
        // The G-buffer pass and its lighting stage are the engine's only scene path, so there is no
        // flag for them: taa/ssao below ask this instead, and the debug view stands in for the
        // lighting stage rather than running alongside it (the two write the HDR target differently).
        bool const shaded_scene = !f.gbuffer_debug && this->deferred_pipeline.has_value() && this->gbuffer_pipeline.has_value();
        // The shadow map is only read by the shading stages. The flat render mode samples nothing
        // (unlit.frag has no lighting include; the lighting stage returns the albedo before any
        // shading), so recording the pass would be pure waste - it measured 0.22 ms of a 0.5 ms frame.
        f.shadow = this->shadow_enabled && this->shadows_enabled && this->shadow_pipeline.has_value() && !f.unlit;
        // Same argument for the cluster pass: flat shading reads no light list, and with no active
        // punctual light there is nothing to sort in the first place.
        f.clustered = this->clustered_lights && this->cluster_pipeline.has_value() && this->light_state.light_count.x > 0.5f && !f.unlit;
        f.taa = this->taa_on && this->taa_resolve.pipeline_ready() && shaded_scene;
        f.ssao = this->ssao_enabled && shaded_scene; // shader-side gate: no pass of its own to skip
        f.bloom = this->bloom_intensity > 0.0f && this->post_hdr_pipeline.has_value() && !f.gbuffer_debug;
        f.fxaa = this->fxaa_on && this->post_fxaa_pipeline.has_value();
        // The transparent pass composites over the shaded frame, so it needs that frame to exist -
        // and it is skipped in the debug view, which shows the G-buffer rather than a frame.
        f.transparent = !f.gbuffer_debug && !this->frame_transparent.empty();
        return f;
    }

    bool runtime::feature_active(std::string_view const name) const noexcept {
        render_features const f = this->active_features();
        if (name == "gbuffer-debug") {
            return f.gbuffer_debug;
        }
        if (name == "ssgi") {
            return f.ssgi;
        }
        if (name == "ssgi_probes") {
            return f.ssgi_probes;
        }
        if (name == "taa") {
            return f.taa;
        }
        if (name == "fxaa") {
            return f.fxaa;
        }
        if (name == "shadow") {
            return f.shadow;
        }
        if (name == "clustered") {
            return f.clustered;
        }
        if (name == "ssao") {
            return f.ssao;
        }
        if (name == "bloom") {
            return f.bloom;
        }
        if (name == "unlit") {
            return f.unlit;
        }
        return false;
    }

    void runtime::warn_missing_feature(std::string_view const key, std::string const& message) {
        // Only meaningful once the startup is complete: the app applies the config to the runtime
        // BEFORE the pipelines exist (main sets the toggles, chores then creates the pipelines), so
        // warning there would claim "TAA has no effect" one line above "TAA pipeline
        // created". The scene set is created on the first recorded frame, i.e. once every optional
        // pipeline exists.
        if (!this->scene_sets.created()) {
            return;
        }
        // At most once per feature per session: main() mirrors the overlay's state into the runtime
        // every frame, so an unconditional log here would print once per FRAME - which is how a
        // diagnostics feature turns into log spam.
        for (std::string const& seen : this->warned_features) {
            if (seen == key) {
                return;
            }
        }
        this->warned_features.emplace_back(key);
        utility::log("gui: {}", message);
    }

    void runtime::set_gbuffer_debug(bool const enabled) noexcept {
        this->gbuffer_debug = enabled;
        if (enabled && (!this->gbuffer_pipeline.has_value() || !this->gbuffer_debug_pipeline.has_value())) {
            this->warn_missing_feature("gbuffer-debug", "the G-buffer debug view has no effect: its pipelines were not created (see the startup log)");
        }
    }

    bool runtime::feature_available(std::string_view const name) const noexcept {
        // The single source of truth for "can this feature run at all this session": the overlay asks
        // it to decide what to offer, log_feature_status() prints it, and both therefore agree.
        if (name == "gbuffer-debug") {
            return this->gbuffer_pipeline.has_value() && this->gbuffer_debug_pipeline.has_value();
        }
        if (name == "taa") {
            return this->taa_resolve.pipeline_ready();
        }
        if (name == "fxaa") {
            return this->post_fxaa_pipeline.has_value();
        }
        if (name == "shadow") {
            return this->shadow_pipeline.has_value();
        }
        if (name == "clustered") {
            return this->cluster_pipeline.has_value();
        }
        if (name == "ssgi") {
            return this->ssgi_pipeline.has_value();
        }
        return false;
    }

    void runtime::log_feature_status() const {
        // One line naming every optional feature, so "why does this switch do nothing?" is answerable
        // from the log alone. `on` means the pipeline exists and the feature CAN run; whether it is
        // currently switched on is the overlay's and the config's business.
        utility::log("features: gbuffer-debug={} ssgi={} taa={} fxaa={} shadow={} clustered-lights={}",
                     this->feature_available("gbuffer-debug") ? "on" : "UNAVAILABLE",
                     this->feature_available("ssgi") ? "on" : "UNAVAILABLE",
                     this->feature_available("taa") ? "on" : "UNAVAILABLE",
                     this->feature_available("fxaa") ? "on" : "UNAVAILABLE",
                     this->feature_available("shadow") ? "on" : "UNAVAILABLE",
                     this->feature_available("clustered") ? "on" : "UNAVAILABLE");
        if (!this->gbuffer_pipeline.has_value() || !this->deferred_pipeline.has_value()) {
            utility::log("features: the G-buffer pass or its lighting stage was not created, so NO SCENE IS DRAWN this session (see the startup log's 'deferred lighting disabled' line)");
        }
    }

    void runtime::set_ssao(bool const enabled, float const radius, float const intensity, uint32_t const samples) noexcept {
        // CPU-side only (the same rule as set_brdf_model / set_clustered_lights): the values are
        // pushed with the deferred lighting stage each frame, so they are safe to change mid-run.
        this->ssao_enabled = enabled;
        this->ssao_radius = std::max(radius, 0.0f);
        this->ssao_intensity = std::clamp(intensity, 0.0f, 1.0f);
        this->ssao_samples = std::clamp(samples, 0u, 16u); // MAX_SSAO_SAMPLES in deferred.frag
        if (enabled && !this->deferred_lit_active()) {
            this->warn_missing_feature("ssao", "screen-space AO has no effect: the G-buffer pass or its lighting stage was not created (see the startup log)");
        }
    }

    void runtime::set_max_fps(double const fps) noexcept {
        double const clamped = fps > 0.0 ? fps : 0.0;
        // The demo calls this every frame to mirror the GUI, so an unchanged value must be a no-op:
        // re-arming the deadline here would push it back to the epoch on every frame and the limiter
        // would never wait for anything (which is exactly what the first version did).
        if (clamped == this->max_fps) {
            return;
        }
        this->max_fps = clamped;
        this->next_frame_deadline = {}; // re-arm: the first frame after a change never waits
        if (this->max_fps > 0.0) {
            utility::log("runtime: frame rate limited to {:.1f} fps", this->max_fps);
        } else {
            utility::log("runtime: frame rate limit removed (uncapped)");
        }
    }

    std::string runtime::cpu_timing_summary() const {
        return this->cpu_timings.summary();
    }

    std::array<double, static_cast<std::size_t>(vulkan::profiling::cpu_phase::count)> runtime::cpu_timing_means() const noexcept {
        return this->cpu_timings.means();
    }

    void runtime::set_unlit(bool const unlit) noexcept {
        // CPU-side only, like the other render-mode flags: the value is pushed with the deferred
        // lighting stage each frame (and the forward path does not need it at all - there the render
        // mode IS the default pipeline).
        this->unlit_active = unlit;
    }

    void runtime::set_clustered_lights(bool const enabled) noexcept {
        // CPU-side only, like set_brdf_model: the flag rides light_state's cluster_grid.w lane and
        // pace_and_acquire() copies light_state into the paced slot's buffer, so the next frame's
        // cluster dispatch and shading both see it (no in-flight buffer is touched).
        this->clustered_lights = enabled;
        if (enabled && !this->cluster_pipeline.has_value()) {
            this->warn_missing_feature("clustered", "clustered light culling has no effect: the cluster compute pipeline was not created, so the shading stage loops EVERY active light instead (see the startup log)");
        }
    }

    void runtime::record_cluster_pass(VkCommandBuffer const command_buffer) {
        core& vk = this->vulkan_core;
        // Feature registry: skipped when no shading stage reads a light list (flat render mode) or
        // when no punctual light is active - there would be nothing to sort, and the shading stage
        // then falls back to looping zero lights.
        if (!this->active_features().clustered || !this->scene_sets.created()) {
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
        VkDescriptorSet const scene_set_handle = this->scene_sets.set(static_cast<uint32_t>(frame_slot));
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
        ++this->shadow_content_version; // a refit changes the maps: every slot must render again
        this->shadow_frustum_valid = true;
        if (!(this->current_aspect > 0.0f) || this->current_ubo.proj[2][2] == 0.0f) {
            // Degenerate camera (the very first frames, before the swapchain has an extent): the
            // frustum corners would come out of an inverse of a singular matrix and the fit would be
            // garbage (measured: a light-space z span of 1631 for two unrelated scenes). Keep the
            // enable_shadows() default this frame and try again next frame.
            this->shadow_frustum_valid = false;
            return;
        }

        glm::vec3 const light_dir = glm::normalize(glm::vec3(this->light_state.light_dir));
        // ---- gather: every caster's light-space AABB, computed ONCE for all cascades ----
        // Casters outside the view still cast into it (a wall behind the camera): fitting only the
        // frustum corners clipped them and sunlight leaked through. Every scene leaf's world AABB is
        // collected here and handed to the fit, which decides per cascade which of them can reach it.
        // Geometry whose shadow cannot reach the view (e.g. the floor slab behind the camera) is left
        // out there.
        this->shadow_caster_world_boxes.clear();
        bool unbounded_caster = false;
        if (this->bound_scene != nullptr) {
            this->shadow_caster_scratch.clear();
            for (scene_tree::scene_node const& root : this->bound_scene->roots) {
                collect_leaf_primitives(root, this->shadow_caster_scratch);
            }
            this->shadow_caster_world_boxes.reserve(this->shadow_caster_scratch.size());
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
                    // a caster whose geometry we genuinely cannot bound: the fit falls back to the
                    // scene sphere below, which is sound but costs resolution
                    unbounded_caster = true;
                    continue;
                }
                this->shadow_caster_world_boxes.emplace_back(wmin, wmax);
            }
        }
        this->shadow_caster_boxes = shadow_fit::fit_casters(this->shadow_caster_world_boxes, light_dir, unbounded_caster);

        // ---- fit: the pure part (vulkan.shadow_fit) ----
        shadow_fit::fit_params params = {};
        params.proj = this->current_proj_unjittered;
        params.view = this->current_ubo.view;
        params.camera_pos = glm::vec3(this->current_ubo.camera_pos);
        params.light_dir = this->light_state.light_dir;
        params.scene_center = this->shadow_scene_center;
        params.scene_radius = this->scene_radius;
        params.caster_extent = this->shadow_caster_extent;
        params.cascades = this->shadow_cascades;
        params.map_size = this->shadow_map_size;
        params.unbounded_caster = unbounded_caster;
        params.log_summary = !this->shadow_cascade_logged;
        params.caster_boxes = this->shadow_caster_boxes;
        shadow_fit::fit_result const fitted = shadow_fit::fit(params);
        if (!fitted.valid) {
            this->shadow_frustum_valid = false;
            return;
        }
        this->shadow_cascade_logged = true;

        // the fit produces exactly the lanes the shader reads; the rest of the UBO is untouched
        this->light_state.light_view_proj = fitted.light_view_proj;
        this->light_state.cascade_splits = glm::vec4(fitted.cascade_splits[0], fitted.cascade_splits[1], fitted.cascade_splits[2], fitted.cascade_splits[3]);
        this->light_state.cascade_texel_world = glm::vec4(fitted.cascade_texel_world[0], fitted.cascade_texel_world[1], fitted.cascade_texel_world[2], fitted.cascade_texel_world[3]);
        this->light_state.cascade_count = static_cast<float>(fitted.cascade_count);
        this->light_state.light_dir = glm::vec4(fitted.light_dir, 1.0f / static_cast<float>(this->shadow_map_size));
    }
    void runtime::set_shadow_cascades(uint32_t const cascades) noexcept {
        uint32_t const clamped = std::clamp(cascades, 1u, vulkan::max_shadow_cascades);
        if (clamped == this->shadow_cascades) {
            return;
        }
        this->shadow_cascades = clamped;
        ++this->shadow_content_version; // a different cascade count refits the splits
        // Growing past the layers we own has to rebuild the images; the descriptor set is rewritten
        // afterwards because binding 8 holds their array view. Shrinking keeps the layers (no second
        // rebuild when the user cycles the combo, and the spare ones simply go unused).
        if (!this->shadow_images.empty() && clamped > this->shadow_allocated_layers) {
            vkDeviceWaitIdle(this->vulkan_core.device);
            this->shadow_images.clear();
            this->shadow_array_views.clear();
            this->shadow_layer_views.clear();
            this->ensure_shadow_resources();
            this->write_light_and_shadow_bindings();
        }
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
        this->light_state = make_directional_light_ubo(scene_center, scene_radius, static_cast<float>(this->shadow_map_size));
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

    void runtime::set_rt_shadows(bool const enabled) noexcept {
        this->rt_shadows = enabled;
        // The lighting stage reads this from the light UBO, so the flag has to be settled before the
        // next frame's paced write - which is why it is set here rather than recomputed per frame. The
        // device check and the pipeline check are folded in: a request that cannot be honoured leaves the
        // cascaded shadow maps running, and the shader never even looks at the visibility image.
        this->light_state.rt_shadows = (enabled && this->rt_shadow_pipeline.has_value() && this->vulkan_core.ray_query_available) ? 1.0f : 0.0f;
    }

    void runtime::set_rt_mask_bake(bool const enabled) noexcept {
        // Read once, when the structures are built (see record_acceleration_structures): the bake is startup
        // work and the structures are built once, so this can only be settled before the first traced frame.
        this->rt_mask_bake = enabled;
    }

    void runtime::set_rt_skin_bake(bool const enabled) noexcept {
        // Unlike the mask bake this is read EVERY frame (the pass runs per frame), so it can be toggled at
        // any time: turning it off leaves the structures holding the last pose the pass wrote, which is the
        // A/B's whole point - the traced shadow either follows the animation or it does not.
        this->rt_skin_bake = enabled;
    }

    bool runtime::rt_structures_wanted() const noexcept {
        // `ssgi_on` rather than `ssgi_on && ssgi_ray_tracing`: the GI tracer is ONE shader that declares
        // the top level structure as a binding whether or not its traced branch runs, and a shader that
        // statically uses a binding needs it written - so a GI frame has to have structures even when it
        // marches. The cost of that is one build (Sponza: 17.8 MiB, ~2 ms) plus a per-frame rebuild
        // (~0.1 ms) for a GI scene that never traces; the alternative is a second shader variant or the
        // nullDescriptor feature (VK_EXT_robustness2), and both are larger changes than this one.
        return this->vulkan_core.ray_query_available && (this->rt_shadows || this->ssgi_on);
    }

    bool runtime::rt_shadows_active() const noexcept {
        return this->rt_shadows && this->vulkan_core.ray_query_available;
    }

    void runtime::record_acceleration_structures(VkCommandBuffer const command_buffer) {
        core& vk = this->vulkan_core;
        if (!this->rt_structures_wanted() || this->rt_structures_attempted) {
            return; // off, unsupported, or already built (see rt_structures_attempted)
        }
        this->rt_structures_attempted = true;

        auto const start = std::chrono::steady_clock::now();
        this->rt_bottom_levels.emplace(vk);
        // The top level structure is per FRAME SLOT (see its class docs): with frames in flight one
        // buffer would be rewritten by the frame being recorded while the previous one still reads it.
        this->rt_top_levels.emplace(vk, vulkan::core::MAX_FRAMES_IN_FLIGHT);
        auto& structures = *this->rt_bottom_levels;

        // One structure per SHADOW CASTER, which is the set the shadow pass itself draws (and the
        // reason it is the right set: a caster can sit off screen and still throw a shadow into the
        // view, so the visible set would be wrong). The geometry is the renderer's own: the build
        // reads the vertex and index buffers through their DEVICE ADDRESSES, so nothing is copied and
        // the structures follow whatever those buffers hold.
        uint32_t skipped_no_address = 0;
        uint32_t skipped_no_stride = 0;
        // The mask bake: how many casters had an alphaMode MASK baked into their geometry, and how many
        // could not be (an allocation failure falls back to the documented solid behaviour rather than
        // failing the whole build).
        uint32_t mask_baked = 0;
        uint32_t skipped_mask_buffers = 0;
        bool mask_bakes_recorded = false;
        // ... and the same two counters for the skinned casters (see the SKINNED branch below).
        uint32_t skinned_baked = 0;
        uint32_t skipped_skin_buffers = 0;
        for (primitive const* caster : this->shadow_casters) {
            if (caster == nullptr) {
                continue;
            }
            VkBufferDeviceAddressInfo vertex_address_info = {};
            vertex_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            VkBufferDeviceAddressInfo index_address_info = vertex_address_info;

            auto const* const vertex_detail = caster->vertex_detail;
            auto const* const index_detail = caster->index_detail;
            if (vertex_detail == nullptr || index_detail == nullptr || vertex_detail->buffer == VK_NULL_HANDLE || index_detail->buffer == VK_NULL_HANDLE) {
                ++skipped_no_address;
                continue;
            }
            // A buffer only has a device address when it was created with SHADER_DEVICE_ADDRESS_BIT,
            // which the primitive uploads add when this device has the extensions - so this is a check
            // on a device that has ray queries but whose buffers were uploaded before the flag... which
            // cannot happen: the buffers are uploaded with the bits whenever the device supports them,
            // regardless of the config. Kept as a guard because the alternative is a validation error
            // per frame instead of one line in the log.
            if (caster->vertex_stride == 0) {
                ++skipped_no_stride;
                continue;
            }
            vertex_address_info.buffer = vertex_detail->buffer;
            index_address_info.buffer = index_detail->buffer;
            VkDeviceAddress const source_vertex_address = vkGetBufferDeviceAddress(vk.device, &vertex_address_info);
            VkDeviceAddress const source_index_address = vkGetBufferDeviceAddress(vk.device, &index_address_info);

            // alphaMode MASK: bake the material's holes into an EXPANDED copy of this caster's vertices and
            // build the structure from that. An inline ray query has no any-hit stage, so a traversal cannot
            // run the material's discard - this pass is where the mask is applied instead, and it is startup
            // work because the structures are built once and a MASK material is a property of the file (see
            // shaders/mask_bake.comp for the rule and for what the mechanism cannot represent).
            VkDeviceAddress mask_address = 0;
            uint32_t mask_stride = 0;
            if (this->rt_mask_bake && this->mask_bake_pipeline.has_value() && this->material_mapped != nullptr) {
                // material_record::flags bit 4 is alphaMode MASK (see vulkan/primitive.cppm; the bits are
                // literals in register_material, so they are literals here too).
                uint32_t const material_index = caster->push.material_index.value;
                material_record const* const material =
                    material_index < this->material_count
                        ? reinterpret_cast<material_record const*>(static_cast<unsigned char const*>(this->material_mapped) + static_cast<std::size_t>(material_index) * sizeof(material_record))
                        : nullptr;
                if (material != nullptr && (material->flags & 16u) != 0u && caster->index_count >= 3u) {
                    // Three vertices per triangle, 32 bytes each: position(3) + normal(3) + uv(2), which is
                    // what the hit shading reads (offsets 0, 3 and 6). GPU-only and never mapped - the bake
                    // fills it and the build reads it.
                    constexpr uint32_t mask_vertex_stride = 32u;
                    uint64_t const expanded_bytes = static_cast<uint64_t>(caster->index_count) * mask_vertex_stride;
                    vk_buffer expanded = vk.vma.create_buffer(nullptr, expanded_bytes, buffer_type::storage_gpu_only, acceleration_structure::build_input_usage);
                    auto const* const expanded_detail = expanded.valid() ? vk.vma.get_buffer_detail(expanded.handle()) : nullptr;
                    if (expanded_detail != nullptr) {
                        VkBufferDeviceAddressInfo const expanded_info = {
                            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = expanded_detail->buffer};
                        mask_address = vkGetBufferDeviceAddress(vk.device, &expanded_info);
                        mask_stride = mask_vertex_stride;

                        mask_bake_push_constants bake = {};
                        auto const halves = [](VkDeviceAddress const address) {
                            return glm::uvec2(static_cast<uint32_t>(address & 0xFFFFFFFFu), static_cast<uint32_t>(address >> 32u));
                        };
                        bake.source_vertices = halves(source_vertex_address);
                        bake.source_indices = halves(source_index_address);
                        bake.destination = halves(mask_address);
                        bake.source_stride = caster->vertex_stride;
                        bake.destination_stride = mask_vertex_stride;
                        bake.index_type = static_cast<uint32_t>(caster->index_type);
                        bake.triangle_count = caster->index_count / 3u;
                        bake.material_index = material_index;
                        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->mask_bake_pipeline->get_pipeline());
                        // The bake's own set - NOT the frame's scene set, which this same command buffer
                        // will have updated by the end of the frame (binding 16). See the member's comment.
                        VkDescriptorSet const bake_set = this->mask_bake_set.get();
                        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->mask_bake_pipeline_layout, 0, 1, &bake_set, 0, nullptr);
                        vkCmdPushConstants(command_buffer, this->mask_bake_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bake), &bake);
                        constexpr uint32_t mask_bake_group = 64; // shaders/mask_bake.comp's local_size_x
                        vkCmdDispatch(command_buffer, (bake.triangle_count + mask_bake_group - 1u) / mask_bake_group, 1, 1);
                        mask_bakes_recorded = true;
                        ++mask_baked;
                        // The buffer outlives this loop: the build below reads it, and a hit's shading reads
                        // its vertices through the instance table for as long as the structures live.
                        this->rt_mask_buffers.push_back(std::move(expanded));
                    } else {
                        ++skipped_mask_buffers;
                    }
                }
            }

            // SKINNED: the pass (below, once every caster is known) writes this caster's deformed vertices
            // into a buffer of its own, the structure is built from that buffer, and every frame after it is
            // REFITTED - which is legal because the vertex order, the index buffer and the triangle count are
            // all the primitive's own: only the bytes change. `skin_base != 0` is the test for "skinned",
            // because index 0 is the identity block every unskinned draw uses (see set_skin_matrices). The
            // stride test is the shader's precondition, not a heuristic: shaders/compute_skin.comp reads the
            // joints at byte 32 and the weights at byte 48 of the engine's 64-byte interleaved vertex, so a
            // caster whose vertices are packed differently is REFUSED (it keeps its bind pose and is counted
            // in the log) rather than skinned with the wrong words.
            constexpr uint32_t skin_source_stride_expected = 64u;
            VkDeviceAddress skin_address = 0;
            uint32_t skin_stride = 0;
            uint32_t skin_source_stride = 0;
            uint32_t skin_vertex_count = 0;
            uint32_t skin_base = 0;
            if (mask_address == 0 && this->rt_skin_bake && this->compute_skin_pipeline.has_value() && caster->push.skin_base != 0 && caster->vertex_count != 0 &&
                caster->vertex_stride == skin_source_stride_expected) {
                constexpr uint32_t skin_vertex_stride = 32u; // position, normal, uv - what hit shading reads
                uint64_t const skinned_bytes = static_cast<uint64_t>(caster->vertex_count) * skin_vertex_stride;
                vk_buffer skinned_vertices = vk.vma.create_buffer(nullptr, skinned_bytes, buffer_type::storage_gpu_only, acceleration_structure::build_input_usage);
                auto const* const skinned_detail = skinned_vertices.valid() ? vk.vma.get_buffer_detail(skinned_vertices.handle()) : nullptr;
                if (skinned_detail != nullptr) {
                    VkBufferDeviceAddressInfo const skinned_info = {
                        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = skinned_detail->buffer};
                    skin_address = vkGetBufferDeviceAddress(vk.device, &skinned_info);
                    skin_stride = skin_vertex_stride;
                    skin_source_stride = caster->vertex_stride;
                    skin_vertex_count = caster->vertex_count;
                    skin_base = caster->push.skin_base;
                    this->rt_skin_buffers.push_back(std::move(skinned_vertices));
                    ++skinned_baked;
                } else {
                    ++skipped_skin_buffers;
                }
            }

            acceleration_structure::geometry_source const source =
                mask_address != 0
                    ? acceleration_structure::geometry_source{.vertex_address = mask_address,
                                                              .vertex_stride = mask_stride,
                                                              .vertex_count = caster->index_count,
                                                              .index_address = 0,
                                                              .index_type = caster->index_type,
                                                              .index_count = caster->index_count}
                : skin_address != 0
                    ? acceleration_structure::geometry_source{.vertex_address = skin_address,
                                                              .vertex_stride = skin_stride,
                                                              .vertex_count = caster->vertex_count,
                                                              .index_address = source_index_address,
                                                              .index_type = caster->index_type,
                                                              .index_count = caster->index_count}
                    : acceleration_structure::geometry_source{.vertex_address = source_vertex_address,
                                                              .vertex_stride = caster->vertex_stride,
                                                              .vertex_count = caster->vertex_count,
                                                              .index_address = source_index_address,
                                                              .index_type = caster->index_type,
                                                              .index_count = caster->index_count};
            // A skinned structure is built ALLOW_UPDATE so the per-frame refit is legal; everything else is
            // built once and never touched again.
            auto const added = structures.add(source, skin_address != 0);
            if (!added) {
                utility::log("ray-traced shadows disabled: {}", added.error());
                this->rt_bottom_levels.reset();
                this->rt_top_levels.reset();
                // The expansion buffers and the caster mapping go with the structures they belong to: a
                // stale mapping would have the instance list read geometry no structure was built from.
                this->rt_mask_buffers.clear();
                this->rt_skin_buffers.clear();
                this->rt_skin_levels.clear();
                this->rt_caster_levels.clear();
                return;
            }
            // Remember which caster got which index: the per-frame instance list walks THIS, so a
            // caster that was skipped above is skipped there too and the two walks cannot disagree. The
            // mask and skin addresses ride along, because that list is what a hit's shading reads the
            // geometry through - a baked or skinned caster must be read from the copy it was built from.
            this->rt_caster_levels.emplace_back(rt_caster_level{.caster = caster,
                                                                .blas_index = added.value(),
                                                                .mask_stride = mask_stride,
                                                                .mask_vertex_address = mask_address,
                                                                .skin_source_address = source_vertex_address,
                                                                .skin_destination_address = skin_address,
                                                                .skin_source_stride = skin_source_stride,
                                                                .skin_destination_stride = skin_stride,
                                                                .skin_vertex_count = skin_vertex_count,
                                                                .skin_base = skin_base});
        }

        // The skinned casters' first skinning pass, recorded here because the BUILD below has to read skinned
        // vertices - and every frame after this one re-skins and REFITS in record_top_level_structure. The
        // refit is not recorded here: this is the frame the structures are created, and a refit against a
        // structure that does not exist yet is illegal.
        if (skinned_baked != 0) {
            for (auto const& built : this->rt_caster_levels) {
                if (built.skin_destination_address != 0) {
                    this->rt_skin_levels.push_back(built.blas_index);
                }
            }
            this->record_compute_skin_pass(command_buffer);
        }

        // Every bake wrote a buffer the build below reads: one barrier covers them all, because every
        // dispatch is recorded before the first build (add() only sizes and allocates; record_build()
        // records). A compute WRITE is not visible to an acceleration structure build without it, and the
        // symptom would be a structure built from an empty buffer - i.e. geometry that stops casting.
        if (mask_bakes_recorded) {
            VkMemoryBarrier2 bake_order = {};
            bake_order.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            bake_order.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bake_order.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            bake_order.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            bake_order.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            VkDependencyInfo const bake_dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                      .pNext = nullptr,
                                                      .dependencyFlags = 0,
                                                      .memoryBarrierCount = 1,
                                                      .pMemoryBarriers = &bake_order,
                                                      .bufferMemoryBarrierCount = 0,
                                                      .pBufferMemoryBarriers = nullptr,
                                                      .imageMemoryBarrierCount = 0,
                                                      .pImageMemoryBarriers = nullptr};
            vkCmdPipelineBarrier2(command_buffer, &bake_dependency);
        }

        if (auto const built = structures.record_build(command_buffer); !built) {
            utility::log("ray-traced shadows disabled: {}", built.error());
            this->rt_bottom_levels.reset();
            this->rt_top_levels.reset();
            return;
        }

        acceleration_structure::build_stats const& stats = structures.last_stats();
        double const host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        utility::log("ray-traced shadows: built {} bottom level structures ({} triangles, {:.1f} MiB + {:.1f} MiB scratch) in {:.1f} ms",
                     stats.geometry_count,
                     stats.triangle_count,
                     static_cast<double>(stats.structure_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(stats.scratch_bytes) / (1024.0 * 1024.0),
                     host_ms);
        if (skipped_no_stride != 0) {
            utility::log("  {} casters skipped (no vertex stride recorded - a primitive not created by make_primitive)", skipped_no_stride);
        }
        if (skipped_no_address != 0) {
            utility::log("  {} casters skipped (no vertex/index buffer)", skipped_no_address);
        }
        if (mask_baked != 0 || skipped_mask_buffers != 0) {
            // The measurement this feature is read with: how much geometry the mask actually removed is a
            // property of the asset (a two-quad MASK plane whose pattern is in the middle keeps every
            // triangle; a vase of flowers loses 40% of them - see docs/gi_hit_shading.md).
            utility::log("ray-traced shadows: {} MASK casters baked into their structures ({} could not be - those stay solid to a ray)", mask_baked, skipped_mask_buffers);
        }
        if (skinned_baked != 0 || skipped_skin_buffers != 0) {
            // The skinned casters are re-skinned and REFITTED every frame (see record_top_level_structure),
            // so this count is also the number of structures a frame's refit touches.
            utility::log("ray-traced shadows: {} skinned casters re-skinned and REFITTED from their deformed vertices every frame ({} could not be - those keep their bind pose)", skinned_baked, skipped_skin_buffers);
        }
    }

    void runtime::record_top_level_structure(VkCommandBuffer const command_buffer) {
        core& vk = this->vulkan_core;
        if (!this->rt_structures_wanted() || !this->rt_bottom_levels.has_value() || !this->rt_top_levels.has_value()) {
            return;
        }
        uint32_t const frame_slot = static_cast<uint32_t>(vk.current_frame);
        auto& levels = *this->rt_bottom_levels;
        auto& top = *this->rt_top_levels;

        // The skinned casters are deformed and their structures REFITTED here, before the instance list is
        // walked (the addresses do not change, so the order does not matter to correctness - but the refit
        // has to be recorded before this frame writes the scene set's binding 16, the ordering the mask
        // bake's own-set comment explains). The pass itself returns false when there is nothing skinned.
        if (this->record_compute_skin_pass(command_buffer)) {
            if (auto const updated = this->rt_bottom_levels->record_update(command_buffer, this->rt_skin_levels); !updated) {
                // Once, and off: a failure here would otherwise log every frame, and a refit is not
                // something to keep attempting against structures the device refused.
                utility::log("runtime: skinned shadow refit disabled: {}", updated.error());
                this->rt_skin_bake = false;
            }
        }

        if (auto const begun = top.begin(frame_slot); !begun) {
            utility::log("runtime: {}", begun.error());
            return;
        }
        // The instance list is the caster set the shadow pass draws, with the world matrix the raster
        // passes use for each caster - the same matrix shadow_geometry_signature() hashes, which is why
        // an animated or moved caster is reflected here for free.
        for (auto const& built : this->rt_caster_levels) {
            primitive const* const caster = built.caster;
            // The addresses a hit-shading path reads the hit triangle from: the same buffers, and the
            // same vkGetBufferDeviceAddress calls, the bottom level build above already used for this
            // caster - so the triangle a shader fetches with them IS the triangle the ray hit. They are
            // the buffers' base addresses (the build applies no offset), which is also what makes them
            // legal as a buffer reference: a buffer's address is aligned, an offset into one need not be.
            //
            // A baked or skinned caster is read from the copy its structure was built from: the mask bake's
            // expanded, non-indexed one (zero index address = a flat vertex list), or the skinned one, which
            // keeps the primitive's own index buffer because its vertex ORDER is unchanged.
            VkDeviceAddress vertex_address = built.mask_vertex_address != 0 ? built.mask_vertex_address : built.skin_destination_address;
            VkDeviceAddress index_address = 0;
            uint32_t vertex_stride = built.mask_vertex_address != 0 ? built.mask_stride : built.skin_destination_stride;
            if (vertex_address == 0) {
                VkBufferDeviceAddressInfo const vertex_address_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = caster->vertex_detail->buffer};
                VkBufferDeviceAddressInfo const index_address_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = caster->index_detail->buffer};
                vertex_address = vkGetBufferDeviceAddress(vk.device, &vertex_address_info);
                index_address = vkGetBufferDeviceAddress(vk.device, &index_address_info);
                vertex_stride = caster->vertex_stride;
            } else if (built.skin_destination_address != 0) {
                VkBufferDeviceAddressInfo const index_address_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = caster->index_detail->buffer};
                index_address = vkGetBufferDeviceAddress(vk.device, &index_address_info);
            }
            acceleration_structure::instance_source const instance = {
                .transform = caster->push.model,
                .blas_index = built.blas_index,
                .record = {.vertex_address = vertex_address,
                           .index_address = index_address,
                           .model = caster->push.model,
                           .vertex_stride = vertex_stride,
                           .index_type = static_cast<uint32_t>(caster->index_type),
                           .material_index = caster->push.material_index.value,
                           .primitive_index = built.blas_index},
            };
            if (auto const added = top.add(levels, instance); !added) {
                utility::log("runtime: {}", added.error());
                return;
            }
        }

        // The top level reads the BOTTOM levels, and on the frame that creates them the two builds are
        // in the same command buffer with nothing between them: without this barrier the driver is free
        // to run the second build's reads against writes the first one has not published. It costs a
        // no-op on every later frame (nothing wrote a bottom level in this buffer), which is cheaper
        // than a flag that would have to track "which frame built them".
        VkMemoryBarrier2 build_order = {};
        build_order.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        build_order.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        build_order.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        build_order.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        build_order.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo const build_order_info = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                   .pNext = nullptr,
                                                   .dependencyFlags = 0,
                                                   .memoryBarrierCount = 1,
                                                   .pMemoryBarriers = &build_order,
                                                   .bufferMemoryBarrierCount = 0,
                                                   .pBufferMemoryBarriers = nullptr,
                                                   .imageMemoryBarrierCount = 0,
                                                   .pImageMemoryBarriers = nullptr};
        vkCmdPipelineBarrier2(command_buffer, &build_order_info);

        if (auto const built = top.record_build(command_buffer); !built) {
            utility::log("runtime: {}", built.error());
            return;
        }

        // Point this slot's binding 16 at the structure that was just built. The scene sets were written
        // before any structure existed (a null acceleration-structure descriptor is not legal without
        // nullDescriptor), so this is where the binding first becomes valid - and the pass that reads it
        // is gated on the same handle.
        if (this->scene_sets.created()) {
            this->write_rt_structure_binding(this->scene_sets.set(frame_slot), top.handle(frame_slot));
        }
        if (!this->rt_top_level_logged) {
            this->rt_top_level_logged = true;
            // The class measured the host cost of the build itself (see build_stats); reporting that
            // rather than a second timer around it keeps one definition of "what the build costs".
            utility::log("ray-traced shadows: {} instances in the top level structure, one instance table entry each ({:.3f} ms host per frame)",
                         top.instance_count(frame_slot),
                         top.last_stats().build_ms);
        }
    }

    void runtime::set_shadow_enabled(bool const enabled) {
        if (this->shadow_enabled == enabled) {
            return;
        }
        this->shadow_enabled = enabled;
        ++this->shadow_content_version;
        // Only the CPU-side flag changes here: the frame loop copies light_state into the paced
        // slot's OWN light buffer (pace_and_acquire), so this call is safe at ANY time (GUI
        // callbacks included) - it never touches memory a frame in flight may be reading. The
        // light UBO's shadow_enabled flag drives pbr.frag: when shadows are off the shader
        // skips calc_shadow entirely (fully lit), so no shadow-map clearing is needed - this
        // avoids the per-frame-slot double-buffer race that clearing once could not fix.
        this->light_state.shadow_enabled = (enabled && this->shadows_enabled) ? 1.0f : 0.0f;
        if (enabled && !this->shadows_enabled) {
            this->warn_missing_feature("shadow-on", "the shadow pass has no effect: enable_shadows() did not succeed (the startup log says why)");
        }
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
            this->warn_missing_feature("fxaa", "FXAA has no effect: the fxaa pipeline was not created (is fxaa.frag.spv present?)");
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

        // The pixels come from the staging buffer record_screenshot_copy() filled while the frame was
        // being recorded (see the class docs): by the time the caller asks, the frame has been
        // submitted, so one wait for the GPU is all that is left - and the staging's owner is
        // vulkan.readback, which is also what sized the buffer and handed out the mapping.
        if (this->screenshot_staging_mapped == nullptr || this->screenshot_readback_extent.width == 0) {
            return std::unexpected(std::string("screenshot: no captured frame (the read-back copy was never recorded)"));
        }
        VkExtent2D const extent = this->screenshot_readback_extent;
        VkDeviceSize const buffer_size = static_cast<VkDeviceSize>(extent.width) * static_cast<VkDeviceSize>(extent.height) * 4u;
        vk.wait_idle();

        frame_image result = {};
        result.width = extent.width;
        result.height = extent.height;
        result.rgba.resize(static_cast<std::size_t>(buffer_size));
        auto const* source = static_cast<unsigned char const*>(this->screenshot_staging_mapped);
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
        // The staging buffer and its mapping are vulkan.readback's; only the IMAGE side is this
        // function's business (the layout transitions, the region, the format the caller will unpack).
        auto const staged = this->readback_staging.stage_for_copy(static_cast<VkDeviceSize>(extent.width) * static_cast<VkDeviceSize>(extent.height) * 4u);
        if (!staged) {
            utility::log("screenshot: read-back staging buffer unavailable");
            return;
        }
        this->screenshot_staging_mapped = staged->mapped;
        this->screenshot_readback_extent = extent;

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
        vkCmdCopyImageToBuffer(command_buffer, vk.swap_chain_images[this->current_image_index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staged->buffer, 1, &region);

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
        // The vertex and index buffers carry the acceleration-structure build-input usage when the
        // device has ray tracing, so a later build can read them through their device addresses
        // instead of a second copy of the geometry. The flag is the DEVICE's, not the config's: the
        // usage bit needs the extension enabled, and a buffer uploaded without it can never be built
        // from - so it is decided where the upload happens, once, and not per frame by whoever wants
        // to trace.
        VkBufferUsageFlags const rt_input_usage = this->vulkan_core.ray_query_available ? acceleration_structure::build_input_usage : 0u;
        result->vertex_buffer = this->vulkan_core.vma.create_buffer(info.vertex_data.data(), info.vertex_data.size_bytes(), vulkan::buffer_type::vertex, rt_input_usage);
        if (!result->vertex_buffer.valid()) {
            utility::panic("failed to create vertex buffer");
        }
        result->vertex_detail = this->vulkan_core.vma.get_buffer_detail(result->vertex_buffer.handle());
        if (result->vertex_detail == nullptr) {
            utility::panic("failed to get vertex buffer detail");
        }

        result->index_buffer = this->vulkan_core.vma.create_buffer(info.index_data.data(), info.index_data.size_bytes(), vulkan::buffer_type::index, rt_input_usage);
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
        // Kept for the acceleration-structure build, which reads the vertex buffer directly and has to
        // be told the stride the interleaved layout uses (nothing else needs it after the upload: the
        // raster pipelines take it from the vertex input state).
        result->vertex_stride = info.vertex_stride;

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
        // Its own motion slot, where advance_motion_transforms() will keep the world matrix this
        // leaf had one frame ago - and which pbr.vert reads to build the object half of TAA's motion
        // vector. Allocated here rather than per draw because the shader needs a STABLE index.
        result->motion_slot_index = this->motion_cursor++;
        result->push.motion_base = result->motion_slot_index;
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

        // The instanced draw owns `count` motion slots, and gets them filled with the SAME matrices
        // right away: an instance's previous transform is its current one, so its object motion reads
        // as zero. That is correct for a static grid, and it is where a moving instanced draw would
        // have to be handled (advance_motion_transforms skips instanced leaves on purpose: their
        // per-instance matrices are a setup-time quantity, not a per-frame one).
        uint32_t const motion_base = this->motion_cursor;
        uint32_t const motion_count = std::min<uint32_t>(count, vulkan::scene_motion_capacity - this->motion_cursor);
        this->motion_cursor += motion_count;
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            auto* const published = static_cast<glm::mat4*>(this->motion_mapped[static_cast<std::size_t>(slot)]);
            if (published != nullptr && motion_count > 0) {
                std::memcpy(published + motion_base, transforms.data(), static_cast<std::size_t>(motion_count) * sizeof(glm::mat4));
            }
        }
        std::copy_n(transforms.data(), motion_count, this->motion_previous.begin() + static_cast<std::ptrdiff_t>(motion_base));

        auto result = std::make_unique<instanced_draw_primitive>();
        // same pipeline semantics as the source geometry it draws (empty = default semantics)
        result->pipeline_name = source.pipeline_name;
        result->source = &source; // geometry owner; must stay in this runtime's scene tree
        result->instance_count = count;
        result->push.material_index = source.push.material_index;
        result->push.flags = 1u; // bit0: pbr.vert picks instances[instance_base + gl_InstanceIndex]
        result->push.instance_base = base;
        result->push.motion_base = motion_base; // slots run parallel to the instance block
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
        // Same build-input usage as create_primitive's, for the same reason (see there).
        VkBufferUsageFlags const rt_input_usage = this->vulkan_core.ray_query_available ? acceleration_structure::build_input_usage : 0u;
        result->vertex_buffer = this->vulkan_core.vma.create_buffer(info.vertex_data.data(), info.vertex_data.size_bytes(), vulkan::buffer_type::vertex, rt_input_usage);
        if (!result->vertex_buffer.valid()) {
            utility::panic("failed to create static vertex buffer");
        }
        result->vertex_detail = this->vulkan_core.vma.get_buffer_detail(result->vertex_buffer.handle());
        if (result->vertex_detail == nullptr) {
            utility::panic("failed to get static vertex buffer detail");
        }
        result->index_buffer = this->vulkan_core.vma.create_buffer(info.index_data.data(), info.index_data.size_bytes(), vulkan::buffer_type::index, rt_input_usage);
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
        result->vertex_stride = info.vertex_stride; // see create_primitive: the AS build reads the buffer

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
        // The whole batch is one draw of one world, so it tracks object motion through ONE slot,
        // exactly like a normal primitive (see create_primitive).
        result->motion_slot_index = this->motion_cursor++;
        result->push.motion_base = result->motion_slot_index;

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
        // The motion cursor is NOT recycled with it: every leaf owns a slot from that cursor, not
        // just the instanced ones, so resetting it here would hand a surviving leaf's slot to the
        // next primitive created. It is monotonic for the runtime's lifetime, and recycling it would
        // need a free list over the slots that just disappeared.
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
        // Content fingerprint of this upload: the only per-frame signal that a skinned caster moved
        // (its push.model is constant, the pose lives in these matrices). XXH3 rather than a byte
        // loop - 1.3 us for an 840-joint rig against 43.7 us, and this runs on every frame.
        this->skin_matrix_hash = utility::xxh3_64bits({static_cast<unsigned char const*>(this->skin_mapped[slot]), bytes});
    }

    void* runtime::morph_scratch() noexcept {
        return this->morph_scratch(this->active_slot);
    }

    void* runtime::morph_scratch(uint32_t const slot) noexcept {
        if (slot >= this->morph_mapped.size()) {
            return nullptr;
        }
        // relaxed: the fetch_add is only there to make concurrent bumps from the animation
        // controller's worker threads well defined and non-lossy (see the member docs) - the value
        // is read back through a plain load in shadow_geometry_signature(), on the frame thread.
        this->morph_revision.fetch_add(1, std::memory_order_relaxed); // no upload hook: assume the caller is about to deform the mesh
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
