module;

#include <GLFW/glfw3.h>
#include <algorithm> // std::min in the resource publication
#include <bit>       // std::bit_cast for the caster world-matrix hash
#include <chrono>
#include <cstring> // std::memcpy, for composing a pass's push block
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <span>   // std::as_bytes for the init_utils calls (the bytes behind a UBO or a zeroed table)
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
import vulkan.init_utils;      // the resource-creation patterns the init/ensure functions below repeat
import vulkan.frame_constants; // one frame's shared constants (see update_frame_constants)
import vulkan.core.pipeline;   // vulkan::make_pipeline for the post-process pipeline

// Route std::pmr allocations through mimalloc for this TU (utility.better_pmr). Idempotent:
// init_pmr() returns the same process-wide singleton no matter which TU calls it first, so
// main.cpp's keep-alive and this one coexist safely. The reference itself is never read; it
// only forces the (dynamic) initialization before any pmr container in this TU is constructed.
[[maybe_unused]] static auto& pmr = utility::init_pmr(); // NOLINT(keep-alive)

namespace {
    // (The `is_srgb_format` predicate that used to live here is `vulkan::is_srgb_format` now, in
    // vulkan.constant_init: two PASSES push the `encode_gamma` lane it decides - the composite and FXAA - and a
    // copy per file is a list of formats that falls behind in one of them.)

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
        : runtime(std::make_shared<core>(options)) {
        // The device root is built here and owned by this runtime; the constructor below is the one that does the
        // work, so a caller that ALREADY has a core takes the same path (see its doc note).
    }

    runtime::runtime(std::shared_ptr<core> shared_core)
        : core_owner{std::move(shared_core)}
        , vulkan_core{*this->core_owner}
        // readback owns GPU resources and is deliberately neither copyable nor movable (two owners of
        // one staging buffer is the bug its deletion prevents), so it must be constructed here - which
        // is why its member declaration sits ABOVE filtered_core's, matching this order. Both only need
        // the core, so the order between them is otherwise free.
        , readback_staging{vulkan_core}
        , filtered_core{core_owner}
        , pass_resources{core_owner} {
        glfwSetWindowUserPointer(this->vulkan_core.window, this);
        glfwSetMouseButtonCallback(this->vulkan_core.window, mouse_button_callback);
        glfwSetCursorPosCallback(this->vulkan_core.window, cursor_pos_callback);
        glfwSetScrollCallback(this->vulkan_core.window, scroll_callback);

        // The recording resources - one primary command buffer per frame slot, plus the secondary
        // buffers (and the per-consumer pools that must own them) the parallel recording stages hand
        // out - are built by init_recording_resources(): they depend on nothing else in this
        // constructor but the two capacities, and nothing else here reads them.
        this->init_recording_resources();

        // Shared scene resources: camera UBO buffers, white fallback texture, texture sampler
        this->init_scene_resources();
        // Every per-image flag that describes this generation starts where the generation's images do.
        // The core has already built this generation's targets (its constructor ran
        // create_hdr_resolve_resources), so the flags can be sized HERE, before any frame records; every
        // later generation gets the very same reset from on_swapchain_recreated - one function, so the
        // two lists cannot drift (which they already had).
        this->reset_image_generation_state();
        // The lobe's per-image first-use state starts where its images do (empty, which reads as "nothing has
        // been transitioned"), and the pass owns it - see vulkan.pass.ssgi_spec.

        // The probe cache's remaining flag starts where the images do: nothing has transitioned the grid out
        // of UNDEFINED, so the tracer's gain is 0 until the TRACE pass - the frame's first reader of the grid
        // - has taken its first-use transition once. (What the grid HOLDS is the pass's own state now:
        // gi_probe_pass::cache_valid, and whether the batch happened is the TRACER's - its own
        // `on_swapchain_recreated` clears it, which is what the recreate_stage call in the swapchain path runs.)
        // (The furnace cube is a new image too - its level is part of the generation reset above.)
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

        // post-process objects: the FXAA pipeline and the two samplers are RAII members, and the post chain's set
        // layout, pipeline layout and two pipelines are NOT here any more - they belong to the post composite PASS
        // (vulkan.pass.post::release_owned), the same rule every extracted pass follows.
        //
        // The block that USED to be here is why this comment exists at all: an earlier edit collapsed it onto the
        // comment line above it, which commented the destroy calls out and leaked them (validation: "VkDevice has
        // 18 leaked objects ... VkPipelineLayout, VkDescriptorSetLayout"). The pool belongs to post_family now and
        // those layouts to the pass, so what is left of the post chain in this destructor is nothing at all.
        // ... and the G-buffer debug view's set layout and pipeline layout are NOT here any more: they are the
        // PASS's (vulkan.pass.gbuffer_debug::release_owned), which built them - and which the deferred lighting
        // stage now asks for the layout through shared_set_layout(1). The FAMILY's pool is still gbuffer_family's,
        // whose destructor destroys it (and the generations it retired).
        // ... and the deferred lighting stage's pipeline layout is NOT here any more: it is the PASS's
        // (vulkan.pass.deferred::release_owned), built by the pass from the two shared set layouts its owner
        // hands it at create time.
        // ... and the GI tracer is NOT here any more: its pipeline layout and its pipeline are the PASS's (see
        // vulkan.pass.ssgi_trace::release_owned), which is what the create/record split is for. The SPATIAL
        // FILTER's left the same way (vulkan.pass.ssgi_spatial), and so did the TEMPORAL resolve's set layout,
        // pipeline layout and pipeline (vulkan.pass.ssgi_temporal) - so no SSGI pipeline handle is torn down in
        // this destructor at all now, and the only GI object still the renderer's is the denoiser's two
        // descriptor FAMILIES (the diffuse and the reflection), which one declaration cannot describe.
        // The bake's pipeline layout is NOT destroyed here: it is the JOB's (vulkan.pass.mask_bake_job), which
        // releases its pipeline layout, its pipeline and its own descriptor set in its own destructor - the
        // same rule the extracted passes follow. The compute-skinning job's layout, pipeline and per-slot sets
        // left the same way (vulkan.pass.compute_skin_job), so this destructor no longer names either of the two
        // traced-feature jobs.
        // The glossy lobe's layout and pipeline are NOT destroyed here any more either: they are the pass's
        // (vulkan.pass.ssgi_spec::release_owned), the same way the probe cache's and the TAA resolve's left.
        // ... and the SPATIAL FILTER's left the same way with it (vulkan.pass.ssgi_spatial::release_owned), so
        // the only SSGI layout this destructor still names is the temporal resolve's - whose pass is next.
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
        camera_ubo initial = {};
        init_utils::create_host_buffers(this->vulkan_core,
                                        vulkan::core::MAX_FRAMES_IN_FLIGHT,
                                        std::as_bytes(std::span(&initial, 1)),
                                        vulkan::buffer_type::uniform_coherent,
                                        "camera ubo buffer",
                                        this->camera_buffers,
                                        &this->camera_mapped);

        // 1x1 white fallback texture, always the first entry of the scene texture array; missing
        // material textures point at it
        constexpr std::array<unsigned char, 4> white_pixels = {255, 255, 255, 255};
        vulkan::image_create_info white_info = {};
        white_info.width = 1;
        white_info.height = 1;
        white_info.mip_levels = 1;
        white_info.array_layers = 1;
        white_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        init_utils::texture_2d white = init_utils::create_texture_2d(this->vulkan_core, std::as_bytes(std::span(white_pixels)), white_info, "white fallback texture");
        this->owned_textures.push_back(std::move(white.image));
        this->owned_texture_views.push_back(std::move(white.view));
        this->white_texture_index = static_cast<uint32_t>(this->texture_array_views.size());
        this->texture_array_views.push_back(*this->owned_texture_views.back());

        // (The sampler the array entries are read through is NOT created here: the maxLod-12 REPEAT
        // sampler the comment that used to sit here described moved into the core, next to the other six
        // - see core::create_samplers / shared_samplers.) The white element above is the one entry this
        // function has to place, because every later texture index is assigned around it.

        // GPU material table: fixed capacity, host-visible (direct mapping); records are appended
        // at registration and read-only for the GPU (set 0 binding 5)
        std::vector<unsigned char> const zeroed_materials(static_cast<size_t>(vulkan::material_capacity) * sizeof(material_record), 0);
        init_utils::create_host_buffer(this->vulkan_core,
                                       std::as_bytes(std::span(zeroed_materials)),
                                       vulkan::buffer_type::storage_coherent,
                                       "material table buffer",
                                       this->material_buffer,
                                       this->material_mapped);

        // Per-instance transform buffer (set 0 binding 6): one mat4 per instance, host-visible;
        // filled by set_instanced_draw() for instanced stress draws (see pbr.vert)
        std::vector<unsigned char> const zeroed_instances(static_cast<size_t>(vulkan::instance_capacity) * sizeof(glm::mat4), 0);
        init_utils::create_host_buffer(this->vulkan_core,
                                       std::as_bytes(std::span(zeroed_instances)),
                                       vulkan::buffer_type::storage_coherent,
                                       "instance transform buffer",
                                       this->instance_buffer,
                                       this->instance_mapped);

        // Per-motion-slot previous world matrices (set 0 binding 13): ONE buffer per frame slot, like
        // the skin and morph buffers below, so a frame in flight never shares the buffer the next
        // frame rewrites. Zero-filled: a leaf's first frame reports "no motion", which is right -
        // nothing was there to move from. motion_previous is the CPU-side copy of what is currently
        // in it, advanced by advance_motion_transforms().
        std::vector<unsigned char> const zeroed_motion(static_cast<size_t>(vulkan::scene_motion_capacity) * sizeof(glm::mat4), 0);
        init_utils::create_host_buffers(this->vulkan_core,
                                        vulkan::core::MAX_FRAMES_IN_FLIGHT,
                                        std::as_bytes(std::span(zeroed_motion)),
                                        vulkan::buffer_type::storage_coherent,
                                        "motion transform buffer",
                                        this->motion_buffers,
                                        &this->motion_mapped);
        this->motion_previous.assign(vulkan::scene_motion_capacity, glm::mat4(1.0f));

        // Per-joint skin matrices (set 0 binding 9): one buffer PER FRAME SLOT (scene_skin_capacity
        // mat4s each, host-visible) so an in-flight frame never shares the buffer the next frame
        // rewrites. Zero-filled initially (the identity block is written by the setup upload).
        std::vector<unsigned char> const zeroed_skins(static_cast<size_t>(vulkan::scene_skin_capacity) * sizeof(glm::mat4), 0);
        init_utils::create_host_buffers(this->vulkan_core,
                                        vulkan::core::MAX_FRAMES_IN_FLIGHT,
                                        std::as_bytes(std::span(zeroed_skins)),
                                        vulkan::buffer_type::storage_coherent,
                                        "skin matrix buffer",
                                        this->skin_buffers,
                                        &this->skin_mapped);

        // Morph data (set 0 binding 10): one buffer PER FRAME SLOT (scene_morph_capacity floats
        // each, host-visible); the caller bakes per-primitive morph blocks (deltas + weights)
        // into every slot's buffer at setup, then rewrites only the active slot's weights per frame.
        // Zero-filled from one shared host vector (each create_buffer copies its own GPU buffer).
        std::vector<unsigned char> const zeroed_morphs(static_cast<size_t>(vulkan::scene_morph_capacity) * sizeof(float), 0);
        init_utils::create_host_buffers(this->vulkan_core,
                                        vulkan::core::MAX_FRAMES_IN_FLIGHT,
                                        std::as_bytes(std::span(zeroed_morphs)),
                                        vulkan::buffer_type::storage_coherent,
                                        "morph data buffer",
                                        this->morph_buffers,
                                        &this->morph_mapped);

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

    // The recording resources: one primary command buffer per frame slot, plus the secondary buffers
    // of the two parallel recording stages and the pools that have to own them. They are pre-allocated
    // so the GPU can read a secondary while this slot's primary executes, and reused every frame, so
    // they must exist before the first recorded frame - but nothing else in the constructor depends on
    // them, and they depend on nothing else but the two capacities below (hence a function of their
    // own). This is also where the command pools would move to the core (the "command pools and
    // secondaries" item of the trim list in docs/pass_chain_plan.md): the SHAPE is policy and stays
    // here, the objects are device resources.
    void runtime::init_recording_resources() {
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
                cascade_recording.push_back(init_utils::create_recording_pool(this->vulkan_core));
            }
            this->shadow_recording.push_back(std::move(cascade_recording));
            std::vector<std::pair<VkCommandPool, vk_command_buffer>> segments;
            segments.reserve(record_workers);
            for (unsigned s = 0; s < record_workers; ++s) {
                segments.push_back(init_utils::create_recording_pool(this->vulkan_core)); // one per worker
            }
            this->main_segments.push_back(std::move(segments));
        }
    }

    // The per-image state a swapchain GENERATION starts from. A freshly created target image is in
    // UNDEFINED, holds nothing, and no pass has written it: that is equally true of generation 0 (this
    // constructor, after the core built the generation's targets) and of every later generation
    // (on_swapchain_recreated), so both call THIS and the two can no longer drift.
    //
    // NOT here, deliberately: the TAA history matrices (image_view_proj). They may only be written from
    // a real camera snapshot (current_ubo), which does not exist yet in the constructor - they stay with
    // the two call sites that have one (on_swapchain_recreated, and set_taa's off -> on edge).
    void runtime::reset_image_generation_state() {
        // The G-buffer depth layout flags are one per swapchain image, and a freshly created depth image
        // is in UNDEFINED (which is what a clear flag says); see ensure_gbuffer_depth_sampled.
        this->gbuffer_depth_written.assign(this->vulkan_core.gbuffer_depth_images.size(), false);
        // The motion-vector images died with the generation as well: clear the layout flag so the first
        // frame of the new generation takes the attachment -> sampled transition (see
        // ensure_velocity_sampled).
        this->velocity_written.assign(this->vulkan_core.velocity_images.size(), false);
        this->gbuffer_targets_written.assign(this->vulkan_core.gbuffer_images[0].size(), false);
        // The GI accumulation starts empty for the same reason (see gi_history_valid): the first frame of
        // a generation has nothing to blend with, and it is reset HERE rather than only on the off -> on
        // edge in set_ssgi, because a run that starts with GI enabled never sees that edge (an empty
        // vector reads as "no history" for every frame, which silently turns the temporal resolve into a
        // pass-through of the raw trace).
        this->gi_history_valid.assign(this->vulkan_core.gi_history_images.size(), false);
        // ... and the furnace cube is a new image too, so its level has to be written again.
        this->furnace_cube_ready = false;
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

        // Light UBO (scene set binding 7): one buffer PER FRAME SLOT (host-visible, mapped), so
        // a frame being rendered never shares the buffer the next frame rewrites. CPU-side
        // content lives in light_state; the frame loop memcpys it into the paced slot's buffer
        // (pace_and_acquire) - see the member docs for the concurrency rationale.
        light_ubo initial = {};
        init_utils::create_host_buffers(this->vulkan_core,
                                        vulkan::core::MAX_FRAMES_IN_FLIGHT,
                                        std::as_bytes(std::span(&initial, 1)),
                                        vulkan::buffer_type::uniform_coherent,
                                        "light ubo buffer",
                                        this->light_buffers,
                                        &this->light_mapped);
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
        init_utils::create_host_buffers(this->vulkan_core,
                                        static_cast<uint32_t>(slots),
                                        std::as_bytes(std::span(zero_counts)),
                                        vulkan::buffer_type::storage_coherent,
                                        "cluster count buffer",
                                        this->cluster_count_buffers,
                                        &this->cluster_count_mapped);
        // The index rows are host-visible for the same reason, but nothing on the CPU ever writes
        // through the mapping: the dispatch fills them, so the mapped list stays a nullptr.
        init_utils::create_host_buffers(this->vulkan_core,
                                        static_cast<uint32_t>(slots),
                                        std::as_bytes(std::span(zero_indices)),
                                        vulkan::buffer_type::storage_coherent,
                                        "cluster index buffer",
                                        this->cluster_index_buffers);
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
                visibility_infos[0].sampler = *this->vulkan_core.post_sampler;
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
                .sampler = *this->vulkan_core.shadow_sampler,
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
        this->scene_sets.write_ibl(this->vulkan_core, this->ibl_ready, view_span, *this->env_sampler, *this->owned_texture_views[0], *this->vulkan_core.texture_sampler);
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
        VkSampler const sampler = *this->vulkan_core.texture_sampler;
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
            init_utils::texture_2d material_texture = init_utils::create_texture_2d(this->vulkan_core, std::as_bytes(tex.data), image_info, "material texture");
            this->owned_textures.push_back(std::move(material_texture.image));
            this->owned_texture_views.push_back(std::move(material_texture.view));
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
        this->ssgi_spec_temporal_family.retire_all();
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
            // THE STAGES ADDED SINCE THIS LIST WAS WRITTEN, and now one call per chain: the tracer's
            // `probe_grid_seen_` and the lobe's per-image first-use flags describe a GENERATION, so a swapchain
            // recreation owes them another first-use batch. Leaving a pass out of this list is invisible until
            // someone resizes the window - exactly the hazard `recreate_stage` exists to remove, which is why the
            // fix is this call and not a second hand-kept flag in the host. The chains make "all of them" the
            // default instead of a list someone has to remember to extend.
            [[maybe_unused]] pass::run_report const gi_trace_recreated = pass::recreate_stage(this->gi_trace_chain.as_stage(), this->make_pass_host());
            [[maybe_unused]] pass::run_report const gi_denoise_recreated = pass::recreate_stage(this->gi_denoise_chain.as_stage(), this->make_pass_host());
        }
        // Every swapchain image's history died with the old generation (and its size may have
        // changed): forget the matrices, so the next frame for each image starts a new accumulation
        // instead of blending in a misaligned one. (WHETHER a history holds anything is the TAA pass's own
        // state, and the call above is what cleared it.)
        std::size_t const image_count = this->vulkan_core.taa_history_images.size();
        this->image_view_proj.assign(image_count, this->current_ubo.view_proj_unjittered);
        // The lobe's per-image first-use state is the PASS's, and the recreate_stage call above is what told it
        // (vulkan.pass.ssgi_spec::on_swapchain_recreated). The tracer's "this generation's probe grid has had
        // its first-use batch" is the same shape and the same call cleared it - which is why there is no host
        // flag for it any more (the renderer asks the pass: `probe_grid_seen`).
        //
        // Every OTHER per-image flag is the generation reset, and it is the very function the constructor
        // calls for generation 0: this list used to be hand-kept in two places (the G-buffer depth flag, the
        // motion-vector flag, the G-buffer target flags, the GI accumulation and the furnace cube), and the
        // two copies had already drifted. `image_view_proj` above stays out of it, because it needs a camera
        // snapshot (current_ubo) - which only this call site and set_taa's off -> on edge have.
        this->reset_image_generation_state();
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
        //    camera update - no per-frame descriptor write exists anymore.
        this->current_aspect = static_cast<float>(vk.swap_chain_extent.width) / static_cast<float>(vk.swap_chain_extent.height);
        this->current_ubo = make_orbit_camera_ubo(this->camera.yaw, this->camera.pitch, this->camera.distance, this->camera.target, this->scene_radius, this->current_aspect);
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
            this->light_state.rt_shadows = (this->rt_shadows && this->pass_ready("rt_shadow") && this->vulkan_core.ray_query_available) ? 1.0f : 0.0f;
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
                bool const clustered = this->clustered_lights && this->pass_ready("cluster") && !degenerate && this->cluster_tiles_x > 0 && this->cluster_tiles_y > 0;
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
        // ... and the frame's shared constants are THIS frame's from here on: the camera record, the light
        // UBO and the fitted scene bounds above are all final at this point, and every pass of the frame reads
        // the same numbers the shaders see (see vulkan.frame_constants).
        this->update_frame_constants();
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
        // THIS FRAME's resources, in the declaration's vocabulary: published before anything resolves, because
        // the per-swapchain-image views are this generation's, an alias (`scene_color`) is decided per frame, and
        // the lazily created shadow map exists only after a scene set does - so this is the first point where
        // "what exists right now" has an answer (see pass::resource_table and runtime::publish_frame_resources).
        this->publish_frame_resources();
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
        //     old flat-list world matrices exactly; the scene transform adds programmatic
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
            // camera identity: the orbit state that shapes the frustum
            bool const scene_changed_this_frame = this->bvh_dirty;
            std::array<float, 7> const key = {
                this->camera.yaw,
                this->camera.pitch,
                this->camera.distance,
                this->camera.target.x,
                this->camera.target.y,
                this->camera.target.z,
                aspect,
            };
            this->camera_moved = key != this->camera_key;

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
            // THE CLUSTER SORT records its own dispatch and its own two buffer barriers now
            // (vulkan.pass.cluster); what is this loop's is WHERE it runs - before the passes that read the
            // bins - and the frame data it is handed, which the chain's owner supplies (see prepare_stage).
            this->prepare_stage("cluster", *command_buffer);
            pass::stage const cluster_stage = {.name = "cluster", .passes = this->cluster_stage, .marks = false};
            [[maybe_unused]] pass::run_report const cluster_report = pass::record_stage(cluster_stage, this->make_pass_host());
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
                // ---- THE SHADOW PASS RECORDS IT (vulkan.pass.shadow) ----
                // What stays HERE is what the pass cannot know: the reuse test above (which is why this whole block
                // is inside it), the map's OWN image and its layer count (the hand-back below covers every
                // allocated layer, including the spare ones), and the bookkeeping that makes the next frame's reuse
                // test true. The frame itself - the per-cascade secondaries, the map's edge and the two callbacks -
                // is built HERE and handed to the pass by whoever owns it (see make_shadow_frame/prepare_stage).
                this->prepare_stage("shadow", *command_buffer);
                {
                    pass::stage const shadow_stage = {.name = "shadow", .passes = this->shadow_stage, .marks = false};
                    [[maybe_unused]] pass::run_report const shadow_report = pass::record_stage(shadow_stage, this->make_pass_host());
                }
                // Hand the cascades back to the shading stages as a sampled array texture: the layers just rendered go
                // depth-attachment -> shader-read (the src masks publish the attachment write), and the SPARE layers a
                // shrank cascade count left behind are covered by the same range - the descriptor's array view spans
                // every allocated layer, so "one barrier, whole array" is the invariant. That count is the IMAGE's,
                // which is why this is the host's and not the pass's.
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
        } else if (this->pass_ready("shadow") && this->shadow_images.size() > static_cast<std::size_t>(frame_slot)) {
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

    // =============================================================================================
    // THE SHADOW PASS's content callback and its scheduler (vulkan.pass.shadow)
    // =============================================================================================
    //
    // THE RESOLVER THAT USED TO BE HERE IS GONE (S3.8): the layers this frame renders were a hand-written
    // `resolve_shadow_pass` because the declaration could name only ONE layer of the map, and the vocabulary that
    // replaced it is a RUN of elements (`render_target::count`) - the declaration now claims every cascade layer
    // the map can hold and the resource table answers how many it HAS this frame, which is the cascade knob's
    // (`ensure_shadow_resources`). What is left here is what the pass genuinely cannot know: which secondaries to
    // record into, and what a caster's draw state is (the scene set, the live depth-bias state, the two-sided
    // policy, the secondary's own begin info) - so that arrives as a callback, and the map's edge with it.
    bool runtime::record_shadow_cascade(void* const owner, VkCommandBuffer const secondary, uint32_t const cascade_index, VkPipeline const pipeline, VkPipelineLayout const pipeline_layout) {
        runtime* const self = static_cast<runtime*>(owner);
        core const& vk = self->vulkan_core;
        // The secondary inherits ONLY the depth attachment (no colour one): dynamic rendering 1.3, single-sampled,
        // viewMask 0. The inheritance struct hangs off VkCommandBufferInheritanceInfo::pNext (NOT the begin info's),
        // and a secondary buffer must always provide inheritance info.
        VkCommandBufferInheritanceRenderingInfo const inheritance = make_inheritance_rendering_info(false, nullptr, vk.depth_format, VK_SAMPLE_COUNT_1_BIT);
        VkCommandBufferInheritanceInfo const inherit = make_inheritance_info(&inheritance);
        VkCommandBufferBeginInfo const begin = make_command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT, &inherit);
        if (vkBeginCommandBuffer(secondary, &begin) != VK_SUCCESS) {
            utility::log("runtime: shadow secondary command buffer begin failed - cascade {} skipped this frame", cascade_index);
            return false;
        }
        // which cascade these casters are projected into: the vertex stage indexes the light UBO's matrix array with
        // it, and a secondary records that itself (state is not inherited from the primary). The offset is the
        // declaration's own - where the scene's push block ends.
        uint32_t const index = cascade_index;
        vkCmdPushConstants(secondary, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, render_resource::shadow_io.push->offset, sizeof(index), &index);
        self->record_shadow_content(secondary, pipeline, pipeline_layout);
        vkEndCommandBuffer(secondary);
        return true;
    }

    void runtime::run_shadow_tasks(void* const owner, std::span<std::function<void()>> const tasks) {
        static_cast<runtime*>(owner)->run_tasks(tasks, vulkan::task_priority::recording);
    }
    void runtime::record_scene(VkCommandBuffer const command_buffer) {
        this->record_scene_attachments(command_buffer);
        this->update_pass_geometry();
        // THE SCENE PASS records the surface write: the instance over its six declared targets, the segmented
        // draw of the visible leaves, and closing the instance - all inside one function now (see
        // vulkan.pass.scene for why that is the point of this extraction).
        if (this->gbuffer_pass_active()) {
            this->prepare_stage("scene", command_buffer);
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
        // viewport is the fixed shadow-map size, which the shadow pass's content callback sets (see the pass).
        // ... and the post chain's pipelines are NOT resynced here any more: they belong to the post composite
        // PASS (vulkan.pass.post), which declares `resync_viewport` for the composite and derives each bloom
        // level's extent from its own declaration - so the viewport comes from the pass's declaration instead of
        // from a list this class maintains. The two blocks that used to be here are what that field replaced.
        // ... and FXAA's pipeline is NOT resynced here either: it is the FXAA PASS's now (vulkan.pass.fxaa),
        // which declares `resync_viewport` like the rest of the post chain.
        // ... and the same for the G-buffer pair: gbuffer_pipeline is the opaque pass's default
        // pipeline (begin_pipeline applies the stored viewport) and gbuffer_debug_pipeline is a
        // fullscreen pass. Neither lives in the named cache above.
        if (this->gbuffer_pipeline) {
            this->gbuffer_pipeline->viewport = full_viewport;
            this->gbuffer_pipeline->scissor = full_scissor;
        }
        // ... and the debug view's pipeline is not resynced here either: it is its PASS's now (vulkan.pass.
        // gbuffer_debug), which declares resync_viewport like every other fullscreen pass in this chain.
        // ... and the deferred lighting stage's is not here either, for a stronger reason than the TAA
        // resolve's below: it is a PASS (vulkan.pass.deferred), it declares `resync_viewport = true`, so the
        // runner sets the viewport and scissor from the extent its own declaration produced. The old
        // `deferred_pipeline->viewport = ...` line existed because a pipeline object caches what
        // begin_pipeline() applies; the pass never calls begin_pipeline, the runner binds the pipeline.
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
    void runtime::record_shadow_content(VkCommandBuffer const command_buffer, VkPipeline const pipeline, VkPipelineLayout const pipeline_layout) const {
        core const& vk = this->vulkan_core;
        uint32_t const frame_slot = static_cast<uint32_t>(vk.current_frame);
        if (this->scene_sets.created()) {
            VkDescriptorSet const scene_set_handle = this->scene_sets.set(static_cast<uint32_t>(frame_slot));
            vkCmdBindDescriptorSets(command_buffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline_layout,
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
        env.bind = [this, pipeline](VkCommandBuffer const cb, std::string_view const /*name*/) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            VkViewport const shadow_viewport = {0.0f, 0.0f, static_cast<float>(this->shadow_map_size), static_cast<float>(this->shadow_map_size), 0.0f, 1.0f};
            VkRect2D const shadow_scissor = {{0, 0}, {this->shadow_map_size, this->shadow_map_size}};
            vkCmdSetViewport(cb, 0, 1, &shadow_viewport);
            vkCmdSetScissor(cb, 0, 1, &shadow_scissor);
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
    // The post chain's PIPELINES are not made here any more: the composite PASS owns them (vulkan.pass.post,
    // built by `pipelines::build_post` inside its create step). What stays is the SAMPLERS, and they are the
    // renderer's for the reason `build_post`'s own comment gives: they belong to the descriptor SETS, which this
    // class still writes (post_family). They have to exist before `create_passes()`, because the pass context
    // hands every pass the six samplers a declaration may choose between.
    // THE FXAA PASS'S FRAME (vulkan.pass.fxaa). What the pass cannot know is which image it reads and which it
    // writes (the LDR image the composite produced, and the swapchain), which variant of the post family's set is
    // the one it reads that image through (set 4, the composite's), and the push block's values.
    //
    // NO OFF PATH IS NEEDED, and that is a difference from the bloom chain worth stating: a frame that cannot
    // resolve this pass leaves the LDR image alone (nobody moves it, nobody samples it) and the swapchain holds
    // what it held. The old recorder would have bound a null descriptor set in that case, which is worse.
    // THE FXAA PASS'S RESOLVER IS GONE (S3): its target (the swapchain), the one image it transitions (the LDR
    // image the composite wrote), the post set it binds and its extent all come from its own declaration against
    // the frame's resource table, and its pipeline from the pass (it builds its own, against the post set layout
    // the context answers). Its push block it composes itself out of the frame's settings and the surface's
    // format, which it cached at create.
    void runtime::ensure_post_descriptors() {
        core& vk = this->vulkan_core;
        if (!this->pass_ready("post_composite")) {
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
                    image_infos[b].sampler = nearest ? *this->vulkan_core.post_nearest_sampler : *this->vulkan_core.post_sampler;
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
        if (!this->post_family.ensure_all(vk.device, this->post_composite.set_layout(), static_cast<uint32_t>(image_count), 5u, 9u, fingerprints, write_sets)) {
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

    // THE DEFERRED LIGHTING PASS'S FRAME (vulkan.pass.deferred). The pass owns the barrier, the two per-image
    // input transitions, the LOAD instance, the two binds, the push and the draw; what the host resolves is the
    // FRAME - which image is lit, which two sets are bound (the shared scene set's frame slot and the G-buffer
    // set's swapchain image) and the values the push carries - plus deciding that this frame cannot run at all.
    //
    // It was `record_lighting_pass`'s prologue, unchanged, including the order of the two checks: the pipeline
    // first, then the G-buffer set. The second one is why this returns false instead of recording something, and
    // the caller is what clears the target in that case (the fallback is about the renderer's own pool).
    // THE DEFERRED LIGHTING STAGE'S RESOLVER AND ITS `ensure_inputs` CALLBACK ARE GONE (S3). Its declaration
    // resolves generically - the shared scene set (0), the shared G-buffer set (1), its own pipeline and a
    // full-frame extent - and its TARGET is the frame's alias `scene_color`, which the per-frame resource table
    // answers (the TAA input while the resolve runs, the HDR image otherwise). Its push block it composes itself
    // out of its own SSAO parameters, the frame's inverse view-projection and the frame's answer to what the traced
    // chain is doing. The two per-image transitions the frame used to carry are the frame's ORDERING rule about the
    // images the G-buffer pass wrote, so they now run in the deferred STAGE's preamble in `record_main_drawcalls` -
    // the same move the ray-traced shadow stage's identical pair made, and the same command-stream position.

    void runtime::clear_scene_color_for_missing_gbuffer(VkCommandBuffer const command_buffer) {
        // The deferred lighting pass did not record because the G-buffer family had no descriptor set for this
        // image (its pool could not give one). The pass cannot do this itself - it never sees the family's pool,
        // and "the owner has no set" is a failure of the owner's pool - so the frame's answer lives here: clear
        // the target so the frame is DEFINED (the post chain samples it) instead of leaving whatever the
        // background/emissive wrote mixed with garbage, and say so once per frame, because a silent black frame
        // is worse than a log line.
        core const& vk = this->vulkan_core;
        uint32_t const index = this->current_image_index;
        if (index >= vk.scene_color_images.size()) {
            return;
        }
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
    }

    // The G-buffer declarations' SAMPLERS: what is left of make_gbuffer_debug_pipeline in the renderer, because the
    // set layout, its pipeline layout and the view pipeline are the debug view's PASS's now. They have to exist
    // before create_passes(), since the pass context hands every pass the six a declaration may choose between.
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
        this->prepare_stage("transparent", command_buffer);
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
        return this->taa_on && this->pass_ready("taa") && this->deferred_lit_active();
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
            if (!this->pass_ready("taa")) {
                this->warn_missing_feature("taa", "TAA has no effect: the taa pipeline was not created (see the startup log)");
            } else if (!this->deferred_lit_active()) {
                this->warn_missing_feature("taa", "TAA has no effect: the G-buffer pass or its lighting stage was not created (see the startup log)");
            }
        }
        // THE TWO BLEND WEIGHTS ARE THE PASS'S PARAMETERS (S3), including their clamps: this setter forwards
        // them and the pass owns what it does with them, the same way the push block it composes is the pass's.
        // Nothing else in the renderer reads them, so there is no second copy to keep in step.
        this->taa_resolve.set_blend(blend_static, blend_min);
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
            return this->pass_ready("gbuffer-debug");
        }
        return this->pass_ready("deferred");
    }

    bool runtime::deferred_lit_active() const noexcept {
        // the debug view wins when both are available: looking at the stored data is an inspection,
        // not a render mode (and the two write the HDR target in incompatible ways)
        return this->gbuffer_pass_active() && !this->gbuffer_debug;
    }

    void runtime::set_gbuffer_channel(int const channel) noexcept {
        // The knob is the PASS's parameter now (with its clamp, which is the same fact as the value): the renderer
        // forwards, the same rule set_taa's blend weights and the composite's GI upsample follow.
        this->gbuffer_debug_view.set_channel(channel);
    }

    void runtime::ensure_gbuffer_descriptors() {
        core& vk = this->vulkan_core;
        if (!this->pass_ready("gbuffer-debug")) {
            return;
        }
        std::size_t const image_count = vk.gbuffer_image_views[0].size();
        if (image_count == 0 || vk.gbuffer_depth_image_views.size() != image_count || vk.hdr_image_views.size() != image_count || vk.gi_image_views.size() != image_count ||
            vk.gi_spec_image_views.size() != image_count || vk.gi_spec_reproject_image_views.size() != image_count ||
            vk.gi_spec_resolve_image_views.size() != image_count ||
            vk.gi_probe_image_views.empty() || vk.gi_probe_sampler.get() == VK_NULL_HANDLE) {
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
                image_infos[b].sampler = storage ? VK_NULL_HANDLE : (b >= 9u && b <= 12u ? *this->vulkan_core.gi_probe_sampler : *this->vulkan_core.gbuffer_sampler);
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
        if (!this->gbuffer_family.ensure(vk.device, this->gbuffer_debug_view.set_layout(), static_cast<uint32_t>(image_count), 1u, static_cast<uint32_t>(signature.size()), signature, write_sets)) {
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
        return this->ssgi_on && this->pass_ready("ssgi_trace") && this->pass_ready("ssgi_temporal") &&
               this->pass_ready("ssgi_spatial") && this->deferred_lit_active() && !this->deferred.unlit();
    }

    void runtime::set_ssgi(bool const enabled, float const intensity, float const radius, uint32_t const rays, uint32_t const steps) noexcept {
        bool const was_on = this->ssgi_on;
        this->ssgi_on = enabled;
        // The ray budget is the PASS's from here on: it owns the values and their clamps (the intensity, the reach
        // as a fraction of the scene radius, the rays and the steps), because it is the only reader - the same rule
        // the TAA resolve's blend weights and the spatial filter's width follow.
        this->ssgi_trace.set_reach(intensity, radius, rays, steps);
        if (enabled && !this->pass_ready("ssgi_trace")) {
            this->warn_missing_feature("ssgi", "screen-space GI has no effect: its compute pipeline was not created (see the startup log)");
        } else if (enabled && !this->pass_ready("ssgi_temporal")) {
            this->warn_missing_feature("ssgi", "screen-space GI has no effect: its temporal resolve was not created (see the startup log)");
        } else if (enabled && !this->pass_ready("ssgi_spatial")) {
            this->warn_missing_feature("ssgi", "screen-space GI has no effect: its spatial filter was not created (see the startup log)");
        }
        if (enabled && !was_on) {
            // A fresh accumulation, on the off -> on EDGE only. Today the one caller is startup
            // (main.cpp applies the config once), so this is mostly a guard for the shape of the
            // setter: it mirrors set_taa, and a future overlay control that calls it every frame must
            // not have the history thrown away on each of those calls - the resolve would show the raw
            // trace forever, which looks like a denoiser running while doing nothing.
            this->gi_history_valid.assign(this->vulkan_core.gi_history_images.size(), false);
            // ... and the lobe's outputs are re-transitioned from UNDEFINED, which is what "switched on" means
            // for a pass that has not run yet this generation: the pass owns the flags, so the host asks it.
            this->ssgi_spec.reset_first_use();
        }
    }

    pass::ssgi_trace_frame runtime::make_ssgi_trace_frame() const noexcept {
        std::size_t const index = this->current_image_index;
        return pass::ssgi_trace_frame{
            // The previous frame's accumulation: the DENOISER's state, which still lives here because the
            // temporal resolve has not been extracted yet (see docs/pass_chain_plan.md).
            .history_valid = index < this->gi_history_valid.size() && this->gi_history_valid[index],
            // Whether the glossy lobe runs after the tracer, which decides who owes the denoiser the hand-off
            // barrier. THE SAME PREDICATE the lobe's own call site uses, evaluated here so the two cannot
            // disagree about which of them hands the raw trace over.
            .specular_next = this->ssgi_specular_active(),
            // Whether this generation's probe grid still needs its first-use batch. The tracer's pass tracks "have
            // I seen it" itself and the HOST asks it (`probe_grid_seen`), which is why there is no second flag
            // here: the batch has to happen exactly once per generation, and two copies of "has it happened" can
            // disagree after a resize.
            .probe_grid_first_use = !this->ssgi_trace.probe_grid_seen() && this->vulkan_core.gi_probe_images.size() == 8,
            // WHICH ORACLE this frame's rays use - the same predicate the lobe's feature is gated on, so the two
            // stages cannot disagree about the path (it decides three lanes of the tracer's push at once).
            .traced_oracle = this->ssgi_traced_active(),
            // ... and whether the probe cache may be READ: active AND written at least once. The gain the pass
            // pushes is zero otherwise, so a grid nothing has deposited into is never sampled - and the "has it
            // been written" half is the probe PASS's own state, which is why the host asks it.
            .probe_ready = this->gi_probe_active() && this->gi_probe.cache_valid(),
            // ... and whether a hit is shaded from its own geometry, which is what makes the instance table's
            // address meaningful to this shader (a zero address is how it is told to read the screen instead).
            .shade_hits = this->ssgi_hit_shading,
        };
    }

    // THE TRACER'S RESOLVER IS GONE (S3.10): its twelve declared images, its two shared sets, its own pipeline and
    // the declaration's `half` extent are all the framework's to resolve now (the probe grid's eight elements and
    // the surface image are ordinary table entries), and the push block it used to be handed is composed by the
    // PASS from `io.constants` (the camera, the projection terms, the scene bounds, the instance table's address,
    // the ray sequence), its own ray budget and the two answers its frame carries (which oracle, and whether the
    // probe cache has been written).

    void runtime::ensure_ssgi_denoise_descriptors() {
        // ONE FAMILY IS LEFT HERE: the REFLECTION's. The diffuse one is the pass's now
        // (vulkan.pass.ssgi_temporal::record writes it from `resolved_io::own_per_image`), and this family shares
        // that pass's set layout because two signals are resolved through one pipeline - which is exactly why a
        // single declaration cannot describe both lists of images, and why this one is still the renderer's.
        core& vk = this->vulkan_core;
        VkDescriptorSetLayout const set_layout = this->ssgi_temporal.set_layout();
        if (!this->pass_ready("ssgi_temporal") || set_layout == VK_NULL_HANDLE) {
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
        // THE COUNT COMES FROM THE DECLARATION, which is what makes it impossible for the pool and the layout to
        // disagree: this number and the layout the PASS generated in its create are both derived from
        // `ssgi_temporal_io` now. (They were two hand-written lists once, and the layer named the mismatch.)
        uint32_t const descriptors_per_set = render_resource::descriptor_counts_for(render_resource::ssgi_temporal_io, render_resource::ssgi_temporal_io.own_set).total();
        // ... and the REFLECTION's resolve, which is the SAME layout with a different list of images - which is
        // what makes it a second family and what a single declaration cannot express. Bindings 2 and 3 (the
        // surface's motion vectors and depth) are unused in mode 1 - its reprojection carries its own depth - but
        // every binding of the layout has to name a real view, so they carry the same ones the diffuse set uses;
        // binding 6 is the lobe's reprojection, which is what mode 1 actually reprojects by.
        std::array<VkImageView, 7> const spec_signature = {vk.gi_spec_image_views[0], vk.gi_spec_history_image_views[0], vk.velocity_image_views[0],
                                                           vk.gbuffer_depth_image_views[0], vk.gi_spec_resolve_image_views[0], vk.gbuffer_image_views[1][0],
                                                           vk.gi_spec_reproject_image_views[0]};
        auto const write_spec_sets = [this](uint32_t const image_index, std::span<VkDescriptorSet const> const sets) {
            std::array<VkImageView, 7> const views = {
                this->vulkan_core.gi_spec_image_views[image_index],
                this->vulkan_core.gi_spec_history_image_views[image_index],
                this->vulkan_core.velocity_image_views[image_index],
                this->vulkan_core.gbuffer_depth_image_views[image_index],
                this->vulkan_core.gi_spec_resolve_image_views[image_index],
                this->vulkan_core.gbuffer_image_views[1][image_index],
                this->vulkan_core.gi_spec_reproject_image_views[image_index]};
            // The write itself is GENERATED from the pass's declaration, exactly as the pass's own family does
            // it: the binding numbers, the descriptor types, the layouts and the sampler are the declaration's,
            // so this family cannot drift from the layout it shares with the pass.
            auto const written = bindings::write_set(this->vulkan_core.device, render_resource::ssgi_temporal_io, render_resource::ssgi_temporal_io.own_set, sets[0], views, {},
                                                     this->shared_samplers());
            if (!written) {
                utility::log("runtime: GI reflection descriptor set: {}", written.error());
            }
        };
        if (!this->ssgi_spec_temporal_family.ensure(vk.device, set_layout, static_cast<uint32_t>(image_count), 1u, descriptors_per_set, spec_signature, write_spec_sets)) {
            utility::log("runtime: GI reflection descriptor sets unavailable - this frame's reflection is not resolved");
        }
    }

    pass::ssgi_temporal_frame runtime::make_ssgi_denoise_frame() noexcept {
        // ONE field and ONE callback: the flag both signals blend by (read HERE, per image, so the owner does not
        // need the renderer's array), and the reflection's own recording (the documented exception - a declaration
        // cannot describe two signals in the same seven slots). The pass calls the callback at the END of its own
        // recording, so the chain stays contiguous.
        std::size_t const index = this->current_image_index;
        bool const history_valid = index < this->gi_history_valid.size() && this->gi_history_valid[index];
        return pass::ssgi_temporal_frame{.history_valid = history_valid,
                                         .record_reflection = &runtime::record_reflection,
                                         .owner = this};
    }

    // THE TEMPORAL RESOLVE'S RESOLVER IS GONE (S3.11) with the three before it: its seven own bindings (whose
    // PER-IMAGE views are the channel that was added for this pass), its two barrier images, its own pipeline and
    // the declaration's `half` extent are all the framework's to resolve, and the push block it used to be handed is
    // composed by the PASS from `io.constants`, the two extents and its own two blend constants. The two shared
    // per-image transitions it was handed as a callback are the FRAME's ordering rule and run between the chain's
    // two halves (see record_main_drawcalls); what is left of its frame is `record_reflection`, the one documented
    // exception.

    void runtime::record_reflection(void* const owner, VkCommandBuffer const command_buffer, bool const history_valid) {
        // THE REFLECTION's own accumulation, recorded for the temporal PASS: it is the renderer's because a
        // declaration cannot describe two signals in the same seven slots (see vulkan.pass.ssgi_temporal's
        // header), so the pass calls back here at the end of its own recording - after mode 0's hand-backs and
        // before the spatial filter's stage, which is exactly where this ran when the frame loop split the chain
        // around it. Its reprojection is the one the lobe published, its history is its own, and the spatial
        // filter's spec_weight lane is what keeps a stale accumulation out of a frame whose lobe did not run.
        runtime& self = *static_cast<runtime*>(owner);
        core& vk = self.vulkan_core;
        std::size_t const index = self.current_image_index;
        bool spec_resolved = false;
        if (self.ssgi_specular_active() && index < vk.gi_spec_resolve_images.size() && vk.gi_spec_history_images.size() == vk.gi_spec_resolve_images.size()) {
            VkDescriptorSet const spec_set = self.ssgi_spec_temporal_family.set(static_cast<uint32_t>(index), 0);
            if (spec_set != VK_NULL_HANDLE) {
                spec_resolved = self.record_ssgi_resolve_pass(command_buffer, spec_set, vk.gi_spec_resolve_images[index], vk.gi_spec_history_images[index], history_valid, 1.0f);
            }
        }
        // THE FRAME'S ANSWER, not the renderer's: the spatial filter's `spec_weight` lane is read from it, and the
        // filter resolves AFTER this callback (the temporal pass calls it at the end of its own recording), so the
        // value it copies out of `frame_facts` is this frame's. See frame_constants::gi_spec_resolved.
        self.frame_facts.gi_spec_resolved = spec_resolved;
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

        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->ssgi_temporal.pipeline_layout(), 0, 1, &set, 0, nullptr);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->ssgi_temporal.pipeline());

        ssgi_temporal_push_constants const push = {
            .history_valid = history_valid ? 1.0f : 0.0f,
            // The two weights are the temporal PASS's constants (see its header): the reflection shares that
            // pass's pipeline and shader, so both lanes have to be the same number or the two signals would be
            // denoised differently.
            .blend_static = pass::ssgi_temporal_pass::blend_static,
            .blend_min = pass::ssgi_temporal_pass::blend_min,
            .depth_scale = this->current_ubo.proj[2][2],
            .depth_offset = this->current_ubo.proj[3][2],
            // Which signal this dispatch resolves: 0.0 = the diffuse bounce, 1.0 = the reflection (see the
            // shader's `glossy`). One pipeline serves both, each with a set and a history of its own.
            .mode = mode,
            .unused1 = 0.0f,
            .unused2 = 0.0f,
            .gi_size = glm::vec4(static_cast<float>(gi_width), static_cast<float>(gi_height),
                                 static_cast<float>(vk.swap_chain_extent.width), static_cast<float>(vk.swap_chain_extent.height))};
        vkCmdPushConstants(command_buffer, this->ssgi_temporal.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

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

    bool runtime::ssgi_traced_active() const noexcept {
        return this->ssgi_active() && this->ssgi_ray_tracing && this->vulkan_core.ray_query_available && this->rt_top_levels.has_value();
    }

    bool runtime::ssgi_specular_active() const noexcept {
        // Hit shading is part of the predicate, and not as a quality preference: without the instance
        // table a glossy ray that LANDS on geometry cannot be shaded, so the pass would have nothing to
        // add for the only samples that make it more than the lighting stage's own term. The table is the
        // same switch the tracer uses, and the pipeline is the PASS's - so this predicate and the pass's own
        // feature are the same answer to "does the lobe run", which is what the tracer relies on.
        return this->ssgi_specular && this->ssgi_hit_shading && this->ssgi_traced_active() && this->pass_ready("ssgi_spec");
    }

    void runtime::set_ssgi_specular(bool const enabled, uint32_t const rays, float const radius) noexcept {
        this->ssgi_specular = enabled;
        // The two values are the PASS's (one reader each), so they are forwarded with their clamps - and the
        // argument for those clamps stays here, where the measurements that chose them are recorded:
        // One is the feature's definition and the low end of the noise/cost trade; the upper bound is a
        // cost guard, not a quality claim - the pass shades every hit it finds, which is the expensive
        // part of this chain.
        // A reach, not a quality knob: below it the reflection falls back to the sky, and past the knee the
        // curve is flat in BOTH effect and cost (measured: -0.907 / -2.126 / -2.350 at 0.12 / 0.5 / 2.0, and
        // the `gi` interval 1.28 / 2.22 / 2.21 ms). The FLOOR is deliberately low enough to be useless: a
        // reach whose ray length lands at or below the ray query's own tmin (a fixed 0.01 world units) makes
        // every sample answer with the environment - the identity configuration the L2.3 acceptance is read
        // in - and a floor above that would not be expressible on a small scene (0.01 scene radii of the
        // helmet's 1.64 is 0.0164 units, i.e. just past tmin, which is why the floor is 0.001 and not 0.01).
        this->ssgi_spec.set_reach(radius, rays);
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
        } else if (enabled && !this->pass_ready("rt_shadow")) {
            this->warn_missing_feature("ssgi", "GI rays are marched, not traced: the ray-traced pipelines were not created");
        }
    }

    void runtime::set_ssgi_bounce(float const gain) noexcept {
        // The knob stops at one: above it the geometric series a diffuse loop forms is not guaranteed to
        // converge (the surfaces' albedos approach one), and the failure mode is a frame that gets
        // brighter every frame rather than a visibly wrong one. The clamp lives with the value now (the
        // TRACER's, which is the pass that pushes the gain).
        this->ssgi_trace.set_bounce(gain);
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
        // The filter's width is the PASS's parameter and the clamp lives with it (the rule the TAA resolve's blend
        // weights settled), so this setter forwards rather than keeping a second copy of the value.
        this->ssgi_spatial.set_sigma(sigma);
    }

    void runtime::set_ssgi_upsample(bool const enabled) noexcept {
        // A consequence of the composite's push block, which is the only place it is read - so the pass owns it and
        // this setter forwards (the same rule as set_taa's two blend weights).
        this->post_composite.set_gi_upsample(enabled);
    }

    // ---- the world-space probe cache (see shaders/gi_probe.comp) ----
    void runtime::set_ssgi_probes(bool const enabled, float const rate, uint32_t const rounds, float const gain) noexcept {
        this->gi_probe_enabled = enabled;
        // The rate is the loop gain of the grid's own cycle (tracer -> resolve -> grid -> tracer): a rate
        // of 1 would make the grid an immediate echo of the frame that read it, so it stops below that. It is
        // the PASS's value and the pass clamps it (see gi_probe_pass::set_rate).
        this->gi_probe.set_rate(rate);
        this->gi_probe_rounds = std::clamp(rounds, 0u, 4u);
        // The dispatch count is the pass's: it owns the update sequence, so the renderer hands it the one
        // number of that sequence that is configuration rather than structure.
        this->gi_probe.set_rounds(this->gi_probe_rounds);
        // The sign is the cache's DIRECTION A/B rather than a mistake: |gain| is the gain, and a negative
        // value looks the cache up along the opposite direction of the ray - the same cell, the other side.
        // It is clamped rather than rejected because it is a documented measurement setting. The value is the
        // TRACER's (it is the lane that weighs the cache against the environment fallback), so it is forwarded.
        this->ssgi_trace.set_probe_gain(gain);
        if (enabled && !this->ssgi_on) {
            this->warn_missing_feature("ssgi", "the probe cache has no effect: it is injected from the screen-space GI chain, which is off");
        } else if (enabled && !this->pass_ready("gi_probe")) {
            this->warn_missing_feature("ssgi", "the probe cache has no effect: it has not built its pipeline (see the startup log)");
        }
    }

    bool runtime::gi_probe_active() const noexcept {
        // The cache is deposited from the screen-space chain's resolved image and lives on the deferred
        // path's G-buffer, so it needs both of those, plus the pass having built what it records with: a
        // build without the cache keeps the tracer's environment-probe fallback and changes nothing else.
        return this->gi_probe_enabled && this->pass_ready("gi_probe") && this->ssgi_active();
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
        // ... and what the passes may NAME, before any of them is created: the registry the pass filter answers
        // `resource()` from (see publish_pass_resources). It is the renderer's half of the channel; the passes'
        // half is that they ask for what their own declaration lists instead of being handed it.
        this->publish_pass_resources();
        // THE GI CHAIN's ORDER, which is data now rather than the line order of the frame loop that records it:
        // each stage reads what the one before it wrote (the lobe corrects the trace, the denoise accumulates it,
        // the filter smooths the accumulation and is what the composite samples). TWO chains rather than one,
        // because the frame has an ordering rule to run BETWEEN the lobe and the temporal resolve - it publishes
        // the G-buffer depth and the motion-vector target that the resolve is the first sampler of (see
        // record_main_drawcalls) - and a frame rule cannot run from outside a chain.
        this->gi_trace_chain.add(this->ssgi_trace);
        this->gi_trace_chain.add(this->ssgi_spec);
        this->gi_denoise_chain.add(this->ssgi_temporal);
        this->gi_denoise_chain.add(this->ssgi_spatial);
        pass::pass_context const build = this->make_pass_context();
        // ONE CREATE STEP OVER EVERY PASS, in the order the OWNING chain holds them (see the member block in the
        // header): `passes` owns the ten passes this renderer has, so its `init` IS the whole create step. A pass
        // that could not build itself reports its own name in `rejected` and stays INACTIVE (its feature predicate
        // is false), which is what makes a startup failure a log line rather than a broken frame.
        pass::run_report const created = this->passes.init(build);
        if (!created.rejected.empty()) {
            utility::log("pass '{}': its declaration was refused by the validator, so it does not run", created.rejected);
        }
        // THE TWO JOBS, and they are created HERE rather than by the application: neither is a frame pass (see
        // their headers - one runs once inside the structure-build command buffer, the other per frame from a
        // caster list), but both are GPU-owning objects constructed from the same context, so they belong in the
        // same step. Before the pass filter existed each had its own entry point in this class, because a pass
        // could not name a resource the renderer owns; now it asks (see publish_pass_resources).
        if (auto const created = this->mask_bake.create(build); !created) {
            utility::log("alphaMode MASK bake unavailable: {} (masked geometry stays solid to a ray)", created.error());
        }
        if (auto const created = this->compute_skin.create(build); !created) {
            utility::log("skinned shadow refit unavailable: {} (a traced shadow keeps the bind pose)", created.error());
        }
    }

    render_resource::shared::sampler_set runtime::shared_samplers() const noexcept {
        // The six samplers a declaration chooses between, as handles. One place, so that two passes cannot end
        // up with two different ideas of "the post sampler".
        // THE SAMPLERS ARE THE DEVICE ROOT'S (core::create_samplers): a sampler has no per-frame state and no owner
        // among the passes, so this function is now a READ of the handles rather than the place that made them - and
        // it stays the single place the renderer maps them onto the declaration layer's hints.
        core const& vk = this->vulkan_core;
        return {.gbuffer = *vk.gbuffer_sampler,
                .probe_grid = *vk.gi_probe_sampler,
                .taa = *vk.taa_sampler,
                .post = *vk.post_sampler,
                .nearest = *vk.post_nearest_sampler,
                .shadow = *vk.shadow_sampler,
                // The bindless texture array's sampler, which only hand-written set code used until the MASK
                // bake had to write the scene layout's binding 1 itself (see sampler_set's doc).
                .textures = *vk.texture_sampler};
    }

    std::span<unsigned char const> runtime::registered_shader(std::string_view const name) const noexcept {
        for (auto const& [registered_name, registered_bytes] : this->registered_shaders) {
            if (registered_name == name) {
                return registered_bytes;
            }
        }
        return {}; // a pass whose shader was never registered builds nothing and says so
    }

    pass::pass_context runtime::make_pass_context() noexcept {
        // THE ONE CONSTRUCTION SITE for a pass's create-time context, and it is a method rather than a block
        // because there were two of them: `create_passes()` built one for the stages, and the two jobs that are
        // not frame passes (the MASK bake, the compute-skinning job) each built their own copy. A second copy of
        // this struct is how a per-pass entry point per job appears, which is what the pass filter exists to
        // remove - so there is one builder now, and everything that constructs a pass uses it.
        return pass::pass_context{
            .device = this->vulkan_core.device,
            .samplers = this->shared_samplers(),
            .shared_set_layout = [](void* owner, uint32_t const set) {
                // THREE shared sets now, and each one is there because a pass needs a layout it does not own:
                // 0 = the scene set (the tracer, the lobe, the spatial filter and the raytraced shadow bind it),
                // 1 = the G-buffer set (the same four, plus the debug view) and 2 = the POST set (the composite,
                // the bloom chain and the FXAA pass). A pass asking for any other index gets "none", which makes
                // it build nothing and say so.
                runtime* const self = static_cast<runtime*>(owner);
                if (set == 0u) {
                    return self->vulkan_core.scene_descriptor_set_layout;
                }
                if (set == 1u) {
                    return self->gbuffer_debug_view.set_layout();
                }
                return set == 2u ? self->post_composite.set_layout() : VkDescriptorSetLayout{VK_NULL_HANDLE}; },
            // ... and the SCENE pipeline layout, which the shadow pass builds its pipeline against (its draw is a subset of the scene's and its per-cascade push goes through the same layout): .shader above is the shape, one callback per thing a pass cannot own.
            .shared_pipeline_layout = [](void* owner) { return static_cast<runtime*>(owner)->vulkan_core.scene_pipeline_layout; },
            .shader = [](void* owner, std::string_view const name) { return static_cast<runtime*>(owner)->registered_shader(name); },
            // The surface's format: a SESSION-STABLE device fact a pipeline that renders into the swapchain must
            // be created with (see pass_context). The post chain needs it today; the graphics passes being
            // extracted need it tomorrow.
            .swap_chain_image_format = this->vulkan_core.swap_chain_image_format,
            // ... and the DEPTH format, which the shadow pass`s pipeline needs (it has a depth attachment and no
            // colour one): the same kind of session-stable device fact, and the second one a context carries.
            .depth_format = this->vulkan_core.depth_format,
            // The two channels a pass uses to build what it owns over resources the RENDERER holds: the handles
            // of the resources this runtime published (`pass_resources`), and a set from the core's pool for a
            // layout it handed out above. Both forward to the filter, which is the object that knows what a pass
            // may reach - the context itself stays a plain struct of callbacks, so the framework still does not
            // depend on `vulkan.core`.
            .resource =
                [](void* owner, render_resource::resource_id const id, uint32_t const element) {
                    runtime* const self = static_cast<runtime*>(owner);
                    resource_handles const handles = self->pass_resources.resource(id, element);
                    return pass::resolved_binding{.view = handles.view, .buffer = handles.buffer, .image = handles.image};
                },
            .descriptor_set = [](void* owner, VkDescriptorSetLayout const layout) { return static_cast<runtime*>(owner)->pass_resources.make_descriptor_set(layout); },
            .frames_in_flight = vulkan::core::MAX_FRAMES_IN_FLIGHT,
            .owner = this,
        };
    }

    void runtime::publish_pass_resources() {
        // WHAT THE RENDERER PUBLISHES FOR ITS PASSES, in the declaration layer's own vocabulary: a pass asks for
        // `resource_id::material_table` or `resource_id::skin_matrices` + a slot, not for a member of this class.
        // Everything here is SESSION-STABLE by the filter's contract (the material table and the texture array
        // are created once and only rewritten; the skin matrix buffers are created once and rewritten per slot),
        // which is what makes them safe for a pass to bind into a descriptor set it writes once at create time.
        if (auto const* const materials = this->vulkan_core.vma.get_buffer_detail(this->material_buffer.handle()); materials != nullptr && this->material_mapped != nullptr) {
            this->pass_resources.register_resource(render_resource::resource_id::material_table, 0, resource_handles{.buffer = materials->buffer});
        }
        if (!this->owned_texture_views.empty()) {
            this->pass_resources.register_resource(render_resource::resource_id::scene_textures, 0, resource_handles{.view = *this->owned_texture_views[0]});
        }
        for (uint32_t slot = 0; slot < this->skin_buffers.size(); ++slot) {
            auto const* const detail = this->vulkan_core.vma.get_buffer_detail(this->skin_buffers[slot].handle());
            if (detail != nullptr) {
                this->pass_resources.register_resource(render_resource::resource_id::skin_matrices, slot, resource_handles{.buffer = detail->buffer});
            }
        }
    }

    void runtime::update_frame_constants() noexcept {
        // THE ONE PLACE THE FACTS TYPE AND THE UBOs HAVE TO AGREE. `vulkan.frame_constants` deliberately does not
        // include `camera_ubo`/`light_ubo` (the pass framework must not depend on vulkan.primitive, which imports
        // core), so the fields are copied here - one site to check, instead of a type relationship spread across
        // two modules that only a reader of both would notice breaking.
        this->frame_facts.view = this->current_ubo.view;
        this->frame_facts.proj = this->current_ubo.proj; // JITTERED: the geometry is sampled at these offsets
        this->frame_facts.view_proj_unjittered = this->current_ubo.view_proj_unjittered;
        this->frame_facts.inv_view_proj = this->current_inv_view_proj;
        this->frame_facts.camera_pos = glm::vec3(this->current_ubo.camera_pos);
        this->frame_facts.scene_center = this->shadow_scene_center;
        this->frame_facts.scene_radius = this->scene_radius;
        // The sun, normalized for the same reason the probe cache's resolve normalizes it (the shader wants a
        // direction, the UBO's own lane stays as the app set it). A zero direction - nothing has set a light yet
        // - is kept as zero rather than turned into a NaN by normalize().
        glm::vec3 const sun = glm::vec3(this->light_state.light_dir);
        this->frame_facts.light_dir = glm::dot(sun, sun) > 0.0f ? glm::vec4(glm::normalize(sun), 0.0f) : glm::vec4(0.0f);
        // ... and the renderer's settings that MORE THAN ONE pass reads (see render_settings for the rule): the
        // post chain's exposure and bloom weights, FXAA's two thresholds, and the GI upsample's silhouette test.
        // The members stay the app-facing ones (`set_exposure` and friends) - this is the frame's copy of them.
        this->frame_facts.settings.exposure = this->exposure_scale;
        this->frame_facts.settings.bloom_intensity = this->bloom_intensity;
        this->frame_facts.settings.bloom_threshold = this->bloom_threshold;
        this->frame_facts.settings.fxaa_subpixel = this->fxaa_subpixel;
        this->frame_facts.settings.fxaa_edge_threshold = this->fxaa_edge_threshold;
        this->frame_facts.settings.gi_depth_sigma = this->gi_spatial_depth_sigma;
        this->frame_facts.settings.gi_normal_power = this->gi_spatial_normal_power;
    }

    void runtime::publish_frame_resources() {
        core const& vk = this->vulkan_core;
        pass::resource_table& table = this->frame_resources;
        table.clear();

        // One per-swapchain-image family: published as the run of views and images the core already holds (see
        // resource_table::publish_family), so the per-image channel is that same run rather than a second copy of
        // it. `element` is the family's own index (the G-buffer's third target, the bloom chain's level 2).
        auto const family = [&table](render_resource::resource_id const id, uint32_t const element, std::span<VkImageView const> views, std::span<VkImage const> images) {
            table.publish_family(id, element, views, images);
        };
        // One resource that exists once (a device-wide image): the instance is still the schema's (0 for
        // device-wide, the slot for a per-frame-slot buffer that happens to exist once).
        auto const single = [&table](render_resource::resource_id const id, uint32_t const element, uint32_t const instance, pass::resolved_binding const binding) {
            table.publish(id, element, instance, binding);
        };
        // One buffer: the RAII wrapper holds a handle and the descriptor needs the VkBuffer behind it, so the
        // lookup goes through vma exactly where the renderer's own binding writes do.
        auto const buffer = [this, &table](render_resource::resource_id const id, uint32_t const instance, vk_buffer const& owned) {
            auto const* const detail = this->vulkan_core.vma.get_buffer_detail(owned.handle());
            if (detail == nullptr) {
                return;
            }
            table.publish(id, 0, instance, pass::resolved_binding{.buffer = detail->buffer});
        };
        auto const image = [this](vk_image const& owned) -> pass::resolved_binding {
            auto const* const detail = this->vulkan_core.vma.get_image_detail(owned.handle());
            return detail == nullptr ? pass::resolved_binding{} : pass::resolved_binding{.image = detail->image};
        };

        // ---- the render-target chain: the image families core owns ----
        family(render_resource::resource_id::swapchain_image, 0, vk.swap_chain_image_views, vk.swap_chain_images);
        family(render_resource::resource_id::hdr, 0, vk.hdr_image_views, vk.hdr_images);
        family(render_resource::resource_id::ldr, 0, vk.ldr_image_views, vk.ldr_images);
        family(render_resource::resource_id::gbuffer_depth, 0, vk.gbuffer_depth_image_views, vk.gbuffer_depth_images);
        family(render_resource::resource_id::velocity, 0, vk.velocity_image_views, vk.velocity_images);
        family(render_resource::resource_id::taa_history, 0, vk.taa_history_image_views, vk.taa_history_images);
        family(render_resource::resource_id::gi_trace, 0, vk.gi_image_views, vk.gi_images);
        family(render_resource::resource_id::gi_resolve, 0, vk.gi_resolve_image_views, vk.gi_resolve_images);
        family(render_resource::resource_id::gi_history, 0, vk.gi_history_image_views, vk.gi_history_images);
        family(render_resource::resource_id::gi_spatial, 0, vk.gi_spatial_image_views, vk.gi_spatial_images);
        family(render_resource::resource_id::gi_spec_trace, 0, vk.gi_spec_image_views, vk.gi_spec_images);
        family(render_resource::resource_id::gi_spec_reproject, 0, vk.gi_spec_reproject_image_views, vk.gi_spec_reproject_images);
        family(render_resource::resource_id::gi_spec_resolve, 0, vk.gi_spec_resolve_image_views, vk.gi_spec_resolve_images);
        family(render_resource::resource_id::gi_spec_history, 0, vk.gi_spec_history_image_views, vk.gi_spec_history_images);
        for (std::size_t target = 0; target < vk.gbuffer_image_views.size() && target < vk.gbuffer_images.size(); ++target) {
            family(render_resource::resource_id::gbuffer_targets, static_cast<uint32_t>(target), vk.gbuffer_image_views[target], vk.gbuffer_images[target]);
        }
        for (std::size_t level = 0; level < vk.bloom_image_views.size() && level < vk.bloom_images.size(); ++level) {
            family(render_resource::resource_id::bloom, static_cast<uint32_t>(level), vk.bloom_image_views[level], vk.bloom_images[level]);
        }
        // `scene_color` is an ALIAS rather than a family of its own: the scene-side passes write the TAA input
        // while the resolve runs and the HDR target otherwise (see scene_target_view), so WHAT THE ID MEANS is
        // decided here, once per frame, in the same place that answers it for the resolvers - which is also why
        // the table is refreshed per frame rather than per generation.
        std::size_t const scene_images = std::min({vk.scene_color_image_views.size(), vk.scene_color_images.size(), vk.hdr_image_views.size(), vk.hdr_images.size()});
        for (std::size_t i = 0; i < scene_images; ++i) {
            single(render_resource::resource_id::scene_color, 0, static_cast<uint32_t>(i),
                   pass::resolved_binding{.view = this->scene_target_view(static_cast<uint32_t>(i)), .buffer = VK_NULL_HANDLE, .image = this->scene_target_image(static_cast<uint32_t>(i))});
        }

        // ---- per FRAME SLOT: the ray-traced visibility image, the shadow map's cascades, the buffers ----
        for (std::size_t slot = 0; slot < vk.rt_shadow_image_views.size() && slot < vk.rt_shadow_images.size(); ++slot) {
            single(render_resource::resource_id::rt_shadow_visibility, 0, static_cast<uint32_t>(slot),
                   pass::resolved_binding{.view = vk.rt_shadow_image_views[slot], .buffer = VK_NULL_HANDLE, .image = vk.rt_shadow_images[slot]});
        }
        // The shadow map's family elements are the CASCADES (see render_resource::shadow_io): the image is one
        // layered depth array per slot and the pass renders one layer at a time, so every layer the image
        // currently HAS is published, by cascade index, with the image behind it for the layer's own barrier.
        for (std::size_t slot = 0; slot < this->shadow_images.size() && slot < this->shadow_layer_views.size(); ++slot) {
            auto const* const detail = this->vulkan_core.vma.get_image_detail(this->shadow_images[slot].handle());
            if (detail == nullptr) {
                continue;
            }
            for (std::size_t layer = 0; layer < this->shadow_layer_views[slot].size(); ++layer) {
                single(render_resource::resource_id::shadow_map, static_cast<uint32_t>(layer), static_cast<uint32_t>(slot),
                       pass::resolved_binding{.view = *this->shadow_layer_views[slot][layer], .buffer = VK_NULL_HANDLE, .image = detail->image});
            }
        }
        // The per-slot buffers, each into its own instance: a frame in flight reads its own copy, which is the
        // whole reason those resources exist per slot (see the member docs).
        uint32_t const slots = static_cast<uint32_t>(vulkan::core::MAX_FRAMES_IN_FLIGHT);
        for (uint32_t slot = 0; slot < slots; ++slot) {
            if (slot < this->camera_buffers.size()) {
                buffer(render_resource::resource_id::camera_ubo, slot, this->camera_buffers[slot]);
            }
            if (slot < this->light_buffers.size()) {
                buffer(render_resource::resource_id::light_ubo, slot, this->light_buffers[slot]);
            }
            if (slot < this->motion_buffers.size()) {
                buffer(render_resource::resource_id::motion_vectors, slot, this->motion_buffers[slot]);
            }
            if (slot < this->skin_buffers.size()) {
                buffer(render_resource::resource_id::skin_matrices, slot, this->skin_buffers[slot]);
            }
            if (slot < this->morph_buffers.size()) {
                buffer(render_resource::resource_id::morph_targets, slot, this->morph_buffers[slot]);
            }
            if (slot < this->cluster_count_buffers.size()) {
                buffer(render_resource::resource_id::cluster_counts, slot, this->cluster_count_buffers[slot]);
            }
            if (slot < this->cluster_index_buffers.size()) {
                buffer(render_resource::resource_id::cluster_indices, slot, this->cluster_index_buffers[slot]);
            }
            // ... and the two that exist ONCE while their schema scope still says per-frame-slot: the material
            // table and the instance transforms are rewritten in place rather than per slot, which is why the
            // same handle is published for every instance.
            buffer(render_resource::resource_id::material_table, slot, this->material_buffer);
            buffer(render_resource::resource_id::instance_table, slot, this->instance_buffer);
        }

        // ---- device-wide: the probe grid's eight elements, its geometry, the cubes and the textures ----
        for (std::size_t element = 0; element < vk.gi_probe_image_views.size() && element < vk.gi_probe_images.size(); ++element) {
            single(render_resource::resource_id::probe_grid, static_cast<uint32_t>(element), 0,
                   pass::resolved_binding{.view = vk.gi_probe_image_views[element], .buffer = VK_NULL_HANDLE, .image = vk.gi_probe_images[element]});
        }
        if (!vk.gi_probe_surface_image_views.empty() && !vk.gi_probe_surface_images.empty()) {
            single(render_resource::resource_id::probe_surface, 0, 0,
                   pass::resolved_binding{.view = vk.gi_probe_surface_image_views[0], .buffer = VK_NULL_HANDLE, .image = vk.gi_probe_surface_images[0]});
        }
        if (!vk.furnace_cube_views.empty() && !vk.furnace_cube_images.empty()) {
            single(render_resource::resource_id::furnace_cube, 0, 0,
                   pass::resolved_binding{.view = vk.furnace_cube_views[0], .buffer = VK_NULL_HANDLE, .image = vk.furnace_cube_images[0]});
        }
        // The white fallback and the array it is element 0 of. The array is BINDLESS (one binding, N descriptors),
        // which the declaration vocabulary cannot index element by element yet - so what is published is the one
        // element that always exists, which is also the one every declaration can name (element 0).
        if (this->white_texture_index < this->owned_textures.size() && this->white_texture_index < this->owned_texture_views.size()) {
            pass::resolved_binding white = image(this->owned_textures[this->white_texture_index]);
            white.view = *this->owned_texture_views[this->white_texture_index];
            single(render_resource::resource_id::white_texture, 0, 0, white);
            single(render_resource::resource_id::scene_textures, 0, 0, white);
        }
        // The IBL triple, in the order set_ibl uploads it: prefiltered environment, irradiance, BRDF LUT.
        if (this->ibl_views.size() >= 3 && this->ibl_images.size() >= 3) {
            pass::resolved_binding env = image(this->ibl_images[0]);
            env.view = *this->ibl_views[0];
            pass::resolved_binding irradiance = image(this->ibl_images[1]);
            irradiance.view = *this->ibl_views[1];
            pass::resolved_binding lut = image(this->ibl_images[2]);
            lut.view = *this->ibl_views[2];
            single(render_resource::resource_id::ibl_env, 0, 0, env);
            single(render_resource::resource_id::ibl_irradiance, 0, 0, irradiance);
            single(render_resource::resource_id::brdf_lut, 0, 0, lut);
        }
        // NOT published: `top_level_structure`. It is an acceleration structure, and `resolved_binding` carries
        // the three handles a set write and a barrier take - a device address is neither. Whoever needs it asks
        // the runtime for it today (see the scene set's binding 16), and a table entry that could not carry the
        // handle would be a claim this type cannot make.
    }

    void runtime::verify_resource_table(pass::frame_pass const& pass, pass::resolved_io const& io) {
        render_resource::pass_io const& decl = pass.io();
        pass::resource_table const& table = this->frame_resources;
        uint32_t checked = 0;
        uint32_t mismatched = 0;
        auto const as_pointer = [](auto const handle) { return static_cast<void const*>(handle); };
        auto const check = [&](render_resource::resource_id const id, uint32_t const element, pass::resolved_binding const& resolved, std::string_view const channel) {
            render_resource::resource_info const* const info = render_resource::find(id);
            if (info == nullptr) {
                return;
            }
            ++checked;
            pass::resolved_binding const published = table.find(id, element, pass::instance_for(info->scope, io.frame));
            if (published.view == resolved.view && published.buffer == resolved.buffer && published.image == resolved.image) {
                return;
            }
            ++mismatched;
            // Bounded, and only while the check's window is open: a wrong entry is a finding to fix, not a reason
            // to fill the log of every frame.
            if (this->resource_check_frames < 3 && this->resource_check_mismatched + mismatched <= 10) {
                utility::log("resource table: pass '{}' {} (resource {}, element {}) resolved as [view {}, buffer {}, image {}] but published as [view {}, buffer {}, image {}]",
                             decl.name, channel, static_cast<int>(id), element,
                             as_pointer(resolved.view), as_pointer(resolved.buffer), as_pointer(resolved.image),
                             as_pointer(published.view), as_pointer(published.buffer), as_pointer(published.image));
            }
        };
        // The own bindings, indexed by their own binding number: the validator requires those to be contiguous
        // from zero, so `own[binding]` IS this binding's handle (see resolved_io).
        for (render_resource::pass_binding const& binding : decl.bindings) {
            if (binding.owner != render_resource::set_owner::own || binding.binding >= io.own.size()) {
                continue;
            }
            check(binding.resource, binding.element, io.own[binding.binding], "own binding");
        }
        // The targets, walked the way `resolve_declaration` walks them: a target claiming a RUN of elements
        // (`render_target::count`) expands to one slot per element, so the declaration's t-th entry is NOT always
        // the t-th slot in `io.targets` - the shadow pass's four cascade layers are one entry there and four here.
        uint32_t target_slot = 0;
        for (render_resource::render_target const& target : decl.targets) {
            for (uint16_t i = 0; i < target.count && target_slot < io.targets.size(); ++i, ++target_slot) {
                check(target.resource, static_cast<uint32_t>(target.element) + i, io.targets[target_slot], "render target");
            }
        }
        for (std::size_t i = 0; i < decl.barrier_images.size() && i < io.barrier_images.size(); ++i) {
            check(decl.barrier_images[i].resource, decl.barrier_images[i].element, io.barrier_images[i], "barrier image");
        }
        for (std::size_t i = 0; i < decl.barrier_buffers.size() && i < io.barrier_buffers.size(); ++i) {
            check(decl.barrier_buffers[i].resource, decl.barrier_buffers[i].element, io.barrier_buffers[i], "barrier buffer");
        }
        this->resource_check_checked += checked;
        this->resource_check_mismatched += mismatched;
        // One line per pass, inside the same window: it says WHICH passes the running scenario put under the
        // check, which is what tells a reader which resolvers the table has been proven against and which ones a
        // different scenario has to exercise.
        if (this->resource_check_frames < 3 && checked > 0 &&
            std::find(this->resource_check_passes.begin(), this->resource_check_passes.end(), &pass) == this->resource_check_passes.end()) {
            this->resource_check_passes.push_back(&pass);
            utility::log("resource table: verified {} declaration handle(s) for pass '{}'", checked, decl.name);
        }
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

    // THE SCENE PASS'S RESOLVER IS GONE (S3): its six targets, its shared set and its extent are resolved from
    // its own declaration against the frame's resource table (see pass::resolve_declaration), and the gate that
    // used to be the first line here - "there is no surface pipeline, or no target generation to draw into" - is
    // now the pass's FEATURE ("scene", answered by `gbuffer_pass_active()` in `feature_active`). The generation
    // half of it is what the table already says: a frame whose targets do not exist has no entry to resolve, so
    // the pass does not run. What is NOT resolved here anymore is `out.own = {}` and `out.push = {}` either -
    // this pass declares neither, and the generic resolver leaves both empty for exactly that case.

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

    // THE TRANSPARENT PASS'S RESOLVER IS GONE TOO (S3), for the same reason as the scene pass's: two declared
    // targets, the shared scene set and a full-frame extent, all of them resolved from the declaration. Its gate
    // - "nothing blended this frame" - is the pass's FEATURE ("transparent", answered by `frame_transparent`
    // being empty in `feature_active`), which is the frame's content rather than the declaration's shape.

    // =============================================================================================
    // THE CHAIN OWNER'S SEAM (see runtime::frame_services and docs/pass_chain_plan.md)
    // =============================================================================================
    //
    // Every builder below reads the RUNTIME's data (the culling's leaves, the structure phase's secondaries, this
    // frame's decisions) and returns a frame the OWNER of the pass hands over. That is the whole split: the
    // runtime knows what a frame IS, the owner knows which pass wants it.

    pass::cluster_frame runtime::make_cluster_frame() noexcept {
        // The sort's only input beyond its own dispatch: the grid the light culling produced this frame, which is
        // the same number the light UBO's cluster_grid carries.
        return pass::cluster_frame{.cluster_count = this->cluster_tiles_x * this->cluster_tiles_y * vulkan::cluster_slice_count};
    }

    pass::shadow_frame runtime::make_shadow_frame() noexcept {
        // The per-cascade secondaries live in per-slot pairs (a command pool is not thread safe), so they are NOT
        // contiguous: they are gathered into the scratch array the returned span points at, and that array is a
        // member because a span over a local would dangle the moment this function returned.
        uint32_t const slot = static_cast<uint32_t>(this->vulkan_core.current_frame);
        uint32_t const cascades = std::clamp(this->shadow_cascades, 1u, vulkan::max_shadow_cascades);
        for (uint32_t cascade = 0; cascade < cascades; ++cascade) {
            this->shadow_secondaries_scratch[cascade] = *this->shadow_recording[slot][cascade].second;
        }
        return pass::shadow_frame{.record_cascade = &runtime::record_shadow_cascade,
                                  .run_tasks = &runtime::run_shadow_tasks,
                                  .owner = this,
                                  .cascades = std::span<VkCommandBuffer const>(this->shadow_secondaries_scratch.data(), cascades),
                                  .map_size = this->shadow_map_size};
    }

    pass::deferred_frame runtime::make_deferred_frame() const noexcept {
        return pass::deferred_frame{.gi_replaces_ambient = this->ssgi_traced_active()};
    }

    pass::composite_frame runtime::make_composite_frame() noexcept {
        return pass::composite_frame{
            // The overlay has no load op of its own, so it is drawn inside whichever instance is the frame's LAST
            // writer: the composite when FXAA is off, the FXAA pass when it runs.
            .after_draw = this->post_fxaa_active() ? nullptr : &runtime::draw_overlay_after,
            .owner = this,
            .write_ldr = this->post_fxaa_active(),
            // ... and the same "what runs this frame" answer the bloom chain's own gate uses, so the weight the
            // pass adds and the gate cannot disagree about whether there is a bloom sum.
            .suppress_bloom = this->active_features().gbuffer_debug,
        };
    }

    pass::fxaa_frame runtime::make_fxaa_frame() noexcept {
        return pass::fxaa_frame{.after_draw = &runtime::draw_overlay_after, .owner = this};
    }

    pass::ssgi_spatial_frame runtime::make_ssgi_spatial_frame() const noexcept {
        return pass::ssgi_spatial_frame{.traced_oracle = this->ssgi_traced_active()};
    }

    runtime::frame_services runtime::make_frame_services(VkCommandBuffer const command_buffer) noexcept {
        // Capture-less lambdas: each one casts the owner back and calls the builder it names, so the owner never
        // sees a member function of this class - it sees the frame.
        return frame_services{
            .owner = this,
            .cmd = command_buffer,
            .image_index = static_cast<uint32_t>(this->current_image_index),
            .make_cluster_frame = [](void* owner) { return static_cast<runtime*>(owner)->make_cluster_frame(); },
            .make_shadow_frame = [](void* owner) { return static_cast<runtime*>(owner)->make_shadow_frame(); },
            .make_scene_frame = [](void* owner) { return static_cast<runtime*>(owner)->make_scene_frame(); },
            .make_transparent_frame = [](void* owner) { return static_cast<runtime*>(owner)->make_transparent_frame(); },
            .make_deferred_frame = [](void* owner) { return static_cast<runtime*>(owner)->make_deferred_frame(); },
            .make_composite_frame = [](void* owner) { return static_cast<runtime*>(owner)->make_composite_frame(); },
            .make_fxaa_frame = [](void* owner) { return static_cast<runtime*>(owner)->make_fxaa_frame(); },
            .make_ssgi_trace_frame = [](void* owner) { return static_cast<runtime*>(owner)->make_ssgi_trace_frame(); },
            .make_ssgi_spec_frame = [](void* owner) { return static_cast<runtime*>(owner)->make_ssgi_spec_frame(); },
            .make_ssgi_denoise_frame = [](void* owner) { return static_cast<runtime*>(owner)->make_ssgi_denoise_frame(); },
            .make_ssgi_spatial_frame = [](void* owner) { return static_cast<runtime*>(owner)->make_ssgi_spatial_frame(); },
            .ensure_gbuffer_targets_sampled =
                [](void* owner, VkCommandBuffer cmd, uint32_t image) { return static_cast<runtime*>(owner)->ensure_gbuffer_targets_sampled(cmd, image); },
            .ensure_gbuffer_depth_sampled =
                [](void* owner, VkCommandBuffer cmd, uint32_t image) { return static_cast<runtime*>(owner)->ensure_gbuffer_depth_sampled(cmd, image); },
            .ensure_velocity_sampled = [](void* owner, VkCommandBuffer cmd, uint32_t image) { return static_cast<runtime*>(owner)->ensure_velocity_sampled(cmd, image); },
            .require_velocity_publish = [](void* owner, uint32_t image) { static_cast<runtime*>(owner)->require_velocity_publish(image); },
            .feature_active = [](void* owner, std::string_view name) { return static_cast<runtime const*>(owner)->feature_active(name); },
        };
    }

    void runtime::set_chain_wiring(chain_wiring const wiring) noexcept {
        this->wiring_ = wiring;
    }

    void runtime::prepare_stage(std::string_view const stage, VkCommandBuffer const command_buffer) {
        if (this->wiring_.prepare == nullptr) {
            return; // no owner: every pass is handed an empty frame, and a pass with one records nothing
        }
        this->wiring_.prepare(this->wiring_.owner, this->make_frame_services(command_buffer), stage);
    }

    void runtime::collect_stage(std::string_view const stage) {
        if (this->wiring_.collect == nullptr) {
            return;
        }
        frame_results results = {};
        this->wiring_.collect(this->wiring_.owner, stage, results);
        // WHAT THE FRAME LOOP DECIDES ON, once per stage that reports: the composite's GI weight is a frame
        // CONSTANT (the composite reads it while recording), and the two flags are the renderer's per-image
        // bookkeeping.
        if (results.gi_resolved) {
            this->frame_facts.gi_resolved = true;
        }
        if (results.gi_temporal_resolved && this->current_image_index < this->gi_history_valid.size()) {
            this->gi_history_valid[this->current_image_index] = true;
        }
        if (results.taa_wrote_history && this->current_image_index < this->image_view_proj.size()) {
            this->image_view_proj[this->current_image_index] = this->current_ubo.view_proj_unjittered;
        }
    }

    void runtime::require_velocity_publish(uint32_t const image_index) {
        if (image_index < this->velocity_written.size()) {
            this->velocity_written[image_index] = false;
        }
    }

    // `runtime::pass_extent` IS GONE (S3.12), and it is the payoff of the last resolver: it was the bridge from a
    // declaration to a size the renderer owns, and it existed because a hand-written resolver had to apply the
    // declaration's own `extent_rule` itself. The framework applies it now (`pass::resolve_extent`, over the
    // `extent_of` callback below), so the bridge had no callers left - the rule is in ONE place, which is what the
    // function's own comment argued for while it was the second copy.

    VkExtent2D runtime::resolve_resource_extent(render_resource::resource_id const id, uint32_t const element) const noexcept {
        core const& vk = this->vulkan_core;
        switch (id) {
        case pass::resource_id::probe_grid:
            // The probe cache's grid: 32 cells on a side, not the frame's size.
            return VkExtent2D{vulkan::gi_probe_grid_extent, vulkan::gi_probe_grid_extent};
        case pass::resource_id::bloom: {
            // A bloom level is HALF the previous one - max(1, swap >> (level + 1)) - which is the SAME formula
            // `core::create_render_targets` created the images with. The clamp is belt-and-braces rather than the
            // contract: the schema's `count` (4) plus the validator's element check is what limits the level, and
            // a shift of 32 or more would be undefined behaviour if one ever got through.
            uint32_t const shift = std::min<uint32_t>(element + 1u, 31u);
            return VkExtent2D{std::max(1u, vk.swap_chain_extent.width >> shift), std::max(1u, vk.swap_chain_extent.height >> shift)};
        }
        default:
            return VkExtent2D{}; // an element of a resource whose extent IS the frame's: nothing to answer
        }
    }

    pass::owned_pipeline runtime::resolve_pipeline(std::string_view const name) const noexcept {
        if (vk_pipeline const* const pipeline = this->get_pipeline(name); pipeline != nullptr) {
            return pass::owned_pipeline{.pipeline = pipeline->get_pipeline(), .layout = pipeline->get_pipeline_layout()};
        }
        // ... AND THEN THE CHAIN'S OWN PASSES, because a chain's stages may SHARE one pipeline: the post chain's
        // four bloom levels record with the composite's R16F variant, and a copy per level would be five identical
        // pipelines. Asking every pass by NAME is what keeps this chain-agnostic - the renderer does not know, and
        // does not need to know, which pass owns what; a pass answers for the names it publishes and nothing else.
        for (pass::pass_chain const* const chain : {&this->passes, &this->gi_trace_chain, &this->gi_denoise_chain}) {
            for (pass::frame_pass* const candidate : chain->as_stage().passes) {
                if (candidate == nullptr) {
                    continue;
                }
                if (pass::owned_pipeline const found = candidate->named_pipeline(name); found.pipeline != VK_NULL_HANDLE) {
                    return found;
                }
            }
        }
        return {};
    }

    VkDescriptorSet runtime::resolve_shared_set(uint32_t const family, uint32_t const element, uint32_t const image_index) {
        core const& vk = this->vulkan_core;
        switch (family) {
        case 0: // the scene set: one per frame SLOT (the camera and light UBOs live there)
            return this->scene_sets.set(static_cast<uint32_t>(vk.current_frame));
        case 1: // the G-buffer set: one per swapchain IMAGE, written on demand by the accessor its users share
            this->ensure_gbuffer_descriptors();
            return this->gbuffer_family.set(image_index, 0);
        case 2:
            // THE POST FAMILY, and the element is what makes it answerable: its five sets are one per STAGE, so
            // the declaration names which one it binds (the four bloom levels are elements 0..3, the composite
            // and FXAA share element 4 - the set the composite writes the LDR image through).
            return this->post_family.set(image_index, element);
        default:
            return VK_NULL_HANDLE;
        }
    }

    pass::resolve_context runtime::make_resolve_context() noexcept {
        return pass::resolve_context{
            .resources = &this->frame_resources,
            .frame = this->pass_frame(),
            .cmd = *this->command_buffers[static_cast<uint32_t>(this->vulkan_core.current_frame)],
            .descriptor_set =
                [](void* owner, uint32_t const family, uint32_t const element, uint32_t const image_index) {
                    return static_cast<runtime*>(owner)->resolve_shared_set(family, element, image_index);
                },
            .extent_of = [](void* owner, render_resource::resource_id const id, uint32_t const element) { return static_cast<runtime*>(owner)->resolve_resource_extent(id, element); },
            .pipeline = [](void* owner, std::string_view const name) { return static_cast<runtime*>(owner)->resolve_pipeline(name); },
            .owner = this,
        };
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
        // TWO THINGS WRAP THE FRAMEWORK'S RESOLUTION, and neither belongs there: this frame's shared constants,
        // which every pass is handed whether or not it reads them yet, and the differential check that kept the
        // resource table honest while the resolvers were still here (see verify_resource_table).
        //
        // THE PER-PASS SWITCH IS GONE (S3.12): every pass in this renderer is resolved by its own DECLARATION now
        // (`frame_pass::resolve`), from the resource table the frame publishes - own bindings, targets, barrier
        // entries, the shared sets it names, its own pipeline and the extent from its behaviour's rule. What is
        // left of the renderer's knowledge is the table's CONTENTS (`publish_frame_resources`) and the frame's
        // constants, which is exactly the split this migration was for: a pass cannot reach a resource its
        // declaration does not name, and the renderer no longer knows which pass wants which image.
        out.constants = this->frame_facts;
        bool const resolved = pass.resolve(this->make_resolve_context(), out);
        if (resolved) {
            this->verify_resource_table(pass, out);
        }
        return resolved;
    }

    // THE TAA RESOLVE'S RESOLVER IS GONE (S3), and it is the first pass whose PARAMETERS moved with it: its
    // four own bindings and its HDR target come from the frame's resource table, its pipeline from the pass (it
    // builds its own), its extent from the declaration's `full` rule, and its push block it composes itself out
    // of `resolved_io::constants` (the projection's two depth terms), `io.extent` (the texel size) and its own
    // two blend weights - which `set_taa` now forwards instead of caching them here. Its old gate is answered by
    // the table (the images), the pass's own pipeline (a null one fails the resolution) and the feature "taa".

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

    bool runtime::pass_ready(std::string_view const name) const noexcept {
        // The chain's own answer, and it is deliberately the CHAIN's: the pass whose declaration carries that name,
        // asked whether it built what it records with (see frame_pass::ready). This is the first of the renderer's
        // questions moved into the declaration vocabulary - the direction the handover needs, because a renderer
        // handed a chain from outside holds no typed member to ask.
        return this->passes.ready(name);
    }

    void runtime::fill_compute_skin_requests() {
        // The job's input, built from the casters this frame's structure phase knows: one request per SKINNED
        // caster (a zero destination means "not skinned: its geometry is what the build read"). The list is a
        // member so a frame does not allocate while recording, and the JOB decides what to do with it.
        this->compute_skin_requests.clear();
        this->compute_skin_requests.reserve(this->rt_caster_levels.size());
        for (auto const& built : this->rt_caster_levels) {
            if (built.skin_destination_address == 0) {
                continue;
            }
            this->compute_skin_requests.push_back(pass::compute_skin_request{
                .source_vertices = built.skin_source_address,
                .destination = built.skin_destination_address,
                .source_stride = built.skin_source_stride,
                .destination_stride = built.skin_destination_stride,
                .vertex_count = built.skin_vertex_count,
                .skin_base = built.skin_base,
            });
        }
    }

    bool runtime::record_compute_skin_pass(VkCommandBuffer const command_buffer) {
        // The knob, the caster list and the frame slot are the RENDERER's; the dispatches and the barrier the
        // acceleration structure build needs after them are the JOB's (vulkan.pass.compute_skin_job). The gate
        // is the moved code's, in the same order: the knob, the job, and whether anything is skinned at all.
        if (!this->rt_skin_bake || !this->compute_skin.ready() || this->rt_skin_levels.empty()) {
            return false;
        }
        this->fill_compute_skin_requests();
        return this->compute_skin.record(command_buffer, static_cast<uint32_t>(this->vulkan_core.current_frame), this->compute_skin_requests);
    }

    // THE RAY-TRACED SHADOW PASS'S RESOLVER IS GONE (S3): its one barrier image comes from the frame's resource
    // table (per-frame-slot instance), its two shared sets from the owner, its pipeline from the PASS (which owns
    // it) and its push block is composed by the pass itself out of `resolved_io::constants` - the first pass for
    // which a push block crossed that line. The two halves of its old gate went where the scene pass's did:
    //
    //  * "this slot's visibility image exists" is the RESOURCE TABLE (no entry, no resolution);
    //  * "this slot's top level structure is built, and the pipeline exists" is the pass's FEATURE
    //    (`feature_active("rt_shadow")`), because an acceleration structure is not a `resolved_binding` - it has
    //    no view, no buffer and no image, only a device address - so it cannot be a table entry at all.
    //
    // THE ONE THING THAT DID NOT MOVE INTO THE PASS is the pair of `ensure_gbuffer_*_sampled` calls it used to
    // make: they are the frame's *ordering* rule ("whoever samples the G-buffer first publishes the G-buffer
    // instance's attachment writes"), and the flags they read are the renderer's. They now run in the rt_shadow
    // STAGE's preamble in `begin_recording`, immediately before the stage records - which is the same position in
    // the command stream, because the stages carry no marks of their own (`make_pass_host` leaves the mark pair
    // null), so nothing is emitted between them.

    // THE SPATIAL FILTER'S RESOLVER IS GONE (S3.9), and it was the simplest of the four GI ones: its two shared sets,
    // its one barrier image and its own pipeline are all its declaration, its extent is the declaration's `half`
    // rule, and the push block it used to be handed is now composed by the PASS from `io.constants` (the projection
    // terms and the two depth/normal criteria the composite's upsample shares), `io.extent`/`io.frame.extent`, its
    // own filter width and two frame answers - which oracle ran, and whether the reflection's accumulation was
    // resolved before this dispatch (`frame_constants::gi_spec_resolved`, the field the reflection writes). The one
    // predicate the frame still owns is `ssgi_traced_active()`, and it arrives in the pass's frame.

    pass::ssgi_spec_frame runtime::make_ssgi_spec_frame() const noexcept {
        // One number: the pass's per-image first-use state is sized from the generation's image count, the same
        // shape the TAA resolve's history flags have.
        return pass::ssgi_spec_frame{.image_count = static_cast<uint32_t>(this->vulkan_core.gi_spec_images.size())};
    }

    // THE GLOSSY LOBE'S RESOLVER IS GONE (S3.10) with the tracer's, and the two went together because they push the
    // same frame facts: three declared images, the same two shared sets, its own pipeline, the declaration's `half`
    // extent - and a push block the PASS composes from `io.constants` and its own reach. Its "no instance table" gate
    // is the frame constant `gi_instance_table`, which the pass checks before recording (the feature registry already
    // refuses such a frame, so the check is the pass being unable to proceed rather than a path the gate depends on).

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
            // surface, so the position is not a detail - it is the ordering constraint. The PASS owns the
            // recording (vulkan.pass.rt_shadow); what is this loop's is the position and the off path below.
            pass::stage const rt_shadow_stage = {.name = "rt_shadow", .passes = this->rt_shadow_stage, .marks = false};
            // THIS STAGE'S ONE FRAME-ORDER DUTY, done by the chain's OWNER now that the passes are its: this stage
            // may be the first sampler of the stored surface this frame, and whoever samples it FIRST publishes the
            // G-buffer instance's attachment writes (the flags are the renderer's, and the idempotent `ensure_*`
            // pair is what makes "first" a fact rather than a promise) - so the owner's `prepare` asks for exactly
            // that, gated on the same feature the runner gates the stage on. Nothing is emitted between here and
            // `record_stage` (the stages carry no marks), so the command stream is unchanged.
            this->prepare_stage("rt_shadow", command_buffer);
            pass::run_report const rt_shadow_report = pass::record_stage(rt_shadow_stage, this->make_pass_host());
            if (rt_shadow_report.recorded == 0 && static_cast<std::size_t>(vk.current_frame) < vk.rt_shadow_images.size() &&
                vk.rt_shadow_images[vk.current_frame] != VK_NULL_HANDLE) {
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
            // THE LIGHTING STAGE IS A PASS (vulkan.pass.deferred), and what the frame still owes it is one answer
            // (whether the traced chain is replacing the ambient) plus the stage's own frame-order duty: it is a
            // sampler of the stored surface, and whoever samples it FIRST publishes the G-buffer instance's
            // attachment writes. Both are the chain OWNER's now (see prepare_stage) - the runtime's `prepare` call
            // sits exactly where the pass's frame used to be set, and nothing is emitted between it and
            // `record_stage` (the stages carry no marks), so the command stream is unchanged.
            this->prepare_stage("deferred", command_buffer);
            pass::stage const deferred_stage = {.name = "deferred", .passes = this->deferred_stage, .marks = false};
            pass::run_report const deferred_report = pass::record_stage(deferred_stage, this->make_pass_host());
            if (deferred_report.recorded == 0) {
                // The pass could not resolve a frame. Inside this branch the only remaining cause is the G-buffer
                // family having no set for this image (the pipeline and the target generation are what
                // deferred_lit_active() just checked), and the frame then has to be cleared rather than left
                // half-written - see the fallback's own comment.
                this->clear_scene_color_for_missing_gbuffer(command_buffer);
            }
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
        // swapchain image. The PASS owns all of that now (vulkan.pass.taa), and the two things below are the
        // part it cannot own yet:
        //
        //  * the G-buffer depth's transition to a sampled layout, whose "was it written this frame" flag
        //    belongs to the G-buffer pass, and
        //  * clearing the motion-vector flag, which is what stops the GI chain (later in the same frame) from
        //    transitioning the velocity image a second time.
        //
        // BOTH ARE THE CHAIN OWNER'S NOW, run from its `prepare` for this stage and gated on the same predicate the
        // runner gates the stage on - so the frame never touches them on a frame the pass does not run (clearing the
        // velocity flag for a frame with no resolve would make the GI tracer sample an image still in ATTACHMENT
        // layout). The runtime's call sits exactly where those two lines were.
        this->prepare_stage("taa", command_buffer);
        {
            pass::stage const taa_stage = {.name = "taa", .passes = this->taa_stage, .marks = false};
            [[maybe_unused]] pass::run_report const taa_report = pass::record_stage(taa_stage, this->make_pass_host());
        }
        // The matrix the NEXT frame's motion vectors are computed against is this frame's, and it is only
        // recorded when the resolve actually wrote a history: a resolve that bailed out (no descriptor set) must
        // not claim one. The pass answers that (`wrote_history`) and the owner reports it through `collect`; the
        // array stays the renderer's because the camera UBO - not TAA - reads it as `prev_view_proj`.
        this->collect_stage("taa");
        this->gpu_mark(command_buffer, gpu_mark_id::taa_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // G-buffer debug mode (an inspection of the stored data, never combined with the lighting
        // stage or TAA): turn one channel into a visible image in the HDR target. THE PASS owns the recording
        // (vulkan.pass.gbuffer_debug); what stays here is the frame - the target's own transition (which has to
        // happen even when the pass cannot draw, because the post chain samples that image), the two per-image
        // hand-backs its frame carries, and the fallback that clears the target when there is no set to draw with.
        if (this->gbuffer_pass_active() && !this->deferred_lit_active()) {
            // The HDR target becomes a colour attachment BEFORE the stage: the G-buffer pass wrote its own targets,
            // so this image was never an attachment this frame, and the pass's instance CLEARs it.
            std::array<VkImageMemoryBarrier2, 1> hdr_barrier = {vulkan::color_attachment_transition};
            hdr_barrier[0].image = vk.hdr_images[this->current_image_index];
            VkDependencyInfo const hdr_dependency = make_image_dependency_info(1, hdr_barrier.data());
            vkCmdPipelineBarrier2(command_buffer, &hdr_dependency);
            // THIS STAGE'S FRAME-ORDER DUTY, the same shape as the lighting stage's above and for the same reason:
            // the debug view runs INSTEAD of the lighting stage, so it is the stage that hands the stored surface
            // and the motion vectors to samplers this frame - and both halves are the chain OWNER's now (the
            // depth's flag-based accessor and the velocity flag's clear, see prepare_stage).
            this->prepare_stage("gbuffer_debug", command_buffer);
            pass::stage const debug_stage = {.name = "gbuffer_debug", .passes = this->gbuffer_debug_stage, .marks = false};
            pass::run_report const debug_report = pass::record_stage(debug_stage, this->make_pass_host());
            if (debug_report.recorded == 0) {
                // Inside this branch the only remaining cause is the G-buffer family having no set for this image
                // (the pipeline is what gbuffer_pass_active() just checked), and the frame then has to be cleared -
                // see the fallback's own comment. Its own transition is part of it.
                this->clear_hdr_for_missing_gbuffer_set(command_buffer);
            }
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

    void runtime::record_overlay_if_enabled(VkCommandBuffer const command_buffer) {
        if (this->debug_gui_shown && this->debug_overlay.is_active()) {
            this->debug_overlay.record(command_buffer);
        }
    }

    // =============================================================================================
    // THE POST CHAIN (vulkan.pass.post)
    // =============================================================================================
    //
    // NO RESOLVER IS LEFT FOR IT (S3). The composite's declaration resolves generically except for the one
    // frame-decided choice (which target and pipeline variant, and the push block that follows from it), which
    // its own `resolve` override makes; its four bloom levels and FXAA resolve ENTIRELY from their declarations -
    // the level they write (a `bloom` element), the level they read (the element before it, or nothing at level
    // 0), their own set of the post family (`shared_set{2, level}`), their extent (the bloom element's size) and
    // the pipeline they record with, which is the COMPOSITE's R16F variant published by name (`post_hdr`) and
    // resolved by asking the chain's passes - see `resolve_pipeline`.
    //
    // WHAT IS NOT RESOLVED, and it is a decision rather than an omission: the HDR target's transition to a sampled
    // layout. The composite reads that image whether or not bloom runs and the bright-pass prefilter reads it too,
    // and on a frame the whole chain is skipped nobody else would move it - so it has exactly one owner and that
    // owner is the frame loop (see render_resource::post_composite_io).

    bool runtime::post_fxaa_active() const noexcept {
        // ONE definition of "the FXAA pass is this frame's last writer": the resolver picks the composite's target
        // and pipeline variant with it, the frame loop decides the overlay's owner with it, the feature registry
        // reports it, and set_fxaa() already folds the pipeline's existence into `fxaa_on` for the same reason. The
        // pipeline is the FXAA PASS's now (vulkan.pass.fxaa), which is why this asks the pass rather than a member.
        return this->fxaa_on && this->pass_ready("fxaa");
    }

    void runtime::draw_overlay_after(void* const owner, VkCommandBuffer const command_buffer) {
        // The composite's frame carries this as a plain function pointer (a pass records into what it is given and
        // knows nothing about this class), so the member is reached through the context that frame also holds.
        static_cast<runtime*>(owner)->record_overlay_if_enabled(command_buffer);
    }

    // =============================================================================================
    // THE G-BUFFER DEBUG VIEW (vulkan.pass.gbuffer_debug)
    // =============================================================================================
    //
    // THE G-BUFFER DEBUG VIEW'S RESOLVER AND ITS `ensure_inputs` CALLBACK ARE GONE (S3). Its declaration resolves
    // generically - the HDR display target, the FOUR images it moves to a sampled layout (three stored surface
    // targets and the motion vectors, all per-image families the table publishes), the shared G-buffer set (1),
    // its own pipeline and a full-frame extent - and its push block is the pass's own (its channel, the frame's two
    // projection terms and a motion gain derived from the frame's width). Nothing here is left but the frame's own
    // transitions, which stay in the frame loop as they were: the HDR target's move to a colour attachment has to
    // happen even on a frame the pass cannot draw, because the post chain samples that image.

    void runtime::clear_hdr_for_missing_gbuffer_set(VkCommandBuffer const command_buffer) {
        // The debug view did not record because the G-buffer family had no descriptor set for this image (its pool
        // could not give one). The pass cannot do this itself - it never sees the family's pool - so the frame's
        // answer lives here: the target is CLEARED, which is what makes such a frame black rather than undefined
        // (the post chain samples that image), and the log line says so once per frame.
        core const& vk = this->vulkan_core;
        uint32_t const index = this->current_image_index;
        if (index >= vk.hdr_images.size()) {
            return;
        }
        utility::log("runtime: gbuffer debug pass has no descriptor set - showing a cleared frame");
        std::array<VkImageMemoryBarrier2, 1> clear_barrier = {vulkan::color_attachment_transition};
        clear_barrier[0].image = vk.hdr_images[index];
        VkDependencyInfo const clear_dependency = make_image_dependency_info(1, clear_barrier.data());
        vkCmdPipelineBarrier2(command_buffer, &clear_dependency);
        VkClearValue clear = {};
        VkRenderingAttachmentInfo const attachment = make_color_attachment_info(vk.hdr_image_views[index], clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
        VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, vk.swap_chain_extent}, true, &attachment, nullptr);
        vkCmdBeginRendering(command_buffer, &rendering_info);
        vkCmdEndRendering(command_buffer);
    }
    bool runtime::record_post_process(VkCommandBuffer const command_buffer) {
        core const& vk = this->vulkan_core;

        // ---- the scene side: close the geometry instance, then the stages that consume the G-buffer
        this->record_scene_tail(command_buffer);

        this->ensure_post_descriptors();
        if (!this->pass_ready("post_composite") || this->post_family.set(static_cast<uint32_t>(this->current_image_index), 4) == VK_NULL_HANDLE) {
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
        //
        // `gi_spec_resolved` is cleared in the same place and for the same reason: the reflection writes it
        // through the temporal pass's callback, and a frame whose reflection did not run must not leave the
        // previous frame's answer for the filter's `spec_weight` lane to read.
        //
        // THE TWO FACTS THE TRACING STAGES SHARE go in here as well, and this is the earliest point they CAN be
        // this frame's: the instance table belongs to a buffer the structure phase above (re)builds, so an address
        // read in `update_frame_constants` - which runs before that phase - could be a buffer this same frame is
        // about to replace. The address is the table this frame HAS and nothing more: the hit-shading gate belongs
        // to the stages that push it, and they do not agree about it (see frame_constants::gi_instance_table).
        this->frame_facts.gi_resolved = false;
        this->frame_facts.gi_spec_resolved = false;
        uint64_t instance_table = 0;
        if (this->rt_top_levels.has_value()) {
            VkBuffer const table = this->rt_top_levels->instance_table(static_cast<uint32_t>(vk.current_frame));
            if (table != VK_NULL_HANDLE) {
                VkBufferDeviceAddressInfo const table_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = table};
                instance_table = vkGetBufferDeviceAddress(vk.device, &table_info);
            }
        }
        this->frame_facts.gi_instance_table = instance_table;
        this->frame_facts.gi_frame_index = this->ssgi_frame;
        if (this->ssgi_active()) {
            // THE GI CHAIN, recorded as TWO calls with the FRAME's rule between them. Why two: the temporal
            // resolve is the first sampler of two shared per-image images (the G-buffer's depth, which its depth
            // guard reads, and the motion-vector target its reprojection reads), and whether each of them still
            // needs its "the G-buffer pass wrote me" publication is the FRAME's per-image bookkeeping - so the
            // owner publishes them from its `prepare` for the denoise half, exactly where the old `ensure_inputs`
            // callback did (after the lobe recorded, before the resolve's dispatch), and the command stream is
            // unchanged.
            //
            // The frames the passes need are given by the owner too, and the two halves are prepared SEPARATELY
            // because two of their values must be read BEFORE the half runs: the tracer's `specular_next` (which
            // decides who owes the denoiser the hand-off barrier) and the temporal's `history_valid`, which the
            // reflection's callback is handed as well so the two signals agree about the frame that created the
            // history.
            //
            // The REFLECTION's descriptor family is still the RENDERER's (the diffuse one is the temporal pass's
            // own), so it is ensured HERE - once per frame, before the chain, exactly where the old
            // `record_ssgi_denoise_pass` ensured it before running the temporal stage.
            this->ensure_ssgi_denoise_descriptors();
            // ---- the trace half: the tracer, then the lobe (the two writers of the raw trace) ----
            this->prepare_stage("gi_trace", command_buffer);
            pass::run_report const gi_trace_report = this->gi_trace_chain.record(this->make_pass_host());
            static_cast<void>(gi_trace_report);
            // ---- the denoise half: the temporal resolve, then the spatial filter. Its `prepare` is also where the
            //      FRAME's rule between the halves runs (both calls are idempotent and consult per-image flags the
            //      frame owns, so on the frames where another stage already published them - the deferred stage's
            //      preamble publishes the depth, the TAA resolve the velocity target - they record nothing at all).
            this->prepare_stage("gi_denoise", command_buffer);
            pass::run_report const gi_denoise_report = this->gi_denoise_chain.record(this->make_pass_host());
            static_cast<void>(gi_denoise_report);
            // ... and the chain's answers, which the owner reports: whether the spatial filter wrote the image the
            // composite samples (that is `gi_resolved`, a frame constant the composite reads while recording) and
            // whether the denoiser produced an accumulation (which the NEXT frame's history flag is set from).
            this->collect_stage("gi_denoise");
            ++this->ssgi_frame; // the next frame's ray sequence must differ (see ssgi_frame)
        }
        if (!this->frame_facts.gi_resolved && index < vk.gi_spatial_images.size() && vk.gi_spatial_images[index] != VK_NULL_HANDLE) {
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

        // ---- THE BLOOM CHAIN, as a stage of FOUR passes (vulkan.pass.post) ----
        // The chain is four instances of one class; the level IS the pass boundary (each binds a different one of
        // the post family's five sets), and what the host resolves for each is in resolve_post_bloom. The RUNNER
        // gates them: the pass declares the feature `bloom`, which is `bloom_intensity > 0` and the chain's
        // pipeline being there - the same predicate that decides whether the composite adds a bloom sum (see
        // resolve_post_composite, where the weight is zeroed for the debug view - which is also what
        // `active_features().gbuffer_debug` means).
        pass::stage const bloom_stage = {.name = "bloom", .passes = this->bloom_stage, .marks = false};
        pass::run_report const bloom_report = pass::record_stage(bloom_stage, this->make_pass_host());
        if (bloom_report.recorded == 0) {
            // THE OFF PATH, and it is the frame loop's because it is about the COMPOSITE's descriptor: its set
            // declares all four bloom levels as inputs and multiplies them by the weight it pushed, and a
            // statically-used binding's image has to be in the layout that descriptor declares whether or not the
            // value is ever read. UNDEFINED as the old layout asserts nothing (their contents are dead on a frame
            // nothing wrote), and this is the same answer the shadow map's spare layers and the GI image's off path
            // give. It hangs on "the stage recorded nothing" rather than on the knob, so a frame whose levels had no
            // descriptor set takes it too.
            for (uint32_t level = 0; level < vulkan::core::bloom_level_count; ++level) {
                std::array<VkImageMemoryBarrier2, 1> barriers = {vulkan::undefined_to_sampling_transition};
                barriers[0].image = vk.bloom_images[level][index];
                VkDependencyInfo const dependency_info = make_image_dependency_info(1, barriers.data());
                vkCmdPipelineBarrier2(command_buffer, &dependency_info);
            }
        }
        // GPU timing: the bloom chain ends here (a disabled chain is just the layout fixups above, so its interval
        // reads ~0).
        this->gpu_mark(command_buffer, gpu_mark_id::bloom_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // ---- THE COMPOSITE, as a stage of one pass ----
        // ITS FRAME CARRIES THE OVERLAY when FXAA is off: the overlay has no load op of its own, so it has to be
        // drawn inside whichever instance is the frame's LAST writer - and when FXAA runs, that is the FXAA pass.
        // Which of the two that is, and whether this frame's bloom sum exists, are the RENDERER's decisions, so the
        // frame is built here (make_composite_frame) and the owner hands it to its pass (prepare_stage).
        this->prepare_stage("post_composite", command_buffer);
        pass::stage const composite_stage = {.name = "post_composite", .passes = this->post_composite_stage, .marks = false};
        [[maybe_unused]] pass::run_report const composite_report = pass::record_stage(composite_stage, this->make_pass_host());
        // GPU timing: the composite (and the debug overlay, when it draws here) is done.
        this->gpu_mark(command_buffer, gpu_mark_id::composite_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // ---- THE FXAA RESOLVE, as a stage of one pass (vulkan.pass.fxaa) ----
        // It is the frame's LAST writer whenever it runs, so it is the pass that carries the overlay - the other
        // half of the split the composite's frame above states. The runner gates it on the feature `fxaa`, which is
        // `post_fxaa_active()`: the same predicate that decided this frame's composite TARGET, so the pass runs
        // exactly when the LDR image is what the composite wrote.
        this->prepare_stage("fxaa", command_buffer);
        pass::stage const fxaa_stage = {.name = "fxaa", .passes = this->fxaa_stage, .marks = false};
        [[maybe_unused]] pass::run_report const fxaa_report = pass::record_stage(fxaa_stage, this->make_pass_host());
        // GPU timing: the FXAA pass (and the overlay it carries when it is the last writer) is done. Without FXAA
        // the composite already ended the frame's display work, so this interval is ~0.
        this->gpu_mark(command_buffer, gpu_mark_id::fxaa_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // true in both paths: the FXAA pass (or the composite, when FXAA is off) wrote the swapchain
        return true;
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
        // The resource table's differential check reports itself HERE, once, at the end of the frame that closes
        // its window - the last point of a frame, so the counters cover EVERY stage (the post and GI stages record
        // inside `end_recording`, which is why this cannot sit there). THREE frames rather than one, because a
        // pass may legitimately resolve only from the second frame on (TAA needs a history), and a check that
        // watched one frame would report those as uncovered. See verify_resource_table: a mismatch is a table
        // entry to fix before that pass's resolver can be deleted.
        ++this->resource_check_frames;
        if (!this->resource_check_reported && this->resource_check_frames >= 3) {
            this->resource_check_reported = true;
            utility::log("resource table: {} entries published, {} declaration handle checks against the resolvers, {} mismatch(es) over {} frame(s)",
                         this->frame_resources.size(),
                         this->resource_check_checked,
                         this->resource_check_mismatched,
                         this->resource_check_frames);
        }
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

    bool runtime::debug_gui_wants_mouse() const noexcept {
        return this->debug_gui_shown && this->debug_overlay.is_active() && this->debug_overlay.wants_mouse();
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
        //
        // BUILT FROM THE DEVICE, not through `core::make_pipeline`: that entry point is the old style - it
        // reached into the core for the flat scene layout, the surface format and the depth format, which is
        // exactly what a caller that is not the runtime (a pass, whose create step has a device and its own
        // layout) cannot do. The four facts are read here instead, and they are the ones that entry point used,
        // so this is a re-expression: same layout, same formats, same single-sampled pipeline.
        std::array<VkFormat, 1> const color_formats = {this->vulkan_core.swap_chain_image_format};
        // ... AND THE BLEND STATE, which the old entry point got from the convenience overload: the FORWARD
        // pipelines' convention is src-alpha blending (alpha is coverage, and an opaque draw's alpha of one
        // reduces the blend math to the source colour), while the span-based form's default is "overwrite".
        // Omitting it is not a no-op - the gate caught it as a changed scenario, and this is why the conversion
        // is a re-expression rather than a rewrite.
        std::array<VkPipelineColorBlendAttachmentState, 1> const blend_attachments = {make_color_blend_attachment()};
        auto make_result = vulkan::make_pipeline(this->vulkan_core.device,
                                                 this->vulkan_core.scene_pipeline_layout,
                                                 std::span<VkFormat const>(color_formats),
                                                 this->vulkan_core.depth_format,
                                                 vertex_shader_code,
                                                 fragment_shader_code,
                                                 VK_SAMPLE_COUNT_1_BIT,
                                                 /*depth_test_enabled=*/true,
                                                 0.0f,
                                                 0.0f,
                                                 0.0f,
                                                 std::span<VkPipelineColorBlendAttachmentState const>(blend_attachments));
        if (!make_result) {
            return fail(std::string(make_result.error()));
        }
        // ... AND THE VIEWPORT THE OLD ENTRY POINT ALSO SAVED, which is not incidental: these pipelines are
        // drawn through `vk_pipeline::begin_pipeline`, which re-emits the cached viewport, and the scene path
        // resyncs it once per frame in update_pass_geometry. Leaving it at its creation-time zero is the
        // zero-width viewport this project has already paid for once - and the gate caught exactly this
        // omission here, as one changed scenario.
        make_result->viewport = {0.0f, 0.0f, static_cast<float>(this->vulkan_core.swap_chain_extent.width), static_cast<float>(this->vulkan_core.swap_chain_extent.height), 0.0f, 1.0f};
        make_result->scissor = {{0, 0}, this->vulkan_core.swap_chain_extent};
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

    void runtime::set_shadow_map_size(uint32_t const size) noexcept {
        // Startup-only: everything that consumes the size (the layered image + its views + the
        // descriptor, the depth pass rendering instance, the pipeline viewport, the light UBO texel
        // size and the fit) is built from it when the scene set is first created, so a change after
        // that cannot take effect - say so instead of pretending otherwise.
        if (this->pass_ready("shadow") || !this->shadow_images.empty()) {
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
        // THE FLAT RENDER FLAG AND THE SSAO SWITCH LIVE IN THE LIGHTING PASS now (they are its parameters), so the
        // feature registry ASKS it - the same shape `feature_active("ssgi_spatial")` uses for the temporal pass's
        // own answer. One copy of each value, and the pass that pushes them is the one that owns them.
        f.unlit = this->deferred.unlit();
        f.gbuffer_debug = this->gbuffer_debug && this->gbuffer_pipeline.has_value() && this->pass_ready("gbuffer-debug");
        f.ssgi = this->ssgi_active();
        // The probe cache is a pass of its own, so its activity is a feature of its own: the tracer asks
        // whether the cache is READY (probe_ready below), and the runner asks whether the pass RUNS.
        f.ssgi_probes = this->gi_probe_active();
        // The G-buffer pass and its lighting stage are the engine's only scene path, so there is no
        // flag for them: taa/ssao below ask this instead, and the debug view stands in for the
        // lighting stage rather than running alongside it (the two write the HDR target differently).
        bool const shaded_scene = !f.gbuffer_debug && this->pass_ready("deferred") && this->gbuffer_pipeline.has_value();
        // The shadow map is only read by the shading stages. The flat render mode samples nothing
        // (unlit.frag has no lighting include; the lighting stage returns the albedo before any
        // shading), so recording the pass would be pure waste - it measured 0.22 ms of a 0.5 ms frame.
        f.shadow = this->shadow_enabled && this->shadows_enabled && this->pass_ready("shadow") && !f.unlit;
        // The ray-traced shadow is a pass of its own, so it is a feature of its own: the knob, a device with ray
        // queries, and its own pipeline. The frame loop gates its STAGE on this, and the light UBO's
        // `rt_shadows` lane (what the lighting stage actually reads) is composed from the same three.
        f.rt_shadow = this->rt_shadows_active() && this->pass_ready("rt_shadow");
        // Same argument for the cluster pass: flat shading reads no light list, and with no active
        // punctual light there is nothing to sort in the first place.
        f.clustered = this->clustered_lights && this->pass_ready("cluster") && this->light_state.light_count.x > 0.5f && !f.unlit;
        f.taa = this->taa_on && this->pass_ready("taa") && shaded_scene;
        f.ssao = this->deferred.ssao_enabled() && shaded_scene; // shader-side gate: no pass of its own to skip
        f.bloom = this->bloom_intensity > 0.0f && this->pass_ready("post_composite") && !f.gbuffer_debug;
        f.fxaa = this->post_fxaa_active();
        // The transparent pass composites over the shaded frame, so it needs that frame to exist -
        // and it is skipped in the debug view, which shows the G-buffer rather than a frame.
        f.transparent = !f.gbuffer_debug && !this->frame_transparent.empty();
        return f;
    }

    bool runtime::feature_active(std::string_view const name) const noexcept {
        render_features const f = this->active_features();
        // THE TWO GATES THAT USED TO BE THE FIRST LINE OF A RESOLVER. Both passes were "always active" before S3,
        // with the renderer's resolver returning false to skip them; the skip is the same, but the question now
        // has one name and one answer, and the pass does not need a resolver to be asked it.
        //
        // What is NOT in these answers is the OTHER half of those old gates - whether this frame's target
        // generation exists. That is what the resource table already says: a frame whose images are not there has
        // no entry for the pass's declared targets, so the pass resolves to false and the runner skips it. Two
        // mechanisms, two questions: "the pass cannot run at all this session" is here, "this frame has not got
        // the resources" is the table.
        if (name == "scene") {
            return this->gbuffer_pass_active();
        }
        if (name == "transparent") {
            return !this->frame_transparent.empty();
        }
        if (name == "gbuffer-debug") {
            return f.gbuffer_debug;
        }
        if (name == "ssgi") {
            return f.ssgi;
        }
        if (name == "ssgi_spatial") {
            // THE CHAIN'S LAST STAGE, and the one pass whose gate is not just "the chain is on": the filter must
            // not filter a STALE accumulation, so it runs only when THIS frame's temporal resolve recorded. The
            // answer comes from the pass that owns it (the temporal pass clears its flag when the host sets the
            // frame, so this cannot read an earlier frame's answer) - which is what replaced the frame loop's
            // `if (record_ssgi_denoise_pass(...))` around the filter's stage.
            //
            // NOTE WHICH FUNCTION THIS IS: the runner asks `feature_active`, NOT `feature_available` (which
            // answers "could this feature run this session" for the overlay and the startup log). The first
            // version of this branch was added to `feature_available` by mistake, and the symptom was precise:
            // the filter was skipped on every frame (`skipped_inactive 1`), so the composite sampled an image
            // nothing had written - a subtly darker frame in the model's region, which is exactly what the gate
            // reported. Two predicates with the same vocabulary and different questions is a trap this file now
            // has a comment about.
            return this->ssgi_active() && this->ssgi_temporal.resolved();
        }
        if (name == "ssgi_probes") {
            return f.ssgi_probes;
        }
        if (name == "ssgi_specular") {
            // The lobe's own feature name, and the FIRST one whose answer is a predicate rather than a struct
            // field: "the knob is on, the frame can shade a hit, and the pass built its pipeline". It has to be
            // this exact predicate because the tracer reads it too (specular_next) to decide who owes the
            // denoiser the hand-off barrier - one answer, so the two passes cannot disagree.
            return this->ssgi_specular_active();
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
        if (name == "rt_shadow") {
            // THE PASS'S GATE, in full: the knob and the extension (`f.rt_shadow`, which is the same predicate
            // the light UBO's flag and the startup log use) PLUS "this frame's structure is built for the slot".
            // The second half is here rather than in the resource table because an acceleration structure is not
            // a `resolved_binding` - it has a device address and no view, buffer or image - so the table cannot
            // carry it. It is the same handle the scene set's binding 16 is written from, which is what keeps a
            // dispatch from ever reading an unwritten descriptor.
            return f.rt_shadow && this->rt_top_levels.has_value() && this->rt_top_levels->handle(static_cast<uint32_t>(this->vulkan_core.current_frame)) != VK_NULL_HANDLE;
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
        if (name == "deferred") {
            // THE LIGHTING STAGE'S OWN GATE, and it is deliberately the SAME predicate the frame loop's branch
            // uses (`deferred_lit_active()`): the runner asks this before resolving the pass, and the frame loop
            // asks it before recording the stage, so one answer means the two cannot disagree about whether the
            // lighting runs this frame. A pass's feature name with NO branch here would record nothing at all
            // (`skipped_inactive 1` every frame) - the trap the ssgi_spatial note above records.
            return this->deferred_lit_active();
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
        if (enabled && (!this->gbuffer_pipeline.has_value() || !this->pass_ready("gbuffer-debug"))) {
            this->warn_missing_feature("gbuffer-debug", "the G-buffer debug view has no effect: its pipelines were not created (see the startup log)");
        }
    }

    bool runtime::feature_available(std::string_view const name) const noexcept {
        // The single source of truth for "can this feature run at all this session": the overlay asks
        // it to decide what to offer, log_feature_status() prints it, and both therefore agree.
        if (name == "gbuffer-debug") {
            return this->gbuffer_pipeline.has_value() && this->pass_ready("gbuffer-debug");
        }
        if (name == "deferred") {
            // AVAILABILITY, not activity: "the pass built its pipeline" (whether it RUNS this frame is
            // `feature_active`'s answer above). This branch did not exist before the lighting stage was a PASS,
            // so it answered false - and chores.cpp's SSAO group, whose visibility asks exactly this question
            // (`feature_available("deferred") && feature_active("ssao")`), was never offered. Found by the
            // extraction, recorded, and the gate is what proves the fix changes no frame.
            return this->pass_ready("deferred");
        }
        if (name == "taa") {
            return this->pass_ready("taa");
        }
        if (name == "fxaa") {
            return this->pass_ready("fxaa");
        }
        if (name == "shadow") {
            return this->pass_ready("shadow");
        }
        if (name == "clustered") {
            // AVAILABILITY, not activity: this is the one feature whose answer is "the pass built a pipeline",
            // while `feature_active` above answers "it runs THIS frame" (a live punctual light is part of that).
            return this->pass_ready("cluster");
        }
        if (name == "ssgi") {
            return this->pass_ready("ssgi_trace");
        }
        if (name == "ssgi_spatial") {
            // AVAILABILITY, not activity: "the chain's last stage built its pipeline", which is what the overlay
            // and the startup log ask. Whether it runs THIS frame is `feature_active`'s answer (the chain is on
            // AND this frame's temporal resolve recorded) - see the note there.
            return this->pass_ready("ssgi_spatial");
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
        if (!this->gbuffer_pipeline.has_value() || !this->pass_ready("deferred")) {
            utility::log("features: the G-buffer pass or its lighting stage was not created, so NO SCENE IS DRAWN this session (see the startup log's 'deferred lighting disabled' line)");
        }
    }

    void runtime::set_ssao(bool const enabled, float const radius, float const intensity, uint32_t const samples) noexcept {
        // CPU-side only (the same rule as set_brdf_model / set_clustered_lights): the values are pushed with the
        // deferred lighting stage each frame, so they are safe to change mid-run. THE VALUES AND THEIR CLAMPS ARE
        // THE PASS'S now (see deferred_pass::set_ssao); this setter forwards and keeps the diagnostic, because
        // "why does this switch do nothing?" is a question about the SESSION rather than about the pass.
        this->deferred.set_ssao(enabled, radius, intensity, samples);
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

    void runtime::set_unlit(bool const unlit) noexcept {
        // CPU-side only, like the other render-mode flags: the value is pushed with the deferred lighting stage
        // each frame (and the forward path does not need it at all - there the render mode IS the default
        // pipeline). THE FLAG IS THE PASS'S now; the renderer's other features (shadow, clustered, bloom) ask the
        // pass for it through `active_features`, so there is one copy.
        this->deferred.set_unlit(unlit);
    }

    void runtime::set_clustered_lights(bool const enabled) noexcept {
        // CPU-side only, like set_brdf_model: the flag rides light_state's cluster_grid.w lane and
        // pace_and_acquire() copies light_state into the paced slot's buffer, so the next frame's
        // cluster dispatch and shading both see it (no in-flight buffer is touched).
        this->clustered_lights = enabled;
        if (enabled && !this->pass_ready("cluster")) {
            this->warn_missing_feature("clustered", "clustered light culling has no effect: the cluster compute pipeline was not created, so the shading stage loops EVERY active light instead (see the startup log)");
        }
    }

    // THE CLUSTERED-LIGHT SORT'S RESOLVER IS GONE (S3), and it needed nothing but the mechanism: its two barrier
    // BUFFERS come from the frame's resource table (per-frame-slot instance), its shared scene set from the owner,
    // its pipeline from the PASS (which owns it), its extent from the declaration's `none` rule and its push block
    // from nowhere - it has none. Its old gates are answered by the two mechanisms that own those questions: "the
    // pipeline exists" is `pass.pipeline()` (a null pipeline is what the generic resolution fails on), "the
    // buffers are there" is the resource table, "the scene set is there" is the shared-set rule, and "the grid was
    // computed this frame" is the pass's own `frame_.cluster_count == 0` guard, which its `record` already had.

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
        if (!this->pass_ready("shadow") || this->light_mapped.empty()) {
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
        this->light_state.rt_shadows = (enabled && this->pass_ready("rt_shadow") && this->vulkan_core.ray_query_available) ? 1.0f : 0.0f;
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
            // run the material's discard - this bake is where the mask is applied instead, and it is startup
            // work because the structures are built once and a MASK material is a property of the file (see
            // shaders/mask_bake.comp for the rule and for what the mechanism cannot represent).
            //
            // WHAT IS STILL THE RENDERER'S HERE, and what is the JOB's: the policy is this loop's - which
            // casters carry a MASK material, and the allocation of the expanded buffer each one is baked into
            // (the buffer outlives the loop: the build below reads it, and hit shading reads its vertices
            // through the instance table for as long as the structures live). The pipeline, its layout, the set
            // it binds and the dispatch are `mask_bake`'s (vulkan.pass.mask_bake_job).
            VkDeviceAddress mask_address = 0;
            uint32_t mask_stride = 0;
            if (this->rt_mask_bake && this->mask_bake.ready() && this->material_mapped != nullptr) {
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
                        this->mask_bake.record(command_buffer,
                                               pass::mask_bake_request{
                                                   .source_vertices = source_vertex_address,
                                                   .source_indices = source_index_address,
                                                   .destination = mask_address,
                                                   .source_stride = caster->vertex_stride,
                                                   .destination_stride = mask_vertex_stride,
                                                   .index_type = static_cast<uint32_t>(caster->index_type),
                                                   .triangle_count = caster->index_count / 3u,
                                                   .material_index = material_index,
                                               });
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
            if (mask_address == 0 && this->rt_skin_bake && this->compute_skin.ready() && caster->push.skin_base != 0 && caster->vertex_count != 0 &&
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
        this->fxaa_on = enabled && this->pass_ready("fxaa");
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
} // namespace vulkan
