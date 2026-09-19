// ============================================================================
// module: vulkan.runtime:constructor  - construction, initialisation and resource creation
//
// The two contiguous runs of runtime.cpp that BUILD a runtime rather than drive one: the constructors
// and the whole init/ensure run (init_scene_resources through ensure_scene_heap_slots), then
// write_light_and_shadow_bindings, set_ibl and register_material. What stayed behind runs per frame or
// is a diagnostic (the two heap probes, which get their own partition).
//
// THE GLFW CALLBACKS CAME WITH THE CONSTRUCTOR because the constructor is what registers them; the
// window lookup they use is the same three-line helper runtime.cpp keeps. The heap-write helpers that
// sit inside the init run (heap_slot_offset, write_heap_grid_image, write_heap_scene_buffer) came too,
// and runtime.cpp keeps a copy of the ones its own frame path still calls. Duplication of small
// file-local helpers across two anonymous namespaces is legal but it IS duplication: the follow-up is to
// publish the heap three from vulkan.core:descriptor_heap, where heap plumbing belongs.
//
// Imports are NOT transitive: this partition imports what the moved code calls, and repeats the pmr
// keep-alive that must run before any pmr container in this TU.
// ============================================================================
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

module vulkan.runtime:constructor;

import :declarations;
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

// Route std::pmr allocations through mimalloc for this TU (utility:better_pmr). Idempotent:
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
        // (The furnace cube is a new image too - its level is part of the generation reset above.)
        // NOTE: the shadow resources (map layers + light UBO buffers) are created LAZILY, by
        // ensure_shadow_resources() from ensure_scene_heap_slots(). The shadow map is a layered 2D array
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

        // post-process objects: the FXAA pipeline and the two samplers are RAII members, and the post chain's
        // two pipelines belong to the post composite PASS (vulkan.pass.post::release_owned), the same rule
        // every extracted pass follows.
        //
        // NO SET LAYOUT, PIPELINE LAYOUT OR DESCRIPTOR POOL IS TORN DOWN HERE ANY MORE, and that is the whole
        // point of the deletion this destructor records: every stage is heap-native, so there is nothing of that
        // kind left in this class to destroy - no scene set layout (it was `core`'s), no post or G-buffer set
        // layout (this class created them for the families it wrote), no pool and no per-image family. What
        // remains of each extracted pass's GPU material is the PASS's, released by its own destructor.

        // Shared scene resources: views/samplers/buffers/images are RAII and free
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
        // these buffers through the shared scene block, so one memcpy per frame replaces the old
        // per-primitive per-frame UBO updates
        camera_ubo initial = {};
        init_utils::create_host_buffers(this->vulkan_core,
                                        vulkan::core::MAX_FRAMES_IN_FLIGHT,
                                        std::as_bytes(std::span(&initial, 1)),
                                        vulkan::buffer_type::uniform_coherent,
                                        "camera ubo buffer",
                                        this->camera_buffers,
                                        &this->camera_mapped,
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT); // heap-bound: see the scene-set block

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

        // THE WHITE ELEMENT NEEDS ITS OWN HEAP DESCRIPTOR HERE, and its absence was a class of black frames.
        // Every texture that reaches the bindless array through register_material has its heap slot written
        // there, but this one is created above that loop and never passes through it - so slot
        // heap_slots::textures + 0 stayed EMPTY and sampled as zero. A material slot with no texture arrives
        // here (Sponza's stone carries no occlusion map), so its ao read 0 - and shading.glsl multiplies BOTH
        // the diffuse ambient and the specular IBL by s.ao, which left those surfaces lit by the sun alone:
        // a black interior, while the metal test assets (whose materials do carry an occlusion map, and whose
        // diffuse term is multiplied away by (1 - metallic) anyway) looked untouched. The startup probe had
        // it in the log the whole time: "the heap-native probe sampled grid slot 16384 ... read back
        // 0x00000000", on the very slot this write fills, while the material table's white record read 0xffff.
        // @note core::heap_slot_offset() is defined below this constructor, so the arithmetic is spelled out: a slot
        //       number is already absolute and the stride is the one every heap array agrees on.
        if (this->vulkan_core.descriptor_heaps.ready()) {
            auto const* const white_detail = this->vulkan_core.vma.get_image_detail(this->owned_textures.back().handle());
            if (white_detail != nullptr) {
                VkImageViewCreateInfo const heap_view = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                                         .pNext = nullptr,
                                                         .flags = 0,
                                                         .image = white_detail->image,
                                                         .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                                         .format = VK_FORMAT_R8G8B8A8_UNORM,
                                                         .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
                                                         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
                VkDeviceSize const white_offset = static_cast<VkDeviceSize>(core::heap_slots::textures + this->white_texture_index) * core::heap_slot_stride;
                if (!this->vulkan_core.descriptor_heaps.write_image(white_offset, heap_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)) {
                    utility::log("descriptor heap: the white fallback texture did not reach grid slot {}", core::heap_slots::textures + this->white_texture_index);
                }
            }
        }

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
                                       this->material_mapped,
                                       // it goes on the descriptor heap, and a heap descriptor for a buffer is an
                                       // ADDRESS RANGE - so this buffer needs a device address
                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

        // ---- THE MATERIAL TABLE'S HEAP DESCRIPTOR, written ONCE here ----
        //
        // A storage buffer descriptor is just an address range, so this is the simplest descriptor in the
        // renderer: no image view to create, no sampler, no embedded sampler. The buffer has a FIXED capacity and
        // is created above, so its address is stable and one write covers it - which is why this is not a
        // per-frame write. It goes at its OWN GRID SLOT (core::heap_slots::materials), which is the same number a
        // heap-native shader bakes as `heap_slots_materials` (shaders/heap_slots.glsl): the write and the read are
        // the same number by construction rather than by review.
        if (this->vulkan_core.descriptor_heaps.ready() && this->vulkan_core.heap_grid_offset != VK_WHOLE_SIZE) {
            auto const* const detail = this->vulkan_core.vma.get_buffer_detail(this->material_buffer.handle());
            if (detail != nullptr) {
                VkBufferDeviceAddressInfo const address_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = detail->buffer};
                VkDeviceAddress const address = vkGetBufferDeviceAddress(this->vulkan_core.device, &address_info);
                bool const written = this->vulkan_core.descriptor_heaps.write_buffer(static_cast<VkDeviceSize>(core::heap_slots::materials) * core::heap_slot_stride,
                                                                                     address,
                                                                                     static_cast<VkDeviceSize>(vulkan::material_capacity) * sizeof(material_record),
                                                                                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
                utility::log("descriptor heap: material table {} (address 0x{:x}, {} records, offset {})",
                             written ? "written" : "NOT written",
                             address,
                             vulkan::material_capacity,
                             static_cast<VkDeviceSize>(core::heap_slots::materials) * core::heap_slot_stride);
            }
        }

        // Per-instance transform buffer (set 0 binding 6): one mat4 per instance, host-visible;
        // filled by set_instanced_draw() for instanced stress draws (see pbr.vert)
        std::vector<unsigned char> const zeroed_instances(static_cast<size_t>(vulkan::instance_capacity) * sizeof(glm::mat4), 0);
        init_utils::create_host_buffer(this->vulkan_core,
                                       std::as_bytes(std::span(zeroed_instances)),
                                       vulkan::buffer_type::storage_coherent,
                                       "instance transform buffer",
                                       this->instance_buffer,
                                       this->instance_mapped,
                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT); // the heap writes this one's address

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
                                        &this->motion_mapped,
                                        // THE HEAP'S REQUIREMENT: a descriptor written as a device ADDRESS range
                                        // needs the buffer to be addressable (VUID-VkBufferDeviceAddressInfo-
                                        // buffer-02601 says so, and validation did, the moment this write went in).
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
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
                                        &this->skin_mapped,
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT); // the heap writes this one's address

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
                                        &this->morph_mapped,
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT); // the heap writes this one's address

        // Reserve table index 0 as the DEFAULT material (white textures + identity factors):
        // registrations that overflow the table degrade to it (see register_material). Done
        // FIRST so it always lands at index 0 - the raw zeroed record at 0 would render black
        // (all factors zero), not white. Safe here: the texture it uploads is registered on the
        // heap as it arrives, exactly as a later material's is.
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
    // secondaries" item of the trim list): the SHAPE is policy and stays
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
        this->rt_binding_written.assign(this->vulkan_core.velocity_images.size(), VK_NULL_HANDLE);
        this->gbuffer_targets_written.assign(this->vulkan_core.gbuffer_images[0].size(), false);
        // a generation has nothing to blend with, and it is reset HERE rather than only on the off -> on
        // vector reads as "no history" for every frame, which silently turns the temporal resolve into a
        // pass-through of the raw trace).
        // ... and its FRAME COUNT restarts with it: the cold-start widening is measured in frames since the
        // accumulation restarted, and a new generation IS that restart (see frame_facts::gi_cold_start).
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

        // Light UBO (scene block slot 7): one buffer PER FRAME SLOT (host-visible, mapped), so
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
                                        &this->light_mapped,
                                        // it goes on the descriptor heap (a heap descriptor for a buffer is its
                                        // device address), so the address has to exist - validation states it as
                                        // VUID-VkBufferDeviceAddressInfo-buffer-02601 the moment it is queried
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    }

    namespace {

        /// THE HEAP'S COPY OF ONE IMAGE, built from the SAME arguments `core::make_image_view` uses (see
        /// vulkan::make_image_view_info in vulkan.constant_init): a heap image descriptor carries a CREATE INFO
        /// rather than a view, and the driver makes the view inside it. That is why this is called where the image
        /// and its view are created - only that site knows the format, the view type and the range.
        bool write_heap_grid_image(core& vk, uint32_t const slot, VkImage const image, VkFormat const format, VkImageViewType const type, VkImageAspectFlags const aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
            if (!vk.descriptor_heaps.ready() || vk.heap_grid_offset == VK_WHOLE_SIZE || image == VK_NULL_HANDLE) {
                return false;
            }
            VkImageViewCreateInfo const view_info = make_image_view_info(image, format, type, aspect, VK_REMAINING_MIP_LEVELS, VK_REMAINING_ARRAY_LAYERS);
            return vk.descriptor_heaps.write_image(core::heap_slot_offset(slot), view_info, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        /**
         * @brief write ONE per-slot binding of the scene block into the GRID - one slot per frame in flight
         *
         * THE HEAP'S BUFFER PATTERN, in one place: a heap descriptor for a buffer IS its address range, and a
         * per-slot binding's entry must name THAT slot's buffer - so this walks the per-slot vector, takes each
         * buffer's device address and writes it at `core::heap_slot_offset(slot_base + slot)`: one GRID slot per frame in
         * flight, which is exactly where the shaders read it (`heap_slots_scene_camera + heap_frame_slot` and its
         * neighbours). The heap is the only path now, so "the heap is not in use" is a startup
         * failure rather than a quiet fall back to a descriptor set.
         */
        void write_heap_scene_buffer(core& vk, std::vector<vk_buffer> const& buffers, uint32_t const slot_base, VkDeviceSize const size, VkDescriptorType const type) {
            if (!vk.descriptor_heaps.ready() || vk.heap_grid_offset == VK_WHOLE_SIZE) {
                return;
            }
            uint32_t written = 0;
            for (uint32_t slot = 0; slot < buffers.size(); ++slot) {
                auto const* const detail = vk.vma.get_buffer_detail(buffers[slot].handle());
                if (detail == nullptr) {
                    continue;
                }
                VkBufferDeviceAddressInfo const info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = detail->buffer};
                VkDeviceAddress const address = vkGetBufferDeviceAddress(vk.device, &info);
                VkDeviceSize const offset = core::heap_slot_offset(slot_base + slot);
                if (vk.descriptor_heaps.write_buffer(offset, address, size, type)) {
                    ++written;
                } else {
                    utility::log("descriptor heap: the per-frame buffer for grid slot {} (frame slot {}) did not fit at offset {}", slot_base, slot, offset);
                }
            }
            // SUCCESS IS LOGGED TOO, and that is not noise: a heap write has no picture to show for itself until
            // the shaders read the heap, so "no failure line" and "the buffers were empty, so nothing was written"
            // look exactly alike. This line is what tells them apart (it is the same reason the texture array and
            // the material table each log their count).
            utility::log("descriptor heap: {} per-frame descriptor(s) written for grid slots {}..{}", written, slot_base, slot_base + (buffers.empty() ? 0u : static_cast<uint32_t>(buffers.size()) - 1u));
        }
    } // namespace

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
                                        &this->cluster_count_mapped,
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT); // heap-bound
        // The index rows are host-visible for the same reason, but nothing on the CPU ever writes
        // through the mapping: the dispatch fills them, so the mapped list stays a nullptr.
        init_utils::create_host_buffers(this->vulkan_core,
                                        static_cast<uint32_t>(slots),
                                        std::as_bytes(std::span(zero_indices)),
                                        vulkan::buffer_type::storage_coherent,
                                        "cluster index buffer",
                                        this->cluster_index_buffers,
                                        nullptr,
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT); // heap-bound

        // The clustered-light pass's TWO buffers, into the heap's per-slot blocks (bindings 11 and 12): this stage
        // is the first whose WHOLE set-0 interface is buffers (camera 0, light 7, counts 11, indices 12), which is
        // what makes it the one to migrate first - no image descriptors, no embedded samplers, nothing but address
        // ranges - and therefore the one that can be verified byte for byte before the image side is attempted.
        // The sizes are the buffers' REAL sizes, not VK_WHOLE_SIZE: a heap buffer descriptor is an
        // address RANGE, and validation states the rule as VUID-VkDeviceAddressRangeKHR-address-11365 - address plus
        // size must stay inside the buffer, which VK_WHOLE_SIZE cannot satisfy.
        write_heap_scene_buffer(this->vulkan_core, this->cluster_count_buffers, core::heap_slots::cluster_counts, static_cast<VkDeviceSize>(vulkan::max_cluster_count) * sizeof(uint32_t), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        write_heap_scene_buffer(this->vulkan_core, this->cluster_index_buffers, core::heap_slots::cluster_indices, static_cast<VkDeviceSize>(vulkan::max_cluster_count) * vulkan::cluster_light_capacity * sizeof(uint32_t), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        write_heap_scene_buffer(this->vulkan_core, this->camera_buffers, core::heap_slots::scene_camera, sizeof(camera_ubo), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        // ... and the rest of the per-frame arrays, from the same place and with the SAME sizes the scene block
        // writes them with (see ensure_scene_heap_slots's write_buffer_binding calls): motion (binding 13), skin (9) and
        // morph (10), each a two-slot array whose slot is the frame's. Bound here rather than in the per-slot loop
        // because a heap descriptor is an ADDRESS: the buffers are allocated once, so their addresses do not
        // change per frame, and only the CONTENTS are rewritten (see the per-frame slot rule in runtime.cppm).
        write_heap_scene_buffer(this->vulkan_core, this->motion_buffers, core::heap_slots::previous_transforms, static_cast<VkDeviceSize>(vulkan::scene_motion_capacity) * sizeof(glm::mat4), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        write_heap_scene_buffer(this->vulkan_core, this->skin_buffers, core::heap_slots::skin_matrices, static_cast<VkDeviceSize>(vulkan::scene_skin_capacity) * sizeof(glm::mat4), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        write_heap_scene_buffer(this->vulkan_core, this->morph_buffers, core::heap_slots::morph_data, static_cast<VkDeviceSize>(vulkan::scene_morph_capacity) * sizeof(float), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        // THE INSTANCE TRANSFORM TABLE (set 0 binding 6) is the odd one: a SINGLE buffer rather than one per frame
        // slot (see runtime.cppm's member), so it takes ONE grid slot instead of a two-slot array - which is what
        // heap_slots::instance_transforms reserved. Written from the same size the scene block gives it.
        if (this->vulkan_core.descriptor_heaps.ready() && this->vulkan_core.heap_grid_offset != VK_WHOLE_SIZE) {
            auto const* const instance_table_detail = this->vulkan_core.vma.get_buffer_detail(this->instance_buffer.handle());
            if (instance_table_detail != nullptr) {
                VkBufferDeviceAddressInfo const instance_table_address_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = instance_table_detail->buffer};
                VkDeviceAddress const instance_table_address = vkGetBufferDeviceAddress(this->vulkan_core.device, &instance_table_address_info);
                if (!this->vulkan_core.descriptor_heaps.write_buffer(core::heap_slot_offset(core::heap_slots::instance_transforms), instance_table_address, static_cast<VkDeviceSize>(vulkan::instance_capacity) * sizeof(glm::mat4), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)) {
                    utility::log("descriptor heap: the instance transform table did not reach grid slot {}", core::heap_slots::instance_transforms);
                }
            }
        }
    }

    void runtime::ensure_scene_heap_slots() {
        if (!this->cluster_count_buffers.empty()) {
            return; // the scene's heap slots are already written (the first primitive asked for them)
        }
        // THERE IS NO SCENE SET TO ENSURE ANY MORE - what this function still does is what the heap needs: the
        // shadow map and the clustered-light buffers must exist before the heap slots that name them are written,
        // and their creation is deferred to here so the app config that sets the cascade count has already run
        // (see the constructor note).
        this->ensure_shadow_resources();
        this->ensure_cluster_buffers();
        // binding 7 (light UBO, per-slot) is bound into each slot's heap block here, together with the shadow map's
        // grid slot.
        this->write_light_and_shadow_bindings();
    }

    void runtime::write_light_and_shadow_bindings() {
        // binding 7 (light UBO) + binding 8 (shadow map): BOTH point at THIS slot's own
        // resources (per-slot light buffers like the camera UBO, per-slot shadow images), so no
        // per-frame re-pointing is needed and an in-flight frame never shares a buffer the next
        // frame rewrites. Only the HEAP half is left: the two bindings are written into this slot's heap block
        // (and the shadow map into its grid slot) rather than into a descriptor set.
        for (int slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
            auto const* light_detail = this->vulkan_core.vma.get_buffer_detail(this->light_buffers[static_cast<std::size_t>(slot)].handle());
            if (light_detail == nullptr) {
                utility::panic("failed to get light ubo buffer detail");
            }
            auto const* shadow_detail = this->vulkan_core.vma.get_image_detail(this->shadow_images[static_cast<std::size_t>(slot)].handle());
            if (shadow_detail == nullptr) {
                utility::panic("failed to get shadow map image detail");
            }

            // THE SHADOW MAP GOES ONTO THE GRID HERE, because an image binding cannot be written the way a buffer
            // binding is: its heap descriptor is a CREATE INFO, rebuilt from the same arguments
            // core::make_depth_array_view uses - this slot's image, the depth format, a 2D-array view and the DEPTH
            // aspect (a colour aspect here would be a validation error, not a wrong picture). Which slot it
            // occupies is the frame's, matching shadow_images[slot].
            if (!write_heap_grid_image(this->vulkan_core, core::heap_slots::shadow_map + static_cast<uint32_t>(slot), shadow_detail->image, this->vulkan_core.depth_format, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_ASPECT_DEPTH_BIT)) {
                utility::log("descriptor heap: the shadow map for frame slot {} did not reach grid slot {}", slot, core::heap_slots::shadow_map + static_cast<uint32_t>(slot));
            }

            // ---- AND THE LIGHT UBO, INTO THE HEAP'S BLOCK FOR THIS SLOT ----
            //
            // A heap descriptor for a buffer IS an address range: take the buffer's device address, write it at
            // the binding's offset inside this slot's block (core reserved the block and computed the offsets), and
            // nothing else changes. What does NOT work this way is an IMAGE binding: a heap image descriptor
            // carries a VkImageViewCreateInfo while a VkDescriptorImageInfo carries a view, not the image and range
            // that create info is made of - which is why the shadow map above is written where its image is known.
            if (this->vulkan_core.descriptor_heaps.ready() && this->vulkan_core.heap_grid_offset != VK_WHOLE_SIZE) {
                VkBufferDeviceAddressInfo const address_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = light_detail->buffer};
                VkDeviceAddress const light_address = vkGetBufferDeviceAddress(this->vulkan_core.device, &address_info);
                VkDeviceSize const heap_offset = core::heap_slot_offset(core::heap_slots::scene_light + slot);
                if (!this->vulkan_core.descriptor_heaps.write_buffer(heap_offset, light_address, sizeof(light_ubo), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)) {
                    utility::log("descriptor heap: the light UBO did not fit slot {}'s block at offset {}", slot, heap_offset);
                }
            }
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
        // ... and the heap's copy of it, at its own grid slot (see docs/descriptor_heap_migration.md): written
        // HERE because this is the site that knows the format and the view type, which is what a heap image
        // descriptor is made of. An image whose BINDING is later repointed (the furnace mode) needs a rewrite
        // beside that change - the heap does not follow a view.
        if (!write_heap_grid_image(this->vulkan_core, core::heap_slots::env_cube, env_detail->image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_CUBE)) {
            utility::log("descriptor heap: the environment cube did not reach grid slot {}", core::heap_slots::env_cube);
        }

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
        if (!write_heap_grid_image(this->vulkan_core, core::heap_slots::irradiance_cube, irr_detail->image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_CUBE)) {
            utility::log("descriptor heap: the irradiance cube did not reach grid slot {}", core::heap_slots::irradiance_cube);
        }

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
        if (!write_heap_grid_image(this->vulkan_core, core::heap_slots::brdf_lut, lut_detail->image, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_VIEW_TYPE_2D)) {
            utility::log("descriptor heap: the BRDF LUT did not reach grid slot {}", core::heap_slots::brdf_lut);
        }

        this->env_sampler = this->vulkan_core.make_sampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, static_cast<float>(info.env_mip_count - 1));
        this->ibl_ready = true;
        // The three images above are already on the heap, written where their images are (see the
        // write_heap_grid_image calls): a heap image descriptor is a CREATE INFO, so there is no separate
        // "point the binding at it" step and nothing to rewrite here.
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
        uint32_t heap_texture_descriptors = 0; // how many went into the descriptor heap (see the log below)
        for (int i = 0; i < 5; ++i) {
            texture_input const& tex = *slots[i].first;
            if (!tex.valid || tex.data.empty()) {
                texture_indices[i] = this->white_texture_index; // white fallback
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

            // A texture is registered ONCE, at scene load, and never rewritten - which is why the texture array is
            // the first binding this renderer puts on the heap: there is no per-frame rewrite and therefore no
            // frame-in-flight hazard to design around. The array starts at its own grid slot
            // (core::heap_slots::textures) and advances one SLOT per texture, i.e. 64 B - NOT the device's
            // imageDescriptorSize: the grid's single stride is what lets a shader index it with
            // `descriptor_stride = 64` (see docs/descriptor_heap_migration.md). The descriptor is a VIEW TO CREATE
            // rather than the view above: VkImageDescriptorInfoEXT carries a VkImageViewCreateInfo and the driver
            // makes the view itself. The values below are the ones core::make_image_view uses, on purpose - a view
            // that differs in mip range would sample a different image than the descriptor-set path.
            //
            // Nothing READS the heap yet, so a failure here is a log line and not a wrong frame - but it is the
            // write path that has to work first.
            if (this->vulkan_core.descriptor_heaps.ready()) {
                auto const* const texture_detail = this->vulkan_core.vma.get_image_detail(this->owned_textures.back().handle());
                if (texture_detail != nullptr) {
                    VkImageViewCreateInfo const heap_view = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                                             .pNext = nullptr,
                                                             .flags = 0,
                                                             .image = texture_detail->image,
                                                             .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                                             .format = slots[i].second,
                                                             .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
                                                             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
                    VkDeviceSize const heap_offset = core::heap_slot_offset(core::heap_slots::textures + index);
                    if (this->vulkan_core.descriptor_heaps.write_image(heap_offset, heap_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)) {
                        ++heap_texture_descriptors;
                    } else {
                        utility::log("descriptor heap: texture {} did not fit the resource heap at offset {}", index, heap_offset);
                    }
                }
            }
        }

        // THE HEAP-NATIVE PROBE, ONCE, now that the bindless array is actually in the heap: it is the first thing
        // in this renderer to read the heap instead of a descriptor set, and its answer goes to the log (see
        // run_heap_probe). Texture slot 0 is the white placeholder, which is registered first and always exists.
        if (heap_texture_descriptors != 0u) {
            this->run_heap_probe(static_cast<uint32_t>(core::heap_slots::textures));
        }
        // ... and its GRAPHICS half, twice: once with the material table's REAL grid slot (which must come back
        // white, the default material's base colour) and once with a deliberately WRONG one (which must not). The
        // pair is the negative proof the mechanism needs - the same draw, the same shader, one different number.
        if (this->vulkan_core.descriptor_heaps.ready() && this->vulkan_core.heap_grid_offset != VK_WHOLE_SIZE) {
            this->run_heap_graphics_probe(static_cast<uint32_t>(core::heap_slots::materials));
            this->run_heap_graphics_probe(static_cast<uint32_t>(core::heap_slots::materials) + 1u);
        }

        // SAY WHAT WENT INTO THE HEAP, because the success path of a heap write is silent by nature (it returns
        // true and writes memory) and "no failure line" is not evidence that anything happened. This is the line a
        // reader checks to know the texture array really is on the heap; the mapping that points a shader stage at
        // it is the step after this one.
        if (heap_texture_descriptors > 0) {
            utility::log("descriptor heap: {} texture descriptors written ({} B each, {} KiB resource heap)",
                         heap_texture_descriptors,
                         this->vulkan_core.descriptor_heaps.limits().image_descriptor_size,
                         this->vulkan_core.descriptor_heaps.resource_size() / 1024);
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
        // MMD's own outline inputs (docs/zzz_shading.md): they reach the outline's two stages through this
        // record, and bit6 is what tells them the model authored an edge at all - black is a legitimate
        // colour and 0 a legitimate size, so the values cannot say it themselves.
        if (info.factors.mmd_edge_present) {
            record.npr_edge = glm::vec4(glm::vec3(info.factors.mmd_edge_color), info.factors.mmd_edge_size);
            record.flags |= 64u; // bit6: the model authored MMD edge data
        }
        // MMD's sphere map (docs/zzz_shading.md): the texture index rides the record's spare 4 bytes and the
        // MODE rides two flag bits, so the record keeps its size and every shader copy keeps its layout.
        // 0 = none, 1 = multiply, 2 = add, 3 = sub-texture.
        record.sphere_index = info.factors.mmd_sphere_index;
        if (info.factors.mmd_unlit) {
            // bit7 is the PAINTED flag the lighting stage reads out of the G-buffer's one-byte material
            // channel, so it has to be bit7 for the same reason the face bit did: a shading flag only
            // reaches the deferred lighting stage if it fits inside that byte.
            record.flags |= 128u;
        }
        if (info.factors.mmd_face) {
            // bit7, NOT a higher bit: the deferred path's only channel for material flags is ONE BYTE in the
            // G-buffer (out_material.a), and the shading stage reads the face there rather than from the
            // record - so the face bit has to live inside that byte. The sphere MODE moved up to bits 8-9 for
            // the same reason, and it loses nothing: the stage that applies the sphere (the G-buffer pass)
            // reads the record directly.
            // ... while the FACE flag rides bit8, which only the G-buffer PASS can see - and that is enough,
            // because the one thing needing it (the redrawn nose mark) is drawn on the albedo there.
            record.flags |= 256u; // bit8: this material is part of the model's FACE block
        }
        record.flags |= static_cast<uint32_t>(std::clamp(info.factors.mmd_sphere_mode, 0, 3)) << 8u;

        // ---- 3. Content-address the record, then append (or degrade on overflow) ----
        // Identical materials (same texture slots, factors and flags) share ONE table entry:
        // registration happens per primitive, so a scene with N primitives over M shared glTF
        // materials would otherwise append N records and burn the table needlessly. The key is
        // the byte-exact 96-byte record carried in a data_block - no hash collisions, because
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

} // namespace vulkan