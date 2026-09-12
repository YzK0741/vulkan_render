// ============================================================================
// module: vulkan.runtime
// module version: 0.19.1  (independent of the app version in CMakeLists project(VERSION))
//
// The renderer core: per-frame-slot frame facade (pace/record/submit phases,
// scene resources, parallel secondary-CB recording). It re-exports its peer
// modules vulkan.scene_tree (scene storage + GPU primitives) and
// vulkan.render_environment (per-worker draw state) - the frame draws through
// both, so they are versioned as ONE unit because they share the scene / draw
// interface and evolve together.
// Depends on vulkan.core (GPU), vulkan.math (IBL) and utility, with the frame
// struct fills coming from vulkan.constant_init.
//
// evolve: bump MAJOR on breaking interface changes, MINOR on additive features,
//         PATCH on internal fixes - independently of the rest of the project.
// ============================================================================
module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>

export module vulkan.runtime;

import vulkan.profiling;
import vulkan.bindings; // the per-image descriptor-set families (the G-buffer debug view's for now)
export import vstd;
export import vulkan.core;
export import vulkan.core.filter;
export import vulkan.scene_tree;         // scene storage + the abstract leaf interface (pure CPU)
export import vulkan.primitive;          // the GPU primitives + material/UBO records (peer module)
export import vulkan.render_environment; // per-worker draw state (peer module)
import utility;
export import vulkan.gui; // optional debug overlay (gui_content): exported so callers can manage panels/widgets via debug_gui()

/**
 * @file runtime.cppm
 * @defgroup vulkan_runtime Vulkan Runtime Facade
 * @brief runtime facade: a thin wrapper exposing all functionality of vulkan::core
 * @note
 *      - use operator-> to access the filtered core view (core_filter, e.g. runtime->get_device())
 *      - the inner core's lifetime is tied to the runtime
 */
namespace vulkan {
    /**
     * @ingroup vulkan_runtime
     * @brief orbit camera state, updated by the mouse callbacks registered in the runtime constructor
     */
    export struct orbit_camera {
        double last_x = 0.0;
        double last_y = 0.0;
        bool dragging = false;
        float yaw = 0.0f;
        // level view: the skybox horizon (the direction parallel to the ground plane) then sits
        // exactly at the screen center, where the primitive is framed against it
        float pitch = 0.0f;
        float distance = 2.2f;
        // point the camera looks at and orbits around (default origin; main may sink it
        // together with the scene so the camera follows the primitive)
        glm::vec3 target = glm::vec3(0.0f);
    };

    /**
     * @ingroup vulkan_runtime
     * @brief outcome of one frame step or of a whole frame (the frame phases); the caller reacts to it
     * @note shared by the split frame steps: each step returns proceed when it succeeded and the
     *       caller may continue to the next step (for the frame path that means a frame was
     *       recorded, submitted and presented). Failures are granular per stage so a caller can
     *       tell WHERE the frame broke (acquire / command-buffer recording / submit / present).
     */
    export enum class frame_status {
        proceed, // step succeeded / a frame was presented; caller continues
        skipped, // not renderable this iteration (window minimized / swapchain recreated); caller yields and retries
        closed,  // the window was closed (ESC or the native close button); caller exits the loop
        // stage-specific failures: a fatal Vulkan error at that point; caller exits the loop
        acquire_failed,         // vkAcquireNextImageKHR failed (other than out-of-date)
        begin_recording_failed, // vkBeginCommandBuffer failed
        end_recording_failed,   // vkEndCommandBuffer failed
        submit_failed,          // vkQueueSubmit failed
        present_failed,         // vkQueuePresentKHR failed (other than out-of-date / suboptimal)
    };

    /**
     * @ingroup vulkan_runtime
     * @brief true for any stage-specific failure (not proceed / skipped / closed)
     */
    export [[nodiscard]] constexpr bool is_failure(frame_status const status) noexcept {
        return status != frame_status::proceed && status != frame_status::skipped && status != frame_status::closed;
    }

    /**
     * @ingroup vulkan_runtime
     * @brief named tiers for tasks submitted to the runtime's shared task pool (run_tasks).
     *
     * Each enumerator names one frame-time parallel stage of the runtime; it maps onto the
     * underlying pool's integer priority (higher numeric values run earlier) and is ALSO that
     * stage's wait group, so distinct stages never block on each other's tasks. Values follow
     * frame causality: the stage that PRODUCES data (animation sampling) carries the higher
     * value so it runs first, the stage that CONSUMES it (command recording / rendering) the
     * lower one - "prepare the data, then render". Add an enumerator per future stage instead
     * of passing magic ints at call sites (slot new stages in by execution order; renumbering
     * is free, these values are compile-time only).
     */
    export enum class task_priority : int {
        animation = 1, // runs first: per-source animation sampling fan-out (animation::controller::update)
        recording = 0, // runs after the data is ready: pass/secondary command-buffer recording (record_main_drawcalls)
        // future frame-time stages (culling, skin upload, ...) slot in between by execution order
    };

    /**
     * @ingroup vulkan_runtime
     * @brief vulkan runtime facade class
     * @note
     *      - use operator-> to access the filtered core view (core_filter, e.g. runtime->get_device())
     *      - default construction performs the whole core initialization (window/instance/device/swap chain etc.)
     *        and registers the orbit camera mouse callbacks on the window
     */
    export class runtime {
        core vulkan_core;

        /**
         * @ingroup vulkan_runtime
         * @brief the CPU frame phases that are measured per frame (see cpu_timing_summary)
         *
         * Above a few hundred fps the frame stops being GPU-bound: measured on the RTX 4060 at
         * 960x720, cutting the GPU frame from 0.52 to 0.41 ms raised fps by 74% while cutting it
         * further to 0.38 ms changed nothing, and halving the shadow map (which halves the shadow
         * pass rasterization) changed neither - so what is left is CPU-side recording,
         * synchronization and presentation. These phases say which: `pace` is the wait for the frame
         * slot, i.e. where GPU/present backpressure surfaces.
         */
        // `cluster` and `shadow` are SUB-measurements of `scene` (which spans the whole
        // record_main_drawcalls() call). The shadow pass records one secondary + one rendering
        // instance + one barrier PER CASCADE on the primary thread, which makes it the first suspect
        // for the scene phase's ~1.3 ms, so it is reported separately - and, being inside `scene`, it
        // is NOT added to the total again.

        // set while the window is iconified; the restore transition recreates the swapchain
        bool was_minimized = false;

        // ---- shared scene resources (single flat descriptor set, see core::init_scene_layouts) ----
        // camera UBO: one buffer per frame slot, updated once per frame, shared by every primitive
        std::vector<vk_buffer> camera_buffers = {};
        std::vector<void*> camera_mapped = {};
        // texture registry: flat entries of the set 0 binding 1 array (raw handles); the owning
        // views / vma images live in the vectors below (vk_image RAII frees the GPU image when
        // the runtime goes away). texture_slot_cache deduplicates uploads by CONTENT (xxh3 of
        // the decoded bytes + format + dimensions): several materials sharing one glTF image
        // (same decoded pixels, different byte copies) all point at the same array slot instead
        // of uploading a copy per material - the loader hands each material its own byte copy,
        // so a pointer key would never match.
        std::vector<VkImageView> texture_array_views = {};
        std::vector<vk_image_view> owned_texture_views = {};
        std::vector<vk_image> owned_textures = {};
        uint32_t white_texture_index = 0;
        std::map<std::tuple<utility::xxh3_digest, VkFormat, std::uint32_t, std::uint32_t, std::uint32_t>, uint32_t> texture_slot_cache = {}; // digest (data_block<16>), format, width, height, mip_levels
        // scene-wide IBL (bindings 2-4): prefiltered env / irradiance / BRDF LUT, uploaded once
        std::vector<vk_image_view> ibl_views = {};
        std::vector<vk_image> ibl_images = {};
        vk_sampler texture_sampler = {};
        vk_sampler env_sampler = {};
        // GPU material table (set 0 binding 5): one material_record per entry (texture indices +
        // factors + flags); primitives only push their material_index. Host-visible, written at
        // registration, read-only for the GPU.
        vk_buffer material_buffer = {};
        void* material_mapped = nullptr;
        uint32_t material_count = 0;
        // content-addressed material dedup + overflow fallback (see register_material):
        // material_slot_cache keys the full material_record bytes (texture indices + factors +
        // flags) as a byte-exact data_block<sizeof(material_record)>, so N primitives sharing
        // one glTF material register ONE record instead of N identical appends (an unordered
        // key: data_block's own FNV-1a hasher + byte-equality); when the table really fills up,
        // later registrations degrade to the reserved default material at index 0 (registered
        // in init_scene_resources) with a one-time log instead of a hard panic.
        std::unordered_map<utility::data_block<sizeof(vulkan::material_record)>, material_id,
                           utility::data_block<sizeof(vulkan::material_record)>::hasher>
            material_slot_cache = {};
        bool material_overflow_logged = false;
        // same degradation policy for the texture array: when scene_texture_capacity distinct
        // images are in use, later slots fall back to the white element (0) with a one-time log
        bool texture_overflow_logged = false;
        // per-instance transforms for instanced primitives (scene set binding 6): one mat4 per
        // instance, host-visible. The buffer is ONE shared region split into per-instanced-
        // primitive slices: make_instanced_primitive() appends its transforms at instance_cursor
        // (mat4 units), hands the slice start to the new primitive via push.instance_base, and
        // advances the cursor, so several instanced primitives coexist without overwriting each
        // other. instance_cursor resets to 0 whenever instanced primitives are cleared (they are
        // the only writers).
        vk_buffer instance_buffer = {};
        void* instance_mapped = nullptr;
        uint32_t instance_cursor = 0;
        // per-joint skin matrices (scene set binding 9): ONE buffer per frame slot, like the
        // camera UBO — each slot's scene set always points at its own buffer, so a frame being
        // rendered never shares the buffer the next frame rewrites. scene_skin_capacity mat4s
        // each, host-visible; indices 0-3 are the identity block (unskinned fallback),
        // per-skin joint blocks follow. Filled per frame by set_skin_matrices(); primitives
        // reference their block via material_push_constants::skin_base
        std::vector<vk_buffer> skin_buffers = {};
        std::vector<void*> skin_mapped = {};
        // morph data (scene set binding 10): ONE buffer per frame slot, like the skin matrices.
        // scene_morph_capacity floats each, host-visible. The caller writes per-primitive blocks
        // (morph deltas + weights) through morph_scratch() and points primitives at them via
        // material_push_constants::morph_* fields
        std::vector<vk_buffer> morph_buffers = {};
        std::vector<void*> morph_mapped = {};
        // per-slot scene descriptor sets: all pipelines share the scene layout, so every frame
        // slot gets one set from it. A set's per-slot bindings (0 camera / 7 light / 8 shadow /
        // 9 skin / 10 morph) always point at that slot's own resources and never change, so an
        // in-flight frame can never observe the next frame's descriptors (no update-after-bind
        // race).
        std::array<vk_descriptor_set, vulkan::core::MAX_FRAMES_IN_FLIGHT> scene_sets = {};
        bool scene_set_created = false;
        // the frame slot paced by the last successful pace_and_acquire();
        // per-frame host writes (set_skin_matrices / morph_scratch) target this slot's buffers
        uint32_t active_slot = 0;
        bool ibl_ready = false;
        // background pass (fullscreen triangle, no depth test): drawn first every frame
        std::optional<vk_pipeline> skybox_pipeline = std::nullopt;

        // ---- GPU pass timing (see gpu_mark / gpu_timing_summary) ----
        // Which pass boundaries a frame marks. The sequence is FIXED: a pass that does not record
        // this frame (shadow off, bloom intensity 0, FXAA off) still writes its mark immediately
        // after the previous one, so the measured interval is 0 and the label-to-interval mapping
        // never shifts. Mark i is written at the END of the pass named by gpu_timing_labels[i],
        // which is why the label table is one entry shorter than the mark list.
        enum class gpu_mark_id : uint32_t {
            frame_begin = 0, // first command of the frame (TOP_OF_PIPE)
            shadow_end,      // after the shadow pass + its sampling barrier
            scene_end,       // after the geometry instance: forward main (opaque + transparent), or
                             // the background + G-buffer pass in the deferred path
            lighting_end,    // after the deferred lighting stage (~0 in the forward path)
            taa_end,         // after the TAA resolve + its history copy (~0 when TAA is off)
            main_end,        // after the last scene-side work of the frame (the debug view, when it runs)
            bloom_end,       // after the bloom prefilter/downsample chain
            composite_end,   // after the composite (exposure + ACES + display encode)
            fxaa_end,        // after the FXAA pass (and the overlay, when FXAA draws it)
            frame_end,       // last command of the frame (the screenshot copy + present barrier)
            count,           // not a mark: the number of marks a frame writes
        };
        static constexpr uint32_t gpu_mark_count = static_cast<uint32_t>(gpu_mark_id::count);
        static_assert(gpu_mark_count <= vulkan::gpu_timing_mark_capacity, "the core's timestamp pool must hold one frame's marks");
        // labels of the intervals between consecutive marks (interval i = mark i -> mark i + 1).
        // new_line starts a new line in the overlay's report, which has to fit one narrow panel
        // row; the log line ignores it and prints everything on one line.
        // "scene" is the geometry instance of whichever path is active, "lighting" is the deferred
        // lighting stage (0 ms in the forward path, where the shading happens inside the scene
        // instance - as does the forward transparent pass, whose cost therefore shows up in "scene"
        // as well), and "debug" is the G-buffer debug view when it runs.
        struct gpu_timing_label {
            std::string_view name;
            bool new_line; // begin a new line in the overlay report
        };
        static constexpr std::array<gpu_timing_label, gpu_mark_count - 1> gpu_timing_labels = {{
            {"shadow", false},
            {"scene", false},
            {"lighting", false},
            {"taa", false},
            {"debug", true},
            {"bloom", false},
            {"composite", false},
            {"fxaa", false},
            {"tail", false},
        }};
        // whether to collect pass timings at all ([render] gpu_timings); the device must be able
        // to timestamp as well, which core::gpu_timing_available() reports
        bool gpu_timings_enabled = true;
        // rolling window of measured intervals: every GPU_TIMING_WINDOW frames the collected
        // samples are averaged, logged, and the window starts over (the GUI label reads the
        // window's running mean, so it stays live instead of dropping to 0 on the reset)
        static constexpr uint32_t GPU_TIMING_WINDOW = 60;
        std::array<double, gpu_mark_count - 1> gpu_timing_sum = {}; // current window's summed ms
        uint32_t gpu_timing_window_frames = 0;                      // frames sampled in the current window
        // The overlay's copy of the last COMPLETED timing window (gpu_timing_summary): a label whose
        // text changes width every frame re-wraps against the panel edge and makes the whole overlay
        // twitch, so this is refreshed once per window and every number is a fixed-width field.
        std::string gpu_timing_report_label = {};
        uint32_t gpu_timing_marks_measured = 0; // intervals the last measured frame had
        void gpu_mark(VkCommandBuffer command_buffer, gpu_mark_id mark, VkPipelineStageFlagBits stage) noexcept;

        // ---- CPU frame phase timing (same 60-frame window as the GPU marks) ----
        // A scope timer rather than manual marks: every phase function has early returns
        // (skipped/minimized/closed) that must still be measured, and an RAII object cannot miss one.
        vulkan::profiling::cpu_phases cpu_timings;
        void collect_gpu_timings(uint32_t slot);

        // ---- G-buffer / deferred path ----
        // M1: the opaque pass writes the G-buffer (three surface targets + the HDR target it adds
        // emissive into) instead of shading; M2 adds the deferred lighting stage that reads it back.
        // The G-buffer pipeline shades nothing: albedo/metallic, normal/roughness and material
        // id/AO/flags go into core::gbuffer_* (1x targets + the pass's own 1x depth), so the opaque
        // pass runs at 1x whatever MSAA the forward path uses.
        std::optional<vk_pipeline> gbuffer_pipeline = std::nullopt;
        // fullscreen debug view of the G-buffer (reads the three targets + depth, writes the HDR
        // target so the ordinary post chain still runs)
        std::optional<vk_pipeline> gbuffer_debug_pipeline = std::nullopt;
        // fullscreen deferred lighting stage: reads the same inputs and ADDS the shading into the
        // HDR target, on top of the sky and the emissive the earlier passes left there
        std::optional<vk_pipeline> deferred_pipeline = std::nullopt;
        // whether the opaque pass writes the G-buffer this frame (see set_gbuffer_debug). Only
        // takes effect once the needed pipelines exist, so the flags can be set before setup ends.
        bool gbuffer_debug = false;
        // whether the deferred lighting stage shades the frame (see set_deferred). With both flags
        // on the debug view wins: inspecting the stored data is not a render mode.
        bool deferred_on = false;
        // which channel the debug view shows (see gbuffer_debug.frag / set_gbuffer_channel)
        int gbuffer_channel_index = 1;
        vk_sampler gbuffer_sampler = {};
        VkDescriptorSetLayout gbuffer_set_layout = VK_NULL_HANDLE;
        VkPipelineLayout gbuffer_pipeline_layout = VK_NULL_HANDLE;
        // the deferred lighting stage needs BOTH sets: set 0 = the shared scene set (camera, IBL,
        // light UBO, shadow map), set 1 = the G-buffer inputs. A set layout is index-agnostic, so the
        // debug view keeps using the same layout object as its set 0.
        VkPipelineLayout deferred_pipeline_layout = VK_NULL_HANDLE;
        // The debug view's sets, one per swapchain image, allocated from a pool this family owns and
        // retires itself (see vulkan.bindings: the pool lifetime rule is only about the pools).
        bindings::image_set_family gbuffer_family;
        // Descriptor pools replaced by a later swapchain generation. A pool may not be destroyed while
        // any RECORDED command buffer still references sets allocated from it - and the per-slot frame
        // command buffers stay recorded (executable) between frames - so a replaced pool is retired
        // here and destroyed with the runtime instead. The post chain and TAA keep their pools here;
        // the G-buffer family owns its own. Destroying them in place was a real
        // VUID-vkDestroyDescriptorPool-descriptorPool-00303
        // ("currently in use by VkCommandBuffer") on every window resize.
        std::vector<VkDescriptorPool> retired_descriptor_pools = {};

        struct gbuffer_debug_push_constants {
            float channel = 1.0f; // 0 albedo, 1 normal, 2 roughness, 3 metallic, 4 ao, 5 id, 6 depth, 7 flags
            float proj_22 = 0.0f; // projection[2][2] / [3][2]: the depth-linearization terms
            float proj_32 = 0.0f;
            float unused = 0.0f;
        };
        struct deferred_push_constants {
            glm::mat4 inv_view_proj = glm::mat4(1.0f); // clip (xy from the pixel, z = depth, w = 1) -> world
            // screen-space ambient occlusion (M6): x = radius, y = intensity (0 = off), z = samples,
            // w = depth bias. It rides the SAME push block because both are per-frame constants of
            // the lighting stage, and the shared range (100 bytes) already covers 80.
            glm::vec4 ssao = glm::vec4(0.5f, 0.0f, 8.0f, 0.02f);
            // 1.0 = the app's default pipeline is the flat "unlit" pass: the deferred lighting stage
            // then writes the G-buffer's albedo instead of shading it, so the render-mode switch
            // means the same thing on both paths (a flat surface has no lighting to defer).
            float unlit = 0.0f;
        };
        // SSAO state (runtime::set_ssao / [render] ssao*): the deferred lighting stage computes the
        // occlusion from the G-buffer depth + normal and folds it into the shade_input's ao, which
        // scales the IBL ambient only. Deferred-only: the forward path has no G-buffer to trace.
        bool ssao_enabled = true;    // master switch (an intensity of 0 is pushed when false)
        float ssao_radius = 0.5f;    // world-space sample radius
        float ssao_intensity = 1.0f; // how much occlusion is applied (1 = full)
        uint32_t ssao_samples = 8;   // samples per pixel, clamped to the shader's MAX_SSAO_SAMPLES
        float ssao_bias = 0.02f;     // view-depth bias that keeps a surface from occluding itself
        // Render mode on the deferred path: the app's default pipeline is the flat "unlit" one, so
        // the lighting stage writes the albedo instead of shading (runtime::set_unlit).
        bool unlit_active = false;
        // the inverse of this frame's view-projection, refreshed with the camera UBO in
        // pace_and_acquire() (the deferred lighting stage reconstructs world positions from depth)
        glm::mat4 current_inv_view_proj = glm::mat4(1.0f);
        // This frame's projection WITHOUT the TAA jitter: the shadow fit extracts near/far from
        // the projection's z-row (a product with the view matrix does not carry those terms) and
        // rebuilds the sub-frustum corners from it.
        glm::mat4 current_proj_unjittered = glm::mat4(1.0f);
        void ensure_gbuffer_descriptors();
        void record_gbuffer_debug_pass(VkCommandBuffer command_buffer);
        void record_deferred_lighting_pass(VkCommandBuffer command_buffer);

        // ---- temporal anti-aliasing (M3, deferred path only) ----
        // TAA replaces MSAA on the deferred path: the projection is jittered per frame (a Halton
        // sequence), the G-buffer writes motion vectors, and a resolve pass blends the current frame
        // with a reprojected, neighborhood-clamped history. The deferred scene writes
        // core::scene_color and the resolve writes the HDR target, so the whole post chain keeps
        // reading exactly what it read before TAA existed. A copy of the resolved frame becomes the
        // next frame's history (no ping-pong, hence no per-frame descriptor rewrites).
        std::optional<vk_pipeline> taa_pipeline = std::nullopt;
        vk_sampler taa_sampler = {};
        VkDescriptorSetLayout taa_set_layout = VK_NULL_HANDLE;
        VkPipelineLayout taa_pipeline_layout = VK_NULL_HANDLE;
        VkDescriptorPool taa_descriptor_pool = VK_NULL_HANDLE;
        uint32_t taa_pool_capacity = 0;
        std::vector<VkDescriptorSet> taa_sets = {};      // one per swapchain image
        std::array<VkImageView, 4> taa_bound_views = {}; // views the current sets point at
        bool taa_on = false;                             // [render] taa
        float taa_blend_static = 0.9f;                   // history weight for a static pixel
        float taa_blend_min = 0.5f;                      // history weight floor under motion
        uint32_t taa_jitter_index = 0;                   // position in the Halton sequence
        // The view-projection each swapchain image's history was rendered with, and whether that
        // history holds anything. Remembered PER IMAGE on purpose: with several swapchain images in
        // rotation, "the previous frame's camera" is not what that image's history was rendered with,
        // and reprojecting against the wrong matrix is exactly what makes a TAA history smear.
        std::vector<glm::mat4> image_view_proj = {};
        std::vector<bool> taa_history_valid = {};
        struct taa_push_constants {
            float history_valid = 0.0f; // 1 = trust the history, 0 = first frame for this image
            float blend_static = 0.9f;  // history weight for a static pixel
            float blend_min = 0.5f;     // history weight floor under motion
            float texel_size_x = 0.0f;  // 1 / target width
            float texel_size_y = 0.0f;  // 1 / target height
            float depth_scale = 0.0f;   // projection[2][2]: the depth-linearization term
            float depth_offset = 0.0f;  // projection[3][2]
            float unused = 0.0f;
        };
        void ensure_taa_descriptors();
        void record_taa_pass(VkCommandBuffer command_buffer);
        /** @brief whether the TAA resolve runs this frame (enabled + deferred lighting + pipeline) */
        [[nodiscard]] bool taa_active() const noexcept;
        /** @brief the image the scene-side passes write into (the TAA input, or the HDR target) */
        [[nodiscard]] VkImage scene_target_image(uint32_t image_index) const noexcept;
        [[nodiscard]] VkImageView scene_target_view(uint32_t image_index) const noexcept;
        /** @brief the sub-pixel jitter for a position in the Halton(2,3) sequence, in PIXELS */
        [[nodiscard]] static glm::vec2 taa_jitter_offset(uint32_t index) noexcept;
        /** @brief whether the opaque pass writes the G-buffer this frame (pipelines present + enabled) */
        [[nodiscard]] bool gbuffer_pass_active() const noexcept;
        /** @brief whether the deferred lighting stage shades this frame (see set_deferred) */
        [[nodiscard]] bool deferred_lit_active() const noexcept;
        /** @brief the swapchain was rebuilt: drop everything that pointed at the old generation
         *         (the debug overlay's backend + the G-buffer descriptor sets, whose views are gone) */
        void on_swapchain_recreated();

        // ---- post-processing: HDR scene target -> exposure + ACES + gamma -> swapchain ----
        // created by make_post_pipeline(); the descriptor sets rebind lazily whenever the
        // swapchain (and with it the per-image HDR resolve targets) is recreated
        struct post_push_constants {
            float exposure = 1.0f;        // linear exposure scale (see set_exposure)
            float bloom_intensity = 0.0f; // bloom blend weight (see set_bloom)
            float bloom_threshold = 0.0f; // bloom bright-pass threshold (see set_bloom)
            float mode = 0.0f;            // 0 = prefilter, 1 = downsample, 2 = composite
            // composite only: 1 = the shader encodes to sRGB itself, 0 = the target is an sRGB
            // attachment and the hardware encodes on write. Filled from the swapchain format every
            // frame - hard-coding either way double-encodes (sRGB attachment) or under-encodes
            // (UNORM attachment) gamma. The FXAA pass also forces it to 1: it renders into the R16F
            // LDR image, which must hold gamma-encoded values for FXAA's luma thresholds.
            float encode_gamma = 0.0f;
            // FXAA lanes (fxaa.frag): the sub-pixel term strength (0 = pure directional blend) and
            // the relative luma contrast below which a pixel counts as flat.
            float fxaa_subpixel = 0.75f;
            float fxaa_edge_threshold = 0.166f;
        };
        std::optional<vk_pipeline> post_pipeline = std::nullopt;
        // The SAME shader pair drives two different color formats, so it needs two pipelines:
        // post_pipeline targets the swapchain (the composite pass) and post_hdr_pipeline targets
        // hdr_format (the bright-pass prefilter and the three downsample passes, which render into
        // the R16F bloom levels). Reusing the swapchain-format pipeline for the HDR passes is a
        // VkPipelineRenderingCreateInfo format mismatch - validation flags it and the write is UB.
        std::optional<vk_pipeline> post_hdr_pipeline = std::nullopt;
        // FXAA pass (fxaa.frag + post.vert): reads the LDR image and writes the swapchain, so it
        // shares the composite's color format - but it is a separate pipeline because its shader
        // statically uses a different binding (5, the LDR image), and descriptor validation is per
        // statically-used binding: putting FXAA into post.frag would make the composite's own set
        // (which points binding 5 at the image it is currently writing) invalid.
        std::optional<vk_pipeline> post_fxaa_pipeline = std::nullopt;
        // FXAA state: enabled + the two knobs the shader takes (see post_push_constants)
        bool fxaa_on = false;
        float fxaa_subpixel = 0.75f;
        float fxaa_edge_threshold = 0.166f;
        vk_sampler post_sampler = {};
        VkDescriptorSetLayout post_set_layout = VK_NULL_HANDLE;
        VkPipelineLayout post_pipeline_layout = VK_NULL_HANDLE;
        VkDescriptorPool post_descriptor_pool = VK_NULL_HANDLE;
        uint32_t post_pool_capacity = 0;                                 // descriptor sets the current pool can hold
        std::vector<VkDescriptorSet> post_prefilter_sets = {};           // HDR -> bloom level 0 (bright pass)
        std::vector<std::array<VkDescriptorSet, 3>> post_down_sets = {}; // level k -> level k+1 (k = 0..2)
        std::vector<VkDescriptorSet> post_sets = {};                     // composite (HDR + all bloom levels)
        std::vector<VkImageView> post_bound_blooms = {};                 // bloom views the current sets point at
        std::vector<VkImageView> post_bound_views = {};                  // HDR views the current sets point at
        std::vector<VkImageView> post_bound_ldr = {};                    // LDR views the current sets point at
        // per-stage render toggles: whether the skybox / shadow pass actually records this frame.
        // Skybox off leaves just the clear color; shadow off skips the depth pass (the shadow map
        // is cleared to fully-lit so the main pass samples "no shadow"). Both default on.
        bool skybox_enabled = true;
        bool shadow_enabled = true;

        // ---- directional shadow mapping (scene set binding 7 light UBO + binding 8 shadow map) ----
        // Shadow map edge length in texels ([render] shadow_map_size). A MEMBER, not a constant,
        // because M7 surfaced it as a config knob - every user of it (the layered image, the depth
        // pass's rendering instance + pipeline viewport, the light UBO's texel size and the fit)
        // reads this value at creation time. Startup-only: set_shadow_map_size() before the scene
        // import (like set_shadow_cascades, the resources are created when the first scene set binds
        // them); changing it afterwards would need the image, the views and the descriptor rewritten.
        uint32_t shadow_map_size = 2048;
        // Layers currently owned by shadow_images: one per ACTIVE cascade (see ensure_shadow_resources).
        // Tracked separately from shadow_cascades because shrinking the count keeps the layers that are
        // already allocated - only growing it costs a rebuild.
        uint32_t shadow_allocated_layers = 0;
        // Frame limiter (see set_max_fps): the instant the next frame may start, advanced by exactly one
        // period per frame so a slow frame resyncs instead of banking debt (which would show up as a
        // burst of catch-up frames) and a fast one does not drift.
        double max_fps = 0.0;
        std::chrono::steady_clock::time_point next_frame_deadline = {};
        // Cascaded shadow maps: ONE 2D-array depth image per frame slot (while slot A is in flight,
        // slot B already rewrites its own map, so the two never race on the same image), with
        // shadow_cascades layers - each layer fitted to its own sub-range of the camera view.
        // Rendering goes through the per-layer views (one cascade = one dynamic rendering instance),
        // sampling through the array view (the fragment shader picks its cascade per pixel).
        std::vector<vk_image> shadow_images = {};                        // layered depth images
        std::vector<vk_image_view> shadow_array_views = {};              // 2D ARRAY views (sampled)
        std::vector<std::vector<vk_image_view>> shadow_layer_views = {}; // per slot: one 2D view per cascade
        // Active cascades (1 = exactly the single-shadow-map behavior; [render] shadow_cascades) and
        // the fraction of a cascade's range over which the shader blends into the next one.
        uint32_t shadow_cascades = 1;
        float shadow_cascade_blend = 0.1f;
        bool shadow_cascade_logged = false; // one-time per-cascade texel-density log
        vk_sampler shadow_sampler = {};     // linear depth-compare (hardware PCF) + clamp-to-edge
        // one-time log for the shadow-caster switch (see begin_recording): scenes over
        // full_scene_shadow_leaf_limit draw a culled subset instead of every leaf - say so once
        // instead of silently changing behavior
        bool shadow_heuristic_logged = false;
        // Light UBO (scene set binding 7): ONE host-visible buffer per frame slot, like the
        // camera UBO - each slot's scene set points at its own buffer, so the per-frame host
        // write into the paced slot's copy can never race a frame still in flight on the other
        // slot. CPU-side light_state mirrors the content: enable_shadows() fills it once,
        // set_shadow_enabled() flips its flag, and pace_and_acquire() copies it into the paced
        // slot's buffer next to the camera UBO - arbitrary-time calls (GUI callbacks included)
        // never touch mapped memory directly.
        std::vector<vk_buffer> light_buffers = {};
        std::vector<void*> light_mapped = {};
        light_ubo light_state = {};
        // linear exposure scale applied before tonemapping; copied into light_state.light_count.y
        // (the light UBO's first unused lane) right before the per-frame UBO upload, and pushed to
        // the skybox pass through the scene layout's push-constant range (see record_main_segment)
        float exposure_scale = 1.0f;
        float bloom_intensity = 0.0f;
        float bloom_threshold = 0.6f;
        // cel/toon shading: quantization steps (0 = plain PBR) and the band edge softness;
        // copied into light_state.light_count.z/w every frame like the exposure lane
        float toon_steps = 0.0f;
        float toon_softness = 0.15f;
        // bloom parameters (see set_bloom): blend weight into the HDR image and the bright-pass
        // threshold subtracted in linear space (0 intensity disables the effect)

        // F12 screenshot request: set edge-triggered by poll_events, consumed by the caller
        // (see consume_screenshot_request / acquire_current_frame_image)
        bool screenshot_requested = false;
        bool screenshot_key_down = false;
        // Read-back copy of the presented image, recorded INSIDE the frame's own command buffer
        // (end_recording, right before the present transition) while the swapchain image is still
        // owned by the app. The old path transitioned the image after vkQueuePresentKHR, which the
        // spec forbids - the presentation engine owns it by then (validation: "performs a layout
        // transition on presentable VkImage ... but the image has not been acquired").
        // screenshot_pending = a copy was recorded and is ready to be read once the submit lands.
        bool screenshot_pending = false;
        // one-time log for "this surface cannot do screenshots" (see record_screenshot_copy)
        bool screenshot_unsupported_logged = false;
        vk_buffer screenshot_readback = {};                        // host-visible TRANSFER_DST staging
        VkBuffer screenshot_readback_buffer = VK_NULL_HANDLE;      // its VkBuffer (vk_buffer::handle() is the allocator's)
        void* screenshot_readback_mapped = nullptr;                // persistent mapping (vma MAPPED_BIT)
        VkDeviceSize screenshot_readback_size = 0;                 // bytes the buffer currently holds
        VkExtent2D screenshot_readback_extent = {0, 0};            // extent the copy was recorded at
        std::optional<vk_pipeline> shadow_pipeline = std::nullopt; // depth-only pass pipeline
        bool shadows_enabled = false;                              // true after enable_shadows() (light UBO filled + pipeline ready)
        // live-tunable depth bias of the shadow pass (dynamic state, set per frame before the
        // depth-only draw): slope-scaled bias removes acne on angled surfaces, the constant
        // factor adds a fixed push; tune from the debug gui when a model shows acne/peter-panning
        float shadow_depth_bias_constant = 0.0f;
        float shadow_depth_bias_slope = 1.5f;
        float shadow_depth_bias_clamp = 0.0f;

        // ---- clustered light culling (M5) ----
        // The cluster COMPUTE pipeline (shaders/light_cluster.comp) sorts the punctual lights into
        // screen tiles x exponential depth slices once per frame; the shading stage then loops only
        // its own cluster's list instead of every active light. Optional: without the shader (or
        // with clustering off) shade_surface() falls back to the brute-force loop, which is exactly
        // what the clustered path is verified against.
        std::optional<vk_pipeline> cluster_pipeline = std::nullopt;
        // Per-cascade shadow recording pairs (one {pool, buffer} per cascade per frame slot). The
        // cascade tasks run CONCURRENTLY on the task pool, and a VkCommandPool is not thread safe, so
        // they may not share one - the same rule the main-pass workers already follow. The shared
        // secondary_command_buffers pool stays for the slot-scoped passes (gui, transparent) that the
        // primary thread records alone.
        std::vector<std::vector<std::pair<VkCommandPool, vk_command_buffer>>> shadow_recording = {};
        // Reused task scratch: the frame builds its task batches into these vectors every frame, so
        // they are members with clear() (capacity kept) instead of a fresh heap allocation per frame
        // (M9 overhead trim: a per-frame allocation plus one std::function per task is a few
        // microseconds of a ~0.35 ms frame on a light scene).
        std::vector<std::function<void()>> shadow_task_scratch = {};
        // Shadow-map reuse (M9): the maps depend on the fitted cascade matrices AND on where the
        // casters actually are, so a slot may skip the pass only when BOTH are unchanged since that
        // slot last rendered. The fit half is a counter bumped by every real refit and by the shadow
        // toggles; the geometry half is a hash of the casters' world matrices, computed per frame -
        // an animated/deformed/programmatic scene changes it (the fit cache does NOT, which is exactly
        // the trap the animated-Fox test caught), while a static scene repeats it.
        uint32_t shadow_content_version = 1;
        std::array<uint32_t, vulkan::core::MAX_FRAMES_IN_FLIGHT> shadow_rendered_version = {};
        std::array<uint64_t, vulkan::core::MAX_FRAMES_IN_FLIGHT> shadow_rendered_models = {};
        // The other half of geometry changed: a SKINNED caster keeps a constant push.model, its
        // pose lives entirely in the joint matrices the vertex shader reads, so those are hashed at
        // upload instead (set_skin_matrices). Morph targets have no upload hook at all - the caller
        // writes into morph_scratch() directly - so handing that scratch out bumps a revision and
        // conservatively blocks reuse. The animated Fox is the regression test for the first half:
        // hashing push.model alone left it with a frozen shadow map.
        uint64_t skin_matrix_hash = 1469598103934665603ull;
        uint32_t morph_revision = 0;
        std::vector<vk_buffer> cluster_count_buffers = {}; // per slot: one uint per cluster
        std::vector<void*> cluster_count_mapped = {};      // their persistent mappings (memset per frame)
        std::vector<vk_buffer> cluster_index_buffers = {}; // per slot: cluster_light_capacity uints per cluster
        bool clustered_lights = true;                      // set_clustered_lights()
        uint32_t cluster_tiles_x = 0;                      // active grid this frame (from the extent)
        uint32_t cluster_tiles_y = 0;

        // shared CPU worker pool for frame-time parallel stages (run_tasks). Sized to the
        // machine (hardware_concurrency()/4, floor 1) instead of per-consumer pools; tasks
        // are grouped by priority so each consumer waits only for its own group. Declared
        // before the scene/pipeline state so the pool outlives what tasks may touch (destructor
        // order is reverse declaration: pipelines etc. go first, the pool joins last).
        static int default_task_pool_threads() noexcept;
        utility::thread_pool task_pool = utility::thread_pool{default_task_pool_threads()};

        // Guards the pipeline registry below (pipelines / pipeline_names / default_pipeline_name):
        // parallel recording workers read it through render_environment's binder (shared locks,
        // concurrent), while make_pipeline / set_default_pipeline / resize resync write it
        // (unique lock). A reader/writer lock because reads vastly outnumber writes. Mutable so
        // const accessors (get_pipeline, the env construction inside the const record steps)
        // can take shared locks. The scene tree, per-frame vectors and camera state are NOT
        // guarded: they are only touched on the frame thread or inside run_tasks' synchronous
        // windows, so locking them would only add hot-path cost.
        mutable std::shared_mutex access_mutex;
        // string keys (not string_view): the runtime owns the pipeline names, so lookups
        // stay valid regardless of the caller's storage lifetime. std::less<> enables heterogeneous
        // lookup, so the string_view-based API (get_pipeline / ...) still works without
        // constructing a std::string per call.
        std::map<std::string, vk_pipeline, std::less<>> pipelines;
        // stable name table mirroring the pipelines map (same order as insertion): every
        // make_pipeline() appends the name, nothing removes. A DEQUE (not vector): push_back
        // never invalidates existing elements, so a pointer to it (handed to
        // render_environment::available for the recording workers' lifetime) stays valid even
        // if another thread registers a pipeline concurrently. Only make_pipeline() writes it,
        // under the unique lock; readers that took the pointer may use it lock-free afterwards.
        std::deque<std::string> pipeline_names = {};
        // name of the runtime's default pipeline: primitives with DEFAULT semantics (empty
        // pipeline_name) draw with it. Set implicitly to the FIRST created pipeline, or
        // explicitly via set_default_pipeline(); the shadow pass never consults it (its env
        // default is the shadow pipeline).
        std::string default_pipeline_name = {};
        // Scene storage: a scene tree of nodes with local transforms + children; every primitive
        // (normal_draw_primitive / instanced_draw_primitive) lives in a node's primitive leaf. The frame
        // record phase walks the tree once per frame: update_world() accumulates world matrices into each
        // leaf (primitive::set_world -> push.model), then each pipeline draws the leaves bound to
        // it (primitive::draw stays polymorphic). This replaces the old flat per-pipeline primitive list.
        //
        // NON-OWNING: the tree belongs to the caller (set_scene() binds it). The caller must keep the
        // tree alive while the runtime is in use and destroy it BEFORE the runtime (the leaves release
        // their GPU resources through this runtime's vma allocator on destruction). Declaration order in
        // the caller (scene after runtime) gives that order automatically.
        scene_tree::scene* bound_scene = nullptr; // user-owned scene the runtime renders
        // conservative radius of the bound scene around the camera target / scene center
        // (set by enable_shadows, which receives it). The camera projection far plane uses it
        // (make_orbit_camera_ubo) so zooming in never clips the scene's far side.
        float scene_radius = 100.0f;
        // optional whole-scene transform applied on top of every root before local transforms
        // (programmatic grouping / demo rotation; identity by default = no visual change)
        glm::mat4 scene_transform = glm::mat4(1.0f);
        // frustum culling of the main pass (BVH over per-leaf world AABBs vs the camera frustum);
        // enabled by default, disable for verification / debugging
        bool frustum_culling = true;
        // culling caches: the BVH is rebuilt only when the scene changed (bvh_dirty), and the
        // culled result is reused while neither the scene nor the camera moved. cull_bvh holds
        // the last built tree (world AABBs are captured at build time and stay valid as long as
        // the scene is unchanged: update_world rewrites the same matrices each frame).
        //
        // Liveness contract (why the raw primitive const* inside the BVH / cull_visible can
        // never dangle): every scene mutation that removes or adds leaves - clear_primitives,
        // make_primitive / make_instanced_primitive, import, set_scene, set_scene_transform,
        // and scene_changed() for callers editing get_scene() directly - sets bvh_dirty. The
        // next begin_recording() then (1) recollects frame_leaves from the CURRENT tree, (2)
        // destroys the old cull_bvh (its stale leaf pointers die with it) and rebuilds from
        // those fresh leaves, and (3) re-runs the cull so cull_visible also drops dead leaves.
        // Between frames the caches are never touched, so a leaf removed mid-frame is safe as
        // long as the removal went through an API that sets bvh_dirty (or the caller invoked
        // scene_changed()). Callers editing the tree behind the runtime's back must call
        // scene_changed() after every structural change or the BVH/cull_visible can outlive a
        // destroyed leaf.
        //
        // Known limitation (documented, accepted): the BVH bounds are the STATIC local AABBs
        // captured at upload/import. Skinned/morphed vertices can move far outside that box
        // (see set_skin_matrices / morph_scratch), so such a primitive may be frustum-culled
        // even while its deformed geometry is on screen. Conservative animation rigs stay
        // inside the authored bounds; culling can also be disabled (frustum_culling = false).
        std::optional<utility::bvh<primitive>> cull_bvh = std::nullopt;
        bool bvh_dirty = true;                                // scene structure/transforms changed -> rebuild
        std::pmr::vector<primitive const*> cull_visible = {}; // last culled result (main-pass set)
        // camera identity for result reuse: yaw, pitch, distance, target.xyz (7 floats)
        std::array<float, 7> camera_key = {};
        bool camera_moved = true; // camera key differs from the last cull frame
        // optional external (glTF/programmatic) camera: when active, the per-frame camera UBO
        // uses these matrices directly (the orbit camera + its mouse controls are ignored) and
        // frustum culling re-runs only when the authored view/projection actually changed
        glm::mat4 external_view = glm::mat4(1.0f);
        glm::mat4 external_proj = glm::mat4(1.0f);
        glm::vec3 external_eye = glm::vec3(0.0f);
        bool external_camera_active = false;
        bool external_camera_changed = true; // set when the override differs -> re-cull
        // one command buffer per frame slot, used and reused every frame
        std::vector<vk_command_buffer> command_buffers;
        // per-slot secondary command buffers for pass recording (stage 2/3 of parallel
        // recording):
        //   - shadow: one shadow-pass CB per frame slot (single segment; the depth-only pass
        //     shares one pipeline, so further splitting buys little - stage 2)
        //   - gui: one overlay CB per frame slot (stage 2)
        //   - transparent: one CB per frame slot for the alpha-blended leaves - recorded on the
        //     PRIMARY thread after the opaque segments (transparent leaves are usually few and
        //     order-sensitive, so they do not join the parallel fan-out), executed last before
        //     the gui overlay so blends compose over the opaque depth
        //   - main: SEGMENT secondaries per frame slot (stage 3): the main pass splits its
        //     visible leaves into up-to-SEGMENT contiguous ranges, each recorded on a pool
        //     worker and executed in order. SEGMENT is sized to the task pool (see
        //     record_main_drawcalls), so the GPU can read all secondaries while the slot's
        //     primary executes - they share the primary's lifetime (reused after the slot's
        //     timeline wait, no per-frame allocation, no pool lock).
        // One SHADOW secondary PER CASCADE (shadow_0 .. shadow_3): a command buffer that was not
        // recorded with SIMULTANEOUS_USE may only be executed once per primary command buffer, so one
        // recorded caster pass cannot be replayed for four cascades - and each cascade needs its own
        // cascade-index push constant, which a secondary must record itself (state is not inherited
        // from the primary).
        enum class secondary_pass : std::size_t { shadow_0 = 0,
                                                  shadow_1 = 1,
                                                  shadow_2 = 2,
                                                  shadow_3 = 3,
                                                  gui = 4,
                                                  transparent = 5,
                                                  count = 6 }; // fixed non-segment slots
        /** @brief the shadow secondary a cascade records into */
        [[nodiscard]] static constexpr secondary_pass shadow_secondary(uint32_t cascade) noexcept {
            return static_cast<secondary_pass>(static_cast<std::size_t>(secondary_pass::shadow_0) + cascade);
        }
        std::vector<std::array<vk_command_buffer, static_cast<std::size_t>(secondary_pass::count)>> secondary_command_buffers;
        // per-slot main-pass parallel segments (stage 3): one {command pool, secondary buffer}
        // PAIR per task-pool worker, in the same style as the vma allocator's command_cache -
        // a VkCommandPool is not thread safe, so every parallel recording thread owns its own
        // pool and the buffer allocated from it, kept together so they can never drift apart.
        // The pair's pool is registered on the core (destroyed by its cleanup AFTER this
        // runtime's RAII vk_command_buffer members free their buffers into those pools). One
        // inner vector per frame slot (the GPU reads the secondaries while the slot's primary
        // executes, so they share the primary's lifetime), index = segment.
        std::vector<std::vector<std::pair<VkCommandPool, vk_command_buffer>>> main_segments;
        // per-frame state shared by the split frame steps (the frame steps call them in order,
        // so an external caller can interleave its own work between the same steps)
        uint32_t current_image_index = 0;                     // swapchain image acquired by pace_and_acquire()
        float current_aspect = 1.0f;                          // swapchain aspect for the frame's UBO + culling
        camera_ubo current_ubo = {};                          // camera UBO snapshot written in pace_and_acquire()
        std::pmr::vector<primitive const*> frame_leaves = {}; // every scene leaf this frame (shadow + cull input)
        // reused scratch for the shadow-frustum caster fit (see update_shadow_frustum): every
        // scene leaf is tested here, not just the visible ones, so off-screen casters count
        std::pmr::vector<primitive const*> shadow_caster_scratch = {};
        std::pmr::vector<primitive const*> frame_visible = {}; // opaque frustum-visible subset (main pass)
        // transparent (alphaMode BLEND) frustum-visible leaves, sorted FAR -> NEAR from the
        // camera each time the cull re-runs: drawn AFTER the opaque pass (depth-write off), so
        // the blend order is back-to-front. Rebuilt in begin_recording together with the cull.
        std::pmr::vector<primitive const*> frame_transparent = {};
        // shadow-pass subset (rebuilt each frame before the shadow recording). SMALL scenes
        // (<= full_scene_shadow_leaf_limit leaves, begin_recording): EVERY leaf - exact and cheap
        // at that size. HEAVY scenes: the camera-visible leaves plus the leaves the BVH reports
        // inside the SHADOW frustum itself. That frustum follows the camera and already contains
        // every caster whose shadow can land in view (update_shadow_frustum merges the per-leaf
        // boxes whose light-space xy overlaps the camera frustum's, which is exact under the
        // orthographic light), so - unlike the old camera-frustum-shifted-by-a-margin heuristic -
        // an off-screen caster such as the wall behind the camera is still included. Instanced /
        // bound-less leaves are always included.
        std::pmr::vector<primitive const*> shadow_casters = {};
        // how far up-light of the camera frustum a caster still matters (its shadow can still
        // reach the view). Scene-scale heuristic: max(1, scene_radius / 8); see enable_shadows.
        float shadow_caster_extent = 1.0f;
        // scene center handed to enable_shadows. update_shadow_frustum falls back to
        // center +- scene_radius when a shadow caster has no world AABB of its own AND is not an
        // instanced draw whose instance matrices we can read (see instanced_world_aabb).
        glm::vec3 shadow_scene_center = glm::vec3(0.0f);
        // Fit cache: rebuilding the light frustum walks every leaf and transforms 8 corners each,
        // which is O(scene) work for a result that only changes when the camera or the scene moves.
        // shadow_fit_view/proj are the camera matrices the current frustum was fitted for;
        // shadow_frustum_valid is cleared by enable_shadows() (new light setup) and bvh_dirty marks
        // a changed scene.
        glm::mat4 shadow_fit_view_proj = glm::mat4(1.0f); // unjittered view * proj the last fit used

        bool shadow_frustum_valid = false;
        // optional Dear ImGui debug overlay; inactive until enable_debug_gui() succeeds. The
        // runtime drives it inside the frame steps (new_frame before recording, record after the
        // runtime's own draw calls) so callers only manage its content via debug_gui().
        gui::gui_content debug_overlay;
        // whether the active overlay is drawn (set_debug_gui_visible / the built-in F1 toggle);
        // hiding keeps the overlay initialized and its panels intact, so showing is instant
        bool debug_gui_shown = true;
        // F1 edge detection for the overlay toggle (true while the key is held, so one press
        // toggles exactly once - see poll_events)
        bool gui_toggle_down = false;
        // filtered view over vulkan_core, exposed via operator-> (external code never sees the raw core)
        core_filter filtered_core;

        /**
         * @ingroup vulkan_runtime
         * @brief collect every leaf primitive under @p node (DFS pre-order) into @p out
         * @note leaves are stored as scene_tree::primitive; every leaf this runtime creates is a
         *       vulkan::primitive (the GPU primitive implements scene_tree::primitive), so the cast is safe
         */
        static void collect_leaf_primitives(scene_tree::scene_node const& node, std::pmr::vector<primitive const*>& out);
        /**
         * @ingroup vulkan_runtime
         * @brief build a normal_draw_primitive from @p info WITHOUT attaching it to the scene tree:
         *        uploads geometry buffers and registers the material (textures + material_record).
         * @param pipeline_name the pipeline the primitive draws with (must already exist)
         * @return the new primitive (caller attaches it into a scene node), or nullptr if the
         *         pipeline does not exist
         * @note make_primitive() is create_primitive() + attach-as-root-leaf; the hierarchy import
         *       (import_scene) attaches leaves to their node instead
         * @note material registration writes every scene set (binding 1), which is only valid
         *       before the first frame or while the runtime is idle - see make_primitive()'s
         *       timing note
         */
        std::unique_ptr<primitive> create_primitive(std::string_view pipeline_name, primitive_create_info const& info);

        /**
         * @ingroup vulkan_runtime
         * @brief begin the frame's rendering on the given command buffer: clears the color
         *        attachment with a dark background and the depth attachment
         * @param command_buffer the command buffer being recorded
         * @param image_index the acquired swapchain image index (selects the attachment views)
         * @note uses vkCmdBeginRendering (dynamic rendering, Vulkan 1.3 core - the only path
         *       the engine supports; device selection requires an apiVersion >= 1.3 device)
         */
        void begin_rendering(VkCommandBuffer command_buffer, uint32_t image_index, VkRenderingFlags flags = 0) const;

        // ---- scene resource management (see the members above) ----
        void init_scene_resources();    // camera UBO buffers + white fallback texture + texture sampler + material table
        void ensure_shadow_resources(); // (lazily) layered shadow map + light UBO buffers
        void ensure_cluster_buffers();  // (lazily) per-slot cluster count/index buffers (M5)
        // Diagnostics for the optional features (see log_feature_status / warn_missing_feature): a
        // toggle whose prerequisites are missing would otherwise do nothing at all - the user clicks,
        // nothing changes, and the log has no trace to explain it. Each message is emitted at most
        // once per session, keyed by the feature name.
        void warn_missing_feature(std::string_view key, std::string const& message);
        std::vector<std::string> warned_features = {};
        void ensure_scene_set();                                                              // lazily create one scene set per frame slot and write all bindings
        void write_ibl_bindings() const;                                                      // (re)write bindings 2-4 on every scene set with the current IBL views / placeholders
        void write_light_and_shadow_bindings();                                               // (re)write binding 7 (light UBO) + binding 8 (shadow map) on every scene set
        void update_all_scene_sets(VkWriteDescriptorSet const* writes, uint32_t write_count); // apply one batch of writes to every slot's scene set
        material_id register_material(primitive_create_info const& info);                     // upload textures into the array, append a material_record, return its index

    public:
        // A non-const runtime exposes a mutable filter (e.g. runtime->get_vma()); a const runtime
        // gets a read-only filter, so mutating operations are impossible through const access.
        core_filter* operator->() noexcept {
            return &this->filtered_core;
        }
        core_filter const* operator->() const noexcept {
            return &this->filtered_core;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief orbit camera state; left-drag rotates, wheel zooms
         */
        orbit_camera camera;

        /**
         * @ingroup vulkan_runtime
         * @brief take over the per-frame camera UBO with an externally computed view/projection
         *        (e.g. a glTF camera, see the demo). While active the orbit camera mouse controls
         *        are ignored and frustum culling re-runs only when the view/projection changed.
         * @param eye camera position in world space (written into the UBO for the shaders)
         * @param view world -> camera space matrix
         * @param proj camera space -> clip space matrix (Vulkan conventions, i.e. with the Y
         *        flip already applied, same as make_orbit_camera_ubo)
         * @note call every frame while the authored camera is in use; identical matrices are
         *       deduplicated (no cull invalidation when nothing changed)
         */
        void set_external_camera(glm::vec3 const& eye, glm::mat4 const& view, glm::mat4 const& proj) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief return to the orbit camera (ignores any previously set external camera)
         */
        void clear_external_camera() noexcept;

        /** @brief whether an external (glTF/programmatic) camera currently drives the UBO */
        [[nodiscard]] bool using_external_camera() const noexcept;

        /** @brief current swapchain aspect ratio (for authored-camera projections) */
        [[nodiscard]] float aspect_ratio() const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief background clear color applied every frame (the skybox is drawn over it, so it
         *        shows only where the environment pass leaves the background uncovered)
         */
        glm::vec3 clear_color = glm::vec3(0.02f, 0.02f, 0.03f);

        /**
         * @ingroup vulkan_runtime
         * @brief run a batch of tasks on the runtime's shared worker pool and block until that
         *        priority group finished. Frame-time parallel stages (the animation controller's
         *        per-source sampling fan-out today, more later) submit their tasks here instead
         *        of owning private pools, so all CPU parallelism shares one pool sized to the
         *        machine (hardware_concurrency()/4 threads, floor 1).
         * @param tasks the batch; each task runs exactly once on a pool worker
         * @param priority the stage this batch belongs to (see task_priority); run_tasks waits
         *        only for this stage's tasks, so parallel stages can share the pool safely
         * @note synchronous: returns only after every task in the batch finished, which is what
         *       the frame phases need (the paced slot is read right after animation sampling)
         */
        void run_tasks(std::span<std::function<void()>> tasks, task_priority priority = task_priority::animation);

        /**
         * @ingroup vulkan_runtime
         * @brief worker count of the shared task pool (what run_tasks() fans out over); callers
         *        that need to slice their work across workers (e.g. the animation controller's
         *        per-source sampling) size their slices to this
         */
        [[nodiscard]] int task_pool_threads() const noexcept {
            return this->task_pool.thread_count();
        }

        /**
         * @ingroup vulkan_runtime
         * @brief construct the runtime: performs the full core initialization (window / instance /
         *        device / swapchain / resources) from @p options (window size, vsync, MSAA), and
         *        registers the orbit camera mouse callbacks on the window
         */
        explicit runtime(core_create_info const& options);

        /**
         * @ingroup vulkan_runtime
         * @brief construct the runtime with default core options (1080x960 window, auto MSAA,
         *        mailbox present mode)
         */
        runtime();

        /**
         * @ingroup vulkan_runtime
         * @brief destroy cached pipelines before the inner core (and thus the VkDevice) is destroyed
         * @note explicit destructor: member destruction order is reverse declaration order, which
         *      currently already destroys pipelines before vulkan_core; making it explicit keeps
         *      that guarantee even if members are reordered later
         */
        ~runtime();

        /**
         * @ingroup vulkan_runtime
         * @brief initialize the Dear ImGui debug overlay on top of this runtime's window
         * @return true when the overlay is active afterwards (initialized, or already active)
         * @note the overlay is drawn inside the runtime frame phases: its
         *       per-frame new_frame/record calls are driven by the runtime once enabled. Call
         *       after the runtime is fully set up (window/device ready). Safe to call again to
         *       re-enable after shutdown; no-op when already active.
         * @note F1 toggles the overlay at runtime (see set_debug_gui_visible): hide/show the whole
         *       overlay without losing its panels, and press it once more to bring it back
         */
        bool enable_debug_gui();

        /**
         * @ingroup vulkan_runtime
         * @brief true while the Dear ImGui debug overlay is active (initialized)
         */
        [[nodiscard]] bool debug_gui_active() const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief whether the active overlay is currently drawn (distinct from debug_gui_active():
         *        the overlay stays initialized while hidden, so showing it again is instant)
         */
        [[nodiscard]] bool debug_gui_visible() const noexcept;
        /**
         * @brief true while the debug overlay owns the mouse: the camera orbit/zoom callbacks are
         *        suppressed then, so dragging an overlay slider cannot rotate or zoom the view
         */
        [[nodiscard]] bool debug_gui_wants_mouse() const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief hide / show the whole debug overlay without touching its panels or widgets
         * @param visible false skips the overlay's per-frame work and leaves the frame to the
         *        scene only; true draws it again
         * @note the flag is remembered while the overlay is inactive and takes effect as soon as
         *       enable_debug_gui() succeeds; the runtime's built-in F1 key toggles this same flag
         */
        void set_debug_gui_visible(bool visible) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief access the debug overlay to manage its content from outside (register panels,
         *        push widgets, show/hide windows)
         * @return the runtime's gui_content (non-const: adding/removing panels mutates it)
         * @note panels added here are drawn every rendered frame by the runtime; add them after
         *       enable_debug_gui() (or any time — they are only drawn while the overlay is active)
         */
        [[nodiscard]] gui::gui_content& debug_gui() noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief drive one whole application frame in a single call by running the frame phases
         *        directly (see below): poll/skip/close, recreate-if-minimized, pace + acquire,
         *        record (world accumulation / culling / shadow / main pass / overlay) and
         *        submit + present, with no opportunity to interleave per-frame host writes.
         * @return frame_status: proceed when a frame was presented; skipped when not renderable
         *         (minimized or swapchain recreated - caller yields and calls again); closed on
         *         window close; one of the stage-specific *_failed values on a fatal Vulkan
         *         error (is_failure() tests for any of them)
         * @note callers that must write per-frame data (animation poses, skin matrices, morph
         *       weights) between pacing and recording call the granular phases themselves in the
         *       same order: poll_events -> recreate_if_minimized -> pace_and_acquire, then the
         *       writes (they may only touch per-slot buffers after pace_and_acquire has waited
         *       the slot's timeline), then begin_recording -> record_main_drawcalls ->
         *       end_recording -> submit_and_present.
         */
        frame_status render_frame();

        /**
         * @ingroup vulkan_runtime
         * @brief the frame slot paced by the last successful pace_and_acquire(); per-frame host
         *        writes (set_skin_matrices / morph_scratch) go into this slot's per-slot buffers
         */
        [[nodiscard]] uint32_t active_frame_slot() const noexcept {
            return this->active_slot;
        }

        // ---- frame phases: one frame is these calls in order. render_frame() runs them all
        //      back to back; callers with per-frame host writes run them at fine granularity
        //      (writes go between pace_and_acquire() and begin_recording()) ----
        /** @brief poll window events; returns closed on ESC/native close, skipped while minimized.
         *         F1 toggles the debug overlay (initializing it on demand when it was disabled at
         *         startup), edge-triggered so one press toggles exactly once */
        frame_status poll_events();
        /** @brief recreate the swapchain if the window was minimized (extent 0) since the last frame */
        void recreate_if_minimized();
        /** @brief wait the frame slot's timeline, acquire the next image, write the camera UBO,
         *         and remember the paced slot in active_frame_slot(); skipped when the swapchain
         *         was recreated, acquire_failed on a fatal acquire error */
        frame_status pace_and_acquire();
        /** @brief begin the slot's command buffer and run the CPU scene prep (world-matrix
         *         accumulation + frustum culling); begin_recording_failed when vkBeginCommandBuffer fails */
        frame_status begin_recording();
        /** @brief record the shadow pass, the attachment transitions and the main scene pass */
        void record_main_drawcalls();

        /**
         * @ingroup vulkan_runtime
         * @brief record the depth-only shadow-pass drawing content into @p command_buffer:
         *        bind the shared scene set + shadow pipeline, set the live depth bias, draw
         *        every scene leaf. The caller frames it (already inside the shadow rendering
         *        instance, depth-only).
         * @note extracted from record_main_drawcalls() so the same content can be recorded
         *       inline (stage 1) or into a per-slot secondary command buffer (stage 2,
         *       parallel recording) - only bind/push/draw commands, no barriers / begin-end.
         */
        /** @brief refit the directional shadow frustum to the current camera view (called once
         *         per frame from pace_and_acquire; skips the work unless the camera or scene moved,
         *         see shadow_frustum_valid) */
        void update_shadow_frustum();
        /**
         * @brief world-space AABB of an INSTANCED leaf (one draw covering many transforms)
         * @param leaf the leaf to bound
         * @param[out] wmin/wmax the union of the source geometry's LOCAL AABB transformed by each of
         *             the leaf's instance matrices (the vertex shader uses the instance matrix as the
         *             whole world matrix, so this is exact)
         * @return false when the leaf is not an instanced draw, its source/bounds are missing, or its
         *         instance slice cannot be read - the caller then has to fall back to a coarser bound
         */
        [[nodiscard]] bool instanced_world_aabb(primitive const& leaf, glm::vec3& wmin, glm::vec3& wmax) const;
        void record_shadow_content(VkCommandBuffer command_buffer) const;

        /**
         * @ingroup vulkan_runtime
         * @brief fingerprint of everything the shadow pass reads as input: the caster count, every
         *        caster's world matrix, the uploaded skin matrices and the morph-scratch revision
         * @return one 64-bit fingerprint, equal across frames exactly when the shadow maps a slot
         *         already holds are still the maps the frame would render
         *
         * The shadow-map reuse compares this against the fingerprint of the frame that last rendered
         * a slot (see record_main_drawcalls). A skinned caster keeps a CONSTANT world matrix - its
         * pose lives entirely in the joint matrices - so the skin upload has to be part of the
         * fingerprint or an animated model keeps a frozen map; morph targets are written through
         * morph_scratch(), which has no upload hook, so it bumps a revision instead. XXH3 per matrix,
         * not a byte loop over the concatenation: this runs on every frame, the reused ones included.
         */
        [[nodiscard]] uint64_t shadow_geometry_signature() const;

        /**
         * @ingroup vulkan_runtime
         * @brief dispatch the clustered-light-culling compute pass (M5) and hand its buffers to the
         *        fragment stages
         * @param command_buffer the frame's primary command buffer (recorded before any rendering)
         *
         * No-op without the cluster pipeline, with clustering off, or before the first paced frame
         * (the grid comes from the swapchain extent). The per-cluster counts were zeroed by the host
         * in pace_and_acquire(), so the pass only appends; the buffer barrier after the dispatch is
         * what makes its SHADER_WRITE visible to the fragment stages that read the lists later in
         * the same submission.
         */
        void record_cluster_pass(VkCommandBuffer command_buffer);

        /**
         * @ingroup vulkan_runtime
         * @brief record the scene pass of the FORWARD path into @p command_buffer: move the HDR
         *        (plus MSAA color) and depth attachments into their render layouts, resync the pass
         *        geometry, then record the opaque leaves - the skybox behind them (segment 0) and
         *        the alpha-blended leaves over the depth they wrote.
         * @param command_buffer the frame's primary command buffer
         *
         * Path 1 of 2 (see record_deferred_scene). Both go through record_opaque_scene(), which owns
         * the segmentation and the secondary lifetime; this one subscribes to the skybox and to the
         * transparent pass, which only make sense when the pass shades as it draws.
         */
        void record_forward_scene(VkCommandBuffer command_buffer);

        /**
         * @ingroup vulkan_runtime
         * @brief record the scene pass of the DEFERRED path into @p command_buffer: move the
         *        single-sampled G-buffer targets (surface + velocity + the pass's own depth) into
         *        their render layouts, resync the pass geometry, then record the opaque leaves into
         *        them - no shading at all.
         * @param command_buffer the frame's primary command buffer
         *
         * Path 2 of 2. It draws neither the skybox (the lighting stage writes the sky into the pixels
         * no geometry covered) nor alpha-blended geometry (blending would have to compose over an
         * already shaded image - a pass of its own, still ahead). record_post_process() turns the
         * G-buffer into the frame afterwards through record_deferred_lighting_pass().
         */
        void record_deferred_scene(VkCommandBuffer command_buffer);

        /**
         * @ingroup vulkan_runtime
         * @brief move the attachments of one scene path into the layouts its rendering instance
         *        declares, before vkCmdBeginRendering - dynamic rendering has no automatic
         *        transitions the way a render pass does
         * @param command_buffer the frame's primary command buffer
         * @param gbuffer_pass true for the deferred path's single-sampled G-buffer target set, false
         *                     for the forward path's HDR (+ MSAA color) and depth
         */
        void record_scene_attachments(VkCommandBuffer command_buffer, bool gbuffer_pass);

        /**
         * @ingroup vulkan_runtime
         * @brief resync every pipeline's cached fullscreen viewport/scissor from the current
         *        swapchain extent, on the primary thread and before the scene content is recorded
         *
         * A resize changes the extent the cached values were built from, and begin_pipeline() pushes
         * the cached values - so they are refreshed once per frame here, for every pipeline that can
         * draw (the scene pipelines, the skybox/post chain, the G-buffer, deferred lighting and TAA).
         */
        void update_pass_geometry();

        /**
         * @ingroup vulkan_runtime
         * @brief record the opaque scene into @p command_buffer: one secondary per task-pool worker
         *        segment (or a single one for a small frame), each inheriting the instance's color +
         *        depth attachments, plus the transparent secondary when @p draw_transparent - the
         *        primary executes them in order inside the rendering instance
         * @param command_buffer the frame's primary command buffer
         * @param gbuffer_pass true when the leaves bind the G-buffer pipelines (deferred path)
         * @param draw_transparent record and execute the alpha-blended leaves (forward path only:
         *        the deferred path has no shaded image to blend over)
         *
         * Shared by both scene paths on purpose - the segmentation, the per-segment secondary
         * lifetime and the execute order are the same work in either; only the pipelines the leaves
         * bind (chosen in record_main_segment() from @p gbuffer_pass) and the two optional extras
         * differ.
         */
        void record_opaque_scene(VkCommandBuffer command_buffer, bool gbuffer_pass, bool draw_transparent);

        /**
         * @ingroup vulkan_runtime
         * @brief record one contiguous slice of the main-pass leaves into @p command_buffer:
         *        bind the shared scene set, then draw the leaves of @p leaves (a sub-range of
         *        frame_visible). When @p draw_skybox the skybox background is drawn first so
         *        the background stays ordered before the scene (segment 0 only); later
         *        segments are pure scene.
         * @note stage 3 of parallel recording: each task-pool worker records one segment into
         *       its own secondary command buffer (see sub_render_task), the primary executes
         *       them in order. Only bind/push/draw commands - caller owns barriers + the
         *       rendering instance.
         */
        /** @brief close the scene rendering instance and record the post-process pass (HDR ->
         *         exposure/tonemap -> swapchain) plus the debug overlay on the final image
         *  @return true when a fullscreen pass actually wrote the SWAPCHAIN image (so it is in
         *          COLOR_ATTACHMENT_OPTIMAL and its contents are this frame's post-processed
         *          result); false when the pass was skipped (no pipeline/descriptors), in which case
         *          the swapchain image was never transitioned and the caller must not pretend
         *          otherwise to the present barrier or the screenshot copy */
        [[nodiscard]] bool record_post_process(VkCommandBuffer command_buffer);
        /** @brief (re)bind the post descriptor sets to the current per-image HDR targets */
        void ensure_post_descriptors();
        /**
         * @brief make sure the screenshot read-back buffer holds @p extent (recreating it when the
         *        swapchain size changed) and keep it persistently mapped
         * @return the mapped host pointer, or nullptr when the buffer is unavailable
         */
        void* ensure_screenshot_readback(VkExtent2D extent);
        /**
         * @brief record the screenshot copy (swapchain image -> read-back buffer) into
         *        @p command_buffer, with the image's own layout transitions around it. Called
         *        from end_recording while the image still belongs to the frame being recorded -
         *        after vkQueuePresentKHR the presentation engine owns it and it must not be
         *        transitioned again.
         */
        void record_screenshot_copy(VkCommandBuffer command_buffer);
        void record_main_segment(VkCommandBuffer command_buffer, std::span<primitive const* const> leaves, bool draw_skybox) const;

        /**
         * @ingroup vulkan_runtime
         * @brief one recording job of the parallel main pass (stage 3): records @p leaves (a
         *        contiguous slice of the frame's visible leaves) into @p command_buffer, a
         *        per-slot SECONDARY command buffer. operator() begins the secondary (inheriting
         *        the main instance's color+depth attachments via dynamic rendering 1.3
         *        inheritance info), records the slice and ends it, so a batch of these can be
         *        posted straight to the shared task pool and the recording group waited on.
         * @note value type (span + handle + formats; no owning pointers), safe to copy into
         *       std::function for the pool; the begin-info is assembled fresh inside operator()
         *       so copies never share dangling pNext chains.
         */
        struct sub_render_task {
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            std::span<primitive const* const> leaves = {};
            bool draw_skybox = false; // segment 0 draws the skybox before its leaves
            // Color attachment formats of the instance this secondary is recorded into, in
            // attachment order: one entry (the HDR target) for the forward pass, the G-buffer set
            // when the opaque pass writes the G-buffer. Held by value because the task outlives the
            // call that builds it (it is moved into the task pool).
            std::array<VkFormat, vulkan::gbuffer_pass_attachment_count> color_formats = {};
            uint32_t color_count = 0;                    // formats in use (1 forward, 4 G-buffer: surface targets + HDR)
            VkFormat depth_format = VK_FORMAT_UNDEFINED; // main depth attachment format
            VkSampleCountFlagBits rasterization_samples = VK_SAMPLE_COUNT_1_BIT;
            runtime const* owner = nullptr; // recording context (scene set / pipeline caches)
            // set to true by operator() when the secondary was actually recorded (begin + end
            // succeeded). Points into a per-frame array owned by the caller of the task batch;
            // the caller waits the recording group before reading it, so no extra sync is
            // needed. The primary must NOT execute a segment whose begin failed.
            std::atomic<bool>* recorded = nullptr;

            void operator()() const; // defined in runtime.cpp (module-private)
        };

        /** @brief the command buffer currently being recorded (between begin_recording() and
         *         end_recording()); internal use for the runtime's own recording */
        [[nodiscard]] VkCommandBuffer active_command_buffer() const noexcept;
        /** @brief end the rendering instance / render pass and finish recording;
         *         end_recording_failed when vkEndCommandBuffer fails */
        frame_status end_recording();
        /** @brief submit the recorded command buffer and present the image (recreating the
         *         swapchain when presentation reports out of date); submit_failed /
         *         present_failed on fatal errors */
        frame_status submit_and_present();

        /**
         * @ingroup vulkan_runtime
         * @brief create a named pipeline from raw SPIR-V and cache it in the runtime. The first
         *        pipeline created becomes the runtime's DEFAULT pipeline (implicitly); primitives
         *        with default semantics draw with it. Every pipeline shares the scene set layout,
         *        so any number of them can coexist in one scene (leaves choose by name).
         * @param pipeline_name the pipeline's name (used by primitives to request it, and by
         *        render_environment to bind it); must be unique
         * @param vertex_shader_code raw SPIR-V binary of the vertex shader
         * @param fragment_shader_code raw SPIR-V binary of the fragment shader
         * @return success, or an error message on failure
         * @note thread-safe (registry guarded), but call OUTSIDE the frame loop: recording
         *       workers read the registry lock-free during a frame (see record_main_segment's
         *       concurrency note), so mid-frame registration would race them. Register at setup
         *       or between frames (while idle).
         */
        std::expected<void, std::string> make_pipeline(
            std::string_view pipeline_name,
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief make @p pipeline_name the runtime's default pipeline (the one default-semantics
         *        primitives draw with; see make_pipeline for the implicit first-pipeline default)
         * @param pipeline_name a pipeline previously created via make_pipeline()
         */
        void set_default_pipeline(std::string_view pipeline_name);

        /**
         * @ingroup vulkan_runtime
         * @brief create the skybox background pipeline: a fullscreen triangle (drawn with
         *        vkCmdDraw(3), no vertex/index buffers) that samples the environment cubemap
         * @param vertex_shader_code raw SPIR-V binary of the skybox vertex shader
         * @param fragment_shader_code raw SPIR-V binary of the skybox fragment shader
         * @return success, or an error message on failure
         * @note drawn first in every frame with depth test/write disabled, so models render over it;
         *       uses the shared scene set (camera UBO binding 0, env cubemap binding 2)
         */
        /**
         * @ingroup vulkan_runtime
         * @brief create the post-process pipeline (HDR scene target -> exposure + ACES tonemap +
         *        gamma -> swapchain): the fullscreen pass runs after the scene rendering instance
         *        closes and before the debug overlay, on a 1x swapchain image
         * @param vertex_shader_code post.vert SPIR-V (synthesizes the fullscreen triangle)
         * @param fragment_shader_code post.frag SPIR-V (sampler2D HDR input + push constants)
         * @note call once after the runtime is set up; callers load the SPIR-V (see
         *       chores::setup_pipeline). Without it the HDR scene target cannot be presented
         */
        std::expected<void, std::string> make_post_pipeline(
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief create the FXAA pipeline (gamma-encoded LDR image -> anti-aliased swapchain)
         * @param vertex_shader_code post.vert SPIR-V (the same fullscreen triangle)
         * @param fragment_shader_code fxaa.frag SPIR-V (sampler2D LDR input + push constants)
         * @note optional: without it set_fxaa() has no effect and the composite keeps writing the
         *       swapchain directly. Requires make_post_pipeline() first (it owns the set layout).
         */
        std::expected<void, std::string> make_fxaa_pipeline(
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code);
        std::expected<void, std::string> make_skybox_pipeline(
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief create the directional shadow pipeline: renders scene geometry depth-only into
         *        the shadow map (no color attachment), from the light's view
         * @param vertex_shader_code raw SPIR-V binary of the shadow vertex shader
         * @param fragment_shader_code raw SPIR-V binary of the shadow fragment shader
         * @return success, or an error message on failure
         * @note depth-only rendering (no color attachment) requires dynamic rendering, which
         *       is core Vulkan 1.3 - the only path the engine supports
         */
        std::expected<void, std::string> make_shadow_pipeline(
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief create the clustered-light-culling compute pipeline (M5)
         * @param compute_shader_code raw SPIR-V binary of shaders/light_cluster.comp
         * @return success, or an error message on failure
         * @note optional: without it (or with clustering off) the shading stage loops every active
         *       light instead, which is the brute-force reference the clustered path is verified
         *       against. The per-slot cluster buffers exist regardless (they are created with the
         *       scene set), so enabling the pass later needs no resource rebuild.
         */
        std::expected<void, std::string> make_cluster_pipeline(std::span<unsigned char const> compute_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief turn clustered light culling on/off (no-op without make_cluster_pipeline())
         * @param enabled true = the shading stage loops only its own cluster's light list
         * @note CPU-side only (the flag rides the light UBO's cluster_grid.w lane): the next frame's
         *       cluster pass and shading both read it, so it is safe to toggle mid-run.
         */
        void set_clustered_lights(bool enabled) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief tell the runtime that its default pipeline is a flat/unlit one
         * @param unlit true = the deferred lighting stage writes the stored albedo, unshaded
         *
         * The forward path switches pipelines, which the runtime default already covers; the deferred
         * path cannot (its G-buffer and lighting stages bind their own pipelines), so its lighting
         * stage needs to be told. Set it together with set_default_pipeline() when the render mode
         * changes, and both paths then agree about what "unlit" means - a shading-free view of the
         * geometry, with the sky still drawn behind it.
         */
        void set_unlit(bool unlit) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief the active cluster grid (tile columns, tile rows) - 0 until the first frame
         */
        [[nodiscard]] std::pair<uint32_t, uint32_t> cluster_grid_extent() const noexcept {
            return {this->cluster_tiles_x, this->cluster_tiles_y};
        }

        /**
         * @ingroup vulkan_runtime
         * @brief enable directional shadow mapping: fills the light UBO with an orthographic
         *        view-proj framing the given scene bounds (plus the light direction, matching
         *        the sky sun). Must be called after the models exist (the shadow pass draws them).
         * @param scene_center world-space center of the shadow frustum (usually the scene bounds
         *        center after the scene offset is applied, i.e. where the models actually sit)
         * @param scene_radius conservative radius covering all shadow casters
         * @note requires make_shadow_pipeline() to have succeeded; no-op otherwise
         */
        void enable_shadows(glm::vec3 const& scene_center, float scene_radius);

        /**
         * @ingroup vulkan_runtime
         * @brief set an extra whole-scene transform applied on top of every scene-tree root
         * @param transform world matrix placed before the roots' local transforms
         * @note identity (the default) leaves rendering untouched; useful for programmatic
         *       grouping / demo rotation of the whole imported scene
         */
        void set_scene_transform(glm::mat4 const& transform);

        /**
         * @ingroup vulkan_runtime
         * @brief enable or disable main-pass frustum culling (BVH vs camera frustum)
         * @param enabled true (default) culls leaves outside the view frustum before drawing
         */
        void set_frustum_culling(bool enabled) noexcept {
            this->frustum_culling = enabled;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief enable or disable drawing the skybox background pass each frame
         * @param enabled true (default) draws the environment skybox; false leaves the clear color
         * @note cheap toggle: only affects command recording, no resource rebuild
         */
        void set_skybox_enabled(bool enabled) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief enable or disable the directional shadow each frame
         * @param enabled true (default) records the shadow pass and samples the map; false skips
         *        the depth pass - pbr.frag then skips calc_shadow entirely (fully lit)
         * @note requires enable_shadows() to have succeeded. Flag-only toggle: nothing is cleared
         *       and no command is recorded; the shader simply stops sampling the stale map
         */
        void set_shadow_enabled(bool enabled);

        /**
         * @ingroup vulkan_runtime
         * @brief select the specular BRDF model preset (pbr.frag, gui "brdf model" combo):
         *        0 = GGX + joint Smith (default), 1 = GGX + height-correlated Smith,
         *        2 = Beckmann + Smith, 3 = Blinn-Phong + Smith
         * @param model preset id (clamped to the valid range)
         * @note CPU-side only: the value rides the light UBO's std140 padding and is copied into
         *       the paced slot's buffer every frame, so this is safe at any time (GUI included)
         */
        void set_brdf_model(int model) noexcept;
        /**
         * @ingroup vulkan_runtime
         * @brief select the diffuse BRDF model (pbr.frag, gui "diffuse model" combo):
         *        0 = Lambert (default), 1 = Oren-Nayar
         * @param model model id (clamped)
         * @note same timing rule as set_brdf_model
         */
        void set_diffuse_model(int model) noexcept;
        /**
         * @ingroup vulkan_runtime
         * @brief linear exposure scale applied to the rendered image before tonemapping
         *        (pbr.frag / skybox.frag read it from the light UBO / skybox push constant)
         * @param exposure multiplier (1 = unchanged; clamped to a sane 0.05 .. 20 range)
         * @note same timing rule as set_brdf_model: CPU-side only, copied into the paced slot
         *       every frame, so this is safe at any time (GUI slider included)
         */
        void set_exposure(float exposure) noexcept;
        /**
         * @ingroup vulkan_runtime
         * @brief the current exposure scale (see set_exposure)
         */
        [[nodiscard]] float exposure() const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief cel/toon shading: quantize the direct-light diffuse falloff (and harden the
         *        shadow edge / the specular lobe) into @p steps bands
         * @param steps 0 = plain PBR (default); 2..8 = band count (rounded, clamped)
         * @param softness band edge width in normalized [0,1] space (0.01..0.5); smaller = harder
         *        cel edges
         * @note same timing rule as set_exposure: CPU-side, copied into the light UBO every frame
         */
        void set_toon_shading(float steps, float softness) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief bloom amount for the post-process pass (bright-pass threshold + blend weight)
         * @param intensity how much of the blurred bright pass is added back (0 disables bloom)
         * @param threshold linear luminance subtracted in the bright pass (visible range 0..0.75:
         *        brightest highlights glow); both are clamped to sane ranges
         * @note same timing rule as set_exposure: CPU-side, copied into the post push constants
         */
        void set_bloom(float intensity, float threshold) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief enable/disable FXAA and set its two knobs
         * @param enabled when true the composite renders into a display-referred (gamma-encoded)
         *        LDR image and an extra fullscreen pass anti-aliases it into the swapchain; when
         *        false the composite writes the swapchain directly (no extra pass, no LDR read)
         * @param subpixel sub-pixel term strength, 0..1 (0 = pure directional blend; higher also
         *        blends away the single-pixel aliasing FXAA leaves on near-axis-aligned edges)
         * @param edge_threshold relative luma contrast below which a pixel counts as flat and is
         *        left untouched (FXAA default 0.166; lower = more edges treated, softer image)
         * @note requires make_fxaa_pipeline(); without it the flag has no effect. CPU-side, copied
         *       into the post push constants every frame (same timing rule as set_exposure)
         */
        void set_fxaa(bool enabled, float subpixel = 0.75f, float edge_threshold = 0.166f) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief cap the render loop at @p fps frames per second (0 or less = uncapped)
         *
         * A frame limiter, not a present-mode choice: with Mailbox - and, measured, even with
         * FIFO_LATEST_READY - the loop free-runs as fast as the CPU can record, which burns most of a
         * core producing frames nobody sees. The wait lives in pace_and_acquire() so it is attributed
         * to the pace phase and never happens while a swapchain image is held. 0 is the default because
         * that is the mode a throughput measurement needs.
         */
        void set_max_fps(double fps) noexcept;

        /** @brief whether the FXAA pass is currently enabled (see set_fxaa) */
        [[nodiscard]] bool fxaa() const noexcept {
            return this->fxaa_on;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief enable/disable the GPU pass timing read-back
         * @param enabled when true (the default) every frame records one timestamp per pass
         *        boundary and the completed measurements are averaged over a 60-frame window,
         *        logged and exposed through gpu_timing_summary(); when false no timestamp is
         *        written and nothing is read back
         * @note the switch only matters on devices that can timestamp at all
         *       (core::gpu_timing_available()); the marks themselves are a handful of
         *       vkCmdWriteTimestamp calls per frame, so the cost of leaving it on is negligible -
         *       this exists for a measurement-free profile run
         */
        void set_gpu_timings(bool enabled) noexcept {
            this->gpu_timings_enabled = enabled;
            // one switch for measure this frame: the CPU phases follow the GPU marks
            this->cpu_timings.set_enabled(enabled);
        }

        /** @brief whether GPU pass timings are being collected (see set_gpu_timings) */
        [[nodiscard]] bool gpu_timings() const noexcept {
            return this->gpu_timings_enabled;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief whether this device can measure GPU pass timings at all
         * @note false when the graphics queue family cannot write timestamps: the timing calls
         *       become no-ops instead of failing, so a caller can always ask
         */
        [[nodiscard]] bool gpu_timings_available() const noexcept {
            return this->vulkan_core.gpu_timing_available();
        }

        /**
         * @ingroup vulkan_runtime
         * @brief report of the current timing window: the mean GPU milliseconds per pass
         * @return e.g. "gpu:  shadow 0.11  main 0.24  bloom 0.01\n     composite 0.08  fxaa 0.00
         *         tail 0.00\n     total 0.43 ms" (broken over lines to fit the narrow overlay
         *         panel), or a short "off"/"unavailable"/"collecting" note
         * @note safe to call every frame (it formats a string from the running window mean) - the
         *       debug overlay binds a label to it
         */
        [[nodiscard]] std::string gpu_timing_summary() const;

        /** @brief how many channels the G-buffer debug view offers (see set_gbuffer_channel) */
        static constexpr int gbuffer_channel_count = 8;

        /**
         * @ingroup vulkan_runtime
         * @brief name the G-buffer pass's default pipeline is bound under while it records
         * @note the G-buffer pipeline is NOT registered in the runtime's named pipeline cache: it
         *       declares three color attachments, so it can only be used inside the G-buffer
         *       rendering instance (the named cache's pipelines all declare the single HDR target
         *       and are usable in the ordinary forward instance). The G-buffer pass therefore hands
         *       its own pipeline to default-semantics leaves under this name - a leaf with explicit
         *       pipeline semantics still requests its own name and is skipped with a log, because
         *       drawing it here would violate the instance's attachment formats.
         */
        static constexpr std::string_view gbuffer_pipeline_name = "gbuffer";

        /**
         * @ingroup vulkan_runtime
         * @brief create the G-buffer pipeline: the deferred path's surface-only fragment stage
         * @param vertex_shader_code raw SPIR-V of pbr.vert (the G-buffer reuses the forward vertex
         *        stage: instancing / skinning / morphing / tangents are identical, only the shading
         *        half differs)
         * @param fragment_shader_code raw SPIR-V of gbuffer.frag
         * @return success, or an error message on failure
         * @note the pipeline declares the three core::gbuffer_formats targets plus the depth format,
         *       so it is only valid inside a rendering instance with exactly those attachments - the
         *       one set_gbuffer_debug() builds. Register it like any other pipeline (it is NOT the
         *       default: the G-buffer pass binds it explicitly).
         */
        std::expected<void, std::string> make_gbuffer_pipeline(std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief create the G-buffer debug-view pipeline (fullscreen: the three targets + the depth
         *        image -> the HDR scene target, so the ordinary post chain still runs)
         * @return success, or an error message on failure
         * @note optional but required for set_gbuffer_debug(true) to take effect; on its own it
         *       changes nothing
         */
        std::expected<void, std::string> make_gbuffer_debug_pipeline(std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief create the deferred lighting pipeline (fullscreen: the three G-buffer targets + the
         *        depth image -> the HDR target, added on top of the background and the emissive)
         * @param vertex_shader_code raw SPIR-V of post.vert (the fullscreen triangle; the lighting
         *        stage has no vertex input of its own)
         * @param fragment_shader_code raw SPIR-V of deferred.frag
         * @return success, or an error message on failure
         * @note optional but required for set_deferred(true) to take effect. It shares the G-buffer
         *       input set layout with the debug view (bound as set 1 here, as set 0 there) and the
         *       shared scene set as set 0.
         */
        std::expected<void, std::string> make_deferred_pipeline(std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief draw the scene's opaque geometry into the G-buffer and shade it in screen space
         *        (the deferred render mode)
         * @param enabled when true the opaque pass writes the G-buffer (1x), the sky is drawn as a
         *        background pass first, and a fullscreen stage then shades every pixel from the
         *        G-buffer through the SAME lighting code the forward path uses
         *        (shaders/shading.glsl), adding the result on top of the background and the emissive.
         *        The post chain is unchanged, so a deferred frame and a forward frame differ only in
         *        where the shading happened - which is what makes them comparable. Without the
         *        pipelines from make_gbuffer_pipeline() + make_deferred_pipeline() the flag has no
         *        effect (the forward path keeps running).
         * @note alphaMode BLEND geometry is NOT drawn in this mode yet: the forward transparent pass
         *       needs a pipeline whose sample count matches the G-buffer depth (1x), so it re-enters
         *       with the 1x/MAA-off path the TAA milestone brings. The runtime logs it once.
         * @note the G-buffer pass runs at 1x whatever MSAA the forward path uses: a multisampled
         *       G-buffer would need per-sample shading, which is the trade the deferred path makes.
         */
        void set_deferred(bool enabled) noexcept;

        /** @brief whether the deferred lighting stage is enabled (see set_deferred) */
        [[nodiscard]] bool deferred() const noexcept {
            return this->deferred_on;
        }

        /** @brief how many jitter positions the Halton(2,3) TAA sequence cycles through */
        static constexpr uint32_t taa_jitter_count = 8;

        /**
         * @ingroup vulkan_runtime
         * @brief create the TAA resolve pipeline (fullscreen: scene color + history + motion vectors +
         *        depth -> the HDR target)
         * @param vertex_shader_code raw SPIR-V of post.vert (the fullscreen triangle)
         * @param fragment_shader_code raw SPIR-V of taa.frag
         * @return success, or an error message on failure
         * @note optional but required for set_taa(true) to take effect
         */
        std::expected<void, std::string> make_taa_pipeline(std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief enable/disable temporal anti-aliasing and set its two blend weights
         * @param enabled when true (and the deferred path is the active render mode) the projection is
         *        jittered every frame, the G-buffer's motion vectors are resolved against a reprojected
         *        history, and the result is what the post chain processes. This is the deferred path's
         *        answer to MSAA: it resolves the sub-pixel detail MSAA would have sampled, AND the
         *        shimmer in motion that no edge filter can remove. Requires make_taa_pipeline(); without
         *        it the flag has no effect.
         * @param blend_static history weight for a pixel that did not move (0.9 = 10% of the current
         *        frame per frame; higher converges smoother but reacts slower to lighting changes)
         * @param blend_min history weight floor once a pixel moves a pixel or more per frame (lower =
         *        trusts the current frame more under motion, which trades smoothing for less ghosting)
         * @note deferred-path only for now: the forward path has no motion vectors, so it keeps its
         *       MSAA answer. The G-buffer motion vectors are camera-only at this milestone, so a
         *       deformed (skinned/morphed) object can ghost slightly - see gbuffer.frag.
         */
        void set_taa(bool enabled, float blend_static = 0.9f, float blend_min = 0.5f) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief screen-space ambient occlusion of the deferred lighting stage (M6)
         * @param enabled master switch (false pushes an intensity of 0, which the shader returns as
         *        an exact 1.0 - an SSAO-off frame is bit for bit the pre-M6 frame)
         * @param radius world-space sample radius (typical: 0.3..1.5, i.e. a fraction of the scene
         *        scale; the ground-truth mismatch of a screen-space AO is that its apparent strength
         *        depends on the view distance)
         * @param intensity how much of the computed occlusion is applied (0..1, 1 = full)
         * @param samples samples per pixel, clamped to the shader's MAX_SSAO_SAMPLES (16)
         * @note deferred-path only: the trace needs the G-buffer depth and normals, which the forward
         *       path does not store. The occlusion scales the IBL ambient (diffuse and specular), not
         *       the direct sun - see shade_surface().
         */
        void set_ssao(bool enabled, float radius = 0.5f, float intensity = 1.0f, uint32_t samples = 8) noexcept;

        /** @brief how many shadow cascades are active (1 = the single-map behavior) */
        [[nodiscard]] uint32_t shadow_cascade_count() const noexcept {
            return this->shadow_cascades;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief set the shadow map's edge length in texels ([render] shadow_map_size)
         * @param size requested edge length; clamped to 256..8192 and rounded to a power of two
         * @note STARTUP-ONLY, like set_shadow_cascades: the layered image, its per-layer views, the
         *       descriptor, the depth pass's rendering instance and the shadow pipeline's viewport
         *       are all created from it, so it must be called before the scene import (the resources
         *       are created lazily by the first scene set). A later call is ignored with a log line
         *       rather than silently taking effect on the next resize.</note>
         */
        void set_shadow_map_size(uint32_t size) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief set how many shadow cascades to fit, render and sample
         * @param cascades 1 (one box over the whole visible range, the historic behavior) up to
         *        vulkan::max_shadow_cascades; clamped
         * @note call it BEFORE the scene is imported (the resources exist by then and the cascade
         *       count only changes which layers are used). The split scheme is a practical
         *       logarithmic/uniform blend, so cascade 0 gets a fraction of the far cascade's texel
         *       size - the whole reason a single map cannot serve near and far at once.
         */
        void set_shadow_cascades(uint32_t cascades) noexcept;

        /** @brief set the cascade blend band (0 disables blending; 0.1 = the last 10% of a cascade) */
        void set_shadow_cascade_blend(float blend) noexcept;

        /** @brief whether TAA is enabled (see set_taa) */
        [[nodiscard]] bool taa() const noexcept {
            return this->taa_on;
        }

        /** @brief the halton jitter position the NEXT frame will use (0..taa_jitter_count-1) */
        [[nodiscard]] uint32_t taa_jitter_position() const noexcept {
            return this->taa_jitter_index;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief draw the scene's opaque geometry into the G-buffer and show a debug view of it
         *        instead of the shaded image
         * @param enabled when true the opaque pass records into the three G-buffer targets (1x) and
         *        a fullscreen pass visualizes one channel into the HDR target, which the post chain
         *        then processes as usual. Without a pipeline from make_gbuffer_pipeline() +
         *        make_gbuffer_debug_pipeline() the flag has no effect (the forward path keeps
         *        running). Takes precedence over set_deferred(true): inspecting the stored data is not
         *        a render mode, so both flags on shows the channels.
         * @note alphaMode BLEND geometry is skipped in this mode: a G-buffer cannot carry a blended
         *       surface, and the transparent pass stays a forward pass around it
         */
        void set_gbuffer_debug(bool enabled) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief log which optional features are actually available this session
         *
         * One line naming every optional pipeline that could NOT be created (deferred lighting,
         * G-buffer debug view, TAA, FXAA, shadow pass, skybox, clustered light culling). Those
         * features are toggled from the debug overlay and from the config, and a toggle whose
         * pipeline is missing does nothing at all - silently, which reads as "this switch is
         * broken". Call it once after the pipelines are created (main does) so the log answers
         * that question up front; `warn_missing_feature` additionally reports a toggle that
         * cannot take effect at the moment it is switched on.
         */
        void log_feature_status() const;

        /**
         * @ingroup vulkan_runtime
         * @brief whether an optional feature is usable this session (its pipeline was created)
         * @param name one of "deferred", "gbuffer-debug", "taa", "fxaa", "shadow", "skybox",
         *        "clustered" - unknown names return false
         *
         * The debug overlay asks this to decide what to offer (see vulkan::gui::widget::visible_when):
         * a control whose pipeline does not exist can never do anything, so it is hidden instead of
         * being shown inert. log_feature_status() prints the same information once at startup.
         */
        [[nodiscard]] bool feature_available(std::string_view name) const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief which render features actually RUN this frame
         *
         * One declared place for "what exists, what it needs, and is it on" - every field is derived
         * from the config/GUI state plus the pipelines that were created, so nothing can disagree.
         * It is the input to pass recording (the shadow pass and the cluster dispatch are skipped
         * when no shading stage would read their output) and to `feature_active()`, which the debug
         * overlay uses to decide what to offer and the log uses to report what ran.
         *
         * The gates that are NOT just "the switch is on" are the interesting ones, and each carries
         * its reason:
         *  - shadow: skipped in the flat (unlit) render mode - nothing samples the map there, so the
         *    pass is pure waste (measured: 0.22 ms, ~45% of a 0.5 ms frame).
         *  - clustered: skipped when no punctual light is active (nothing to sort) and in unlit mode.
         *  - taa/ssao: deferred-path features (they consume the G-buffer).
         *  - bloom: off with an intensity of 0 or while the G-buffer debug view is up.
         *  - transparent: the forward transparent pass, which the G-buffer pass replaces.
         */
        struct render_features {
            bool unlit = false;         // the flat render mode (no lighting anywhere)
            bool gbuffer_debug = false; // the opaque pass stores the G-buffer for the debug view
            bool deferred = false;      // the opaque pass stores the G-buffer and lighting is deferred
            bool shadow = false;        // record the directional shadow pass
            bool clustered = false;     // record the cluster compute pass
            bool taa = false;           // resolve TAA
            bool ssao = false;          // the lighting stage applies screen-space AO (shader-side gate)
            bool bloom = false;         // run the bloom chain
            bool fxaa = false;          // run the final FXAA pass
            bool skybox = false;        // draw the forward skybox background pass
            bool transparent = false;   // the forward transparent pass has work to record
        };
        [[nodiscard]] render_features active_features() const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief the last completed CPU phase window, as the overlay label shows it
         * @return one line per phase (pace / begin / scene / post / submit) with fixed-width
         *         millisecond fields, updated once per 60-frame window - the CPU counterpart of
         *         gpu_timing_summary(), and the instrument for the range where the frame is no
         *         longer GPU-bound (see cpu_phase)
         */
        [[nodiscard]] std::string cpu_timing_summary() const;

        /**
         * @ingroup vulkan_runtime
         * @brief per-phase CPU milliseconds averaged over the window in progress (0 when empty)
         * @note raw access for diagnostics/tests; the overlay uses cpu_timing_summary()
         */
        [[nodiscard]] std::array<double, static_cast<std::size_t>(vulkan::profiling::cpu_phase::count)> cpu_timing_means() const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief whether a feature is available AND switched on right now (name-keyed)
         * @param name "deferred", "gbuffer-debug", "taa", "fxaa", "shadow", "skybox", "clustered",
         *        "ssao", "bloom", "unlit" - the same vocabulary as feature_available()
         *
         * This is what the overlay gates its controls on (`vulkan::gui::widget::visible_when`): a
         * control is offered exactly when switching it could change the frame. feature_available()
         * answers the different question "could this ever run this session", which is what the
         * startup log reports.
         */
        [[nodiscard]] bool feature_active(std::string_view name) const noexcept;

        /** @brief whether the opaque pass currently writes the G-buffer (see set_gbuffer_debug) */
        [[nodiscard]] bool gbuffer_debug_enabled() const noexcept {
            return this->gbuffer_debug;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief select the channel the G-buffer debug view shows
         * @param channel 0 albedo, 1 world normal, 2 roughness, 3 metallic, 4 ambient occlusion,
         *        5 material id (colorized), 6 linearized depth, 7 raw material flags; clamped into
         *        [0, gbuffer_channel_count - 1]
         */
        void set_gbuffer_channel(int channel) noexcept;

        /** @brief the channel the G-buffer debug view shows (see set_gbuffer_channel) */
        [[nodiscard]] int gbuffer_channel() const noexcept {
            return this->gbuffer_channel_index;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief one captured frame image: tightly packed 8-bit RGBA (top-left origin, row-major)
         */
        struct frame_image {
            uint32_t width = 0;
            uint32_t height = 0;
            std::vector<unsigned char> rgba = {}; // width * height * 4, R G B A
        };

        /**
         * @ingroup vulkan_runtime
         * @brief read back the frame captured by the last screenshot request
         * @return the captured image (8-bit RGBA, swizzled from the swapchain format), or an error
         *         string when nothing was captured / the swapchain format is unsupported
         * @note the pixels were already copied GPU-side, into a persistent host-visible buffer, by
         *       record_screenshot_copy() while the frame was recorded (see consume_screenshot_request),
         *       so this only waits for the device to idle and swizzles - no transition of a
         *       presentable image (the WSI owns it after vkQueuePresentKHR). Still a low-frequency
         *       debug feature: expect a visible hitch, do not call per frame. Requires a 4-channel
         *       8-bit swapchain format (BGRA/RGBA, sRGB or UNORM); the sRGB encoding is preserved,
         *       so the PNG matches what was on screen.
         */
        std::expected<frame_image, std::string> acquire_current_frame_image();

        /**
         * @ingroup vulkan_runtime
         * @brief ask whether a frame was captured for the pending screenshot request
         * @return true when the read-back copy has been recorded and is ready to be read by
         *         acquire_current_frame_image() (which consumes it) - so the caller's pattern stays
         *         `if (consume_screenshot_request()) { acquire_current_frame_image(); ... }`
         * @note the request itself (F12 edge in poll_events, or request_screenshot()) is consumed
         *       during recording: end_recording() records the copy into that frame's own command
         *       buffer while the swapchain image still belongs to it. A request whose frame could
         *       not be recorded (zero-sized swapchain) stays pending for a later frame.
         */
        [[nodiscard]] bool consume_screenshot_request() noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief request a screenshot without the F12 key (scripted / automatic captures)
         * @note sets exactly the request that the F12 edge trigger sets, so the caller still
         *       consumes it through consume_screenshot_request() and owns the read-back + save;
         *       lets a headless-ish run (e.g. `--capture-frames`) capture a frame with no human
         *       at the keyboard
         */
        void request_screenshot() noexcept {
            this->screenshot_requested = true;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief set the active punctual lights (point/spot, pbr.frag's direct-light loop).
         *        Each light is evaluated through the same BRDF path as the directional sun
         *        (inverse-square falloff, optional smooth range cutoff, spot cone mask) and
         *        never casts a shadow in this version.
         * @param lights the lights to enable. Every `vulkan::punctual_light` field is honored:
         *        `position`, linear `color` (radiance = color * intensity), `range` (0 =
         *        infinite falloff, otherwise a smooth cutoff at this distance), and for spot
         *        lights `spot = true` + `spot_direction` + `spot_outer_cos` (cos of the outer
         *        half-angle; the shader derives the soft inner cone as mix(outer, 1, 0.6)).
         *        Entries beyond `vulkan::max_punctual_lights` (4) are dropped.
         * @note same timing rule as set_brdf_model: CPU-side only, copied into the paced
         *       slot's buffer every frame, so safe at any time (GUI included). The demo GUI
         *       currently exposes two POINT lights (no spot toggle); programmatic callers can
         *       set spot lights directly.
         */
        void set_point_lights(std::span<punctual_light const> lights) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief set the live depth bias of the directional shadow pass (applied every frame via
         *        vkCmdSetDepthBias before the depth-only draw)
         * @param constant_factor fixed depth bias added to every fragment's depth
         * @param slope_factor slope-scaled bias (depth units per depth-unit of surface slope) -
         *        the main acne control for angled surfaces
         * @param clamp maximum depth bias magnitude (0 = no clamp)
         * @note cheap: only changes the value recorded into the command buffer, no pipeline or
         *       resource rebuild; useful to chase shadow acne / peter-panning per model
         */
        void set_shadow_depth_bias(float constant_factor, float slope_factor, float clamp) noexcept {
            this->shadow_depth_bias_constant = constant_factor;
            this->shadow_depth_bias_slope = slope_factor;
            this->shadow_depth_bias_clamp = clamp;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief mark the scene tree as changed (structure or per-node local transforms edited
         *        through get_scene(), e.g. programmatic animation): the culling BVH is rebuilt on
         *        the next frame. Internal scene mutations (import / make / clear /
         *        set_scene_transform) invalidate automatically.
         */
        void scene_changed() noexcept {
            this->bvh_dirty = true;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief print the scene tree (names + local-transform marker + leaf primitive pipeline)
         *        to the log, one indented line per node, plus a shape summary
         * @note diagnostic helper: shows whether an import rebuilt the real hierarchy (gltf
         *       node names and nesting) or a flat list of root leaves (pipeline names)
         */
        void log_scene_tree() const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief bind the scene tree this runtime renders. The tree is owned by the caller
         *        (never by the runtime): it must stay alive while the runtime is in use and be
         *        destroyed before the runtime goes away, because the leaves' GPU buffers are
         *        released through the runtime's vma allocator when the tree is destroyed.
         * @param scene the caller-owned scene tree to render
         * @note bind before creating/importing primitives and before the first frame; the
         *       culling BVH is invalidated so the next frame rebuilds it
         */
        void set_scene(scene_tree::scene& scene) noexcept {
            this->bound_scene = &scene;
            this->bvh_dirty = true; // a new tree's world AABBs must be (re)collected
        }

        /**
         * @ingroup vulkan_runtime
         * @brief access the scene tree (roots + children + per-node local transforms) for
         *        programmatic whole-group / subtree transforms
         * @note the tree structure is fixed after import (no reallocation of scene or the
         *       children vectors while nodes are only edited in place), so pointers/references
         *       into the tree stay valid until the next make_primitive / import / clear call
         */
        [[nodiscard]] scene_tree::scene& get_scene() noexcept {
            if (this->bound_scene == nullptr) {
                utility::panic("runtime::get_scene() called before set_scene() bound a scene");
            }
            return *this->bound_scene;
        }
        [[nodiscard]] scene_tree::scene const& get_scene() const noexcept {
            if (this->bound_scene == nullptr) {
                utility::panic("runtime::get_scene() called before set_scene() bound a scene");
            }
            return *this->bound_scene;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief get a cached pipeline by its name
         * @param pipeline_name the name passed to make_pipeline()
         * @return pointer to the cached pipeline, or nullptr if no pipeline with that name exists
         */
        [[nodiscard]] vk_pipeline const* get_pipeline(std::string_view pipeline_name) const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief upload the scene-wide IBL resources (prefiltered env / irradiance / BRDF LUT)
         *        into the shared scene set; call it before creating models that use IBL
         * @param info precomputed split-sum IBL bytes (see vulkan::generate_* helpers)
         * @note the images are uploaded once and shared by every primitive (they used to be
         *       duplicated per primitive)
         * @note call before the first frame, or only while the runtime is idle (no frame in
         *       flight): this rewrites bindings 2-4 on every scene set, and the scene sets are
         *       no longer update-after-bind - updating a set an in-flight frame may read is a
         *       spec violation. Mid-loop IBL swaps need wait_idle() first.
         */
        void set_ibl(ibl_input const& info);

        /**
         * @ingroup vulkan_runtime
         * @brief upload the scene-wide skin matrices (scene set binding 9) into the frame slot
         *        paced by the last pace_and_acquire(): the caller fills the buffer layout
         *        [identity block (4 mat4s) | per-skin joint blocks] and calls this once per frame
         *        AFTER pace_and_acquire(); primitives reference their block start via
         *        material_push_constants::skin_base (0 = the identity block: unskinned draws)
         * @param matrices at most scene_skin_capacity mat4s; anything beyond the capacity is dropped
         * @note host-visible copy, no Vulkan objects involved; the per-slot buffers guarantee an
         *       in-flight frame never shares the buffer being rewritten
         */
        void set_skin_matrices(std::span<glm::mat4 const> matrices);

        /**
         * @ingroup vulkan_runtime
         * @brief like set_skin_matrices() but into an explicit slot buffer (used for setup-time
         *        uploads that must be visible to every slot, e.g. the identity block before the
         *        render loop starts)
         */
        void set_skin_matrices(std::span<glm::mat4 const> matrices, uint32_t slot);

        /**
         * @ingroup vulkan_runtime
         * @brief host-visible scratch memory of the ACTIVE frame slot's morph buffer (scene set
         *        binding 10, scene_morph_capacity floats each). The caller lays out per-primitive
         *        morph blocks (per vertex per target pos-delta/nrm-delta floats, then the
         *        per-target weights) and points primitives at their block through
         *        material_push_constants::morph_base / morph_targets / morph_vertices (see those
         *        fields for the layout convention).
         * @return the mapped base of the slot paced by the last pace_and_acquire(), or nullptr when
         *         the morph buffer is unavailable
         */
        [[nodiscard]] void* morph_scratch() noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief like morph_scratch() but for an explicit slot buffer (used for setup-time bakes
         *        that must be duplicated into every slot's buffer before the render loop starts)
         */
        [[nodiscard]] void* morph_scratch(uint32_t slot) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief create a primitive and attach it to the scene tree as a new leaf (root node).
         * @param pipeline_name the pipeline the primitive draws with (must already exist)
         * @param info geometry and material textures
         * @return pointer to the created primitive (owned by the scene tree), or nullptr if the
         *         pipeline does not exist
         * @note the leaf's local transform is @p info.model_matrix and its world is identity
         *       (the frame record phase runs update_world before drawing, so primitive::set_world writes
         *       the same matrix into push.model as before)
         * @note call before the first frame, or only while the runtime is idle (no frame in
         *       flight): registering a material appends binding-1 texture entries to every scene
         *       set, and the scene sets are no longer update-after-bind - updating a set an
         *       in-flight frame may read is a spec violation. Mid-loop creation needs wait_idle()
         *       first.
         */
        primitive* make_primitive(std::string_view pipeline_name, primitive_create_info const& info);

        /**
         * @ingroup vulkan_runtime
         * @brief collect every leaf primitive of the given pipeline (DFS over the scene tree)
         * @param pipeline_name the pipeline name passed to make_primitive()
         * @return models whose leaf node name matches @p pipeline_name, in scene-tree order
         */
        [[nodiscard]] std::vector<primitive const*> get_primitives(std::string_view pipeline_name) const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief append an instanced_draw_primitive: draws @p source's geometry once per transform
         *        in ONE draw call per frame (per-instance matrices in scene set binding 6)
         * @param source any primitive of this runtime (its geometry is drawn transforms.size() times;
         *        it must stay in the runtime's primitive list while the instanced primitive is drawn)
         * @param transforms one world matrix per instance (fully places the source geometry)
         * @return pointer to the appended instanced primitive, or nullptr if nothing was appended
         */
        primitive* make_instanced_primitive(primitive const& source, std::span<glm::mat4 const> transforms);

        /**
         * @ingroup vulkan_runtime
         * @brief append a static_draw_primitive: OWNS one merged vertex/index buffer and draws
         *        a chunk table over it — one buffer bind, then one offset draw per chunk with
         *        each chunk's own material. The primitive-level form of a static scene: N
         *        static sub-meshes cost 1 bind + N draws instead of N binds + N draws.
         * @param info merged geometry + chunk table (the packer's output; empty chunks draw the
         *        whole merged range once, degenerating to a plain normal draw)
         * @return pointer to the appended static primitive, or nullptr if nothing was appended
         * @note the whole batch shares one world (push.model from update_world); per-chunk
         *       placement needs separate batches or baked chunk models
         */
        primitive* make_static_draw(static_draw_create_info const& info);

        /**
         * @ingroup vulkan_runtime
         * @brief batch-import a scene by traversing the retained node hierarchy (structural
         *        node stream) and its drawables (geometry stream) together.
         *        @p nfirst must model vulkan::scene_node_iterator: DFS pre-order over every
         *        scene node INCLUDING transform-only nodes, exposing get_name() /
         *        get_local_transform() / get_depth() / get_drawable_count(). @p dfirst must
         *        model vulkan::scene_drawable_iterator (++ plus geometry/material getters);
         *        the loader keeps both streams over the same pool in the same order, so each
         *        node's get_drawable_count() drawables are the next entries of the drawable
         *        stream. The runtime drives the traversal: it rebuilds the node tree into
         *        scene_tree::scene (one scene_node per loader node, named, with its local
         *        transform; a node's drawable becomes a primitive leaf attached to that node —
         *        extra primitives of one node become identity-local child leaves), uploads
         *        buffers and registers materials. No glTF (or any scene format) knowledge
         *        lives in the runtime.
         * @param nfirst,nlast iterator pair over the scene's node hierarchy
         * @param dfirst,dlast iterator pair over the scene's drawables
         * @param offset translation applied to every scene ROOT node's local transform
         *        (e.g. -scene_center + sink); children inherit it through update_world
         * @return counts of imported primitives and materials
         * @note call before the first frame, or only while the runtime is idle (no frame in
         *       flight): importing registers materials, which appends binding-1 texture entries
         *       to every scene set, and the scene sets are no longer update-after-bind -
         *       updating a set an in-flight frame may read is a spec violation. Mid-loop imports
         *       need wait_idle() first.
         */
        template <class NI, class DI>
            requires scene_node_iterator<NI> && scene_drawable_iterator<DI>
        scene_import_result import_scene(NI nfirst, NI nlast, DI dfirst, DI dlast, glm::vec3 const& offset) {
            if (this->bound_scene == nullptr) {
                utility::panic("runtime::import_scene() called before set_scene() bound a scene");
            }
            scene_import_result result = {};
            uint32_t const materials_before = this->material_count;
            // converts a pure image_source (e.g. the glTF loader's image_view) into the
            // internal texture_input with the slot's upload format; invalid images -> white
            auto const to_texture = [](auto const& image, VkFormat const format) {
                texture_input out = {};
                if (image.valid) {
                    out.data = image.data;
                    out.width = image.width;
                    out.height = image.height;
                    out.mip_levels = image.mip_levels;
                    out.format = format;
                    out.valid = true;
                }
                return out;
            };
            // per-node drawable -> primitive_create_info (reads the next drawable of the stream)
            auto const fill_info = [&](DI& drawable, primitive_create_info& info) {
                auto const vertex = drawable.get_vertex();
                auto const index = drawable.get_index();
                info.vertex_data = vertex.data;
                info.vertex_stride = vertex.stride;
                info.vertex_count = vertex.count;
                info.index_data = index.data;
                info.index_type = index.width == 4 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
                info.index_count = index.count;
                info.albedo = to_texture(drawable.get_albedo(), VK_FORMAT_R8G8B8A8_SRGB);
                info.metallic_roughness = to_texture(drawable.get_metallic_roughness(), VK_FORMAT_R8G8B8A8_UNORM);
                info.normal = to_texture(drawable.get_normal(), VK_FORMAT_R8G8B8A8_UNORM);
                info.occlusion = to_texture(drawable.get_occlusion(), VK_FORMAT_R8G8B8A8_UNORM);
                info.emissive = to_texture(drawable.get_emissive(), VK_FORMAT_R8G8B8A8_SRGB); // glTF emissive textures are sRGB
                auto const factors = drawable.get_factors();
                info.factors.base_color_factor = factors.base_color_factor;
                info.factors.emissive_factor = factors.emissive_factor;
                info.factors.metallic_factor = factors.metallic_factor;
                info.factors.roughness_factor = factors.roughness_factor;
                info.factors.normal_scale = factors.normal_scale;
                info.factors.occlusion_strength = factors.occlusion_strength;
                info.factors.alpha_cutoff = factors.alpha_cutoff;
                info.factors.alpha_mask = factors.alpha_mask;
                info.factors.alpha_blend = factors.alpha_blend;
                info.double_sided = drawable.get_double_sided();
            };
            // attach one leaf primitive to @p node (geometry from the next drawable of the stream);
            // returns the created primitive or nullptr if the pipeline is missing
            auto const attach_leaf = [&](scene_tree::scene_node& node, DI& drawable) -> primitive* {
                if (!(drawable != dlast)) {
                    utility::panic(std::source_location::current(), "drawable stream ended before the node tree did");
                }
                primitive_create_info info = {};
                fill_info(drawable, info);
                ++drawable;
                ++result.primitive_count;
                std::unique_ptr<primitive> created = this->create_primitive("pbr", info);
                if (created == nullptr) {
                    return nullptr;
                }
                if (node.primitive_leaf == nullptr) {
                    node.primitive_leaf = std::move(created);
                } else {
                    // a glTF node can carry several primitives; scene_node has one leaf slot, so
                    // extra primitives become identity-local child leaves (world unchanged)
                    scene_tree::scene_node extra;
                    extra.name = node.name + "/prim";
                    extra.primitive_leaf = std::move(created);
                    node.children.push_back(std::move(extra));
                }
                return static_cast<primitive*>(node.primitive_leaf ? node.primitive_leaf.get() : node.children.back().primitive_leaf.get());
            };

            // DFS over the loader's node stream, rebuilding parent/child edges with an explicit
            // stack of ancestors: ancestors[d] holds the scene_node at depth d on the path to
            // the current node. The loader emits nodes in DFS pre-order, so when a node arrives
            // at depth d its parent is the ancestor at depth d-1 (pop everything deeper first).
            std::vector<scene_tree::scene_node*> ancestors = {}; // ancestors[d] = node at depth d
            for (; nfirst != nlast; ++nfirst) {
                std::size_t const depth = nfirst.get_depth();
                // pop ancestors deeper than the arriving node's parent level (their subtrees are done)
                while (ancestors.size() > depth) {
                    ancestors.pop_back();
                }
                scene_tree::scene_node node;
                node.name = std::string(nfirst.get_name());
                node.source_index = nfirst.get_source_index(); // asset node index (e.g. glTF): animation targets map onto the tree through it
                node.local = nfirst.get_local_transform();
                if (depth == 0) {
                    node.local = glm::translate(glm::mat4(1.0f), offset) * node.local; // scene root gets the offset
                }
                // attach under the parent (depth-1) or as a new scene root
                if (depth == 0) {
                    this->bound_scene->roots.push_back(std::move(node));
                    ancestors.assign(1, &this->bound_scene->roots.back());
                } else {
                    if (ancestors.size() != depth) {
                        utility::panic(std::source_location::current(), "node tree stream: broken ancestor stack");
                    }
                    scene_tree::scene_node* const parent = ancestors[depth - 1];
                    parent->children.push_back(std::move(node));
                    ancestors.resize(depth + 1);
                    ancestors[depth] = &parent->children.back();
                }
                scene_tree::scene_node* const current = ancestors[depth];
                // consume this node's drawables (the stream is aligned node-for-node)
                std::size_t const drawable_count = nfirst.get_drawable_count();
                for (std::size_t i = 0; i < drawable_count; ++i) {
                    if (attach_leaf(*current, dfirst) == nullptr) {
                        utility::panic(std::source_location::current(), "failed to import drawable (pipeline 'pbr' missing)");
                    }
                }
            }
            this->bvh_dirty = true; // new leaves attached -> culling BVH must be rebuilt
            result.material_count = this->material_count - materials_before;
            return result;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief destroy and remove all models of a pipeline, no-op if the pipeline has none
         * @param pipeline_name the pipeline name passed to make_primitive()
         */
        void clear_primitives(std::string_view pipeline_name);
    };
} // namespace vulkan