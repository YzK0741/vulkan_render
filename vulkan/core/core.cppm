// ============================================================================
// module: vulkan.core
// module version: 0.3.0  (independent of the app version in CMakeLists project(VERSION))
//
// GPU scaffolding: instance / device / swapchain / VMA / pipeline / descriptor
// plumbing (core.vma / core.pipeline / core.filter / core.init_utils submodules
// are part of this unit). Standalone Vulkan wrapper; depends on VMA + utility,
// with the struct-fill conventions coming from vulkan.constant_init.
//
// evolve: bump MAJOR on breaking interface changes, MINOR on additive features,
//         PATCH on internal fixes - independently of the rest of the project.
// ============================================================================
module;

#include <GLFW/glfw3.h>
#include <array>
#include <vulkan/vulkan.h>

export module vulkan.core;
import utility;
export import vstd;
export import vulkan.core.handles;
export import vulkan.core.vma;
export import vulkan.core.vma.handles;

/**
 * @file core.cppm
 */

namespace vulkan {
    /**
     * @defgroup vulkan_core Vulkan Core Objects Manager
     * @brief manages core vulkan objects and windows instance init and destroy.
     * @note
     *      - RAII
     *      - includes VMA decorator, which is defined in ./vma/vma.cppm
     *
     * @warning
     *      - do not call the init or create function, just use the members or other functions
     *      - no thread-safe
     */

    /**
     * @ingroup vulkan_core
     * @brief agreed flat scene descriptor set layout, shared by all pipelines (see shaders/pbr.frag):
     *        set 0 binding 0 = CameraUBO (uniform buffer; one per frame slot, each slot's set
     *              points at its own - static, no per-frame descriptor writes),
     *              binding 1 = sampler2D textures[] (runtime array, partially bound + non-uniform index),
     *              binding 2/3/4 = prefiltered env / irradiance / BRDF LUT (combined image samplers),
     *              binding 5 = Material materials[] (storage buffer: per-material texture indices + factors),
     *              binding 6 = mat4 instance transforms[] (storage buffer, per-instance world matrices),
     *              binding 7 = LightUBO (uniform buffer: directional light view-proj + direction),
     *              binding 8 = shadow map (sampler2D, NEAREST; manual 3x3 percentage-closer
     *              filtering in pbr.frag — no depth-comparison / hardware PCF),
     *              binding 9 = mat4 skin matrices[] (storage buffer: identity block + per-skin joints),
     *              binding 10 = float morph data[] (storage buffer: per-primitive morph deltas + weights)
     * @note hardcoded instead of parsed from SPIR-V: the indexed layout is flat, so pipelines
     *       skip descriptor / push constant parsing and share one layout object
     */
    export constexpr uint32_t scene_texture_capacity = 128;
    // material_push_constants: 6 uints + aligned mat4 = 96 bytes, see vulkan/scene_tree/scene_tree.cppm
    export constexpr uint32_t scene_push_constant_size = 96;

    /**
     * @brief format of the HDR scene target the forward pass renders into and the post-process
     *        pass samples: the MSAA color images use it, and each swapchain image owns one
     *        single-sample resolve target in it (see core::create_hdr_resolve_resources)
     */
    export constexpr VkFormat hdr_format = VK_FORMAT_R16G16B16A16_SFLOAT;

    /**
     * @ingroup vulkan_core
     * @brief how many color targets the G-buffer pass writes (see gbuffer_formats)
     */
    export constexpr uint32_t gbuffer_target_count = 3;

    /**
     * @ingroup vulkan_core
     * @brief formats of the G-buffer targets, in attachment order (= the fragment output locations
     *        of shaders/gbuffer.frag), and the reason the deferred path is cheap to store:
     *        - 0 RGBA8_UNORM: albedo.rgb (base color, linear) + metallic in a
     *        - 1 RGBA16F: world normal.xyz (no encoding - the conservative layout trades 4 bytes per
     *          pixel for not having to reason about octahedral precision) + roughness in a
     *        - 2 RGBA8_UNORM: material_id low/high byte + ambient occlusion + material flags
     *        16 bytes per pixel in total; the depth is the pass's own single-sampled depth image.
     * @note every target is single-sampled (1x) on purpose: a G-buffer cannot be multisampled
     *       without per-sample shading, which is exactly what the deferred path trades MSAA for
     *       (the anti-aliasing story is TAA/FXAA on the lit image instead). The forward path keeps
     *       its MSAA targets - they are separate images, so both can coexist for an A/B.
     */
    export constexpr std::array<VkFormat, gbuffer_target_count> gbuffer_formats = {
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R8G8B8A8_UNORM,
    };

    /**
     * @ingroup vulkan_core
     * @brief color attachments the G-buffer pass declares: the three surface targets above plus the
     *        HDR scene target, which the pass ADDS the emissive term into
     * @note emissive is lighting-independent, so it does not belong to the deferred lighting stage -
     *       and it needs the material's emissive texture and the fragment's UVs, neither of which the
     *       G-buffer stores. Adding it in the base pass is what commercial deferred renderers do (the
     *       G-buffer pass writes the surface and adds emissive to the scene color), and it is why the
     *       pass's fourth attachment is blended ONE/ONE while the three surface targets are
     *       overwritten. The HDR attachment loads (not clears), so the sky drawn before the pass
     *       survives under the emissive.
     */
    export constexpr uint32_t gbuffer_pass_attachment_count = gbuffer_target_count + 1;

    /**
     * @ingroup vulkan_core
     * @brief how many GPU timing marks one frame may write (the query pool is sized
     *        MAX_FRAMES_IN_FLIGHT * this, and each frame slot owns its own contiguous range)
     * @note a mark is one vkCmdWriteTimestamp; the frame's pass boundaries use a handful of them,
     *       and the remaining capacity is headroom for the passes later milestones add. A frame
     *       that records more marks than this silently stops marking (the extra passes are simply
     *       not measured) instead of overflowing into the next slot's range.
     */
    export constexpr uint32_t gpu_timing_mark_capacity = 16;

    /**
     * @ingroup vulkan_core
     * @brief GPU durations of one completed frame, in mark order (see core::mark_gpu_timing)
     * @note entry i is the time between mark i and mark i + 1, so a frame that wrote
     *       @p mark_count marks yields mark_count - 1 durations
     */
    export struct gpu_timing_result {
        std::array<double, gpu_timing_mark_capacity> milliseconds = {}; // elapsed per consecutive mark pair
        uint32_t mark_count = 0;                                        // marks the frame wrote (0 = no measurement)
    };

    /**
     * @ingroup vulkan_core
     * @brief everything the core needs at construction, decoupled from the caller (the runtime
     *        assembles this, typically from the app's startup config)
     * @note fields mirror the app_config render settings; defaults keep the historic behavior
     */
    export struct core_create_info {
        int window_width = 1080;
        int window_height = 960;
        std::string window_title = "vulkan_render"; // GLFW window title
        // vsync: false (default) prefers VK_PRESENT_MODE_MAILBOX_KHR, true prefers FIFO_KHR
        bool vsync = false;
        // MSAA sample count: 0 (default) = auto (device max usable), otherwise a fixed count
        // (2/4/8/...); the core clamps to the device's max usable when the requested count is
        // not supported
        int msaa_samples = 0;
        // Vulkan validation layers + debug messenger (instance layer VK_LAYER_KHRONOS_validation
        // and the VK_EXT_debug_utils messenger); off by default - the caller (app_config) keeps
        // the historic Debug-on / Release-off default and can override it per build
        bool validation_layers = false;
        // optional caller-provided window: when set, the core binds to that window instead of
        // creating its own - it does NOT call glfwInit/glfwCreateWindow, keeps no ownership and
        // never destroys it. The caller owns the window, must have initialized GLFW and created
        // it Vulkan-capable (GLFW_NO_API) before constructing the core; the size/title fields
        // above are ignored in this mode. The core still installs nothing on the window itself
        // (no GLFW callbacks), so any caller-side callbacks keep working.
        std::optional<GLFWwindow*> window = std::nullopt;
    };

    export struct core : utility::enable_stack_destruct {
        // creation options this core was built with (window size, vsync, msaa); the window is
        // created from them and the swap chain / MSAA targets honor them
        core_create_info create_options = {};

        VkInstance instance = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        // properties of the picked physical device (VkPhysicalDeviceProperties: limits such as
        // timestampPeriod, bufferImageGranularity, maxPushConstantsSize + the device name).
        // Filled in init_device_and_queue() from the capabilities query it already runs - no
        // second vkGetPhysicalDeviceProperties round trip.
        VkPhysicalDeviceProperties device_properties = {};
        uint32_t graphics_family_index = 0;
        uint32_t present_family_index = 0;
        VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
        void init_instance() noexcept;

        GLFWwindow* window = nullptr;
        void init_window(int width, int height, std::string_view window_name = "") noexcept;

        // ---- facade operations (keep raw Vulkan / GLFW calls out of the caller) ----
        void wait_idle() const noexcept;                              // vkDeviceWaitIdle
        void set_window_title(std::string_view title) const noexcept; // glfwSetWindowTitle

        VkSurfaceKHR surface = VK_NULL_HANDLE;
        void init_surface() noexcept;

        VkQueue graphics_queue = VK_NULL_HANDLE;
        VkQueue present_queue = VK_NULL_HANDLE;
        uint32_t graphics_queue_family = VK_QUEUE_FAMILY_IGNORED;
        void init_device_and_queue() noexcept;

        VkSwapchainKHR swap_chain = {};
        std::vector<VkImage> swap_chain_images = {};
        VkFormat swap_chain_image_format = {};
        VkExtent2D swap_chain_extent = {};
        // Whether the swapchain images were created with VK_IMAGE_USAGE_TRANSFER_SRC_BIT (i.e. the
        // surface supports it). The screenshot read-back copies from a swapchain image and is only
        // legal when this is true - see init_swap_chain().
        bool swapchain_transfer_src_supported = false;

        void init_swap_chain() noexcept;

        std::vector<VkImageView> swap_chain_image_views = {};

        void init_image_views() noexcept;

        // MSAA related
        VkSampleCountFlagBits msaa_samples = VK_SAMPLE_COUNT_1_BIT; // no MSAA by default
        std::vector<VkImage> color_images = {};                     // MSAA color buffer images
        std::vector<VkDeviceMemory> color_image_memories = {};
        std::vector<VkImageView> color_image_views = {}; // MSAA image views
        VkFormat color_format = VK_FORMAT_UNDEFINED;
        // HDR resolve targets (one per swapchain image): the MSAA scene pass resolves into
        // them (format hdr_format) and the post-process pass samples them
        std::vector<VkImage> hdr_images = {};
        std::vector<VkDeviceMemory> hdr_image_memories = {};
        std::vector<VkImageView> hdr_image_views = {};
        void create_hdr_resolve_resources();
        // create_hdr_resolve_resources() runs again on every swapchain recreation (it rebuilds the
        // HDR/LDR/bloom targets); its teardown must be pushed onto the cleanup stack only once, or
        // the stack grows one identical lambda per resize.
        bool resolve_cleanup_registered = false;
        // bloom targets: a 4-level chain (1/2, 1/4, 1/8, 1/16 of the swapchain extent, min 1x1),
        // one chain per swapchain image; the post pass prefilters into level 0, downsamples
        // through the levels and composites a weighted sum of all of them
        static constexpr uint32_t bloom_level_count = 4;
        std::array<std::vector<VkImage>, bloom_level_count> bloom_images = {};
        std::array<std::vector<VkDeviceMemory>, bloom_level_count> bloom_image_memories = {};
        std::array<std::vector<VkImageView>, bloom_level_count> bloom_image_views = {};
        // Display-referred (LDR) targets, one per swapchain image: with FXAA enabled the post
        // composite renders here instead of straight into the swapchain, FXAA reads it back and
        // writes the swapchain. hdr_format (R16F) even though the values are display range: FXAA
        // needs a *gamma-encoded* image to run its luma thresholds on, and a 16F target lets the
        // composite store that encoding itself (an sRGB attachment would decode it again on read,
        // and 8-bit would band).
        std::vector<VkImage> ldr_images = {};
        std::vector<VkDeviceMemory> ldr_image_memories = {};
        std::vector<VkImageView> ldr_image_views = {};

        // ---- G-buffer targets (see gbuffer_formats): one set per swapchain image, single-sampled,
        // written by the G-buffer pass and sampled by the deferred lighting / debug view. They are
        // created and destroyed with the HDR/LDR/bloom targets (create_hdr_resolve_resources +
        // recreate_swap_chain), so a resize rebuilds them in the same step.
        std::array<std::vector<VkImage>, gbuffer_target_count> gbuffer_images = {};
        std::array<std::vector<VkDeviceMemory>, gbuffer_target_count> gbuffer_image_memories = {};
        std::array<std::vector<VkImageView>, gbuffer_target_count> gbuffer_image_views = {};
        // The G-buffer pass needs its own depth image: the main depth image follows MSAA, and a
        // dynamic rendering instance requires every attachment to have the same sample count (a
        // 1x G-buffer over an 8x depth attachment is invalid, and a multisampled G-buffer is the
        // thing the deferred path exists to avoid). Single-sampled, sampled (the lighting pass
        // reads it), cleared by the G-buffer pass like the main depth.
        std::vector<VkImage> gbuffer_depth_images = {};
        std::vector<VkDeviceMemory> gbuffer_depth_image_memories = {};
        std::vector<VkImageView> gbuffer_depth_image_views = {};
        void create_msaa_image(
            uint32_t width,
            uint32_t height,
            VkFormat format,
            VkSampleCountFlagBits num_samples,
            VkImageTiling tiling,
            VkImageUsageFlags usage,
            VkMemoryPropertyFlags properties,
            VkImage& image,
            VkDeviceMemory& image_memory) const noexcept;

        VkFormat depth_format = {};
        std::vector<VkImage> depth_images = {};
        std::vector<VkDeviceMemory> depth_image_memories = {};
        std::vector<VkImageView> depth_image_views = {};
        void create_depth_image(VkImage& image, VkDeviceMemory& image_memory, VkImageView& image_view) const noexcept;
        void create_depth_resources() noexcept;

        void create_color_resources();

        VkCommandPool command_pool = {};
        std::vector<VkCommandBuffer> command_buffers = {};
        void create_command_pool() noexcept;

        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        void create_descriptor_pool() noexcept;

        // shared scene layouts (see the scene_texture_capacity / scene_push_constant_size docs above);
        // all pipelines are created against scene_pipeline_layout, so one descriptor set works for all
        VkDescriptorSetLayout scene_descriptor_set_layout = VK_NULL_HANDLE;
        VkPipelineLayout scene_pipeline_layout = VK_NULL_HANDLE;
        void init_scene_layouts() noexcept;

        vma_allocator vma = {};

        // ---- frame synchronization (timeline semaphores; see create_sync_objects) ----
        // vkAcquireNextImageKHR and vkQueuePresentKHR both require BINARY semaphores:
        //   - image_available_semaphores: binary, per frame slot (acquire signal)
        //   - present_ready_semaphores: binary, ONE PER SWAPCHAIN IMAGE — present may run on a
        //     separate queue, so a per-slot binary could be re-signaled before the previous
        //     present consumed it; a swapchain image is only re-acquired after its present
        //     finished, which keeps this per-image gate safe across queues
        //   - frame_done_semaphores / frame_done_values: TIMELINE, per frame slot, counting
        //     submissions — submit() signals it (GPU completion), wait_frame_slot() is the host
        //     pacing wait that used to be vkWaitForFences
        std::vector<VkSemaphore> image_available_semaphores = {}; // binary, per frame slot (acquire)
        std::vector<VkSemaphore> present_ready_semaphores = {};   // binary, per swapchain image (present wait)
        std::vector<VkSemaphore> frame_done_semaphores = {};      // timeline, per frame slot
        std::vector<uint64_t> frame_done_values = {};             // last signaled value per slot (host bookkeeping)

        size_t current_frame = 0;

        void to_next_frame() noexcept;
        void wait_frame_slot(uint32_t slot) const; // host wait until this slot's last submission completed

        static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

        void create_sync_objects();

        // ---- GPU pass timing (VK_QUERY_TYPE_TIMESTAMP) ----
        // A timestamp pool with one contiguous range of gpu_timing_mark_capacity queries per frame
        // slot: the frame records vkCmdResetQueryPool + one vkCmdWriteTimestamp per pass boundary
        // and the reader converts consecutive marks into milliseconds after the slot's submission
        // completed (see mark_gpu_timing / read_gpu_timings). Timestamps need no feature bit, but a
        // queue family that cannot write them reports timestampValidBits == 0, and the tick length
        // comes from the device limits - either missing means gpu_timing_supported stays false and
        // every timing call is a no-op, so callers do not have to check the device themselves.
        VkQueryPool timestamp_query_pool = VK_NULL_HANDLE;
        float timestamp_period_ns = 0.0f;  // ns per tick (VkPhysicalDeviceLimits::timestampPeriod)
        uint32_t timestamp_valid_bits = 0; // graphics family counter width (0 = cannot timestamp)
        bool gpu_timing_supported = false;
        // marks the CURRENT recording of each slot has written (reset by begin_gpu_timing). Also
        // read back as "how many queries to fetch" for the submission that just completed, because
        // a slot is only read after it was paced and before it is recorded again.
        std::array<uint32_t, MAX_FRAMES_IN_FLIGHT> gpu_timing_marks = {};
        // frame_done value each slot's timings were last read for: a slot is read at most once per
        // submission, so a frame that hits an early return cannot fetch the same results twice
        std::array<uint64_t, MAX_FRAMES_IN_FLIGHT> gpu_timing_read_value = {};
        void create_timestamp_query_pool() noexcept;

        /**
         * @ingroup vulkan_core
         * @brief open this frame's timing range: reset the slot's queries and forget the previous
         *        frame's marks
         * @param command_buffer the frame's command buffer (the reset is recorded on the GPU
         *        timeline, which keeps it off the host/GPU race the pool would otherwise have)
         * @param slot the frame slot being recorded
         * @note call once per frame, before any mark and outside a dynamic rendering instance;
         *       a no-op when the device cannot timestamp
         */
        void begin_gpu_timing(VkCommandBuffer command_buffer, uint32_t slot) noexcept;

        /**
         * @ingroup vulkan_core
         * @brief write one timing mark into this frame's range
         * @param command_buffer the frame's command buffer
         * @param slot the frame slot being recorded
         * @param stage pipeline stage the mark resolves at: callers use TOP_OF_PIPE for the first
         *        mark of the frame and BOTTOM_OF_PIPE for every pass boundary, so mark i + 1 minus
         *        mark i is exactly how long pass i took
         * @note a no-op when the device cannot timestamp or the frame already wrote
         *       gpu_timing_mark_capacity marks
         */
        void mark_gpu_timing(VkCommandBuffer command_buffer, uint32_t slot, VkPipelineStageFlagBits stage) noexcept;

        /**
         * @ingroup vulkan_core
         * @brief convert a completed submission's marks into milliseconds
         * @param slot the frame slot to read (its last submission must have completed - pace the
         *        slot first, see wait_frame_slot)
         * @return the durations between consecutive marks, or a zero mark_count when timings are
         *         unavailable, the slot never submitted, or this submission was already read
         * @note never waits on the GPU and never blocks: vkGetQueryPoolResults is called without
         *       VK_QUERY_RESULT_WAIT_BIT, and an unavailable result reports "no measurement"
         *       instead of stalling
         */
        gpu_timing_result read_gpu_timings(uint32_t slot);

        /** @brief whether this device can measure GPU pass timings (see begin_gpu_timing) */
        [[nodiscard]] bool gpu_timing_available() const noexcept {
            return this->gpu_timing_supported;
        }

        core();
        explicit core(core_create_info const& options);
        ~core();

        vk_command_buffer make_command_buffer() const;
        /** @brief allocate a SECONDARY command buffer (recorded inside a dynamic rendering
         *         instance, executed there via vkCmdExecuteCommands) */
        vk_command_buffer make_secondary_command_buffer() const;
        /** @brief like make_secondary_command_buffer() but allocated from @p pool (a per-thread
         *         pool from make_command_pool(); the RAII wrapper frees into that same pool) */
        vk_command_buffer make_secondary_command_buffer(VkCommandPool pool) const;
        /**
         * @brief create an extra graphics command pool (RESET flag set, graphics queue family)
         *        whose lifetime is tied to this core (destroyed by the registered cleanup).
         *        Parallel recording needs one pool PER RECORDING THREAD - a single pool's
         *        command buffers must not be begun concurrently on different threads.
         * @note not const: registers the pool's destruction on this core (like create_command_pool)
         */
        VkCommandPool make_command_pool();
        vk_descriptor_set make_descriptor_set(VkDescriptorSetLayout layout) const;

        std::optional<vk_shader_module> make_shader_module(std::span<unsigned char> shader) const noexcept;

        /**
         * @ingroup vulkan_core
         * @brief create an image view covering the whole image (all mip levels and layers)
         * @param image the image to view
         * @param format the view format
         * @param type view type (VK_IMAGE_VIEW_TYPE_2D / VK_IMAGE_VIEW_TYPE_CUBE ...)
         * @return raii vk_image_view owning the created view
         */
        vk_image_view make_image_view(VkImage image, VkFormat format, VkImageViewType type) const;

        /**
         * @ingroup vulkan_core
         * @brief create a 2D depth image view (DEPTH aspect) over the whole image
         * @param image the image to view
         * @param format the view format (a depth format)
         * @return raii vk_image_view owning the created view
         * @note the regular make_image_view uses the COLOR aspect; depth images (e.g. the shadow
         *       map) need the DEPTH aspect to be sampled as depth
         */
        vk_image_view make_depth_image_view(VkImage image, VkFormat format) const;

        /**
         * @ingroup vulkan_core
         * @brief create a linear/min-linear sampler with the given wrap mode
         * @param address_mode wrap mode applied to all three axes
         * @param max_lod maximum mip level the sampler may access
         * @return raii vk_sampler owning the created sampler
         */
        vk_sampler make_sampler(VkSamplerAddressMode address_mode, float max_lod) const;

        /**
         * @ingroup vulkan_core
         * @brief create the shadow map sampling sampler (NEAREST + clamp-to-edge)
         * @return raii vk_sampler owning the created sampler
         * @note pbr.frag does manual percentage-closer filtering: it fetches the stored depth
         *       with this NEAREST sampler at a few neighbor texels and averages the comparisons,
         *       so no depth-comparison/linear-filter format feature is required
         */
        vk_sampler make_shadow_sampler() const;

        /**
         * @ingroup vulkan_core
         * @brief create a depth-only graphics pipeline (no color attachment, single sample),
         *        used by the shadow pass to render depth into the shadow map
         * @param vertex_shader_code raw SPIR-V binary of the vertex shader
         * @param fragment_shader_code raw SPIR-V binary of the fragment shader
         * @param depth_format depth attachment format (dynamic rendering only)
         * @param depth_bias_constant_factor constant rasterization depth bias added to depth
         * @param depth_bias_slope_factor slope-scaled depth bias (removes shadow acne on angled surfaces)
         * @param depth_bias_clamp maximum depth bias magnitude (0 = no clamp)
         * @return vk_pipeline on success, error message on failure
         */
        std::expected<vk_pipeline, std::string_view> make_depth_pipeline(
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code,
            VkFormat depth_format,
            float depth_bias_constant_factor = 0.0f,
            float depth_bias_slope_factor = 0.0f,
            float depth_bias_clamp = 0.0f) const;

        void recreate_swap_chain();
        // one-time log for the "recreation deferred because the window has no drawable size" case
        // (see recreate_swap_chain); reset as soon as a recreation actually runs
        bool zero_extent_recreation_logged = false;

        /**
         * @ingroup vulkan_core
         * @brief submit the recorded command buffer for the current frame slot
         * @param command_buffer the command buffer to submit
         * @param image_index the acquired swapchain image index (unused: present waits the same
         *        per-slot timeline that this submit signals)
         * @return the result of vkQueueSubmit
         */
        VkResult submit(VkCommandBuffer command_buffer, uint32_t image_index);

        /**
         * @ingroup vulkan_core
         * @brief present the rendered swapchain image; waits on the current frame slot's
         *        timeline (signaled by submit())
         * @param image_index the swapchain image index to present
         * @return the result of vkQueuePresentKHR
         */
        VkResult present(uint32_t image_index) const;

        std::expected<vk_pipeline, std::string_view> make_pipeline(
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code,
            bool depth_test_enabled = true) const;

        /**
         * @ingroup vulkan_core
         * @brief create the G-buffer pipeline: the shared scene layout, the three gbuffer_formats
         *        color targets and a single-sampled depth attachment
         * @param vertex_shader_code raw SPIR-V of the vertex stage (pbr.vert: instancing / skinning /
         *        morphing are identical to the forward path)
         * @param fragment_shader_code raw SPIR-V of the fragment stage (gbuffer.frag: writes the
         *        three targets and shades nothing)
         * @return vk_pipeline on success, error message on failure
         * @note single-sampled on purpose (a G-buffer cannot be multisampled without per-sample
         *       shading), so this pipeline may NOT be recorded into the forward pass's instance:
         *       its attachments are the core::gbuffer_* targets and the pass that owns them
         */
        std::expected<vk_pipeline, std::string_view> make_gbuffer_pipeline(
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code) const;
    };
} // namespace vulkan