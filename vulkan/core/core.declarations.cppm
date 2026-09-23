// ============================================================================
// module: vulkan.core:declarations  - the INTERFACE PARTITION of vulkan.core
//
// The declarations live here so the implementation halves can be partitions: a partition does not
// see the primary interface on its own, so every implementation partition imports THIS one.
// The primary (vulkan/core/core.cppm) is the three lines that re-export it, and `import vulkan.core;`
// is unaffected anywhere - which is the whole point of paying for the split.
//
// MEASURED BEFORE COMMITTING TO IT: under this project's -Werror, clang 22 REJECTS the textbook
// pattern (the primary importing its implementation partitions) with
// -Wimport-implementation-partition-unit-in-interface-unit. The viable variant, verified on a scratch
// project with these exact flags, is this one: the primary imports ONLY interface partitions, and the
// implementation partitions are listed in CMake (which compiles every unit) without being imported by
// the interface. Their definitions are still archived and still link.
// ============================================================================
// ============================================================================
// module: vulkan.core
// module version: 0.24.0  (independent of the app version in CMakeLists project(VERSION))
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
#include <memory>
#include <vulkan/vulkan.h>

export module vulkan.core:declarations;
import utility;
export import vstd;
export import vulkan.core.handles;
export import :vma;
export import :descriptor_heap;
export import :vma_handles;

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
     * @brief agreed flat layout of the scene block, shared by every pipeline (see shaders/pbr.frag):
     *        set 0 binding 0 = CameraUBO (uniform buffer; one per frame slot, each slot's block
     *              points at its own - static, no per-frame descriptor writes),
     *              binding 1 = sampler2D textures[] (runtime array, partially bound + non-uniform index),
     *              binding 2/3/4 = prefiltered env / irradiance / BRDF LUT (combined image samplers),
     *              binding 5 = Material materials[] (storage buffer: per-material texture indices + factors),
     *              binding 6 = mat4 instance transforms[] (storage buffer, per-instance world matrices),
     *              binding 7 = LightUBO (uniform buffer: directional light view-proj + direction),
     *              binding 8 = shadow map (sampler2DArrayShadow, one array layer per cascade; LINEAR
     *              min/mag with compareEnable = VK_TRUE, so the hardware does the 2x2 comparison and
     *              shading.glsl's calc_shadow_cascade() averages a 3x3 grid of those taps),
     *              binding 9 = mat4 skin matrices[] (storage buffer: identity block + per-skin joints),
     *              binding 10 = float morph data[] (storage buffer: per-primitive morph deltas + weights),
     *              binding 11/12 = uint cluster light counts[] / uint cluster light indices[] (the
     *              clustered-culling result: written by the cluster compute pass, read by the
     *              fragment stage),
     *              binding 13 = mat4 previous world matrices[] (one per motion slot; the vertex stage
     *              reads its own entry so the fragment stage can build TAA's motion vector for a
     *              MOVING object, not only for camera motion)
     * @note hardcoded instead of parsed from SPIR-V: the indexed layout is flat, so pipelines skip
     *       descriptor / push constant parsing and every stage shares this one binding contract, which the
     *       frame's heap carries
     */
    export constexpr uint32_t scene_texture_capacity = 128;
    // material_push_constants: 6 uints + aligned mat4 = 96 bytes, see vulkan/scene_tree/scene_tree.cppm
    export constexpr uint32_t scene_push_constant_size = 96;
    /**
     * @ingroup vulkan_core
     * @brief offset of the SECOND push constant range of the shared scene layout, right after the
     *        per-primitive material block
     * @note currently one uint: the cascade index the shadow pass is rendering. It is a separate
     *       range rather than extra fields in the material block because that block is exactly 96
     *       bytes and the two together would exceed the 128 bytes the spec guarantees every
     *       implementation provides (the engine does not rely on a vendor's larger limit).
     */
    export constexpr uint32_t scene_cascade_push_offset = scene_push_constant_size;
    export constexpr uint32_t scene_cascade_push_size = sizeof(uint32_t);

    /**
     * @brief format of the HDR scene target the deferred lighting stage renders into and the post-process
     *        pass samples: the scene color target uses it, and each swapchain image owns one
     *        single-sample resolve target in it (see core::create_render_targets)
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
     *       without per-sample shading, which is the trade that makes TAA the anti-aliasing
     *       (the anti-aliasing story is TAA/FXAA on the lit image instead).
     */
    export constexpr std::array<VkFormat, gbuffer_target_count> gbuffer_formats = {
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R8G8B8A8_UNORM,
    };

    /**
     * @ingroup vulkan_core
     * @brief motion-vector format: the fourth G-buffer target, written by the G-buffer pass and read
     *        by TAA (RG16F because a motion vector is a signed sub-pixel quantity in UV space and
     *        8-bit would quantize it to ~1/255 of the screen - coarser than the jitter TAA exists to
     *        resolve)
     */
    export constexpr VkFormat gbuffer_velocity_format = VK_FORMAT_R16G16_SFLOAT;

    /**
     * @ingroup vulkan_core
     * @brief color attachments the G-buffer pass declares: the three surface targets above, the
     *        motion-vector target, and the scene-color target it ADDS the emissive term into
     * @note emissive is lighting-independent, so it does not belong to the deferred lighting stage -
     *       and it needs the material's emissive texture and the fragment's UVs, neither of which the
     *       G-buffer stores. Adding it in the base pass is what commercial deferred renderers do (the
     *       G-buffer pass writes the surface and adds emissive to the scene color), and it is why that
     *       last attachment is blended ONE/ONE while the surface and velocity targets are overwritten.
     *       It is CLEARed to zero by the instance, and the deferred lighting stage adds the lighting
     *       (and the sky, where no geometry wrote depth) on top.
     */
    export constexpr uint32_t gbuffer_pass_attachment_count = gbuffer_target_count + 2; // + velocity + scene color

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
        // vsync: true (default) prefers VK_PRESENT_MODE_FIFO_LATEST_READY and falls back to FIFO - the
        // frame goes out at the display's rate and the acquire blocks instead of spinning, which is what
        // keeps an idle window off the CPU. false prefers VK_PRESENT_MODE_MAILBOX_KHR, the uncapped path
        // a throughput measurement needs (see the [render] vsync note in config.example.toml).
        bool vsync = true;
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

    /**
     * @brief the device, its allocator and every resource created on it - the renderer's lifetime ROOT
     *
     * `std::enable_shared_from_this` is here because the core is the one object that can be SHARED: a caller may
     * hold it (see `runtime`'s constructor that takes a `shared_ptr<core>`), and the resources it owns are
     * created through it, so a creation site can hand out a reference-counted handle to the device it is
     * building on. It does NOT mean a `core` must be heap-allocated: nothing calls `shared_from_this()` yet.
     */
    export struct core : utility::enable_stack_destruct, std::enable_shared_from_this<core> {
        // creation options this core was built with (window size, vsync); the window and the
        // swap chain honor them
        core_create_info create_options = {};

        VkInstance instance = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        // properties of the picked physical device (VkPhysicalDeviceProperties: limits such as
        // timestampPeriod, bufferImageGranularity, maxPushConstantsSize + the device name).
        // Filled in init_device_and_queue() from the capabilities query it already runs - no
        // second vkGetPhysicalDeviceProperties round trip.
        VkPhysicalDeviceProperties device_properties = {};
        // Ray tracing is optional and comes from the device, not from a build option: when this is
        // false the extensions were not enabled (see device_capabilities) and every ray-traced path
        // skips itself. The properties carry the two limits the AS builder needs - the scratch
        // buffer's required address alignment and the per-level instance/geometry caps - and are a
        // plain data holder like device_properties above (assigned from the query, never passed to
        // Vulkan), so they are zero-initialized rather than carrying a fixed sType.
        bool ray_query_available = false;
        /// @brief whether the RT pipeline (traceRaysEXT + a shader binding table) can be used at all:
        ///        the extension set AND the feature, enabled together at device creation
        bool ray_tracing_pipeline_available = false;
        /// @brief whether VK_EXT_opacity_micromap is enabled: the micromap state that makes an
        ///        alphaMode MASK surface opaque/transparent per microtriangle. Needs the RT pipeline.
        bool opacity_micromap_available = false;
        VkPhysicalDeviceAccelerationStructurePropertiesKHR acceleration_structure_properties = {};
        /// @brief the micromap subdivision levels the device allows (queried with its feature)
        VkPhysicalDeviceOpacityMicromapPropertiesEXT opacity_micromap_properties = {};
        /// @brief the SBT numbers (handle size, region base alignment, handle alignment, recursion depth)
        ///        a ray-tracing pipeline's shader binding table has to be built against
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_pipeline_properties = {};
        /**
         * @brief VK_EXT_mesh_shader: whether the device runs a MESH pipeline at all, and the dispatch command
         *        that goes with it (docs/mesh_shaders.md)
         * @note the command is fetched through `vkGetDeviceProcAddr` because `vkCmdDrawMeshTasksEXT` is an
         *       EXTENSION command that the loader's import library does not export - calling it directly is a
         *       link error (`undefined symbol: vkCmdDrawMeshTasksEXT`), which is how this was learned. It is
         *       resolved once here rather than per use, and it is null on a device without the extension.
         * @note this is the FEATURE half only (extension + `meshShader`); whether a pass may use it also depends
         *       on the push budget the stage block needs, and that decision is the RUNTIME's (see
         *       runtime::mesh_shaders), because a pass is not the only thing that could want it.
         */
        bool mesh_shader_available = false;
        PFN_vkCmdDrawMeshTasksEXT mesh_dispatch = nullptr;
        uint32_t graphics_family_index = 0;
        uint32_t present_family_index = 0;
        VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;

        GLFWwindow* window = nullptr;

        // ---- facade operations (keep raw Vulkan / GLFW calls out of the caller) ----
        void wait_idle() const noexcept;                              // vkDeviceWaitIdle
        void set_window_title(std::string_view title) const noexcept; // glfwSetWindowTitle

        VkSurfaceKHR surface = VK_NULL_HANDLE;

        VkQueue graphics_queue = VK_NULL_HANDLE;
        VkQueue present_queue = VK_NULL_HANDLE;
        uint32_t graphics_queue_family = VK_QUEUE_FAMILY_IGNORED;

        VkSwapchainKHR swap_chain = {};
        std::vector<VkImage> swap_chain_images = {};
        VkFormat swap_chain_image_format = {};
        VkExtent2D swap_chain_extent = {};
        // Whether the swapchain images were created with VK_IMAGE_USAGE_TRANSFER_SRC_BIT (i.e. the
        // surface supports it). The screenshot read-back copies from a swapchain image and is only
        // legal when this is true - see init_swap_chain().
        bool swapchain_transfer_src_supported = false;

        // MSAA used to live here. It is gone with the forward path that was its only consumer: a
        // G-buffer cannot be multisampled without per-sample shading, so the scene has always
        // rendered at 1x, and with no second path there is nothing left for the setting to select.
        // The anti-aliasing story is TAA (and FXAA) on the shaded image instead.
        std::vector<VkImageView> swap_chain_image_views = {};

        VkFormat color_format = VK_FORMAT_UNDEFINED;
        // HDR scene targets (one per swapchain image, format hdr_format): the lighting stage (or the
        // TAA resolve, when TAA is on) writes them, and the post-process pass samples them
        std::vector<VkImage> hdr_images = {};
        std::vector<VkDeviceMemory> hdr_image_memories = {};
        std::vector<VkImageView> hdr_image_views = {};
        // create_render_targets() runs again on every swapchain recreation (it rebuilds the
        // HDR/LDR/bloom/G-buffer targets); its teardown must be pushed onto the cleanup stack only
        // once, or the stack grows one identical lambda per resize.
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
        // created and destroyed with the HDR/LDR/bloom targets (create_render_targets +
        // recreate_swap_chain), so a resize rebuilds them in the same step.
        std::array<std::vector<VkImage>, gbuffer_target_count> gbuffer_images = {};
        std::array<std::vector<VkDeviceMemory>, gbuffer_target_count> gbuffer_image_memories = {};
        std::array<std::vector<VkImageView>, gbuffer_target_count> gbuffer_image_views = {};
        // The G-buffer pass has its own depth image rather than sharing the main one: a dynamic
        // rendering instance requires every attachment to have the same sample count, and keeping
        // them separate lets the G-buffer depth be SAMPLED later while the main one is never read.
        // Single-sampled, sampled (the lighting pass
        // reads it), cleared by the G-buffer pass like the main depth.
        std::vector<VkImage> gbuffer_depth_images = {};
        std::vector<VkDeviceMemory> gbuffer_depth_image_memories = {};
        std::vector<VkImageView> gbuffer_depth_image_views = {};
        // Motion vectors (gbuffer_velocity_format), one per swapchain image: written by the G-buffer
        // pass, read by the TAA resolve.
        std::vector<VkImage> velocity_images = {};
        std::vector<VkDeviceMemory> velocity_image_memories = {};
        std::vector<VkImageView> velocity_image_views = {};

        // ---- the stochastic punctual lighting chain's images ----
        // Its raw trace, its accumulation, its history and its spatial filter's output (four half-resolution
        // families) stood here. They went with the chain. The half-resolution pair below belongs to the
        // stochastic punctual lighting chain, which is a different feature that happens to share the size.
        // The stochastic PUNCTUAL LIGHTING chain's first image (see docs/megalights.md): the raw estimate
        // the trace writes, at half resolution - STORAGE for the compute pass that
        // writes it and SAMPLED for the lighting stage that adds it. One per swapchain image, because what
        // it holds depends on the frame's jittered camera.
        std::vector<VkImage> ml_images = {};
        std::vector<VkDeviceMemory> ml_image_memories = {};
        std::vector<VkImageView> ml_image_views = {};
        // ... and the temporal resolve's two, the same half resolution: the ACCUMULATION it writes (what the
        // lighting stage samples) and the history that becomes next frame's input - the latter written only by
        // a copy, so TRANSFER_DST plus SAMPLED and nothing else, exactly like the GI history beside it.
        std::vector<VkImage> ml_resolve_images = {};
        std::vector<VkDeviceMemory> ml_resolve_image_memories = {};
        std::vector<VkImageView> ml_resolve_image_views = {};
        std::vector<VkImage> ml_history_images = {};
        std::vector<VkDeviceMemory> ml_history_image_memories = {};
        std::vector<VkImageView> ml_history_image_views = {};
        // per-cell surface-offset image stood here, with the sampler that read them. The cache was the traced
        // chain's answer for the hits the screen cannot resolve; it is gone.
        // The furnace verification mode's constant environment: one texel per face, all six faces at the
        // mode's level. One element vectors rather than a scalar handle so the teardown paths that already
        // know how to destroy a target set can be reused unchanged. Its CONTENTS come from a clear, which
        // together with the binding that points the IBL at it is the next slice; until then nothing samples
        // it, which is what keeps this addition invisible.
        std::vector<VkImage> furnace_cube_images = {};
        std::vector<VkDeviceMemory> furnace_cube_memories = {};
        std::vector<VkImageView> furnace_cube_views = {};

        // ---- ray-traced sun visibility (see the shaders/rt_shadow.* pipeline stages) ----
        // FULL resolution, one per FRAME SLOT rather than per swapchain image: it is written and read
        // within one frame, and BOTH ends live in the frame's scene block, which is per slot. A
        // per-image image would have to be paired there with a per-slot top level structure, and
        // the same image can be recorded on either slot - so the two are different lifetimes and mixing
        // them would be wrong on exactly the frames where they disagree.
        std::vector<VkImage> rt_shadow_images = {};
        std::vector<VkDeviceMemory> rt_shadow_image_memories = {};
        std::vector<VkImageView> rt_shadow_image_views = {};

        // ---- temporal anti-aliasing (see runtime::set_taa) ----
        // The scene color TAA resolves FROM, one per swapchain image: when TAA is on, the geometry
        // and lighting stages write this image instead of the HDR target, and the TAA resolve blends
        // it with the history into the HDR target - which keeps the whole post chain (bloom,
        // composite, FXAA) reading exactly what it read before TAA existed.
        std::vector<VkImage> scene_color_images = {};
        std::vector<VkDeviceMemory> scene_color_image_memories = {};
        std::vector<VkImageView> scene_color_image_views = {};
        // The previous RESOLVED frame, one per swapchain image, read by the TAA resolve as history.
        // It is a separate image rather than a copy of the HDR target because a pass cannot sample the
        // image it renders into: the TAA resolve writes the HDR target (for the post chain) and the
        // runtime copies that into this image afterwards, which is one vkCmdCopyImage per frame - no
        // ping-pong, no per-frame descriptor rewrites.
        // TRANSFER_DST | SAMPLED: it is only ever written by the copy and read by the resolve.
        std::vector<VkImage> taa_history_images = {};
        std::vector<VkDeviceMemory> taa_history_image_memories = {};
        std::vector<VkImageView> taa_history_image_views = {};
        /**
         * @brief create a single-sampled device-local image with its memory, and return both
         * @param width / @param height the extent in texels
         * @param format the image format
         * @param tiling OPTIMAL or LINEAR (staging images that are mapped on the host)
         * @param usage the usage flags the image is created with
         * @param properties the memory type the image is bound to
         * @note every target the engine creates is single-sampled: the only multisampled images it
         *       ever had were the forward path's, and that path is gone. The sample count is fixed
         *       rather than a parameter so there is one less thing a caller can get wrong.
         */

        /**
         * @ingroup vulkan_core
         * @brief create a single-sampled 3D target image (the same allocation path as the 2D one)
         * @param width / @p height / @p depth the three extents, in texels
         * @note its own entry point rather than a defaulted fourth parameter on create_target_image:
         *       the two differ in exactly one field of VkImageCreateInfo (imageType), and a caller that
         *       reads `create_target_image_3d(w, h, d, ...)` cannot pass a depth of 1 by accident and
         *       then sample the result as a volume.
         */
        /**
         * @ingroup vulkan_core
         * @brief create a single-sampled CUBE target: a six-layer 2D array with CUBE_COMPATIBLE set
         * @param size the edge length of one face, in texels (all six faces are the same size)
         * @note its own entry point rather than a generalised array helper, for the same reason
         *       create_target_image_3d has one: a cube is six layers AND the compatibility flag, and a caller
         *       that got one of those wrong would have an image the sampler refuses.
         */

        VkFormat depth_format = {};
        std::vector<VkImage> depth_images = {};
        std::vector<VkDeviceMemory> depth_image_memories = {};
        std::vector<VkImageView> depth_image_views = {};

        VkCommandPool command_pool = {};

        /** @brief create the shared samplers above: device-level, reference-counted by nobody, destroyed with core */

        // ---- the SHARED samplers, created once with the device (see create_samplers) ----
        //
        // THEY LIVE HERE because a sampler is a device-level object with no per-frame state and no owner among the
        // passes: a pass DECLARES one by hint (see render_resource::shared::sampler_set) and the renderer hands over
        // the handles, so the object's owner has to be the device root - the same argument every image in this class
        // answers. Before this they were scattered across the runtime's scene setup, a pipeline builder and two
        // ensure_* functions - the "naming accident" this comment records.
        //
        // `env_sampler` is deliberately NOT here: its max_lod is the app's environment mip count, not a device fact.
        vk_sampler texture_sampler = {};      // the bindless texture array: REPEAT, and all its mip levels
        vk_sampler gbuffer_sampler = {};      // the G-buffer's stored surface: NEAREST, clamp (exact texel centres)
        vk_sampler taa_sampler = {};          // the TAA resolve: LINEAR magnification, NEAREST minification
        vk_sampler post_sampler = {};         // the post chain and the FXAA filter: LINEAR, clamp
        vk_sampler post_nearest_sampler = {}; // the composite's GI upsample: NEAREST, clamp (depths are not colours)
        vk_sampler shadow_sampler = {};       // the cascaded map: depth-compare + LINEAR (hardware PCF), clamp
        /**
         * @brief the CREATE INFO of @ref texture_sampler, kept because the descriptor heap needs it as an
         *        EMBEDDED SAMPLER: a heap binding for a combined image sampler takes its sampler from a
         *        VkSamplerCreateInfo (the driver creates one), not from a VkSampler, and it has to be the SAME
         *        sampler the descriptor-set path uses - a different max_lod alone changes which mip is read.
         * @note filled where the sampler is created (create_samplers), so the two cannot drift.
         */
        VkSamplerCreateInfo texture_sampler_info = {};
        /**
         * @brief the CREATE INFO of every shared sampler, in the order shaders/heap_slots.glsl names them, because
         *        a heap descriptor for a sampler IS a create info - and the heap is created LATER in the
         *        constructor than these samplers are (create_samplers runs first), so the infos have to be kept
         *        here to be written onto the sampler grid once it exists.
         * @note index 0 texture (linear, repeat mips), 1 post (linear, clamp, one mip), 2 gbuffer (nearest, clamp),
         *       3 post-nearest (nearest, clamp), 4 taa (linear mag / nearest min, clamp), 5 shadow (depth compare).
         */
        std::array<VkSamplerCreateInfo, 6> shared_sampler_infos = {};

        vma_allocator vma = {};
        /// the device-wide descriptor heap (see vulkan/core/descriptor_heap): one resource heap and one
        /// sampler heap, created right after the device so every pass can be written against it
        descriptor_heap descriptor_heaps = {};
        /// the heap's limits, copied out of the capability query (see core::init_device_and_queue) because the
        /// heap itself is created in the constructor, after vma.init() - the capabilities are not in scope there
        heap_limits descriptor_heap_limits = {};
        /**
         * @brief the reserved heap blocks, in bytes, or VK_WHOLE_SIZE when the heap is not in use
         *
         * @note THE LAYOUT IS OWNED HERE, and that is not tidiness: the heap's contents are written by the RUNTIME
         *       (it is what knows the textures and the material table) while the SLOT NUMBERS that point the
         *       shaders at them are named in `shaders/heap_slots.glsl`. Both sides have to use the same number, and
         *       a mismatch - a write at one offset, a shader reading another - is invisible to validation and shows
         *       up only as a wrong picture. So the blocks are reserved once, here, and published; the startup log
         *       prints the whole table, which is how a drift between the two sides is seen.
         */
        VkDeviceSize heap_texture_array_offset = VK_WHOLE_SIZE;
        VkDeviceSize heap_material_table_offset = VK_WHOLE_SIZE;

        /**
         * @brief THE SLOT GRID: where every descriptor this renderer uses lives, in SLOTS of @ref heap_slot_stride
         *        bytes, and why an index is a slot number rather than a byte offset
         *
         * @note A HEAP-NATIVE SHADER CAN ONLY NAME `array[index]`, which GL_EXT_descriptor_heap resolves to
         *       `heapBase + index * stride`. Every such array aliases the heap FROM OFFSET 0, so an index IS a
         *       byte offset divided by that array's stride - and the only way a shader can find its data at a
         *       compile-time-known index is for the data to sit on a grid whose base is FIXED. Both heaps start
         *       their grid at 1 MiB (past any reserved window a driver reports here - 94 KiB of resource heap,
         *       64 KiB of sampler heap - and the same number on every device), the stride is 64 B, and every
         *       array below is a CONTIGUOUS RUN OF SLOTS.
         *
         * @note @ref heap_slot_stride IS AN OVERRIDE, not the device's stride: the resources are mixed kinds
         *       (16 B buffers, 32 B images at this device), and one power-of-two stride covering all of them is
         *       what makes a single grid possible - the shader declares `descriptor_stride = 64` and the host
         *       writes 64 B apart. A device whose largest descriptor exceeded it would make the grid ambiguous,
         *       so that is checked once at startup and the grid is REFUSED rather than wrong.
         *
         * @note THESE NUMBERS ARE MIRRORED IN shaders/heap_slots.glsl and they have to be, because the shader
         *       bakes its base indices. A drift is invisible to validation and shows up as a wrong picture, so
         *       the constructor logs this table (see the `descriptor heap: slot grid` line).
         */
        static constexpr VkDeviceSize heap_slot_stride = 64;
        /// THE GRID'S BYTE OFFSET FOR A SLOT (see core::heap_slots and docs/descriptor_heap_migration.md): every
        /// descriptor is 64 B from the next, so a write HERE and a heap-native shader's `array[slot]` with
        /// `descriptor_stride = 64` are the same address by construction. There is no second stride to disagree
        /// with - which is exactly what the older per-slot block could not promise, because it mixed the device's
        /// 16 B buffer stride with its 32 B image stride and put each binding at its own offset.
        /// @note A SLOT NUMBER IS ALREADY ABSOLUTE: `core::heap_slots::x` includes `heap_slot_base`, so this is a
        ///       multiply and nothing else. The first version added `heap_grid_offset` as well and doubled the
        ///       1 MiB base - every write landed past the heap and was refused, which the heap's own bounds check
        ///       reported (`... did not fit at offset 2130432`).
        /// @note THIS IS THE ONE COPY: it used to be duplicated in runtime:constructor, runtime:frames and
        ///       runtime.cpp, which is what a helper defined in terms of the stride below invites. It belongs
        ///       beside the constant it multiplies, and it is exported so those units can drop their own.
        [[nodiscard]] static constexpr VkDeviceSize heap_slot_offset(uint32_t const slot) noexcept {
            return static_cast<VkDeviceSize>(slot) * heap_slot_stride;
        }

        static constexpr uint32_t heap_slot_base = 16384;  // 1 MiB / 64 B: the grid's slot 0
        static constexpr uint32_t heap_slot_count = 1024;  // 703 slots are in use, the rest is room to grow
        static constexpr uint32_t heap_image_capacity = 8; // per-swapchain-image arrays (3-4 images in practice)
        /// The SAMPLER heap is a second grid with its own base, and it cannot share the resource one: the API caps
        /// the sampler heap at 128 KiB, so 64 KiB - the reserved window the embedded-sampler path requires - is the
        /// largest base it can have. Its stride is the device's own sampler descriptor size (32 B here).
        static constexpr uint32_t heap_sampler_base = 2048; // 64 KiB / 32 B
        static constexpr VkDeviceSize heap_sampler_stride = 32;
        struct heap_slots {
            static constexpr uint32_t textures = heap_slot_base + 0u;    // binding 1, the bindless array
            static constexpr uint32_t materials = heap_slot_base + 512u; // binding 5
            /**
             * @brief the top level structure, binding 16, as a TWO-SLOT array - and the reason it is not beside the
             *        per-frame buffers above
             *
             * @note THE TLAS IS REBUILT EVERY FRAME (see runtime::write_rt_structure_binding, which is called per
             *       frame slot), so a single slot would hold one frame's structure while the other frame is still
             *       in flight - the same hazard the camera and light UBOs have, and they are two slots each for it.
             *       This was first laid out as ONE slot beside the per-frame buffers, which is the kind of mistake
             *       a picture cannot show until the shaders read the heap: the fix is cheap now and would have been
             *       a silent wrong image later. It lives at the END of the used region because growing it in place
             *       would renumber every array after it.
             */
            static constexpr uint32_t tlas = heap_slot_base + 703u;
            static constexpr uint32_t scene_camera = heap_slot_base + 514u;        // binding 0, per frame slot
            static constexpr uint32_t scene_light = heap_slot_base + 516u;         // binding 7, per frame slot
            static constexpr uint32_t cluster_counts = heap_slot_base + 518u;      // binding 11, per frame slot
            static constexpr uint32_t cluster_indices = heap_slot_base + 520u;     // binding 12, per frame slot
            static constexpr uint32_t instance_transforms = heap_slot_base + 522u; // binding 6, per frame slot
            static constexpr uint32_t previous_transforms = heap_slot_base + 524u; // binding 13, per frame slot
            static constexpr uint32_t skin_matrices = heap_slot_base + 526u;       // binding 9, per frame slot
            static constexpr uint32_t morph_data = heap_slot_base + 528u;          // binding 10, per frame slot
            static constexpr uint32_t mask_instances = heap_slot_base + 530u;      // binding 17, per frame slot
            static constexpr uint32_t env_cube = heap_slot_base + 532u;            // binding 2
            static constexpr uint32_t irradiance_cube = heap_slot_base + 533u;     // binding 3
            static constexpr uint32_t brdf_lut = heap_slot_base + 534u;            // binding 4
            static constexpr uint32_t shadow_map = heap_slot_base + 535u;          // binding 8, per image
            static constexpr uint32_t rt_visibility = heap_slot_base + 543u;       // binding 15, per image
            /**
             * @brief the SAME image as @ref rt_visibility, as a STORAGE descriptor instead of a sampled one
             *
             * @note TWO DESCRIPTORS FOR ONE IMAGE, and not redundancy: SAMPLED_IMAGE and STORAGE_IMAGE are different
             *       descriptor kinds and no single heap descriptor is both, while this image is WRITTEN by the
             *       ray-traced visibility pass and SAMPLED by the lighting stage. It lives at the end of the used
             *       region for the same reason the TLAS does - growing an array in place would renumber every array
             *       after it.
             */
            static constexpr uint32_t rt_visibility_storage = heap_slot_base + 711u;
            static constexpr uint32_t gbuffer_albedo = heap_slot_base + 551u; // per image, then five in a row
            static constexpr uint32_t gbuffer_normal = heap_slot_base + 559u;
            static constexpr uint32_t gbuffer_material = heap_slot_base + 567u;
            static constexpr uint32_t gbuffer_depth = heap_slot_base + 575u;
            static constexpr uint32_t gbuffer_velocity = heap_slot_base + 583u;
            static constexpr uint32_t ml_trace = heap_slot_base + 591u; // per image: megalights' chain
            /// @brief the same two images written as STORAGE descriptors (see @ref rt_visibility_storage)
            /// @note their compute passes WRITE them and the lighting stage SAMPLES them, and no single heap
            ///       descriptor is both kinds - so the trace and the resolve each need a second slot, at the end of
            ///       the used region for the same reason the others are there.
            static constexpr uint32_t ml_trace_storage = heap_slot_base + 719u;
            static constexpr uint32_t ml_resolved_storage = heap_slot_base + 735u;
            /**
             * @brief the joint blocks as they were ONE FRAME AGO, per frame slot: the deformation half of a
             *        motion vector
             *
             * @note A SECOND per-frame family rather than more slots inside @ref skin_matrices, because a
             *       vertex's motion vector needs the matrices the PREVIOUS frame drew with and the current
             *       buffer has already been overwritten with this frame's by the time the frame records.
             *       The layout, the indices and the frame-slot rule are identical to the current family's -
             *       that is what lets the shader read the same `skin_base` from this slot and lets the
             *       runtime publish into the CURRENT frame slot's buffer (see
             *       runtime::advance_motion_deformations), exactly as @ref previous_transforms does for the
             *       world matrices. It lives at the END of the used region for the same reason the TLAS and
             *       the two storage twins do: growing an array in place would renumber every array after it.
             */
            static constexpr uint32_t skin_matrices_previous = heap_slot_base + 743u;
            static constexpr uint32_t ml_history = heap_slot_base + 599u;
            static constexpr uint32_t ml_resolved = heap_slot_base + 607u;
            static constexpr uint32_t taa_current = heap_slot_base + 623u; // per image: TAA's pair
            static constexpr uint32_t taa_history = heap_slot_base + 631u;
            static constexpr uint32_t post_color = heap_slot_base + 639u; // per image: the post chain
            static constexpr uint32_t bloom_l0 = heap_slot_base + 647u;
            static constexpr uint32_t bloom_l1 = heap_slot_base + 655u;
            static constexpr uint32_t bloom_l2 = heap_slot_base + 663u;
            static constexpr uint32_t bloom_l3 = heap_slot_base + 671u;
            static constexpr uint32_t display_color = heap_slot_base + 695u;
        };
        /// @brief the byte offset of slot 0 (see @ref heap_slots), or VK_WHOLE_SIZE when the heap is not in use
        VkDeviceSize heap_grid_offset = VK_WHOLE_SIZE;

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
         * @brief create a 2D ARRAY depth image view covering every layer (DEPTH aspect)
         * @param image the layered depth image
         * @param format the view format (a depth format)
         * @return raii vk_image_view owning the created view
         * @note the cascaded shadow map is ONE layered image sampled as an array: a fragment shader
         *       picks its cascade per pixel, and dynamic indexing of a sampler array would require
         *       dynamically uniform indices, while a texture-array layer is just a coordinate
         */
        vk_image_view make_depth_array_view(VkImage image, VkFormat format) const;

        /**
         * @ingroup vulkan_core
         * @brief create a 2D depth image view of ONE layer (DEPTH aspect), for rendering into it
         * @param image the layered depth image
         * @param format the view format (a depth format)
         * @param layer the array layer to view
         * @return raii vk_image_view owning the created view
         */
        vk_image_view make_depth_layer_view(VkImage image, VkFormat format, uint32_t layer) const;

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

        // The clustered-light-culling compute pipeline is NOT here any more: the CLUSTER PASS owns it
        // (vulkan.pass.cluster builds it through vulkan.pipelines::build_cluster from the shared scene block
        // layout). `core::make_cluster_pipeline` built it against the core's own scene pipeline layout, which a
        // pass cannot own - and a pipeline only that pass names is that pass's to build and to release.

        /**
         * @ingroup vulkan_core
         * @brief rebuild the swapchain and every per-generation target
         * @return true when a NEW generation was actually built; false when the recreation was
         *         DEFERRED because the window has no drawable size (a minimized window reports a 0x0
         *         currentExtent, and vkCreateSwapchainKHR rejects that).
         *
         * THE RETURN VALUE IS NOT DECORATION. A caller that treats a deferred call as a rebuild
         * invalidates all the per-image state - the temporal histories, the descriptor families, the
         * layout flags - for a generation that still exists, and pays a full re-convergence for a
         * non-event: a minimize/restore dropped the GI and TAA history twice, once for the deferred
         * recreate and once for the real one.
         */
        [[nodiscard]] bool recreate_swap_chain();
        // one-time log for the "recreation deferred because the window has no drawable size" case
        // (see recreate_swap_chain); reset as soon as a recreation actually runs
        bool zero_extent_recreation_logged = false;

        /**
         * @ingroup vulkan_core
         * @brief submit the recorded command buffer for the current frame slot
         * @param command_buffer the command buffer to submit
         * @param image_index the acquired swapchain image index: it selects the per-image binary
         *        semaphore this submit signals for present to wait on
         * @return the result of vkQueueSubmit
         *
         * Signals two semaphores. This slot's TIMELINE (GPU completion, and the host pacing that
         * wait_frame_slot() blocks on) and present_ready_semaphores[image_index]. The second is a
         * BINARY semaphore per swapchain IMAGE rather than per frame slot because vkQueuePresentKHR
         * cannot wait a timeline semaphore, and present may run on a separate queue - keying the gate
         * on the image means an image is only re-acquired after its own present finished, which keeps
         * a re-signal from racing across queues.
         */
        VkResult submit(VkCommandBuffer command_buffer, uint32_t image_index);

        /**
         * @ingroup vulkan_core
         * @brief present the rendered swapchain image, waiting on that image's binary present-ready
         *        semaphore (signaled by submit())
         * @param image_index the swapchain image index to present
         * @return the result of vkQueuePresentKHR
         */
        VkResult present(uint32_t image_index) const;

        /**
         * @ingroup vulkan_core
         * @brief create the G-buffer pipeline: the shared scene layout, the three gbuffer_formats
         *        color targets and a single-sampled depth attachment
         * @param vertex_shader_code raw SPIR-V of the vertex stage (pbr.vert: instancing / skinning /
         *        morphing are identical to what the forward path did)
         * @param fragment_shader_code raw SPIR-V of the fragment stage (gbuffer.frag: writes the
         *        three targets and shades nothing)
         * @param first_stage the stage that emits the geometry: VERTEX for the input-assembler path (the default),
         *        MESH for a MESH entry that fetches its own vertices (docs/mesh_shaders.md step 2) - the fragment
         *        stage is the same shader either way, which is what makes the two pipelines comparable
         * @return vk_pipeline on success, error message on failure
         * @note single-sampled on purpose (a G-buffer cannot be multisampled without per-sample
         *       shading), so this pipeline may NOT be recorded into an instance whose attachments are
         *       the HDR target: its own attachments are the core::gbuffer_* targets and the pass that owns them
         */
        std::expected<vk_pipeline, std::string_view> make_gbuffer_pipeline(
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code,
            VkShaderStageFlagBits first_stage = VK_SHADER_STAGE_VERTEX_BIT) const;

    private:
        // ---- THE INITIALIZATION STEPS, and they are private because the constructor is their only caller:
        //      the order in which they run IS the initialization order (see the constructor in core.cpp), and a
        //      caller that could run one on its own would be able to build half a core. The exception is the
        //      swapchain recreation path, which is a member of this class and re-runs three of them. What stays
        //      PUBLIC is the state they produce (the device, the queues, the samplers, the image views and the
        //      layouts) plus the facade operations (wait_idle, set_window_title).
        void init_instance() noexcept;
        void init_window(int width, int height, std::string_view window_name = "") noexcept;
        void init_surface() noexcept;
        void init_device_and_queue() noexcept;
        void init_swap_chain() noexcept;
        void init_image_views() noexcept;
        void create_render_targets();
        /**
         * @brief create a single-sampled device-local image with its memory, and return both
         * @param width / @param height the extent in texels
         * @param format the image format
         * @param tiling OPTIMAL or LINEAR (staging images that are mapped on the host)
         * @param usage the usage flags the image is created with
         * @param properties the memory type the image is bound to
         * @note every target the engine creates is single-sampled: the only multisampled images it
         *       ever had were the forward path's, and that path is gone. The sample count is fixed
         *       rather than a parameter so there is one less thing a caller can get wrong.
         */
        void create_target_image(
            uint32_t width,
            uint32_t height,
            VkFormat format,
            VkImageTiling tiling,
            VkImageUsageFlags usage,
            VkMemoryPropertyFlags properties,
            VkImage& image,
            VkDeviceMemory& image_memory) const noexcept;
        /**
         * @ingroup vulkan_core
         * @brief create a single-sampled CUBE target: a six-layer 2D array with CUBE_COMPATIBLE set
         * @param size the edge length of one face, in texels (all six faces are the same size)
         * @note its own entry point rather than a generalised array helper, for the same reason
         *       create_target_image_3d has one: a cube is six layers AND the compatibility flag, and a caller
         *       that got one of those wrong would have an image the sampler refuses.
         */
        void create_target_image_cube(
            uint32_t size,
            VkFormat format,
            VkImageTiling tiling,
            VkImageUsageFlags usage,
            VkMemoryPropertyFlags properties,
            VkImage& image,
            VkDeviceMemory& image_memory) const noexcept;
        void create_target_image_3d(
            uint32_t width,
            uint32_t height,
            uint32_t depth,
            VkFormat format,
            VkImageTiling tiling,
            VkImageUsageFlags usage,
            VkMemoryPropertyFlags properties,
            VkImage& image,
            VkDeviceMemory& image_memory) const noexcept;
        void create_depth_image(VkImage& image, VkDeviceMemory& image_memory, VkImageView& image_view) const noexcept;
        void create_depth_resources() noexcept;
        void create_color_resources();
        void create_command_pool() noexcept;
        void create_samplers();
        void create_sync_objects();
        void create_timestamp_query_pool() noexcept;
    };
} // namespace vulkan