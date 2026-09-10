// ============================================================================
// module: vulkan.constant_init
// module version: 0.2.0  (independent of the app version in CMakeLists project(VERSION))
//
// Compile-time Vulkan info-struct conventions: constexpr factories + constinit
// "transition" defaults for the structs the engine fills identically everywhere
// (object create infos, command-buffer / secondary inheritance, fixed-function
// pipeline state, per-frame layout transitions). Top-level module: depends on
// nothing but the Vulkan headers, so any Vulkan module can embed it.
//
// evolve: bump MAJOR on breaking interface changes, MINOR on additive features,
//         PATCH on internal fixes - independently of the rest of the project.
// ============================================================================
module;

#include <vulkan/vulkan.h>

export module vulkan.constant_init;

/**
 * @defgroup vulkan_constant_init Vulkan Info-Struct Builders (fixed conventions)
 * @file constant_init.cppm
 *
 * @brief constexpr constructors and constinit defaults for the Vulkan info structs the engine
 *        fills the same way everywhere.
 *
 * Top-level module (sibling of vulkan.core): it depends on nothing but the Vulkan headers, so
 * any Vulkan module can use it. The name sets it apart from vulkan.core.init_utils - that
 * module performs the initialization PROCEDURES (instance/device/swapchain), while this one
 * holds the compile-time CONSTANTS of those calls: the fixed field values ("constant init").
 *
 * The engine never hand-fills these structs at call sites: every fill is either
 *  - a constexpr factory returning the struct by value (each factory lists EVERY member with
 *    designated initializers, so the zero/unused fields are explicit and identical to a
 *    `= {}` zero-init), or
 *  - a constinit "transition" default the caller copies and then overrides only the fields
 *    that actually differ (e.g. the target image of a per-frame layout transition).
 *
 * Pointer members always point at caller-owned data (never at locals of the factory itself).
 * This module is header-only in effect: all definitions live in the interface, so callers can
 * constant-fold the factories.
 */
export namespace vulkan {
    // ---- Object create infos (one line per object; fields fixed by engine convention) ----

    /**
     * @brief command pool that allows per-buffer reset (the engine resets/re-records buffers)
     */
    constexpr VkCommandPoolCreateInfo make_command_pool_info(uint32_t const queue_family) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = queue_family};
    }
    /** @brief plain binary semaphore (vkAcquireNextImageKHR / vkQueuePresentKHR need binary) */
    constexpr VkSemaphoreCreateInfo make_binary_semaphore_info() noexcept {
        return {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0};
    }
    /** @brief timeline semaphore starting at 0 (frame-slot submission counting + host pacing) */
    constexpr VkSemaphoreTypeCreateInfo make_timeline_semaphore_type_info() noexcept {
        return {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                .pNext = nullptr,
                .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                .initialValue = 0};
    }
    /**
     * @brief linear/mipmap sampler with repeat-style addressing
     * @param address_mode applied to U/V/W
     * @param max_lod upper clamp (e.g. the texture's mip count - 1)
     */
    constexpr VkSamplerCreateInfo make_texture_sampler_info(VkSamplerAddressMode const address_mode, float const max_lod) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .magFilter = VK_FILTER_LINEAR,
                .minFilter = VK_FILTER_LINEAR,
                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                .addressModeU = address_mode,
                .addressModeV = address_mode,
                .addressModeW = address_mode,
                .mipLodBias = 0.0f,
                .anisotropyEnable = VK_FALSE,
                .maxAnisotropy = 1.0f,
                .compareEnable = VK_FALSE,
                .compareOp = VK_COMPARE_OP_NEVER,
                .minLod = 0.0f,
                .maxLod = max_lod,
                .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
                .unnormalizedCoordinates = VK_FALSE};
    }
    /**
     * @brief shadow-map sampler: LINEAR min/mag gives HARDWARE percentage-closer filtering on a
     *        sampler2DShadow (compareOp matches pbr.frag's "not deeper than stored depth")
     */
    constexpr VkSamplerCreateInfo make_shadow_sampler_info() noexcept {
        return {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .magFilter = VK_FILTER_LINEAR,
                .minFilter = VK_FILTER_LINEAR,
                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .mipLodBias = 0.0f,
                .anisotropyEnable = VK_FALSE,
                .maxAnisotropy = 1.0f,
                .compareEnable = VK_TRUE,
                .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
                .minLod = 0.0f,
                .maxLod = 0.0f,
                .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
                .unnormalizedCoordinates = VK_FALSE};
    }
    /** @brief unsignaled fence (host waits after one-shot upload submits) */
    constexpr VkFenceCreateInfo make_fence_info() noexcept {
        return {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0};
    }
    /**
     * @brief query pool of @p count queries, all of type @p query_type
     * @param query_type TIMESTAMP for the GPU pass timings, OCCLUSION for visibility queries
     * @param count number of queries in the pool (a timestamp pool also fixes how many marks a
     *        frame may write: see core::gpu_timing_mark_capacity)
     */
    constexpr VkQueryPoolCreateInfo make_query_pool_info(VkQueryType const query_type, uint32_t const count) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queryType = query_type,
                .queryCount = count,
                .pipelineStatistics = 0};
    }
    /** @brief host-visible staging buffer: TRANSFER_SRC only, exclusive sharing */
    constexpr VkBufferCreateInfo make_staging_buffer_info(VkDeviceSize const size) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = size,
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr};
    }
    /**
     * @brief one-queue device queue create info (one queue of @p queue_family)
     * @param queue_priorities caller-owned array of @p queue_family's queue priorities
     */
    constexpr VkDeviceQueueCreateInfo make_device_queue_info(uint32_t const queue_family, float const* queue_priorities) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queueFamilyIndex = queue_family,
                .queueCount = 1,
                .pQueuePriorities = queue_priorities};
    }
    /**
     * @brief 2D image view with identity component swizzle, from mip 0 / layer 0
     * @param image the image to view
     * @param format the image's format
     * @param view_type usually 2D
     * @param aspect_mask color or depth(-stencil)
     * @param level_count mip levels in the view (VK_REMAINING_MIP_LEVELS for the whole image)
     * @param layer_count array layers in the view (VK_REMAINING_ARRAY_LAYERS for the whole image)
     */
    constexpr VkImageViewCreateInfo make_image_view_info(VkImage const image, VkFormat const format, VkImageViewType const view_type, VkImageAspectFlags const aspect_mask, uint32_t const level_count, uint32_t const layer_count) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .image = image,
                .viewType = view_type,
                .format = format,
                .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
                .subresourceRange = {aspect_mask, 0, level_count, 0, layer_count}};
    }

    // ---- Command buffer / submit infos ----

    /**
     * @brief command buffer allocation: @p level buffers from @p pool
     * @param pool command pool to allocate from
     * @param level primary or secondary
     */
    constexpr VkCommandBufferAllocateInfo make_command_buffer_allocate_info(VkCommandPool const pool, VkCommandBufferLevel const level) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = pool,
                .level = level,
                .commandBufferCount = 1};
    }
    /**
     * @brief command buffer begin info
     * @param flags usage flags (0 for plain inline recording, ONE_TIME_SUBMIT for one-shot
     *        uploads, RENDER_PASS_CONTINUE for secondaries)
     * @param inheritance secondary-buffer inheritance info (nullptr for primary buffers)
     */
    constexpr VkCommandBufferBeginInfo make_command_buffer_begin_info(VkCommandBufferUsageFlags const flags, VkCommandBufferInheritanceInfo const* inheritance) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = flags,
                .pInheritanceInfo = inheritance};
    }
    /**
     * @brief one-shot submit of a single command buffer with no semaphores
     * @param command_buffers pointer to the caller's VkCommandBuffer
     */
    constexpr VkSubmitInfo make_submit_info(VkCommandBuffer const* command_buffers) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = command_buffers,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = nullptr};
    }

    // ---- Secondary-command-buffer inheritance (dynamic rendering 1.3) ----

    /**
     * @brief VkCommandBufferInheritanceInfo shell; only the pNext chain (the rendering info)
     *        differs per buffer
     * @param p_next points at the VkCommandBufferInheritanceRenderingInfo (caller-owned)
     */
    constexpr VkCommandBufferInheritanceInfo make_inheritance_info(void const* p_next) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
                .pNext = p_next,
                .renderPass = VK_NULL_HANDLE,
                .subpass = 0,
                .framebuffer = VK_NULL_HANDLE,
                .occlusionQueryEnable = VK_FALSE,
                .queryFlags = 0,
                .pipelineStatistics = 0};
    }
    /**
     * @brief dynamic-rendering inheritance: which attachments a secondary may assume
     * @param has_color_attachment depth-only secondaries (shadow pass) pass false
     * @param color_format_ptr caller-owned color format (ignored when has_color_attachment
     *        is false)
     * @param depth_format the depth attachment's format (VK_FORMAT_UNDEFINED if none)
     * @param rasterization_samples follows MSAA (1 for the single-sampled shadow map)
     */
    constexpr VkCommandBufferInheritanceRenderingInfo make_inheritance_rendering_info(bool const has_color_attachment, VkFormat const* color_format_ptr, VkFormat const depth_format, VkSampleCountFlagBits const rasterization_samples) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .viewMask = 0,
                .colorAttachmentCount = has_color_attachment ? 1u : 0u,
                .pColorAttachmentFormats = has_color_attachment ? color_format_ptr : nullptr,
                .depthAttachmentFormat = depth_format,
                .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
                .rasterizationSamples = rasterization_samples};
    }
    /**
     * @brief one barrier pass: VkDependencyInfo with only image memory barriers
     * @param image_barrier_count number of barriers
     * @param barriers caller-owned barrier array
     */
    constexpr VkDependencyInfo make_image_dependency_info(uint32_t const image_barrier_count, VkImageMemoryBarrier2 const* barriers) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = nullptr,
                .dependencyFlags = 0,
                .memoryBarrierCount = 0,
                .pMemoryBarriers = nullptr,
                .bufferMemoryBarrierCount = 0,
                .pBufferMemoryBarriers = nullptr,
                .imageMemoryBarrierCount = image_barrier_count,
                .pImageMemoryBarriers = barriers};
    }

    // ---- Dynamic rendering attachment infos ----

    /**
     * @brief depth attachment of a rendering instance: loadOp CLEAR with the far-plane value
     *        (1.0, stencil 0 - the engine clears every attachment on load) and no resolve
     * @param image_view the depth image view
     * @param store_op DONT_CARE for the transient main depth buffer, STORE for the shadow map
     *        (its contents must survive for the main pass to sample)
     */
    constexpr VkRenderingAttachmentInfo make_depth_attachment_info(VkImageView const image_view, VkAttachmentStoreOp const store_op) noexcept {
        VkClearValue clear_value = {};
        clear_value.depthStencil = {1.0f, 0};
        return {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = image_view,
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = store_op,
                .clearValue = clear_value};
    }
    /**
     * @brief color attachment of a rendering instance: COLOR_ATTACHMENT_OPTIMAL layout,
     *        loadOp CLEAR + storeOp STORE (the swapchain image is presented afterwards)
     * @param image_view the (MSAA or swapchain) color image view
     * @param clear_value the runtime clear color
     * @param resolve_mode NONE without MSAA, AVERAGE with MSAA
     * @param resolve_image_view swapchain resolve target when MSAA, VK_NULL_HANDLE otherwise.
     *        NOTE: the resolve layout must not be PRESENT_SRC_KHR
     *        (VUID-VkRenderingAttachmentInfo-imageView-06146) - the swapchain image moves to
     *        PRESENT_SRC_KHR only after vkCmdEndRendering
     */
    constexpr VkRenderingAttachmentInfo make_color_attachment_info(VkImageView const image_view, VkClearValue const clear_value, VkResolveModeFlagBits const resolve_mode, VkImageView const resolve_image_view) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = image_view,
                .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .resolveMode = resolve_mode,
                .resolveImageView = resolve_image_view,
                .resolveImageLayout = resolve_mode == VK_RESOLVE_MODE_NONE ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = clear_value};
    }
    /**
     * @brief dynamic-rendering instance: one layer, at most one color attachment
     * @param flags e.g. VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT
     * @param render_area the area to render into (the pipelines apply their own viewport/scissor)
     * @param has_color_attachment depth-only passes (the shadow map) pass false
     * @param color_attachments pointer to the single color attachment when present, else nullptr
     * @param depth_attachment the depth attachment, or nullptr
     */
    constexpr VkRenderingInfo make_rendering_info(VkRenderingFlags const flags, VkRect2D const render_area, bool const has_color_attachment, VkRenderingAttachmentInfo const* color_attachments, VkRenderingAttachmentInfo const* depth_attachment) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = flags,
                .renderArea = render_area,
                .layerCount = 1,
                .viewMask = 0,
                .colorAttachmentCount = has_color_attachment ? 1u : 0u,
                .pColorAttachments = has_color_attachment ? color_attachments : nullptr,
                .pDepthAttachment = depth_attachment,
                .pStencilAttachment = nullptr};
    }

    // ---- Fixed-function pipeline state (engine-wide conventions) ----

    /** @brief one shader stage of a graphics pipeline (entry point "main", no specialization) */
    constexpr VkPipelineShaderStageCreateInfo make_shader_stage(VkShaderModule const module, VkShaderStageFlagBits const stage) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = stage,
                .module = module,
                .pName = "main",
                .pSpecializationInfo = nullptr};
    }
    /** @brief triangle-list input assembly */
    constexpr VkPipelineInputAssemblyStateCreateInfo make_input_assembly_state() noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                .primitiveRestartEnable = VK_FALSE};
    }
    /** @brief viewport/scissor are dynamic state: only the counts are static */
    constexpr VkPipelineViewportStateCreateInfo make_viewport_state() noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .viewportCount = 1,
                .pViewports = nullptr,
                .scissorCount = 1,
                .pScissors = nullptr};
    }
    /** @brief dynamic state list (caller-owned array) */
    constexpr VkPipelineDynamicStateCreateInfo make_dynamic_state(VkDynamicState const* states, uint32_t const count) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .dynamicStateCount = count,
                .pDynamicStates = states};
    }
    /**
     * @brief fill rasterization, back-face cull, CCW front face, optional slope-scaled depth bias
     * @param depth_bias_enabled the bias is dynamic state on shadow pipelines; the static
     *        factors only matter while it is never set dynamically
     */
    constexpr VkPipelineRasterizationStateCreateInfo make_rasterization_state(bool const depth_bias_enabled, float const depth_bias_constant_factor, float const depth_bias_slope_factor, float const depth_bias_clamp) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_BACK_BIT,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .depthBiasEnable = depth_bias_enabled ? VK_TRUE : VK_FALSE,
                .depthBiasConstantFactor = depth_bias_constant_factor,
                .depthBiasClamp = depth_bias_clamp,
                .depthBiasSlopeFactor = depth_bias_slope_factor,
                .lineWidth = 1.0f};
    }
    /**
     * @brief depth test + write (disabled for background passes such as the skybox, which draw
     *        first and must not occlude later geometry); compare LESS_OR_EQUAL
     */
    constexpr VkPipelineDepthStencilStateCreateInfo make_depth_stencil_state(bool const depth_test_enabled) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .depthTestEnable = depth_test_enabled ? VK_TRUE : VK_FALSE,
                .depthWriteEnable = depth_test_enabled ? VK_TRUE : VK_FALSE,
                .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE,
                .front = {},
                .back = {},
                .minDepthBounds = 0.0f,
                .maxDepthBounds = 0.0f};
    }
    /**
     * @brief standard alpha blending, ALWAYS enabled: with src alpha == 1 (an opaque material)
     *        the blend math reduces to the source color exactly, so opaque draws are
     *        pixel-identical whether or not blending is on. Blended (transparent) materials
     *        carry alpha < 1 and are drawn depth-write-off in the transparent pass - no second
     *        pipeline needed.
     */
    constexpr VkPipelineColorBlendAttachmentState make_color_blend_attachment() noexcept {
        return {.blendEnable = VK_TRUE,
                .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                .colorBlendOp = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                .alphaBlendOp = VK_BLEND_OP_ADD,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    }
    /**
     * @brief blend state for one attachment; depth-only pipelines have no color attachment
     * @param attachment caller-owned blend attachment (ignored when has_color_attachment false)
     */
    constexpr VkPipelineColorBlendStateCreateInfo make_color_blend_state(bool const has_color_attachment, VkPipelineColorBlendAttachmentState const* attachment) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .logicOpEnable = VK_FALSE,
                .logicOp = VK_LOGIC_OP_COPY,
                .attachmentCount = has_color_attachment ? 1u : 0u,
                .pAttachments = has_color_attachment ? attachment : nullptr,
                .blendConstants = {1.0f, 1.0f, 1.0f, 1.0f}};
    }
    /** @brief multisample state; rasterizationSamples follows MSAA */
    constexpr VkPipelineMultisampleStateCreateInfo make_multisample_state(VkSampleCountFlagBits const samples) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .rasterizationSamples = samples,
                .sampleShadingEnable = VK_FALSE,
                .minSampleShading = 0.0f,
                .pSampleMask = nullptr,
                .alphaToCoverageEnable = VK_FALSE,
                .alphaToOneEnable = VK_FALSE};
    }
    /**
     * @brief dynamic-rendering attachment declaration (replaces render pass + subpass)
     * @param color_format_ptr pointer to the caller's color format: the returned info holds
     *        that pointer, so it must outlive the struct
     * @param depth_format the depth attachment format (VK_FORMAT_UNDEFINED for depth-only)
     */
    constexpr VkPipelineRenderingCreateInfo make_rendering_create_info(bool const has_color_attachment, VkFormat const* color_format_ptr, VkFormat const depth_format) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                .pNext = nullptr,
                .viewMask = 0,
                .colorAttachmentCount = has_color_attachment ? 1u : 0u,
                .pColorAttachmentFormats = has_color_attachment ? color_format_ptr : nullptr,
                .depthAttachmentFormat = depth_format,
                .stencilAttachmentFormat = VK_FORMAT_UNDEFINED};
    }

    // ---- Per-frame image layout transitions (constinit defaults) ----
    // Every frame moves the same images between the same layouts; only the target image
    // differs per barrier. Each role below is therefore a default that call sites copy and
    // then override .image on (immutable by design - the invariants must not be retargeted in
    // place). The attachment transitions discard the old contents: loadOp CLEAR makes
    // UNDEFINED as oldLayout valid whatever the image's actual current layout is - no
    // per-frame layout tracking needed.
    /** @brief UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL, color-attachment write (MSAA color image + swapchain resolve target) */
    inline constexpr VkImageMemoryBarrier2 color_attachment_transition = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
        .srcAccessMask = 0,
        .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = VK_NULL_HANDLE,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    /** @brief UNDEFINED -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL, depth write (main depth buffer + the shadow map) */
    inline constexpr VkImageMemoryBarrier2 depth_attachment_transition = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
        .srcAccessMask = 0,
        .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = VK_NULL_HANDLE,
        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
    };
    /** @brief depth attachment -> SHADER_READ_ONLY_OPTIMAL, fragment-shader sampled read (the shadow map back to the main pass) */
    inline constexpr VkImageMemoryBarrier2 shadow_map_sampling_transition = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = VK_NULL_HANDLE,
        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
    };
    /** @brief color attachment -> SHADER_READ_ONLY_OPTIMAL, fragment-shader sampled read (the HDR scene target into the post-process pass) */
    inline constexpr VkImageMemoryBarrier2 hdr_sampling_transition = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = VK_NULL_HANDLE,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    /** @brief UNDEFINED -> SHADER_READ_ONLY_OPTIMAL for a DEPTH image: keep the shadow map's sampled
     *         descriptor valid on frames where the shadow pass does not run (shadows toggled off).
     *         pbr.frag always binds binding 8 and decides at runtime whether to sample it, and a
     *         descriptor must point at an image in the layout it declares - leaving the map in
     *         UNDEFINED made every such frame a VUID. Contents are irrelevant (the shader returns
     *         "fully lit"), so UNDEFINED as the old layout is correct. */
    inline constexpr VkImageMemoryBarrier2 undefined_to_depth_sampling_transition = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
        .srcAccessMask = 0,
        .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = VK_NULL_HANDLE,
        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
    };
    /** @brief UNDEFINED -> SHADER_READ_ONLY_OPTIMAL: make a transient target readable without
     *         claiming a layout it may not be in (the bloom chain when the passes are skipped - the
     *         composite still samples those bindings statically, so the layout must be valid, but the
     *         contents are multiplied by zero) */
    inline constexpr VkImageMemoryBarrier2 undefined_to_sampling_transition = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
        .srcAccessMask = 0,
        .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = VK_NULL_HANDLE,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    /** @brief COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL, screenshot read-back copy
     *         (vkCmdCopyImageToBuffer). Recorded INSIDE the frame's own command buffer, while the
     *         swapchain image is still owned by the app: after vkQueuePresentKHR the presentation
     *         engine owns it and transitioning it again violates the WSI rules. */
    inline constexpr VkImageMemoryBarrier2 color_attachment_to_transfer_transition = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = VK_NULL_HANDLE,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    /** @brief TRANSFER_SRC_OPTIMAL -> COLOR_ATTACHMENT_OPTIMAL, hand the screenshotted image back
     *         to the frame so present_transition (COLOR_ATTACHMENT -> PRESENT_SRC) still applies */
    inline constexpr VkImageMemoryBarrier2 transfer_to_color_attachment_transition = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = VK_NULL_HANDLE,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    /** @brief UNDEFINED -> PRESENT_SRC_KHR: present a frame whose post-process pass was skipped, so
     *         the swapchain image never entered COLOR_ATTACHMENT_OPTIMAL (contents are undefined -
     *         this only exists to hand the WSI a validly-laid-out image instead of lying about the
     *         old layout, which is what present_transition assumes) */
    inline constexpr VkImageMemoryBarrier2 undefined_to_present_transition = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
        .srcAccessMask = 0,
        .dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
        .dstAccessMask = 0,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = VK_NULL_HANDLE,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    /** @brief COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR (dynamic rendering has no finalLayout) */
    inline constexpr VkImageMemoryBarrier2 present_transition = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
        .dstAccessMask = 0,
        .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = VK_NULL_HANDLE,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
} // namespace vulkan
