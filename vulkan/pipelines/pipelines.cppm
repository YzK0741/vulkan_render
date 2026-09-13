// module version: 0.10.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pipelines/pipelines.cppm
 * @defgroup vulkan_pipelines Per-Pass Pipeline Builders
 * @brief The engine's per-pass pipelines: one builder per pass, next to the generic builder in
 *        vulkan.core.pipeline (that one knows HOW to build a pipeline, this one knows what each pass's
 *        pipeline looks like - formats, sample counts, blend state, push-constant ranges).
 *
 * Extracted from vulkan.runtime, whose implementation had grown past 4900 lines. The builders are
 * stateless: the caller passes the core and the set layouts the pass reuses and gets the created
 * handles back. Ownership stays with the runtime, which keeps the members and the call ORDER - the
 * order matters because the set layout a pass needs is owned by the pass that creates it (fxaa needs
 * the post layout, deferred the G-buffer layout, and both say so in their error messages).
 *
 * The pipeline handles are std::optional because vk_pipeline is an RAII owner with no default
 * constructor, which is also how the runtime holds them.
 */

module;

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

export module vulkan.pipelines;

import vulkan.core;
import vulkan.core.pipeline; // vk_pipeline

namespace vulkan::pipelines {
    /// what build_post() creates: the post set layout, its pipeline layout and the two composites
    export struct post_owned {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> composite; // tonemap + bloom sum, writes the swapchain
        std::optional<vk_pipeline> hdr;       // the same pass writing an HDR target instead (FXAA on)
    };

    /// what build_gbuffer_debug() creates: the G-buffer set layout (owned here), its pipeline layout
    /// and the debug view pipeline
    export struct gbuffer_owned {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> debug;
    };

    /// what build_taa() creates
    export struct taa_owned {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> resolve;
    };

    /// what build_ssgi() creates: the tracer's COMPUTE pipeline. It owns no set layout - it binds
    /// the shared scene set plus the G-buffer set, which already carries the radiance sampler and
    /// the GI storage image it needs - so only the pipeline layout is new.
    export struct ssgi_owned {
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> trace;
    };

    export std::expected<post_owned, std::string> build_post(core& vk, uint32_t push_constant_size, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    export std::expected<gbuffer_owned, std::string> build_gbuffer_debug(core& vk, uint32_t push_constant_size, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    export std::expected<taa_owned, std::string> build_taa(core& vk, uint32_t push_constant_size, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    export std::expected<ssgi_owned, std::string> build_ssgi(core& vk, VkDescriptorSetLayout scene_layout, VkDescriptorSetLayout gbuffer_layout, uint32_t push_constant_size, std::span<unsigned char const> compute_shader_code);
    /// the spatial half of the same denoiser: same two set layouts, same shape, its own push block
    export std::expected<ssgi_owned, std::string> build_ssgi_spatial(core& vk, VkDescriptorSetLayout scene_layout, VkDescriptorSetLayout gbuffer_layout, uint32_t push_constant_size, std::span<unsigned char const> compute_shader_code);
    /// the ray-traced sun shadow: the same two set layouts as the GI tracer (the scene set carries the
    /// camera, the light UBO and - when the device has ray tracing - the top level structure; the
    /// G-buffer set carries the surface the ray starts from)
    export std::expected<ssgi_owned, std::string> build_rt_shadow(core& vk, VkDescriptorSetLayout scene_layout, VkDescriptorSetLayout gbuffer_layout, uint32_t push_constant_size, std::span<unsigned char const> compute_shader_code);

    /// what build_ssgi_temporal() creates: the denoiser's set layout (it owns one - its inputs are
    /// the trace, the history, the motion vectors and the depth, which no other pass groups together)
    export struct ssgi_temporal_owned {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> resolve;
    };

    export std::expected<ssgi_temporal_owned, std::string> build_ssgi_temporal(core& vk, uint32_t push_constant_size, std::span<unsigned char const> compute_shader_code);

    /// what build_gi_probe() creates: the world-space probe cache's pass, which owns its set layout for
    /// the same reason the denoiser's resolve does - the things it binds (this frame's resolved
    /// screen-space GI, the depth, and the two grid images it ping-pongs between) are grouped by no
    /// other pass. It binds no scene set: the push block carries the projection it needs.
    export struct gi_probe_owned {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> pass;
    };

    export std::expected<gi_probe_owned, std::string> build_gi_probe(core& vk, VkDescriptorSetLayout scene_layout, uint32_t push_constant_size, std::span<unsigned char const> compute_shader_code);

    /// the passes that reuse a layout someone else owns, so theirs comes in as a parameter
    export std::expected<vk_pipeline, std::string> build_fxaa(core& vk, VkPipelineLayout post_pipeline_layout, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    export struct deferred_owned {
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> lighting;
    };

    /// the additive blend state comes in as a parameter: the helper that builds it is a local of the
    /// runtime, next to the passes whose blend modes it describes
    export std::expected<deferred_owned, std::string> build_deferred(core& vk, VkDescriptorSetLayout scene_layout, VkDescriptorSetLayout gbuffer_layout, uint32_t push_constant_size, std::span<VkPipelineColorBlendAttachmentState const> color_blend, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    // post: the composite chain's owner. The set layout, its pipeline layout and the two fullscreen
    // pipelines (one per color format the chain renders into) are created here; the sampler stays with
    // the runtime, which owns the descriptor sets that use it. The caller passes the size of its push
    // constant block because that structure is the runtime's (it must match post.frag).
    std::expected<post_owned, std::string> build_post(core& vk, uint32_t const push_constant_size, std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        post_owned out;

        // binding 0 = the pass input (HDR for the prefilter, the previous bloom level for a
        // downsample), bindings 1..4 = the four bloom levels, binding 5 = the gamma-encoded LDR image
        // (the FXAA pass, which shares this layout), binding 6 = the screen-space GI image and 7/8 =
        // the G-buffer depth and world normal, which only the composite reads - the depth and normal
        // are there for its joint-bilateral upsample of the half-resolution GI, and a prefilter or
        // downsample set points all three at views it does not care about.
        std::array<VkDescriptorSetLayoutBinding, 9> bindings = {};
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
        if (vkCreateDescriptorSetLayout(vk.device, &layout_info, nullptr, &out.set_layout) != VK_SUCCESS) {
            return fail("post: descriptor set layout creation failed");
        }

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &out.set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("post: pipeline layout creation failed");
        }

        // TWO variants, one per color format the chain renders into: the composite writes the
        // swapchain, the bright-pass prefilter and the downsample passes write the R16F bloom levels. A
        // pipeline's rendering color format must match its attachment, so one swapchain-format pipeline
        // was a validation error for the HDR passes.
        auto const make_post_variant = [&](VkFormat const color_format) -> std::expected<vk_pipeline, std::string> {
            auto pipeline_result = vulkan::make_pipeline(
                vk.device, out.pipeline_layout, color_format, VK_FORMAT_UNDEFINED, vertex_shader_code, fragment_shader_code, VK_SAMPLE_COUNT_1_BIT, false, true, 0.0f, 0.0f, 0.0f);
            if (!pipeline_result) {
                return std::unexpected(std::string(pipeline_result.error()));
            }
            return std::move(pipeline_result).value();
        };

        auto composite_pipeline = make_post_variant(vk.swap_chain_image_format);
        if (!composite_pipeline) {
            return fail(std::move(composite_pipeline.error()));
        }
        out.composite = std::move(composite_pipeline).value();

        auto hdr_pipeline = make_post_variant(vulkan::hdr_format);
        if (!hdr_pipeline) {
            return fail(std::move(hdr_pipeline.error()));
        }
        out.hdr = std::move(hdr_pipeline).value();
        return out;
    }
    // gbuffer debug: the owner of the G-buffer set layout. deferred reuses that layout, which is why
    // the runtime creates this pipeline first and says so in deferred's error message. The nearest
    // sampler the debug view needs stays with the runtime, next to the descriptor sets that use it.
    std::expected<gbuffer_owned, std::string> build_gbuffer_debug(core& vk, uint32_t const push_constant_size, std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        gbuffer_owned out;

        // albedo, normal, material, depth, velocity: the four the debug view displays plus the
        // motion-vector target. The deferred lighting stage binds this SAME layout as its set 1
        // and its shader declares only the first four, which is legal - a binding a shader does
        // not statically use does not need a descriptor written.
        //
        // 5..8 belong to the screen-space GI passes: the direct-radiance image the tracer samples at a
        // hit (5), the half-resolution raw trace it writes (6), the accumulated image the spatial
        // filter reads (7) and the filtered image that filter writes and the composite samples (8). 6
        // and 8 are STORAGE images rather than samplers because a compute pass writes a storage image,
        // and because those passes run at half the composite's resolution - the two ends of each are
        // different kinds of thing.
        //
        // 9 is the world-space probe cache, a sampler3D the TRACER samples for a hit the screen cannot
        // answer (see shaders/gi_probe.comp). It lives here rather than in the probe pass's own set
        // because the pass that reads it is the tracer, which binds this set.
        std::array<VkDescriptorSetLayoutBinding, 10> bindings = {};
        for (uint32_t b = 0; b < bindings.size(); ++b) {
            bindings[b].binding = b;
            bindings[b].descriptorType = (b == 6u || b == 8u) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[b].descriptorCount = 1;
            // EVERY binding lists COMPUTE as well as FRAGMENT. The layout is shared by four consumers
            // and only the shaders know which binding each of them uses: the lighting stage and the
            // debug view are fragment stages, while the GI tracer and the GI spatial filter are COMPUTE
            // stages that read the stored surface (albedo, normal and depth) and the GI images
            // directly. A binding a shader does not statically use needs no descriptor, but one it
            // DOES use has to name the stage here.
            bindings[b].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[b].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(vk.device, &layout_info, nullptr, &out.set_layout) != VK_SUCCESS) {
            return fail("gbuffer debug: descriptor set layout creation failed");
        }

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &out.set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("gbuffer debug: pipeline layout creation failed");
        }

        VkFormat const hdr_format_only = vulkan::hdr_format;
        auto pipeline_result = vulkan::make_pipeline(
            vk.device, out.pipeline_layout, hdr_format_only, VK_FORMAT_UNDEFINED, vertex_shader_code, fragment_shader_code, VK_SAMPLE_COUNT_1_BIT, false, true, 0.0f, 0.0f, 0.0f);
        if (!pipeline_result) {
            return fail(std::string(pipeline_result.error()));
        }
        out.debug = std::move(pipeline_result).value();
        return out;
    }
    // taa: the resolve pass' owner. It writes the HDR target, so its rendering color format is hdr_format
    // (a span of one), and its sampler is the odd one out - linear magnification, nearest minification,
    // because the resolve upsamples the scene color but must not average neighbouring history texels.
    std::expected<taa_owned, std::string> build_taa(core& vk, uint32_t const push_constant_size, std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        taa_owned out;

        // FOUR bindings, because four is what shaders/taa.frag declares (current color, history,
        // velocity, depth) and what runtime::ensure_taa_descriptors writes. The count is not cosmetic:
        // it is also this layout's descriptor count, so a layout declaring one binding more than the
        // family is SIZED for makes that family's pool too small for its own allocation. The validation
        // layer named it - "Trying to allocate 15 of VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
        // descriptors from VkDescriptorPool ..., but this pool only has a total of 12 descriptors for
        // this type" (3 images x 5 bindings against a pool built for 3 x 4, because image_set_family
        // sizes its pool from the signature's length) - and a driver that enforces the rule would fail
        // the allocation, which runtime::ensure_taa_descriptors turns into TAA silently switching itself
        // off rather than a frame that is merely missing a descriptor.
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
        if (vkCreateDescriptorSetLayout(vk.device, &layout_info, nullptr, &out.set_layout) != VK_SUCCESS) {
            return fail("taa: descriptor set layout creation failed");
        }

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &out.set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("taa: pipeline layout creation failed");
        }

        std::array<VkFormat, 1> const color_formats = {vulkan::hdr_format};
        auto pipeline_result = vulkan::make_pipeline(
            vk.device, out.pipeline_layout, std::span<VkFormat const>(color_formats), VK_FORMAT_UNDEFINED, vertex_shader_code, fragment_shader_code, VK_SAMPLE_COUNT_1_BIT, false, 0.0f, 0.0f, 0.0f);
        if (!pipeline_result) {
            return fail(std::string(pipeline_result.error()));
        }
        out.resolve = std::move(pipeline_result).value();
        return out;
    }

    // ssgi: a COMPUTE pipeline. Two set layouts rather than one (the shared scene set and the
    // G-buffer set) because that is where its inputs already are: the camera UBO, the stored
    // surface, the direct-radiance image and the GI image it writes. Creating a third layout for
    // this pass alone would mean duplicating four descriptor writes to gain nothing.
    std::expected<ssgi_owned, std::string> build_ssgi(core& vk, VkDescriptorSetLayout const scene_layout, VkDescriptorSetLayout const gbuffer_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        ssgi_owned out;

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        std::array<VkDescriptorSetLayout, 2> const set_layouts = {scene_layout, gbuffer_layout};
        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(set_layouts.size());
        pipeline_layout_info.pSetLayouts = set_layouts.data();
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("ssgi: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, vk.device);
        if (!module.has_value()) {
            return fail("ssgi: compute shader module creation failed");
        }
        VkPipelineShaderStageCreateInfo stage_info = {};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = **module;
        stage_info.pName = "main"; // the SPIR-V entry point, as everywhere else

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = out.pipeline_layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("ssgi: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, out.pipeline_layout, vk.device);
        return out;
    }

    // The ray-traced sun shadow: a COMPUTE pipeline over the same two set layouts the GI tracer uses,
    // because its inputs are in the same places (the camera block and the light UBO in the scene set,
    // the stored surface in the G-buffer set) plus the top level structure, which lives in the scene set
    // as binding 16 when the device has ray tracing. Nothing new is created here beyond the layout.
    std::expected<ssgi_owned, std::string> build_rt_shadow(core& vk, VkDescriptorSetLayout const scene_layout, VkDescriptorSetLayout const gbuffer_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        ssgi_owned out;

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        std::array<VkDescriptorSetLayout, 2> const set_layouts = {scene_layout, gbuffer_layout};
        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(set_layouts.size());
        pipeline_layout_info.pSetLayouts = set_layouts.data();
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("rt shadow: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, vk.device);
        if (!module.has_value()) {
            return fail("rt shadow: compute shader module creation failed");
        }
        VkPipelineShaderStageCreateInfo stage_info = {};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = **module;
        stage_info.pName = "main";

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = out.pipeline_layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("rt shadow: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, out.pipeline_layout, vk.device);
        return out;
    }
    // The GI spatial filter: the same two set layouts the tracer binds (the shared scene set and the
    // G-buffer set, which carries the normal, the depth, the accumulated image it reads and the
    // filtered image it writes), so only the pipeline layout and the push block are new.
    std::expected<ssgi_owned, std::string> build_ssgi_spatial(core& vk, VkDescriptorSetLayout const scene_layout, VkDescriptorSetLayout const gbuffer_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        ssgi_owned out;

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        std::array<VkDescriptorSetLayout, 2> const set_layouts = {scene_layout, gbuffer_layout};
        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(set_layouts.size());
        pipeline_layout_info.pSetLayouts = set_layouts.data();
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("ssgi spatial: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, vk.device);
        if (!module.has_value()) {
            return fail("ssgi spatial: compute shader module creation failed");
        }
        VkPipelineShaderStageCreateInfo stage_info = {};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = **module;
        stage_info.pName = "main";

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = out.pipeline_layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("ssgi spatial: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, out.pipeline_layout, vk.device);
        return out;
    }

    // The GI temporal resolve: its own set layout, because it groups four things no other pass puts
    // together (the raw trace, the accumulated history, the motion vectors and the depth). It binds
    // no scene set: the push block carries the two projection terms its depth guard needs.
    std::expected<ssgi_temporal_owned, std::string> build_ssgi_temporal(core& vk, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        ssgi_temporal_owned out;

        // 0..3 are sampler bindings (trace, history, motion vectors, depth); 4 is the STORAGE image
        // the resolve writes, which is why one binding differs from the rest.
        std::array<VkDescriptorSetLayoutBinding, 5> bindings = {};
        for (uint32_t b = 0; b < bindings.size(); ++b) {
            bindings[b].binding = b;
            bindings[b].descriptorType = b == 4u ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[b].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(vk.device, &layout_info, nullptr, &out.set_layout) != VK_SUCCESS) {
            return fail("ssgi temporal: descriptor set layout creation failed");
        }

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &out.set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("ssgi temporal: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, vk.device);
        if (!module.has_value()) {
            return fail("ssgi temporal: compute shader module creation failed");
        }
        VkPipelineShaderStageCreateInfo stage_info = {};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = **module;
        stage_info.pName = "main";

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = out.pipeline_layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("ssgi temporal: vkCreateComputePipelines failed");
        }
        out.resolve = vk_pipeline(pipeline, out.pipeline_layout, vk.device);
        return out;
    }

    // The world-space probe cache: injection and propagation share one pipeline (the mode is a push
    // constant), so there is one shader and one layout. 0..2 are the samplers it reads - this frame's
    // resolved GI, the depth, and the grid image it is reading - and 3 is the STORAGE 3D image it
    // writes, plus the per-cell surface offsets the filter tests visibility with, which is why two bindings
    // differ from the rest.
    std::expected<gi_probe_owned, std::string> build_gi_probe(core& vk, VkDescriptorSetLayout const scene_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        gi_probe_owned out;

        // THREE bindings: the grid being read, the grid being written, and the per-cell surface offsets the
        // propagation tests visibility with. Two more (a screen-space GI sampler and the G-buffer depth)
        // went with the screen-space injection that used them, and the shader no longer declares them - a
        // binding the layout names and the shader does not is a slot nothing can be checked against.
        std::array<VkDescriptorSetLayoutBinding, 3> bindings = {};
        for (uint32_t b = 0; b < bindings.size(); ++b) {
            bindings[b].binding = b;
            bindings[b].descriptorType = (b == 1u || b == 2u) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[b].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(vk.device, &layout_info, nullptr, &out.set_layout) != VK_SUCCESS) {
            return fail("gi probe: descriptor set layout creation failed");
        }

        // The shared scene set comes FIRST, because the tracing this pass will do needs what the tracer
        // already has: the top level structure, the material records, the texture array, the light UBO.
        std::array<VkDescriptorSetLayout, 2> const set_layouts = {scene_layout, out.set_layout};

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(set_layouts.size());
        pipeline_layout_info.pSetLayouts = set_layouts.data();
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("gi probe: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, vk.device);
        if (!module.has_value()) {
            return fail("gi probe: compute shader module creation failed");
        }
        VkPipelineShaderStageCreateInfo stage_info = {};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = **module;
        stage_info.pName = "main";

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = out.pipeline_layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("gi probe: vkCreateComputePipelines failed");
        }
        out.pass = vk_pipeline(pipeline, out.pipeline_layout, vk.device);
        return out;
    }
    std::expected<deferred_owned, std::string> build_deferred(core& vk, VkDescriptorSetLayout const scene_layout, VkDescriptorSetLayout const gbuffer_layout, uint32_t const push_constant_size, std::span<VkPipelineColorBlendAttachmentState const> const color_blend, std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        deferred_owned out;
        std::array<VkDescriptorSetLayout, 2> const set_layouts = {scene_layout, gbuffer_layout};
        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;
        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(set_layouts.size());
        pipeline_layout_info.pSetLayouts = set_layouts.data();
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("deferred: pipeline layout creation failed");
        }
        std::array<VkFormat, 1> const color_formats = {vulkan::hdr_format};
        auto pipeline_result = vulkan::make_pipeline(
            vk.device, out.pipeline_layout, std::span<VkFormat const>(color_formats), VK_FORMAT_UNDEFINED, vertex_shader_code, fragment_shader_code, VK_SAMPLE_COUNT_1_BIT, false, 0.0f, 0.0f, 0.0f, color_blend);
        if (!pipeline_result) {
            return fail(std::string(pipeline_result.error()));
        }
        out.lighting = std::move(pipeline_result).value();
        return out;
    }

    std::expected<vk_pipeline, std::string> build_fxaa(core& vk, VkPipelineLayout const post_pipeline_layout, std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        auto pipeline_result = vulkan::make_pipeline(
            vk.device, post_pipeline_layout, vk.swap_chain_image_format, VK_FORMAT_UNDEFINED, vertex_shader_code, fragment_shader_code, VK_SAMPLE_COUNT_1_BIT, false, true, 0.0f, 0.0f, 0.0f);
        if (!pipeline_result) {
            return fail(std::string(pipeline_result.error()));
        }
        return std::move(pipeline_result).value();
    }
} // namespace vulkan::pipelines