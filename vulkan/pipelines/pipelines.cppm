// module version: 0.19.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pipelines/pipelines.cppm
 * @defgroup vulkan_pipelines Per-Pass Pipeline Builders
 * @brief The engine's per-pass pipelines: one builder per pass, next to the generic builder in
 *        vulkan.core.pipeline (that one knows HOW to build a pipeline, this one knows what each pass's
 *        pipeline looks like - formats, sample counts, blend state, push-constant ranges).
 *
 * Extracted from vulkan.runtime, whose implementation had grown past 4900 lines. The builders are
 * stateless, and EVERY ONE OF THEM NOW TAKES A `VkDevice` rather than the whole core: a device is what a
 * caller that owns one has (a pass's create step gets exactly that, see vulkan.pass::pass_context), and the
 * two builders that also need the surface's format - the composite and FXAA, which write the swapchain image -
 * are handed it as a parameter. That is the whole of what they used to reach into the core for. The caller
 * passes the set layouts the pass reuses and gets the created handles back. Ownership stays with whoever asked
 * for the build (the runtime today, a pass once its three-piece has moved), which keeps the members and the
 * call ORDER - the order matters because the set layout a pass needs is owned by the pass that creates it
 * (fxaa needs the post layout, deferred the G-buffer layout, and both say so in their error messages).
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
import vulkan.bindings;      // make_set_layout: a declaration generates the layout (see bindings.cppm)
import vulkan.render_resource;

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

    /// what build_taa() creates: the pipeline layout and the resolve pipeline. The SET layout comes IN as a
    /// parameter, because it is the PASS's (see vulkan.pass.taa) - this builder is handed the one the pass
    /// generated from its declaration, which is why the caller must have run its create step first.
    export struct taa_owned {
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

    export std::expected<post_owned, std::string> build_post(VkDevice device, VkFormat swap_chain_format, uint32_t push_constant_size, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    export std::expected<gbuffer_owned, std::string> build_gbuffer_debug(VkDevice device, uint32_t push_constant_size, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    export std::expected<taa_owned, std::string> build_taa(VkDevice device, VkDescriptorSetLayout pass_set_layout, uint32_t push_constant_size, std::span<unsigned char const> vertex_shader_code,
                                                           std::span<unsigned char const> fragment_shader_code);
    export std::expected<ssgi_owned, std::string> build_ssgi(VkDevice device, VkDescriptorSetLayout scene_layout, VkDescriptorSetLayout gbuffer_layout, uint32_t push_constant_size,
                                                             std::span<unsigned char const> compute_shader_code);
    /// the spatial half of the same denoiser: same two set layouts, same shape, its own push block
    export std::expected<ssgi_owned, std::string> build_ssgi_spatial(VkDevice device, VkDescriptorSetLayout scene_layout, VkDescriptorSetLayout gbuffer_layout, uint32_t push_constant_size, std::span<unsigned char const> compute_shader_code);
    /// the glossy lobe (see shaders/ssgi_spec.comp) - the tracer's set layouts and its own push block
    export std::expected<ssgi_owned, std::string> build_ssgi_spec(VkDevice device, VkDescriptorSetLayout scene_layout, VkDescriptorSetLayout gbuffer_layout, uint32_t push_constant_size,
                                                                  std::span<unsigned char const> compute_shader_code);
    /// the ray-traced sun shadow: the same two set layouts as the GI tracer (the scene set carries the
    /// camera, the light UBO and - when the device has ray tracing - the top level structure; the
    /// G-buffer set carries the surface the ray starts from)
    export std::expected<ssgi_owned, std::string> build_rt_shadow(VkDevice device, VkDescriptorSetLayout scene_layout, VkDescriptorSetLayout gbuffer_layout, uint32_t push_constant_size, std::span<unsigned char const> compute_shader_code);
    /// the mask bake: a compute pass over the shared scene set only (the material table and the texture
    /// array), which collapses the triangles a material's alphaMode MASK cuts out and writes the expanded
    /// vertices a bottom level structure is then built from - see shaders/mask_bake.comp
    export std::expected<ssgi_owned, std::string> build_mask_bake(VkDevice device, VkDescriptorSetLayout scene_layout, uint32_t push_constant_size, std::span<unsigned char const> compute_shader_code);
    /// the compute skinning pass: the same shape again, over the scene set's per-joint matrices - see
    /// shaders/compute_skin.comp
    export std::expected<ssgi_owned, std::string> build_compute_skin(VkDevice device, VkDescriptorSetLayout scene_layout, uint32_t push_constant_size, std::span<unsigned char const> compute_shader_code);
    /// the clustered-light sort (shaders/light_cluster.comp): the shared scene set alone, and NO push constants
    /// at all - the shader reads the light UBO and writes the two cluster buffers through that set's bindings
    /// 11 and 12, which is why this builder takes no push size. It is the first compute pipeline in this module
    /// that came out of `vulkan.core` (where it was built against the core's own scene pipeline layout).
    export std::expected<ssgi_owned, std::string> build_cluster(VkDevice device, VkDescriptorSetLayout scene_layout, std::span<unsigned char const> compute_shader_code);

    /// what build_ssgi_temporal() creates: the pipeline layout and the resolve pipeline. The SET layout comes IN
    /// as a parameter now - it is generated from the denoiser's own DECLARATION (see
    /// render_resource::ssgi_temporal_io), which is what stops the seven bindings and the family's descriptor
    /// pool count from drifting apart: they were written by hand in two places once and the validation layer
    /// named the mismatch ("Trying to allocate 15 of VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER descriptors ...
    /// but this pool only has a total of 12").
    export struct ssgi_temporal_owned {
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> resolve;
    };

    export std::expected<ssgi_temporal_owned, std::string> build_ssgi_temporal(VkDevice device, VkDescriptorSetLayout pass_set_layout, uint32_t push_constant_size,
                                                                               std::span<unsigned char const> compute_shader_code);

    /// what build_gi_probe() creates: the world-space probe cache's pipeline layout and pipeline. The SET
    /// layout comes IN as a parameter rather than being built here, because it is the PASS's (see
    /// vulkan.pass.gi_probe): the pass generates it from its own declaration, and this builder is handed it -
    /// which is why the caller must have run the pass's create step first.
    export struct gi_probe_owned {
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> pass;
    };

    export std::expected<gi_probe_owned, std::string> build_gi_probe(VkDevice device, VkDescriptorSetLayout scene_layout, VkDescriptorSetLayout pass_set_layout, uint32_t push_constant_size,
                                                                     std::span<unsigned char const> compute_shader_code);

    /// the passes that reuse a layout someone else owns, so theirs comes in as a parameter
    export std::expected<vk_pipeline, std::string> build_fxaa(VkDevice device, VkFormat swap_chain_format, VkPipelineLayout post_pipeline_layout, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    export struct deferred_owned {
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> lighting;
    };

    /// the additive blend state comes in as a parameter: the helper that builds it is a local of the
    /// runtime, next to the passes whose blend modes it describes
    export std::expected<deferred_owned, std::string> build_deferred(VkDevice device, VkDescriptorSetLayout scene_layout, VkDescriptorSetLayout gbuffer_layout, uint32_t push_constant_size, std::span<VkPipelineColorBlendAttachmentState const> color_blend, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    // post: the composite chain's owner. The set layout, its pipeline layout and the two fullscreen
    // pipelines (one per color format the chain renders into) are created here; the sampler stays with
    // the runtime, which owns the descriptor sets that use it. The caller passes the size of its push
    // constant block because that structure is the runtime's (it must match post.frag).
    std::expected<post_owned, std::string> build_post(VkDevice device, VkFormat const swap_chain_format, uint32_t const push_constant_size, std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
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
        if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &out.set_layout) != VK_SUCCESS) {
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
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("post: pipeline layout creation failed");
        }

        // TWO variants, one per color format the chain renders into: the composite writes the
        // swapchain, the bright-pass prefilter and the downsample passes write the R16F bloom levels. A
        // pipeline's rendering color format must match its attachment, so one swapchain-format pipeline
        // was a validation error for the HDR passes.
        auto const make_post_variant = [&](VkFormat const color_format) -> std::expected<vk_pipeline, std::string> {
            auto pipeline_result = vulkan::make_pipeline(
                device, out.pipeline_layout, color_format, VK_FORMAT_UNDEFINED, vertex_shader_code, fragment_shader_code, VK_SAMPLE_COUNT_1_BIT, false, true, 0.0f, 0.0f, 0.0f);
            if (!pipeline_result) {
                return std::unexpected(std::string(pipeline_result.error()));
            }
            return std::move(pipeline_result).value();
        };

        auto composite_pipeline = make_post_variant(swap_chain_format);
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
    std::expected<gbuffer_owned, std::string> build_gbuffer_debug(VkDevice device, uint32_t const push_constant_size, std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
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
        // 9..12 are the world-space probe cache's four SH-2 coefficient images, sampler3D images the TRACER
        // samples for a hit the screen cannot answer (see shaders/gi_probe.comp and shaders/probe_sh.glsl).
        // They live here rather than in the probe pass's own set because the pass that READS them is the
        // tracer, which binds this set - and the probe pass's own set binds the other half of the
        // ping-pong. Four bindings rather than one because a cell holds four coefficients per channel.
        //
        // 13 and 14 are the GLOSSY lobe's own two outputs (shaders/ssgi_spec.comp): the correction it
        // traced this frame, and the reprojection of the surface that reflection found. Both are STORAGE
        // images - the lobe writes them, no descriptor of this set ever samples them - and they exist
        // because a reflection cannot be accumulated with the diffuse signal's reprojection (see the L2.3
        // motion section of docs/gi_hit_shading.md).
        //
        // 15 is the reflection's own ACCUMULATION (shaders/ssgi_temporal.comp's mode 1), which the spatial
        // filter samples and sums the diffuse one into. A sampler, unlike its two inputs - the resolve
        // writes it, this set only reads it.
        std::array<VkDescriptorSetLayoutBinding, 16> bindings = {};
        for (uint32_t b = 0; b < bindings.size(); ++b) {
            bindings[b].binding = b;
            bindings[b].descriptorType = (b == 6u || b == 8u || b == 13u || b == 14u) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
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
        if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &out.set_layout) != VK_SUCCESS) {
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
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("gbuffer debug: pipeline layout creation failed");
        }

        VkFormat const hdr_format_only = vulkan::hdr_format;
        auto pipeline_result = vulkan::make_pipeline(
            device, out.pipeline_layout, hdr_format_only, VK_FORMAT_UNDEFINED, vertex_shader_code, fragment_shader_code, VK_SAMPLE_COUNT_1_BIT, false, true, 0.0f, 0.0f, 0.0f);
        if (!pipeline_result) {
            return fail(std::string(pipeline_result.error()));
        }
        out.debug = std::move(pipeline_result).value();
        return out;
    }
    // taa: the resolve pass' owner. It writes the HDR target, so its rendering color format is hdr_format
    // (a span of one), and its sampler is the odd one out - linear magnification, nearest minification,
    // because the resolve upsamples the scene color but must not average neighbouring history texels.
    std::expected<taa_owned, std::string> build_taa(VkDevice const device, VkDescriptorSetLayout const pass_set_layout, uint32_t const push_constant_size, std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        taa_owned out;

        // THE SET LAYOUT IS THE PASS'S (see vulkan.pass.taa): four combined-image-sampler bindings generated
        // from `render_resource::taa_io`, so the layout the fragment stage sees and the declaration cannot
        // drift. The count is not cosmetic: it is also what the pass's descriptor family sizes its pool from,
        // and the validation layer has already named that pair's failure once ("Trying to allocate 15 of
        // VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER descriptors from VkDescriptorPool ..., but this pool only
        // has a total of 12 descriptors for this type" - 3 images x 5 bindings against a pool built for 3 x 4).
        if (pass_set_layout == VK_NULL_HANDLE) {
            return fail("taa: the pass has no set layout yet (its create step must run first)");
        }

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &pass_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("taa: pipeline layout creation failed");
        }

        std::array<VkFormat, 1> const color_formats = {vulkan::hdr_format};
        auto pipeline_result = vulkan::make_pipeline(
            device, out.pipeline_layout, std::span<VkFormat const>(color_formats), VK_FORMAT_UNDEFINED, vertex_shader_code, fragment_shader_code, VK_SAMPLE_COUNT_1_BIT, false, 0.0f, 0.0f, 0.0f);
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
    std::expected<ssgi_owned, std::string> build_ssgi(VkDevice const device, VkDescriptorSetLayout const scene_layout, VkDescriptorSetLayout const gbuffer_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
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
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("ssgi: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, device);
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
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("ssgi: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, out.pipeline_layout, device);
        return out;
    }

    // The mask bake (see shaders/mask_bake.comp): a compute pipeline over the shared scene set ALONE, because
    // everything it needs is there - the material table for the alpha texture's index and the cutoff, and the
    // bindless texture array to sample it. It owns no set layout, like the tracer, and it is the only compute
    // pass here whose output is not an image: it writes vertices into a buffer the acceleration structure is
    // then built from.
    std::expected<ssgi_owned, std::string> build_mask_bake(VkDevice device, VkDescriptorSetLayout const scene_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        ssgi_owned out;

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &scene_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("mask bake: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, device);
        if (!module.has_value()) {
            return fail("mask bake: compute shader module creation failed");
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
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("mask bake: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, out.pipeline_layout, device);
        return out;
    }

    // The compute skinning pass (see shaders/compute_skin.comp): the same shape as the mask bake above and
    // for the same reason - it reads only the shared scene set (here the per-joint matrices at binding 9)
    // and owns no set layout of its own.
    std::expected<ssgi_owned, std::string> build_compute_skin(VkDevice device, VkDescriptorSetLayout const scene_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        ssgi_owned out;

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &scene_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("compute skin: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, device);
        if (!module.has_value()) {
            return fail("compute skin: compute shader module creation failed");
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
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("compute skin: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, out.pipeline_layout, device);
        return out;
    }

    // The clustered-light sort: the shared scene set alone and NO push range, because the shader reads the
    // light UBO and the cluster buffers through that set's own bindings (7, 11 and 12) and takes no constants.
    // The pipeline layout is still this pass's OWN - built here from the set layout its owner hands over -
    // rather than the core's scene pipeline layout, which is what the old `core::make_cluster_pipeline` used:
    // the two are equivalent for this pipeline (same set layout, and the shader uses no push constants), and
    // owning it is what lets the pass release it.
    std::expected<ssgi_owned, std::string> build_cluster(VkDevice const device, VkDescriptorSetLayout const scene_layout, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        ssgi_owned out;

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &scene_layout;
        pipeline_layout_info.pushConstantRangeCount = 0;
        pipeline_layout_info.pPushConstantRanges = nullptr;
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("cluster: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, device);
        if (!module.has_value()) {
            return fail("cluster: compute shader module creation failed");
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
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("cluster: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, out.pipeline_layout, device);
        return out;
    }

    // The ray-traced sun shadow: a COMPUTE pipeline over the same two set layouts the GI tracer uses,
    // because its inputs are in the same places (the camera block and the light UBO in the scene set,
    // the stored surface in the G-buffer set) plus the top level structure, which lives in the scene set
    // as binding 16 when the device has ray tracing. Nothing new is created here beyond the layout.
    std::expected<ssgi_owned, std::string> build_rt_shadow(VkDevice device, VkDescriptorSetLayout const scene_layout, VkDescriptorSetLayout const gbuffer_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
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
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("rt shadow: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, device);
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
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("rt shadow: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, out.pipeline_layout, device);
        return out;
    }
    // The GI spatial filter: the same two set layouts the tracer binds (the shared scene set and the
    // G-buffer set, which carries the normal, the depth, the accumulated image it reads and the
    // filtered image it writes), so only the pipeline layout and the push block are new.
    std::expected<ssgi_owned, std::string> build_ssgi_spatial(VkDevice device, VkDescriptorSetLayout const scene_layout, VkDescriptorSetLayout const gbuffer_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
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
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("ssgi spatial: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, device);
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
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("ssgi spatial: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, out.pipeline_layout, device);
        return out;
    }

    // The glossy lobe (see shaders/ssgi_spec.comp): the tracer's two set layouts again and a push block of
    // its own. It is a separate pass rather than a branch in the tracer for two reasons - the tracer's push
    // block is exactly 128 bytes (the smallest range Vulkan guarantees) and has no lane left for a ray
    // count, and a pass that is not recorded cannot perturb the frame at all, which is a stronger statement
    // than "a branch that arithmetically cancels".
    std::expected<ssgi_owned, std::string> build_ssgi_spec(VkDevice const device, VkDescriptorSetLayout const scene_layout, VkDescriptorSetLayout const gbuffer_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
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
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("ssgi spec: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, device);
        if (!module.has_value()) {
            return fail("ssgi spec: compute shader module creation failed");
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
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("ssgi spec: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, out.pipeline_layout, device);
        return out;
    }

    // The GI temporal resolve: its own set layout, because it groups four things no other pass puts
    // together (the raw trace, the accumulated history, the motion vectors and the depth). It binds
    // no scene set: the push block carries the two projection terms its depth guard needs.
    std::expected<ssgi_temporal_owned, std::string> build_ssgi_temporal(VkDevice const device, VkDescriptorSetLayout const pass_set_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        ssgi_temporal_owned out;

        // THE SET LAYOUT IS THE DECLARATION'S (render_resource::ssgi_temporal_io, generated by the caller with
        // bindings::make_set_layout): seven bindings, one of which is the storage image the resolve writes.
        // Handing it in is what makes the LAYOUT and the family's pool count the same fact.
        if (pass_set_layout == VK_NULL_HANDLE) {
            return fail("ssgi temporal: the declaration produced no set layout");
        }

        VkPushConstantRange push_range = {};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &pass_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("ssgi temporal: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, device);
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
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("ssgi temporal: vkCreateComputePipelines failed");
        }
        out.resolve = vk_pipeline(pipeline, out.pipeline_layout, device);
        return out;
    }

    // The world-space probe cache: injection and propagation share one pipeline (the mode is a push
    // constant), so there is one shader and one layout. 0..2 are the samplers it reads - this frame's
    // resolved GI, the depth, and the grid image it is reading - and 3 is the STORAGE 3D image it
    // writes, plus the per-cell surface offsets the filter tests visibility with, which is why two bindings
    // differ from the rest.
    std::expected<gi_probe_owned, std::string> build_gi_probe(VkDevice const device, VkDescriptorSetLayout const scene_layout, VkDescriptorSetLayout const pass_set_layout, uint32_t const push_constant_size,
                                                              std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        gi_probe_owned out;

        // THE SET LAYOUT IS THE PASS'S. Nine bindings - the four SH-2 coefficient images being READ (0..3),
        // the four being WRITTEN (4..7), and the per-cell surface offsets the propagation tests visibility
        // with (8) - generated from `vulkan.render_resource::gi_probe_io` by the pass that declares them
        // (`vulkan.pass.gi_probe`), which is also where the descriptor WRITES come from. One fact, one
        // source: that is what removed the pair of hand-written halves that drifted twice in this project's
        // history (a pool sized for four descriptors per set while the layout asked for five, and a binding
        // whose type changed without its writer noticing).
        //
        // THAT THE GENERATED LAYOUT IS THE ONE THIS FUNCTION USED TO CARRY IS ASSERTED BY THE CAPTURE GATE,
        // not by a comment: `sponza_gi` is the scenario that runs with the probe cache ON, so its frame is
        // compared byte for byte across the change.
        if (pass_set_layout == VK_NULL_HANDLE) {
            return fail("gi probe: the pass has no set layout yet (its create step must run first)");
        }

        // The shared scene set comes FIRST, because the tracing this pass will do needs what the tracer
        // already has: the top level structure, the material records, the texture array, the light UBO.
        std::array<VkDescriptorSetLayout, 2> const set_layouts = {scene_layout, pass_set_layout};

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
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("gi probe: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, device);
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
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("gi probe: vkCreateComputePipelines failed");
        }
        out.pass = vk_pipeline(pipeline, out.pipeline_layout, device);
        return out;
    }
    std::expected<deferred_owned, std::string> build_deferred(VkDevice device, VkDescriptorSetLayout const scene_layout, VkDescriptorSetLayout const gbuffer_layout, uint32_t const push_constant_size, std::span<VkPipelineColorBlendAttachmentState const> const color_blend, std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
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
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out.pipeline_layout) != VK_SUCCESS) {
            return fail("deferred: pipeline layout creation failed");
        }
        std::array<VkFormat, 1> const color_formats = {vulkan::hdr_format};
        auto pipeline_result = vulkan::make_pipeline(
            device, out.pipeline_layout, std::span<VkFormat const>(color_formats), VK_FORMAT_UNDEFINED, vertex_shader_code, fragment_shader_code, VK_SAMPLE_COUNT_1_BIT, false, 0.0f, 0.0f, 0.0f, color_blend);
        if (!pipeline_result) {
            return fail(std::string(pipeline_result.error()));
        }
        out.lighting = std::move(pipeline_result).value();
        return out;
    }

    std::expected<vk_pipeline, std::string> build_fxaa(VkDevice device, VkFormat const swap_chain_format, VkPipelineLayout const post_pipeline_layout, std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        auto pipeline_result = vulkan::make_pipeline(
            device, post_pipeline_layout, swap_chain_format, VK_FORMAT_UNDEFINED, vertex_shader_code, fragment_shader_code, VK_SAMPLE_COUNT_1_BIT, false, true, 0.0f, 0.0f, 0.0f);
        if (!pipeline_result) {
            return fail(std::string(pipeline_result.error()));
        }
        return std::move(pipeline_result).value();
    }
} // namespace vulkan::pipelines
