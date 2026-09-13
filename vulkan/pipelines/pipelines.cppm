// module version: 0.4.0  (independent of the app version in CMakeLists project(VERSION))

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

    /// what build_ssgi_temporal() creates: the denoiser's set layout (it owns one - its inputs are
    /// the trace, the history, the motion vectors and the depth, which no other pass groups together)
    export struct ssgi_temporal_owned {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> resolve;
    };

    export std::expected<ssgi_temporal_owned, std::string> build_ssgi_temporal(core& vk, uint32_t push_constant_size, std::span<unsigned char const> compute_shader_code);

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
        // (the FXAA pass, which shares this layout), binding 6 = the screen-space GI image, which
        // only the composite reads - so a prefilter or downsample set points it at the HDR view too.
        std::array<VkDescriptorSetLayoutBinding, 7> bindings = {};
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
        // 5 and 6 belong to the screen-space GI tracer: the direct-radiance image it samples at a
        // hit, and the half-resolution image it writes. 6 is a STORAGE image rather than a sampler
        // because a compute pass writes a storage image, and because the tracer runs at half the
        // composite's resolution - the two ends of it are different kinds of thing.
        std::array<VkDescriptorSetLayoutBinding, 7> bindings = {};
        for (uint32_t b = 0; b < bindings.size(); ++b) {
            bindings[b].binding = b;
            bindings[b].descriptorType = b == 6u ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[b].descriptorCount = 1;
            // EVERY binding lists COMPUTE as well as FRAGMENT. The layout is shared by three consumers
            // and only the shaders know which binding each of them uses: the lighting stage and the
            // debug view are fragment stages, while the GI tracer is a COMPUTE stage that reads the
            // stored surface (albedo, normal and depth) directly. A binding a shader does not
            // statically use needs no descriptor, but one it DOES use has to name the stage here.
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