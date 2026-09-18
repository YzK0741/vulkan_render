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
    /**
     * @brief the POST set's layout, on its own
     *
     * SEPARATE FROM `build_post` BECAUSE THE LAYOUT IS NOT THE PASS'S: the nine bindings describe how the RENDERER
     * writes the post sets (HDR, the four bloom levels, the LDR image, the filtered GI, the G-buffer's depth and
     * normal), and the runtime owns that family - so it creates the layout, hands it to the passes that need a
     * pipeline layout around it (`pass_context::shared_set_layout(owner, 2)`) and uses the same object for the
     * family it writes. One layout, one owner.
     */
    export std::expected<VkDescriptorSetLayout, std::string> make_post_set_layout(VkDevice device);

    /// what build_post() creates: the pipeline layout (around the post set layout it is handed) and the two composites
    export struct post_owned {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE; // the one it was handed, not one it made
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> composite; // tonemap + bloom sum, writes the swapchain
        std::optional<vk_pipeline> hdr;       // the same pass writing an HDR target instead (FXAA on)
    };

    /**
     * @brief the G-BUFFER set's layout, on its own
     *
     * SEPARATE FROM `build_gbuffer_debug` FOR THE SAME REASON `make_post_set_layout` is separate from `build_post`:
     * the sixteen bindings describe how the RENDERER writes the G-buffer sets (the stored surface, the chain's
     * images, a removed pass's coefficient volumes, the lobe's two outputs and the reflection's accumulation), and
     * the runtime owns that family - so the runtime creates the layout, hands it to every pass that binds the set
     * (`pass_context::shared_set_layout(owner, 1)`) and uses the same object for the family it writes.
     */
    export std::expected<VkDescriptorSetLayout, std::string> make_gbuffer_set_layout(VkDevice device);

    /// @brief what build_gbuffer_debug() creates: the pipeline layout and the debug view's pipeline
    export struct gbuffer_owned {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE; // the one it was handed, not one it made
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

    /// what a compute builder that owns its PIPELINE LAYOUT returns: that layout plus the COMPUTE pipeline,
    /// and no set layout of its own - the sets it binds belong to the scene and the pass.
    export struct compute_pipeline_owned {
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> trace;
    };

    export std::expected<post_owned, std::string> build_post(VkDevice device, VkDescriptorSetLayout post_set_layout, VkFormat swap_chain_format, uint32_t push_constant_size,
                                                             std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    export std::expected<gbuffer_owned, std::string> build_gbuffer_debug(VkDevice device, VkDescriptorSetLayout gbuffer_set_layout, uint32_t push_constant_size, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    export std::expected<taa_owned, std::string> build_taa(VkDevice device, VkDescriptorSetLayout pass_set_layout, uint32_t push_constant_size, std::span<unsigned char const> vertex_shader_code,
                                                           std::span<unsigned char const> fragment_shader_code);

    /// the ray-traced sun shadow: the same two set layouts the traced compute passes use (the scene set carries the
    /// camera, the light UBO and - when the device has ray tracing - the top level structure; the
    /// G-buffer set carries the surface the ray starts from)
    export std::expected<compute_pipeline_owned, std::string> build_two_set_compute(VkDevice device, VkDescriptorSetLayout scene_layout, VkDescriptorSetLayout gbuffer_layout, uint32_t push_constant_size, std::span<unsigned char const> compute_shader_code);
    /// the stochastic punctual lighting trace (shaders/megalights_trace.comp): the same two set layouts again,
    /// with the estimator's own push block - see docs/megalights.md
    export std::expected<compute_pipeline_owned, std::string> build_megalights_trace(VkDevice device, VkDescriptorSetLayout scene_layout, VkDescriptorSetLayout gbuffer_layout, uint32_t push_constant_size,
                                                                                     std::span<unsigned char const> compute_shader_code);
    /// the stochastic chain's temporal resolve (shaders/megalights_temporal.comp): two set layouts again - the
    /// shared G-buffer set at index 0 and the pass's own at index 1 - with the accumulation's own push block
    export std::expected<compute_pipeline_owned, std::string> build_megalights_temporal(VkDevice device, VkDescriptorSetLayout gbuffer_layout, VkDescriptorSetLayout pass_set_layout, uint32_t push_constant_size,
                                                                                        std::span<unsigned char const> compute_shader_code);
    /// the mask bake: a compute pass over the shared scene set only (the material table and the texture
    /// array), which collapses the triangles a material's alphaMode MASK cuts out and writes the expanded
    /// vertices a bottom level structure is then built from - see shaders/mask_bake.comp
    export std::expected<compute_pipeline_owned, std::string> build_mask_bake(VkDevice device, VkDescriptorSetLayout scene_layout, uint32_t push_constant_size, std::span<unsigned char const> compute_shader_code);
    /// the compute skinning pass: the same shape again, over the scene set's per-joint matrices - see
    /// shaders/compute_skin.comp
    export std::expected<compute_pipeline_owned, std::string> build_compute_skin(VkDevice device, VkDescriptorSetLayout scene_layout, uint32_t push_constant_size, std::span<unsigned char const> compute_shader_code);
    /// the clustered-light sort (shaders/light_cluster.comp): the shared scene set alone, and NO push constants
    /// at all - the shader reads the light UBO and writes the two cluster buffers through that set's bindings
    /// 11 and 12, which is why this builder takes no push size. It is the first compute pipeline in this module
    /// that came out of `vulkan.core` (where it was built against the core's own scene pipeline layout).
    export std::expected<compute_pipeline_owned, std::string> build_cluster(VkDevice device, std::span<unsigned char const> compute_shader_code);

    /**
     * @brief the HEAP-NATIVE probe's pipeline: the first one in this renderer created the heap way
     * @param device the logical device
     * @param compute_shader_code the probe's SPIR-V (see shaders/heap_probe.comp)
     * @return the pipeline, or the reason it could not be created
     * @note NO SET LAYOUT AND NO PIPELINE LAYOUT, which is not a simplification but the flag's requirement:
     *       "the pipeline layout must be NULL and shader resources will be sourced from a descriptor heap". The
     *       probe's parameters therefore reach the shader through vkCmdPushDataEXT (see descriptor_heap::push_data)
     *       and not through vkCmdPushConstants, which needs a layout to push to.
     */
    export std::expected<compute_pipeline_owned, std::string> build_heap_probe(VkDevice device, std::span<unsigned char const> compute_shader_code);

    /// the probe's target: one size for the image, the viewport, the scissor and the readback, so a mismatch
    /// between them is impossible rather than merely unlikely
    export inline constexpr uint32_t heap_probe_extent = 4u;

    /**
     * @brief the GRAPHICS half of the heap-native probe: a heap-flagged, layout-less pipeline over two stages
     * @param device the logical device
     * @param colour_format the format the probe renders into (dynamic rendering, like every pass here)
     * @param vertex_code / @param fragment_code the probe's SPIR-V (see shaders/heap_probe.vert / .frag)
     * @return the pipeline, or the reason it could not be created
     * @note no vertex input, no blend and a static viewport: the probe's subject is the FRAGMENT stage reading the
     *       heap through a graphics pipeline at all, and every one of those would be a second thing that could be
     *       wrong. The flag and the null layout are the rule the compute probe established.
     */
    export std::expected<vk_pipeline, std::string> build_heap_probe_graphics(VkDevice device, VkFormat colour_format, std::span<unsigned char const> vertex_code, std::span<unsigned char const> fragment_code);

    /// what build_resolve_pipeline() creates: the pipeline layout and the resolve pipeline. The SET layout comes IN
    /// as a parameter now - it is generated from the denoiser's own DECLARATION (see
    /// its own DECLARATION), which is what stops the bindings and the family's descriptor pool count from
    /// drifting apart: written by hand in two places once, the validation layer named the mismatch.
    export struct resolve_pipeline_owned {
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> resolve;
    };

    export std::expected<resolve_pipeline_owned, std::string> build_resolve_pipeline(VkDevice device, VkDescriptorSetLayout pass_set_layout, uint32_t push_constant_size,
                                                                                     std::span<unsigned char const> compute_shader_code);

    /// the passes that reuse a layout someone else owns, so theirs comes in as a parameter
    export std::expected<vk_pipeline, std::string> build_fxaa(VkDevice device, VkFormat swap_chain_format, VkPipelineLayout post_pipeline_layout, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    /**
     * @brief the SHADOW pass's depth-only pipeline, built against the layout the caller hands it
     *
     * WHY IT TAKES A DEVICE AND A Layout rather than being `core::make_depth_pipeline`: that method is a member of
     * the `core` object (it reads `this->device`, `this->scene_pipeline_layout` and the caller's depth format), and
     * a PASS reaches neither - the device, the layout and the format all arrive through `pass_context`. The three
     * BIAS factors are parameters for the same reason the depth format is: they are the pipeline's, not the
     * device's, and the depth pass is the one pipeline in this renderer created with slope-scaled bias.
     */
    export std::expected<vk_pipeline, std::string> build_shadow(VkDevice device, VkPipelineLayout pipeline_layout, VkFormat depth_format, float depth_bias_constant_factor, float depth_bias_slope_factor,
                                                                float depth_bias_clamp, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    /**
     * @brief what the FXAA pass's own create step needs: ITS pipeline layout, created from the SET layout it is
     *        handed, and the anti-aliasing pipeline
     *
     * WHY A SECOND ENTRY POINT rather than the one above: a pass may not reach another pass's pipeline layout, and
     * the post chain's belongs to the composite. `build_taa` is the same shape (the pass's set layout comes in and
     * the layout is created around it), and the descriptor SET is unaffected either way - it is the post family's
     * set 4, allocated from the composite's layout, and two layouts created from identically-defined set layouts
     * are compatible for that set.
     */
    export struct fxaa_owned {
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> antialias;
    };
    export std::expected<fxaa_owned, std::string> build_fxaa_owned(VkDevice device, VkFormat swap_chain_format, VkDescriptorSetLayout pass_set_layout, uint32_t push_constant_size,
                                                                   std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
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
    std::expected<VkDescriptorSetLayout, std::string> make_post_set_layout(VkDevice const device) {
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
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &layout) != VK_SUCCESS) {
            return std::unexpected("post: descriptor set layout creation failed");
        }
        return layout;
    }

    std::expected<post_owned, std::string> build_post(VkDevice const device, VkDescriptorSetLayout const post_set_layout, VkFormat const swap_chain_format,
                                                      uint32_t const push_constant_size, std::span<unsigned char const> const vertex_shader_code,
                                                      std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        post_owned out;

        // THE LAYOUT IS HANDED IN, not made here: it belongs to whoever WRITES the sets (the runtime owns the post
        // family - see make_post_set_layout).
        if (post_set_layout == VK_NULL_HANDLE) {
            return fail("post: the post set layout was not provided");
        }
        out.set_layout = post_set_layout;

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
    // gbuffer debug: a pass that binds the G-buffer set, whose LAYOUT the runtime owns (see
    // make_gbuffer_set_layout) because the runtime writes every one of its sets. The nearest sampler the debug view
    // needs stays with the runtime, next to the descriptor sets that use it.
    std::expected<VkDescriptorSetLayout, std::string> make_gbuffer_set_layout(VkDevice const device) {
        // albedo, normal, material, depth, velocity: the four the debug view displays plus the
        // motion-vector target. The deferred lighting stage binds this SAME layout as its set 1
        // and its shader declares only the first four, which is legal - a binding a shader does
        // not statically use does not need a descriptor written.
        //
        // 5 is the direct-radiance image the lighting stage samples at a hit; 6 and 7 are the STOCHASTIC
        // PUNCTUAL LIGHTING chain (docs/megalights.md): 6 is the storage image its trace writes and 7 is the
        // sampler the lighting stage adds the temporal resolve's output through - the two ends of one
        // half-resolution signal. Together with albedo, normal, material, depth and velocity that is EIGHT
        // bindings. A set that once carried sixteen is back to eight, which is why the
        // lighting chain's pair sits at 6/7 rather than 16/17: a descriptor set layout's binding numbers need
        // not be contiguous, but the owner writes them by index, so a gap is not free.
        std::array<VkDescriptorSetLayoutBinding, 8> bindings = {};
        for (uint32_t b = 0; b < bindings.size(); ++b) {
            bindings[b].binding = b;
            bindings[b].descriptorType = (b == 6u) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[b].descriptorCount = 1;
            // EVERY binding lists COMPUTE as well as FRAGMENT. The layout is shared by several consumers
            // and only the shaders know which binding each of them uses: the lighting stage and the
            // debug view are fragment stages, while the lighting chain's resolve is a COMPUTE stage that
            // reads the stored surface (albedo, normal and depth) directly. A binding a shader does not
            // statically use needs no descriptor, but one it DOES use has to name the stage here.
            // FRAGMENT, COMPUTE *and the three ray-tracing stages*: the shadow pass traces through a
            // ray-tracing pipeline whose raygen samples this set, and a binding a shader statically uses has to
            // name that shader's stage here (see the scene set layout's note).
            bindings[b].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
            bindings[b].pImmutableSamplers = nullptr;
        }
        VkDescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &layout) != VK_SUCCESS) {
            return std::unexpected("gbuffer debug: descriptor set layout creation failed");
        }
        return layout;
    }

    std::expected<gbuffer_owned, std::string> build_gbuffer_debug(VkDevice const device, VkDescriptorSetLayout const gbuffer_set_layout, uint32_t const push_constant_size,
                                                                  std::span<unsigned char const> const vertex_shader_code,
                                                                  std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        gbuffer_owned out;

        // THE LAYOUT IS HANDED IN (see make_gbuffer_set_layout): its eighteen bindings are how the OWNER fills the
        // sets this pass and the GI passes bind.
        if (gbuffer_set_layout == VK_NULL_HANDLE) {
            return fail("gbuffer debug: the G-buffer set layout was not provided");
        }
        out.set_layout = gbuffer_set_layout;

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

    // A traced COMPUTE pipeline over TWO set layouts rather than one (the shared scene set and the
    // G-buffer set) because that is where its inputs already are: the camera UBO, the stored surface and
    // the images it writes. Creating a third layout for one pass would duplicate descriptor writes to gain
    // nothing.

    // The mask bake (see shaders/mask_bake.comp): a compute pipeline over the shared scene set ALONE, because
    // everything it needs is there - the material table for the alpha texture's index and the cutoff, and the
    // bindless texture array to sample it. It owns no set layout, like every traced compute pass, and it is the only compute
    // pass here whose output is not an image: it writes vertices into a buffer the acceleration structure is
    // then built from.
    std::expected<compute_pipeline_owned, std::string> build_mask_bake(VkDevice device, VkDescriptorSetLayout const scene_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        compute_pipeline_owned out;

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

        // THE HEAP FLAG IS NOT OPTIONAL WHEN THE LAYOUT IS NULL: validation's rule is "both or neither", and it
        // says so exactly - "pCreateInfos[0].flags (VkPipelineCreateFlags2(0)) does not include
        // VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT while layout is VK_NULL_HANDLE"
        // (VUID-VkComputePipelineCreateInfo-None-11367), measured on the first run of the migrated renderer. The
        // flag is a flags2 bit, past the 32-bit `flags` field, so it reaches a classic create call through
        // VkPipelineCreateFlags2CreateInfo - the shape the graphics path and the two probes already use.
        VkPipelineCreateFlags2CreateInfo const heap_flags = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
        };
        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.pNext = &heap_flags;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = VK_NULL_HANDLE; // heap-native stages: a layout would contradict them (see docs)

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
    std::expected<compute_pipeline_owned, std::string> build_compute_skin(VkDevice device, VkDescriptorSetLayout const scene_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        compute_pipeline_owned out;

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

        // THE HEAP FLAG IS NOT OPTIONAL WHEN THE LAYOUT IS NULL: validation's rule is "both or neither", and it
        // says so exactly - "pCreateInfos[0].flags (VkPipelineCreateFlags2(0)) does not include
        // VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT while layout is VK_NULL_HANDLE"
        // (VUID-VkComputePipelineCreateInfo-None-11367), measured on the first run of the migrated renderer. The
        // flag is a flags2 bit, past the 32-bit `flags` field, so it reaches a classic create call through
        // VkPipelineCreateFlags2CreateInfo - the shape the graphics path and the two probes already use.
        VkPipelineCreateFlags2CreateInfo const heap_flags = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
        };
        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.pNext = &heap_flags;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = VK_NULL_HANDLE; // heap-native stages: a layout would contradict them (see docs)

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("compute skin: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, out.pipeline_layout, device);
        return out;
    }

    // The clustered-light sort: NO set, NO layout and NO push range. Every stage of the frame is heap-native
    // (see docs/descriptor_heap_handover.md), so this pipeline is created with VK_NULL_HANDLE and the heap flag;
    // the shader reads the light UBO and the cluster buffers out of the scene block by slot, and the slot itself
    // travels in the stage push block (shaders/heap_slots.glsl). Owning the pipeline is all that is left to own.
    std::expected<compute_pipeline_owned, std::string> build_cluster(VkDevice const device, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        compute_pipeline_owned out;

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, device);
        if (!module.has_value()) {
            return fail("cluster: compute shader module creation failed");
        }
        VkPipelineShaderStageCreateInfo stage_info = {};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = **module;
        stage_info.pName = "main";

        // THE HEAP FLAG IS WHAT MAKES A NULL LAYOUT LEGAL, and it is set unconditionally: this stage is
        // heap-native, so its layout is VK_NULL_HANDLE and the flag is what validation demands for that.
        VkPipelineCreateFlags2CreateInfo pipeline_flags = {};
        pipeline_flags.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
        pipeline_flags.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.pNext = &pipeline_flags; // the heap flag rides in flags2 (its bit is past the 32-bit `flags` field)
        pipeline_info.stage = stage_info;
        pipeline_info.layout = VK_NULL_HANDLE; // heap-native: a layout would contradict the flag (VUID ...-11367)

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("cluster: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, out.pipeline_layout, device);
        return out;
    }

    std::expected<compute_pipeline_owned, std::string> build_heap_probe(VkDevice const device, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        compute_pipeline_owned out;

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, device);
        if (!module.has_value()) {
            return fail("heap probe: compute shader module creation failed");
        }
        VkPipelineShaderStageCreateInfo stage_info = {};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = **module;
        stage_info.pName = "main";

        // The heap flag is a flags2 bit (0x1000000000, past the 32-bit `flags` field), so it arrives through
        // VkPipelineCreateFlags2CreateInfo - and it REQUIRES layout = VK_NULL_HANDLE, which is the whole point:
        // with the flag set the pipeline layout is not read at all, and the shader's resources come from the heap.
        VkPipelineCreateFlags2CreateInfo pipeline_flags = {};
        pipeline_flags.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
        pipeline_flags.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.pNext = &pipeline_flags;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = VK_NULL_HANDLE;

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("heap probe: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, VK_NULL_HANDLE, device);
        return out;
    }

    // The SHARED two-set compute builder: the shape several traced passes have in common - the camera block and
    // the light UBO in the scene set (plus the top level structure at binding 16 when the device has ray
    // tracing), the stored surface in the G-buffer set - so the caller's only variable is the push block size.
    // (Its old name, build_rt_shadow, is gone with the ray-query shadow pass: the shadow traces through a real
    // ray-tracing PIPELINE now, which is a different builder below.)
    std::expected<vk_pipeline, std::string> build_heap_probe_graphics(VkDevice const device, VkFormat const colour_format, std::span<unsigned char const> const vertex_code, std::span<unsigned char const> const fragment_code) {
        using fail = std::unexpected<std::string>;
        auto const vertex_module = make_shader_module(vertex_code, device);
        if (!vertex_module.has_value()) {
            return fail("heap probe (graphics): vertex shader module creation failed");
        }
        auto const fragment_module = make_shader_module(fragment_code, device);
        if (!fragment_module.has_value()) {
            return fail("heap probe (graphics): fragment shader module creation failed");
        }
        std::array<VkPipelineShaderStageCreateInfo, 2> stages = {};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = **vertex_module;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = **fragment_module;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo const vertex_input = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                                                                   .pNext = nullptr,
                                                                   .flags = 0,
                                                                   .vertexBindingDescriptionCount = 0,
                                                                   .pVertexBindingDescriptions = nullptr,
                                                                   .vertexAttributeDescriptionCount = 0,
                                                                   .pVertexAttributeDescriptions = nullptr};
        VkPipelineInputAssemblyStateCreateInfo const input_assembly = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                                       .pNext = nullptr,
                                                                       .flags = 0,
                                                                       .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                                                       .primitiveRestartEnable = VK_FALSE};
        VkViewport const viewport = {.x = 0.0f, .y = 0.0f, .width = static_cast<float>(heap_probe_extent), .height = static_cast<float>(heap_probe_extent), .minDepth = 0.0f, .maxDepth = 1.0f};
        VkRect2D const scissor = {.offset = {0, 0}, .extent = {heap_probe_extent, heap_probe_extent}};
        VkPipelineViewportStateCreateInfo const viewport_state = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                                                  .pNext = nullptr,
                                                                  .flags = 0,
                                                                  .viewportCount = 1,
                                                                  .pViewports = &viewport,
                                                                  .scissorCount = 1,
                                                                  .pScissors = &scissor};
        VkPipelineRasterizationStateCreateInfo const rasterization = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                                      .pNext = nullptr,
                                                                      .flags = 0,
                                                                      .depthClampEnable = VK_FALSE,
                                                                      .rasterizerDiscardEnable = VK_FALSE,
                                                                      .polygonMode = VK_POLYGON_MODE_FILL,
                                                                      .cullMode = VK_CULL_MODE_NONE,
                                                                      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                                                                      .depthBiasEnable = VK_FALSE,
                                                                      .depthBiasConstantFactor = 0.0f,
                                                                      .depthBiasClamp = 0.0f,
                                                                      .depthBiasSlopeFactor = 0.0f,
                                                                      .lineWidth = 1.0f};
        VkPipelineMultisampleStateCreateInfo const multisample = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                                  .pNext = nullptr,
                                                                  .flags = 0,
                                                                  .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                                                                  .sampleShadingEnable = VK_FALSE,
                                                                  .minSampleShading = 1.0f,
                                                                  .pSampleMask = nullptr,
                                                                  .alphaToCoverageEnable = VK_FALSE,
                                                                  .alphaToOneEnable = VK_FALSE};
        VkPipelineColorBlendAttachmentState const blend_attachment = {.blendEnable = VK_FALSE,
                                                                      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
                                                                      .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
                                                                      .colorBlendOp = VK_BLEND_OP_ADD,
                                                                      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                                                                      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                                                                      .alphaBlendOp = VK_BLEND_OP_ADD,
                                                                      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
        VkPipelineColorBlendStateCreateInfo const colour_blend = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                                  .pNext = nullptr,
                                                                  .flags = 0,
                                                                  .logicOpEnable = VK_FALSE,
                                                                  .logicOp = VK_LOGIC_OP_COPY,
                                                                  .attachmentCount = 1,
                                                                  .pAttachments = &blend_attachment,
                                                                  .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}};
        VkPipelineRenderingCreateInfo rendering = {};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &colour_format;
        // The heap flag is a flags2 bit and the rendering struct hangs off it, so both travel in one pNext chain.
        VkPipelineCreateFlags2CreateInfo flags = {};
        flags.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
        flags.pNext = &rendering;
        flags.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

        VkGraphicsPipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.pNext = &flags;
        pipeline_info.stageCount = static_cast<uint32_t>(stages.size());
        pipeline_info.pStages = stages.data();
        pipeline_info.pVertexInputState = &vertex_input;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterization;
        pipeline_info.pMultisampleState = &multisample;
        pipeline_info.pColorBlendState = &colour_blend;
        pipeline_info.layout = VK_NULL_HANDLE; // required by the flag, exactly as for the compute probe

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("heap probe (graphics): vkCreateGraphicsPipelines failed");
        }
        return vk_pipeline(pipeline, VK_NULL_HANDLE, device);
    }

    std::expected<compute_pipeline_owned, std::string> build_two_set_compute(VkDevice device, VkDescriptorSetLayout const scene_layout, VkDescriptorSetLayout const gbuffer_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        compute_pipeline_owned out;

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

        // THE HEAP FLAG IS NOT OPTIONAL WHEN THE LAYOUT IS NULL: validation's rule is "both or neither", and it
        // says so exactly - "pCreateInfos[0].flags (VkPipelineCreateFlags2(0)) does not include
        // VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT while layout is VK_NULL_HANDLE"
        // (VUID-VkComputePipelineCreateInfo-None-11367), measured on the first run of the migrated renderer. The
        // flag is a flags2 bit, past the 32-bit `flags` field, so it reaches a classic create call through
        // VkPipelineCreateFlags2CreateInfo - the shape the graphics path and the two probes already use.
        VkPipelineCreateFlags2CreateInfo const heap_flags = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
        };
        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.pNext = &heap_flags;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = VK_NULL_HANDLE; // heap-native stages: a layout would contradict them (see docs)

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("rt shadow: vkCreateComputePipelines failed");
        }
        out.trace = vk_pipeline(pipeline, out.pipeline_layout, device);
        return out;
    }
    /**
     * @brief what the ray-tracing shadow pipeline builder returns
     *
     * `group_count` is the number of shader groups the pipeline created, in the order the SBT must follow
     * (raygen, miss, hit): the shader binding table itself is the CALLER's, because the group handles it is
     * filled with are per-pipeline data and its regions have to outlive this call.
     */
    export struct ray_tracing_pipeline_owned {
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> pipeline;
        uint32_t group_count = 0;
    };

    /**
     * @brief the ray-traced sun shadow's PIPELINE: one raygen, one miss and one triangles hit group
     *
     * WHY A PIPELINE AND NOT AN INLINE RAY QUERY, which is what this pass used to run: a ray query has no
     * any-hit stage, so an alphaMode MASK surface is SOLID to the ray. A traced ray can run an any-hit shader
     * for exactly that test, and it can consult an opacity micromap, which is the hardware form of the same
     * question (per-microtriangle opacity, with the any-hit shader as the fallback for its 'unknown' states).
     * Both of those live in the hit group this creates.
     *
     * Recursion depth is 1: the shadow ray answers a yes/no question and the traversal terminates on the first
     * hit (`gl_RayFlagsTerminateOnFirstHitEXT` in the raygen), so there is nothing for a second level to do.
     */
    export std::expected<ray_tracing_pipeline_owned, std::string> build_rt_shadow_ray_tracing(VkDevice device, VkDescriptorSetLayout scene_layout, VkDescriptorSetLayout gbuffer_layout,
                                                                                              uint32_t push_constant_size, std::span<unsigned char const> raygen_code,
                                                                                              std::span<unsigned char const> closest_hit_code, std::span<unsigned char const> miss_code,
                                                                                              std::span<unsigned char const> any_hit_code) {
        using fail = std::unexpected<std::string>;
        ray_tracing_pipeline_owned out;

        VkPushConstantRange push_range = {};
        // The any-hit stage is in this mask even before it reads a push constant: the range is what the SHADER
        // may read, and an alpha test that needs the material's index or the scene's alpha cutoff finds it there
        // rather than needing this layout rebuilt (a pipeline layout is not something to churn per feature).
        push_range.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
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

        std::optional<vk_shader_module> const raygen = make_shader_module(raygen_code, device);
        std::optional<vk_shader_module> const closest_hit = make_shader_module(closest_hit_code, device);
        std::optional<vk_shader_module> const miss = make_shader_module(miss_code, device);
        std::optional<vk_shader_module> const any_hit = make_shader_module(any_hit_code, device);
        if (!raygen.has_value() || !closest_hit.has_value() || !miss.has_value() || !any_hit.has_value()) {
            return fail("rt shadow: shader module creation failed");
        }

        VkPipelineShaderStageCreateInfo const raygen_stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                              .pNext = nullptr,
                                                              .flags = 0,
                                                              .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                                                              .module = **raygen,
                                                              .pName = "main",
                                                              .pSpecializationInfo = nullptr};
        VkPipelineShaderStageCreateInfo const closest_hit_stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                                   .pNext = nullptr,
                                                                   .flags = 0,
                                                                   .stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                                                                   .module = **closest_hit,
                                                                   .pName = "main",
                                                                   .pSpecializationInfo = nullptr};
        VkPipelineShaderStageCreateInfo const miss_stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                            .pNext = nullptr,
                                                            .flags = 0,
                                                            .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
                                                            .module = **miss,
                                                            .pName = "main",
                                                            .pSpecializationInfo = nullptr};
        VkPipelineShaderStageCreateInfo const any_hit_stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                               .pNext = nullptr,
                                                               .flags = 0,
                                                               .stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                                                               .module = **any_hit,
                                                               .pName = "main",
                                                               .pSpecializationInfo = nullptr};
        // FOUR STAGES, THREE GROUPS: the any-hit shader sits at index 3 and is named by the hit group below rather
        // than becoming a group of its own, which is what keeps the caller's shader binding table regions and
        // their addressing unchanged by this step.
        std::array<VkPipelineShaderStageCreateInfo, 4> const stages = {raygen_stage, miss_stage, closest_hit_stage, any_hit_stage};

        // THE GROUP ORDER IS THE SBT'S ORDER: group 0 is the raygen, group 1 the miss shader, group 2 the hit
        // group. The caller's regions follow exactly this order, which is why the count is returned with them.
        // The hit group names BOTH of its stages: the closest-hit shader answers the ray and the any-hit shader
        // is the one that may refuse the intersection first, which is the whole reason this pipeline exists.
        VkRayTracingShaderGroupCreateInfoKHR const raygen_group = {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                                                   .pNext = nullptr,
                                                                   .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                                                                   .generalShader = 0,
                                                                   .closestHitShader = VK_SHADER_UNUSED_KHR,
                                                                   .anyHitShader = VK_SHADER_UNUSED_KHR,
                                                                   .intersectionShader = VK_SHADER_UNUSED_KHR,
                                                                   .pShaderGroupCaptureReplayHandle = nullptr};
        VkRayTracingShaderGroupCreateInfoKHR const miss_group = {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                                                 .pNext = nullptr,
                                                                 .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                                                                 .generalShader = 1,
                                                                 .closestHitShader = VK_SHADER_UNUSED_KHR,
                                                                 .anyHitShader = VK_SHADER_UNUSED_KHR,
                                                                 .intersectionShader = VK_SHADER_UNUSED_KHR,
                                                                 .pShaderGroupCaptureReplayHandle = nullptr};
        VkRayTracingShaderGroupCreateInfoKHR const hit_group = {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                                                .pNext = nullptr,
                                                                .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                                                                .generalShader = VK_SHADER_UNUSED_KHR,
                                                                .closestHitShader = 2,
                                                                .anyHitShader = 3, // the ANY-HIT stage of this same group: see shaders/rt_shadow.rahit
                                                                .intersectionShader = VK_SHADER_UNUSED_KHR,
                                                                .pShaderGroupCaptureReplayHandle = nullptr};
        std::array<VkRayTracingShaderGroupCreateInfoKHR, 3> const groups = {raygen_group, miss_group, hit_group};

        VkPipelineCreateFlags2CreateInfo const rt_heap_flags = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
        };
        VkRayTracingPipelineCreateInfoKHR const pipeline_info = {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
                                                                 .pNext = &rt_heap_flags,
                                                                 .flags = 0,
                                                                 .stageCount = static_cast<uint32_t>(stages.size()),
                                                                 .pStages = stages.data(),
                                                                 .groupCount = static_cast<uint32_t>(groups.size()),
                                                                 .pGroups = groups.data(),
                                                                 .maxPipelineRayRecursionDepth = 1,
                                                                 .pLibraryInfo = nullptr,
                                                                 .pLibraryInterface = nullptr,
                                                                 .pDynamicState = nullptr,
                                                                 .layout = VK_NULL_HANDLE, // heap-native stages: a layout would contradict them
                                                                 .basePipelineHandle = VK_NULL_HANDLE,
                                                                 .basePipelineIndex = -1};
        // THE EXTENSION ENTRY POINT COMES FROM THE DEVICE, not from the link line: `vulkan-1`'s import library
        // does not export an extension command (the acceleration-structure module loads its five the same way),
        // so a direct call is an undefined symbol at link time rather than a missing feature at runtime.
        auto const create_ray_tracing = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));
        if (create_ray_tracing == nullptr) {
            return fail("rt shadow: the device did not publish vkCreateRayTracingPipelinesKHR");
        }
        VkPipeline pipeline = VK_NULL_HANDLE;
        if (create_ray_tracing(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("rt shadow: vkCreateRayTracingPipelinesKHR failed");
        }
        out.pipeline = vk_pipeline(pipeline, out.pipeline_layout, device);
        out.group_count = static_cast<uint32_t>(groups.size());
        return out;
    }

    // The stochastic punctual lighting trace (shaders/megalights_trace.comp): the same two set layouts and the
    // same compute-pipeline shape as the passes above, with a push block of its own. It FORWARDS to the ray-traced
    // shadow's builder rather than repeating twenty lines of Vulkan, and it exists as its own name because a
    // caller reading `build_rt_shadow` inside this pass's create() would have to check that the two are still the
    // same shape - which is exactly the kind of coupling a name is for.
    std::expected<compute_pipeline_owned, std::string> build_megalights_trace(VkDevice device, VkDescriptorSetLayout const scene_layout, VkDescriptorSetLayout const gbuffer_layout, uint32_t const push_constant_size,
                                                                              std::span<unsigned char const> const compute_shader_code) {
        return build_two_set_compute(device, scene_layout, gbuffer_layout, push_constant_size, compute_shader_code);
    }
    // ... and the chain's temporal resolve: the same two-layout shape, with the shared G-buffer set FIRST
    // because that is index 0 in its declaration (its own bindings are set 1) - the order of the arguments IS
    // the set numbering, which is why this forwarder takes them in that order rather than reusing the one above.
    std::expected<compute_pipeline_owned, std::string> build_megalights_temporal(VkDevice device, VkDescriptorSetLayout const gbuffer_layout, VkDescriptorSetLayout const pass_set_layout, uint32_t const push_constant_size,
                                                                                 std::span<unsigned char const> const compute_shader_code) {
        return build_two_set_compute(device, gbuffer_layout, pass_set_layout, push_constant_size, compute_shader_code);
    }
    // The GI spatial filter: the same two set layouts the tracer binds (the shared scene set and the
    // G-buffer set, which carries the normal, the depth, the accumulated image it reads and the
    // filtered image it writes), so only the pipeline layout and the push block are new.

    // The temporal resolve: its own set layout, because it groups things no other pass puts together
    // (this frame's estimate, the accumulated history, the motion vectors and the depth). It binds no scene
    // set: the push block carries the two projection terms its depth guard needs.
    std::expected<resolve_pipeline_owned, std::string> build_resolve_pipeline(VkDevice const device, VkDescriptorSetLayout const pass_set_layout, uint32_t const push_constant_size, std::span<unsigned char const> const compute_shader_code) {
        using fail = std::unexpected<std::string>;
        resolve_pipeline_owned out;

        // THE SET LAYOUT IS THE DECLARATION'S (generated by the caller with bindings::make_set_layout): one of
        // its bindings is the storage image the resolve writes. Handing it in is what makes the LAYOUT and the
        // family's pool count the same fact.
        if (pass_set_layout == VK_NULL_HANDLE) {
            return fail("temporal resolve: the declaration produced no set layout");
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
            return fail("temporal resolve: pipeline layout creation failed");
        }

        std::optional<vk_shader_module> const module = make_shader_module(compute_shader_code, device);
        if (!module.has_value()) {
            return fail("temporal resolve: compute shader module creation failed");
        }
        VkPipelineShaderStageCreateInfo stage_info = {};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = **module;
        stage_info.pName = "main";

        // THE HEAP FLAG IS NOT OPTIONAL WHEN THE LAYOUT IS NULL: validation's rule is "both or neither", and it
        // says so exactly - "pCreateInfos[0].flags (VkPipelineCreateFlags2(0)) does not include
        // VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT while layout is VK_NULL_HANDLE"
        // (VUID-VkComputePipelineCreateInfo-None-11367), measured on the first run of the migrated renderer. The
        // flag is a flags2 bit, past the 32-bit `flags` field, so it reaches a classic create call through
        // VkPipelineCreateFlags2CreateInfo - the shape the graphics path and the two probes already use.
        VkPipelineCreateFlags2CreateInfo const heap_flags = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
        };
        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.pNext = &heap_flags;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = VK_NULL_HANDLE; // heap-native stages: a layout would contradict them (see docs)

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            return fail("temporal resolve: vkCreateComputePipelines failed");
        }
        out.resolve = vk_pipeline(pipeline, out.pipeline_layout, device);
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

    std::expected<vk_pipeline, std::string> build_shadow(VkDevice const device, VkPipelineLayout const pipeline_layout, VkFormat const depth_format, float const depth_bias_constant_factor,
                                                         float const depth_bias_slope_factor, float const depth_bias_clamp, std::span<unsigned char const> const vertex_shader_code,
                                                         std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        // No color attachment, depth test + write, single-sampled, and the slope-scaled bias the shadow pass needs
        // (it removes acne on surfaces angled away from the light, in units of depth per depth-unit of slope - the
        // numbers are the pass's and the caller's, not this builder's).
        auto result = vulkan::make_pipeline(device,
                                            pipeline_layout,
                                            VK_FORMAT_UNDEFINED,
                                            depth_format,
                                            vertex_shader_code,
                                            fragment_shader_code,
                                            VK_SAMPLE_COUNT_1_BIT,
                                            true,  // depth test + write
                                            false, // no color attachment
                                            depth_bias_constant_factor,
                                            depth_bias_slope_factor,
                                            depth_bias_clamp);
        if (!result) {
            return fail(std::string(result.error()));
        }
        return std::move(result).value();
    }
    std::expected<fxaa_owned, std::string> build_fxaa_owned(VkDevice const device, VkFormat const swap_chain_format, VkDescriptorSetLayout const pass_set_layout, uint32_t const push_constant_size,
                                                            std::span<unsigned char const> const vertex_shader_code, std::span<unsigned char const> const fragment_shader_code) {
        using fail = std::unexpected<std::string>;
        fxaa_owned out;
        // The set layout is the POST CHAIN's, which the pass is handed by its owner (the composite pass owns it);
        // what this creates around it is the pass's OWN pipeline layout, because a pipeline's layout is what its
        // binds and pushes go through and a pass may not borrow another pass's.
        if (pass_set_layout == VK_NULL_HANDLE) {
            return fail("fxaa: the post set layout is missing (the composite pass's create step must run first)");
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
            return fail("fxaa: pipeline layout creation failed");
        }

        // The anti-aliasing pipeline renders into the SWAPCHAIN, so its declared colour format is the surface's.
        auto pipeline_result = vulkan::make_pipeline(
            device, out.pipeline_layout, swap_chain_format, VK_FORMAT_UNDEFINED, vertex_shader_code, fragment_shader_code, VK_SAMPLE_COUNT_1_BIT, false, true, 0.0f, 0.0f, 0.0f);
        if (!pipeline_result) {
            vkDestroyPipelineLayout(device, out.pipeline_layout, nullptr);
            out.pipeline_layout = VK_NULL_HANDLE;
            return fail(std::string(pipeline_result.error()));
        }
        out.antialias = std::move(pipeline_result).value();
        return out;
    }
} // namespace vulkan::pipelines