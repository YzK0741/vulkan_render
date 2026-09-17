// The post chain's implementation: the composite (HDR + the weighted bloom levels -> exposure -> ACES ->
// display) and the bloom chain's four levels (the bright-pass prefilter into level 0, then one downsample per
// level). Moved out of `runtime::record_composite` and `runtime::record_bloom_chain` UNCHANGED in behaviour - the
// same barriers in the same order, the same clear instances over the same views, the same one-set binds, the same
// 52-byte push block with the same lanes and the same fullscreen draw - so the capture gate decides the move on
// all twelve scenarios, every one of which runs the composite and (measured, see docs/pass_chain_plan.md) the
// bloom chain as well.

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

module vulkan.pass.post;

import vulkan.constant_init;
import vulkan.pipelines; // build_post: the set layout, the pipeline layout and the chain's two pipelines
import utility;

namespace vulkan::pass {

    // =============================================================================================
    // THE COMPOSITE
    // =============================================================================================

    post_composite_pass::~post_composite_pass() {
        this->release_owned();
    }

    void post_composite_pass::release_owned() noexcept {
        // The order they were made: the set layout first, the pipeline layout FROM it, the two pipelines from
        // that.
        this->composite_.reset();
        this->hdr_.reset();
        if (this->pipeline_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->device_, this->pipeline_layout_, nullptr);
            this->pipeline_layout_ = VK_NULL_HANDLE;
        }
        // THE SET LAYOUT IS NOT DESTROYED HERE: it is the OWNER's (the runtime creates it from the same nine
        // bindings and uses it for the family it writes - see make_post_set_layout), so this pass only holds a view
        // of it and clears that view.
        this->set_layout_ = VK_NULL_HANDLE;
    }

    render_resource::pass_io const& post_composite_pass::io() const noexcept {
        return render_resource::post_composite_io;
    }

    vulkan::pass::behaviour const& post_composite_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view post_composite_pass::feature() const noexcept {
        // NO FEATURE NAME, and that is deliberate rather than an oversight: the runner only asks a pass's feature
        // when it has one, and this pass's gate is the HOST's - `record_post_process` returns before the chain
        // when the pipelines or the frame's composite set are missing. A name here would be a second copy of that
        // question, and the deferred step measured what a pass feature name with no branch in the runtime does
        // (the runner skips the pass on every frame and the frame goes unlit).
        return {};
    }

    void post_composite_pass::create(pass_context const& context) {
        if (context.device == VK_NULL_HANDLE) {
            return;
        }
        if (this->device_ != VK_NULL_HANDLE && this->device_ != context.device) {
            this->release_owned();
        }
        this->device_ = context.device;
        if (this->set_layout_ != VK_NULL_HANDLE) {
            return; // already built for this device
        }
        std::span<unsigned char const> const vertex_spirv = context.shader != nullptr ? context.shader(context.owner, vertex_shader_name) : std::span<unsigned char const>{};
        std::span<unsigned char const> const fragment_spirv = context.shader != nullptr ? context.shader(context.owner, fragment_shader_name) : std::span<unsigned char const>{};
        if (vertex_spirv.empty() || fragment_spirv.empty()) {
            utility::log("post chain disabled: the owner has no {} or {}", vertex_shader_name, fragment_shader_name);
            return;
        }
        // THE POST SET LAYOUT COMES FROM THE CONTEXT, not from this pass: the nine bindings describe how the
        // OWNER writes the post sets (see `pass_context::shared_set_layout` and make_post_set_layout), and this
        // pass only needs a pipeline layout built around them.
        VkDescriptorSetLayout const post_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 2) : VK_NULL_HANDLE;
        if (post_layout == VK_NULL_HANDLE) {
            utility::log("post chain disabled: the owner has no layout for the post set this pass binds");
            return;
        }
        // The SURFACE's format is one of the two pipelines' (the other renders into R16F bloom levels and into
        // the LDR image FXAA reads), and it is a session-stable device fact the context carries for exactly this
        // kind of reason (see pass_context::swap_chain_image_format).
        auto built = pipelines::build_post(context.device, post_layout, context.swap_chain_image_format, static_cast<uint32_t>(sizeof(post_push_constants)), vertex_spirv, fragment_spirv);
        if (!built) {
            utility::log("post chain disabled: {}", built.error());
            this->release_owned();
            return;
        }
        this->set_layout_ = built->set_layout;
        this->pipeline_layout_ = built->pipeline_layout;
        this->composite_ = std::move(built->composite);
        this->hdr_ = std::move(built->hdr);
        // The surface's format is cached here because the `encode_gamma` lane is a consequence of it (see
        // resolve): a session-stable device fact, which is exactly what a create step may keep.
        this->swap_chain_format_ = context.swap_chain_image_format;
        utility::log("SUCCESS: post chain pipelines created (the composite for the swapchain and the R16F variant)");
    }

    void post_composite_pass::on_swapchain_recreated(pass_host const&) {
        // Nothing to reset: the two pipelines depend on the surface's FORMAT (a session-stable device fact) and
        // on nothing whose size changes, and the sets this pass binds belong to the post family, whose owner
        // retires them (the runner calls this so a pass that owned a family could not forget to - see
        // recreate_stage).
    }

    bool post_composite_pass::pipeline_ready() const noexcept {
        return this->composite_.has_value() && this->hdr_.has_value() && this->set_layout_ != VK_NULL_HANDLE;
    }

    VkPipeline post_composite_pass::pipeline() const noexcept {
        return this->composite_pipeline();
    }

    owned_pipeline post_composite_pass::named_pipeline(std::string_view const name) const noexcept {
        // THE ONE NAME THIS PASS PUBLISHES TO ITS SIBLINGS: the bloom levels' `post_hdr` (see the class note and
        // frame_pass::named_pipeline). Its own name is answered by `pipeline()` above. The LAYOUT travels with the
        // pipeline - the four levels bind their set and push the chain's block through it, and it is the one
        // object all five stages were built against.
        return name == bloom_pipeline_name ? owned_pipeline{.pipeline = this->hdr_pipeline(), .layout = this->pipeline_layout_} : owned_pipeline{};
    }

    VkPipeline post_composite_pass::composite_pipeline() const noexcept {
        return this->composite_.has_value() ? this->composite_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipeline post_composite_pass::hdr_pipeline() const noexcept {
        return this->hdr_.has_value() ? this->hdr_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout post_composite_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    void post_composite_pass::set_frame(composite_frame const& frame) noexcept {
        this->frame_ = frame;
    }

    void post_composite_pass::set_overlay(draw_callback const overlay) noexcept {
        // The host's hook, installed once by whoever owns the passes. It is STORED, not put in the frame: what the
        // frame decides per frame is whether THIS pass is the frame's last writer (see prepare_frame).
        this->overlay_ = overlay;
    }

    void post_composite_pass::prepare_frame(frame_facts const& facts) noexcept {
        // WHO WRITES THE LDR IMAGE is the fact that decides both lanes at once: when FXAA resolves, this pass is
        // not the frame's last writer, so it writes the intermediate and FXAA carries the overlay; when it does
        // not, this pass is the last writer and draws the overlay itself.
        composite_frame frame = {};
        frame.after_draw = facts.fxaa_resolves ? draw_callback{} : this->overlay_;
        frame.write_ldr = facts.fxaa_resolves;
        frame.suppress_bloom = facts.debug_view;
        this->set_frame(frame);
    }

    bool post_composite_pass::resolve(resolve_context const& context, resolved_io& out) const {
        if (!resolve_declaration(*this, context, out)) {
            return false;
        }
        // THE FRAME DECIDES THE TARGET AND THE PIPELINE TOGETHER (see this override's declaration note): with
        // FXAA on, the composite writes the R16F LDR image with the R16F variant; otherwise it writes the
        // declared swapchain image and the FXAA pass never runs. The LDR image is not in the declaration - one
        // `render_target` names one resource - so it comes from the frame's table, in the declaration's own
        // vocabulary, exactly as the declared one did.
        if (!this->frame_.write_ldr) {
            return this->fill_push(out, false);
        }
        render_resource::resource_info const* const ldr = render_resource::find(resource_id::ldr);
        if (ldr == nullptr) {
            return false;
        }
        resolved_binding const target = context.resources->find(resource_id::ldr, 0, instance_for(ldr->scope, out.frame));
        if (target.view == VK_NULL_HANDLE || target.image == VK_NULL_HANDLE) {
            return false; // no LDR image this generation: do not record a composite that cannot write anywhere
        }
        out.target_storage[0] = target;
        out.pipeline_storage[0] = this->hdr_pipeline();
        return this->fill_push(out, true);
    }

    bool post_composite_pass::fill_push(resolved_io& out, bool const writing_ldr) const {
        // The lanes are the frame's settings (more than one pass reads them - see
        // vulkan.frame_constants::render_settings), this pass's own GI upsample switch, and the TWO that follow
        // from the target choice: `encode_gamma` (an R16F target needs the shader to encode, an sRGB swapchain
        // attachment does the transfer in hardware) and the GI weight, which is the frame's answer to whether the
        // chain resolved.
        render_settings const& settings = out.constants.settings;
        post_push_constants const push = {
            .exposure = settings.exposure,
            // Zero while the G-buffer debug view is up (the frame says so - see composite_frame::suppress_bloom),
            // for the reason that field records.
            .bloom_intensity = this->frame_.suppress_bloom ? 0.0f : settings.bloom_intensity,
            .bloom_threshold = settings.bloom_threshold,
            .encode_gamma = writing_ldr ? 1.0f : (vulkan::is_srgb_format(this->swap_chain_format_) ? 0.0f : 1.0f),
            .fxaa_subpixel = settings.fxaa_subpixel,
            .fxaa_edge_threshold = settings.fxaa_edge_threshold,
        };
        static_assert(sizeof(push) == render_resource::post_push_bytes, "the composed push block must be the size the declaration promises");
        std::memcpy(out.push_storage.data(), &push, sizeof(push));
        out.push = std::span<std::byte const>(out.push_storage.data(), sizeof(push));
        return true;
    }

    void post_composite_pass::record(resolved_io const& io) {
        if (!this->pipeline_ready() || io.targets.empty() || io.pipelines.empty() || io.pipelines[0] == VK_NULL_HANDLE || io.pipeline_layout == VK_NULL_HANDLE ||
            io.shared.post == VK_NULL_HANDLE || io.push.size() < sizeof(post_push_constants) || io.extent.width == 0 || io.extent.height == 0) {
            return; // the runner resolves all of this or skips the pass (see runtime::resolve_post_composite)
        }
        VkImage const target = io.targets[0].image;
        VkImageView const target_view = io.targets[0].view;
        if (target == VK_NULL_HANDLE || target_view == VK_NULL_HANDLE) {
            return;
        }
        // The target becomes a colour attachment BEFORE the instance (a pipeline barrier may not be recorded
        // inside one) with UNDEFINED as its old layout: the instance CLEARs it, so whatever it held is dead - and
        // on the FXAA path the target is the LDR image, which the previous frame's FXAA pass left in a sampled
        // layout, which is exactly the claim UNDEFINED does not make.
        VkImageMemoryBarrier2 to_attachment = vulkan::color_attachment_transition;
        to_attachment.image = target;
        VkDependencyInfo const attachment_dependency = make_image_dependency_info(1, &to_attachment);
        vkCmdPipelineBarrier2(io.cmd, &attachment_dependency);
        // The push block the host filled, with the pass's own stage lane written last: `mode = 2` is what makes
        // post.frag composite instead of prefiltering or downsampling.
        post_push_constants push = {};
        std::memcpy(&push, io.push.data(), sizeof(push));
        push.mode = 2.0f;
        VkClearValue clear = {};
        VkRenderingAttachmentInfo const attachment = make_color_attachment_info(target_view, clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
        VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, io.extent}, true, &attachment, nullptr);
        vkCmdBeginRendering(io.cmd, &rendering_info);
        vkCmdSetCullMode(io.cmd, VK_CULL_MODE_NONE); // the synthetic triangle has no facing to cull
        VkDescriptorSet const set = io.shared.post;  // the post set, resolved by the host from the frame
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, io.pipeline_layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(io.cmd, 3, 1, 0, 0);
        // INSIDE the instance, between the draw and its end: the debug overlay composites a UI over the image
        // this draw just wrote and has no load op of its own, so it can be neither a pass nor outside the
        // instance (see composite_frame::after_draw - the pass leaves this empty when FXAA is the frame's last
        // writer instead, which its own frame decided in prepare_frame).
        if (this->frame_.after_draw.valid()) {
            this->frame_.after_draw.record(this->frame_.after_draw.owner, io.cmd);
        }
        vkCmdEndRendering(io.cmd);
    }

    // =============================================================================================
    // THE BLOOM CHAIN's FOUR LEVELS
    // =============================================================================================

    post_bloom_pass::post_bloom_pass(uint32_t const level) noexcept
        : level_(level < render_resource::post_bloom_io.size() ? level : 0u)
        , io_(&render_resource::post_bloom_io[this->level_]) {
        // The behaviour is a MEMBER because the ELEMENT is part of it: four instances of this class cannot share
        // one declaration of "which resource my extent comes from". Everything else is every other fullscreen
        // pass's: the runner sets the viewport and scissor from the extent the rule produced.
        this->behaviour_ = vulkan::pass::behaviour{
            .kind = behaviour_kind::fullscreen,
            .extent = extent_rule::resource,
            .extent_of = resource_id::bloom,
            .extent_of_element = static_cast<uint16_t>(this->level_),
            .pipelines = pipeline_names,
            .resync_viewport = true,
        };
    }

    post_bloom_pass::~post_bloom_pass() = default;

    render_resource::pass_io const& post_bloom_pass::io() const noexcept {
        return *this->io_;
    }

    vulkan::pass::behaviour const& post_bloom_pass::behaviour() const noexcept {
        return this->behaviour_;
    }

    std::string_view post_bloom_pass::feature() const noexcept {
        // THE BLOOM CHAIN'S OWN GATE, and the renderer already answers this name: `bloom_intensity > 0` and the
        // chain's pipeline being there. It has to be a feature rather than a resolver that fails, because the
        // OFF path is not "nothing happens": the host moves the four levels to a sampled layout so the
        // composite's descriptor (which declares all four as inputs) is valid on a frame no level was written -
        // and that off path is the host's, triggered by this stage recording none of its passes.
        return "bloom";
    }

    void post_bloom_pass::create(pass_context const&) {
        // NOTHING TO BUILD, and that is this pass's whole shape: it records with the chain's R16F pipeline, which
        // the COMPOSITE owns - one set layout and one pipeline layout serve all five stages, so a copy per level
        // would be five identical sets of objects and five chances to disagree about the push block. Its target
        // is an element of its declaration and its extent comes from the declaration's rule, which is why it has
        // no error path here either.
    }

    void post_bloom_pass::on_swapchain_recreated(pass_host const&) {
        // Nothing to reset: this pass owns no handle at all (its pipeline arrives through resolved_io::pipelines
        // and its target through its own declaration).
    }

    uint32_t post_bloom_pass::level() const noexcept {
        return this->level_;
    }

    void post_bloom_pass::record(resolved_io const& io) {
        if (io.targets.empty() || io.pipelines.empty() || io.pipelines[0] == VK_NULL_HANDLE || io.pipeline_layout == VK_NULL_HANDLE || io.shared.post == VK_NULL_HANDLE ||
            io.extent.width == 0 || io.extent.height == 0) {
            return; // the runner resolves all of this or skips the pass (see frame_pass::resolve)
        }
        VkImage const target = io.targets[0].image;
        VkImageView const target_view = io.targets[0].view;
        if (target == VK_NULL_HANDLE || target_view == VK_NULL_HANDLE) {
            return;
        }
        // THE INPUT FIRST, in the order the old chain moved things: the level this stage READS becomes a sample.
        // Only levels 1..3 declare one - level 0's input is the HDR target, which the frame loop moved before the
        // chain (it is the one owner of that transition, because the composite reads HDR too and because it is
        // needed on the frames the bloom chain is skipped entirely).
        if (!io.barrier_images.empty() && io.barrier_images[0].image != VK_NULL_HANDLE) {
            VkImageMemoryBarrier2 to_sampling = vulkan::hdr_sampling_transition;
            to_sampling.image = io.barrier_images[0].image;
            VkDependencyInfo const sampling_dependency = make_image_dependency_info(1, &to_sampling);
            vkCmdPipelineBarrier2(io.cmd, &sampling_dependency);
        }
        // ... then the level this stage WRITES, with UNDEFINED as its old layout: the instance CLEARs it.
        VkImageMemoryBarrier2 to_attachment = vulkan::color_attachment_transition;
        to_attachment.image = target;
        VkDependencyInfo const attachment_dependency = make_image_dependency_info(1, &to_attachment);
        vkCmdPipelineBarrier2(io.cmd, &attachment_dependency);
        // The push block is composed HERE (S3): the frame's three settings (exposure, the bloom weight and the
        // bright-pass threshold - all of them lanes more than one pass pushes, which is why they are frame
        // settings), the struct's defaults for every lane this stage does not read - which is what the renderer
        // pushed before the move - and the stage's own lane: 0 for the bright-pass prefilter, 1 for every
        // downsample, because the bright-pass test runs on the first level only.
        render_settings const& settings = io.constants.settings;
        post_push_constants push = {
            .exposure = settings.exposure,
            .bloom_intensity = settings.bloom_intensity,
            .bloom_threshold = settings.bloom_threshold,
            .mode = this->level_ == 0u ? 0.0f : 1.0f,
        };
        VkClearValue clear = {};
        VkRenderingAttachmentInfo const attachment = make_color_attachment_info(target_view, clear, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE);
        VkRenderingInfo const rendering_info = make_rendering_info(0, {{0, 0}, io.extent}, true, &attachment, nullptr);
        vkCmdBeginRendering(io.cmd, &rendering_info);
        vkCmdSetCullMode(io.cmd, VK_CULL_MODE_NONE);
        VkDescriptorSet const set = io.shared.post; // THIS level's set of the post family, resolved by the host
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, io.pipeline_layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(io.cmd, 3, 1, 0, 0);
        vkCmdEndRendering(io.cmd);
        // THE HAND-BACK, and it is the deepest level's because it has no successor to do it for it: the composite
        // samples ALL FOUR levels, so the last one has to be left in a sampled layout. The levels before it are
        // moved by the next level's input transition above - which is why this is one barrier and not four.
        if (this->level_ + 1u == render_resource::post_bloom_io.size()) {
            VkImageMemoryBarrier2 hand_back = vulkan::hdr_sampling_transition;
            hand_back.image = target;
            VkDependencyInfo const hand_back_dependency = make_image_dependency_info(1, &hand_back);
            vkCmdPipelineBarrier2(io.cmd, &hand_back_dependency);
        }
    }

} // namespace vulkan::pass
