// The probe cache's implementation: everything the pass does between "the runner resolved this frame" and
// "the grid holds this frame's observation", plus what it builds for itself. Moved out of `vulkan.runtime`
// unchanged in behaviour - the dispatch sequence, the barrier order, the light-change reset, the pipeline and
// the log lines are the ones that were there, which is what makes the capture gate able to decide the move.

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

module vulkan.pass.gi_probe;

import vulkan.render_resource;
import vulkan.constant_init;
import vulkan.pipelines;
import utility;

namespace vulkan::pass {

    gi_probe_pass::~gi_probe_pass() {
        this->release_owned();
    }

    void gi_probe_pass::release_owned() noexcept {
        // ORDER MATTERS: the set layout was created first, the pipeline layout FROM it, the pipeline from
        // that - so they are released in the order they were made. The pipeline is RAII (`vk_pipeline` owns
        // the VkPipeline and not the layout), the two layouts are raw handles this pass destroys itself.
        this->pipeline_.reset();
        if (this->pipeline_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->device_, this->pipeline_layout_, nullptr);
            this->pipeline_layout_ = VK_NULL_HANDLE;
        }
        if (this->set_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(this->device_, this->set_layout_, nullptr);
            this->set_layout_ = VK_NULL_HANDLE;
        }
    }

    render_resource::pass_io const& gi_probe_pass::io() const noexcept {
        return render_resource::gi_probe_io;
    }

    vulkan::pass::behaviour const& gi_probe_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view gi_probe_pass::feature() const noexcept {
        // The feature name the renderer's registry answers: it is what keeps "is the cache running" a
        // runtime decision (its switch, its pipeline, the GI chain it is deposited from) while the pass only
        // says WHICH feature it belongs to.
        return "ssgi_probes";
    }

    VkDescriptorSetLayout gi_probe_pass::set_layout() const noexcept {
        return this->set_layout_;
    }

    bool gi_probe_pass::pipeline_ready() const noexcept {
        return this->pipeline_.has_value();
    }

    VkPipeline gi_probe_pass::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout gi_probe_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    void gi_probe_pass::set_rounds(uint32_t const rounds) noexcept {
        this->rounds_ = rounds;
    }

    bool gi_probe_pass::cache_valid() const noexcept {
        return this->cache_valid_;
    }

    void gi_probe_pass::create(pass_context const& context) {
        if (context.device == VK_NULL_HANDLE) {
            return; // no device, nothing to build on (the create step is a no-op before the core exists)
        }
        if (this->device_ != VK_NULL_HANDLE && this->device_ != context.device) {
            // A NEW DEVICE GENERATION: everything this pass built belongs to the old one. Releasing first is
            // what makes `create` correct on every generation rather than only on the first.
            this->release_owned();
        }
        this->device_ = context.device;
        this->samplers_ = context.samplers;
        if (this->set_layout_ != VK_NULL_HANDLE) {
            return; // already built for this device
        }
        // WHAT THE PASS NEEDS FROM ITS OWNER, all of it at create time and none of it a capability: the device
        // (above), the layout of the SHARED set its pipeline layout must be built against, and its own
        // shader's SPIR-V. It owns the three objects it makes from them.
        VkDescriptorSetLayout const scene_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 0) : VK_NULL_HANDLE;
        std::span<unsigned char const> const spirv = context.shader != nullptr ? context.shader(context.owner, shader_name) : std::span<unsigned char const>{};
        if (scene_layout == VK_NULL_HANDLE) {
            utility::log("world-space probe cache disabled (the tracer keeps its environment fallback): the shared scene set layout is not there yet");
            return;
        }
        if (spirv.empty()) {
            utility::log("world-space probe cache disabled (the tracer keeps its environment fallback): the owner has no {}", shader_name);
            return;
        }
        // THE LAYOUT IS GENERATED FROM THE PASS'S OWN DECLARATION, which is the other half of the sequence
        // that generated the writes below: the nine bindings this pass used to spell out - four coefficient
        // images read, four written, the per-cell surface - are the declaration's own set, so the layout the
        // shader sees and the declaration cannot drift. That the generated layout is the one the renderer
        // carried is decided by the capture gate: `sponza_gi` runs with this pass ON.
        std::expected<VkDescriptorSetLayout, std::string> const layout =
            bindings::make_set_layout(context.device, render_resource::gi_probe_io, render_resource::gi_probe_io.own_set);
        if (!layout.has_value()) {
            utility::log("world-space probe cache disabled (the tracer keeps its environment fallback): {}", layout.error());
            return;
        }
        this->set_layout_ = *layout;
        // ... and the pipeline, whose LAYOUT is built from the declaration too: the shared scene set at 0, the
        // pass's own at 1, and the push range the declaration carries (the `static_assert` at the end of this
        // module ties that number to the struct the pass pushes).
        auto built = pipelines::build_gi_probe(context.device, scene_layout, this->set_layout_, render_resource::gi_probe_io.push->size, spirv);
        if (!built) {
            utility::log("world-space probe cache disabled (the tracer keeps its environment fallback): {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->pass);
        // The line the renderer used to print for this pass, now printed by the thing that built it.
        utility::log("SUCCESS: probe cache pipeline created (injection + propagation, world-space GI)");
    }

    void gi_probe_pass::on_swapchain_recreated(pass_host const&) {
        // A new target generation: the grid images are new (created with the swapchain), so the cache holds
        // nothing worth sampling until this pass has run in it, and the light it was filled under is
        // meaningless. The family keeps its sets: `ensure_all` re-points them when the views change, and a
        // pool is retired rather than destroyed because recorded command buffers may still name its sets.
        this->cache_valid_ = false;
        this->light_dir_valid_ = false;
    }

    void gi_probe_pass::record(resolved_io const& io) {
        if (io.own.size() < own_binding_count || io.push.size() < sizeof(push_constants) || io.pipeline_layout == VK_NULL_HANDLE || io.pipelines.empty() ||
            this->set_layout_ == VK_NULL_HANDLE) {
            return; // the runner resolves all of this or skips the pass; an unresolved frame records nothing
        }
        // The declaration, resolved: elements 0..3 are the side being READ and 4..7 the side being WRITTEN,
        // element 8 the per-cell geometry. Which IMAGE each of them is came from the host (it maps the
        // declaration's elements onto the core's images); which HALF of the ping-pong each of the family's two
        // sets reads and writes is the one thing that differs between them, and it is this pass's own fact.
        std::array<VkImageView, 4> const read_views = {io.own[0].view, io.own[1].view, io.own[2].view, io.own[3].view};
        std::array<VkImageView, 4> const write_views = {io.own[4].view, io.own[5].view, io.own[6].view, io.own[7].view};
        std::array<VkImageView, 1> const surface_view = {io.own[8].view};
        // Three fingerprints, one per KIND of thing a set points at - the four coefficients, the four of the
        // other side, and the per-cell geometry. The family compares what a set points at, and a span of four
        // says "these four, in this order", which is what a set with four coefficient bindings needs.
        std::array<std::span<VkImageView const>, 3> const fingerprints = {read_views, write_views, surface_view};
        auto const write_sets = [&io, this](uint32_t const /*image_index*/, std::span<VkDescriptorSet const> const sets) {
            // Set 0 writes the cache and reads the other side, set 1 the other way round - the entire
            // ping-pong, and the reason the propagation needs no descriptor write between its dispatches.
            // Everything else - the nine binding numbers, their descriptor types, their counts, their GENERAL
            // layouts (all nine, see the declaration: the grid's sides stay in GENERAL for the whole update)
            // and the fact that the four sampled ones take the probe grid's sampler - is generated from the
            // declaration by `bindings::write_set`.
            for (uint32_t which = 0; which < sets.size(); ++which) {
                uint32_t const read_side = (1u - which) * 4u;
                uint32_t const write_side = which * 4u;
                std::array<VkImageView, 9> const views = {
                    io.own[read_side + 0u].view,
                    io.own[read_side + 1u].view,
                    io.own[read_side + 2u].view,
                    io.own[read_side + 3u].view,
                    io.own[write_side + 0u].view,
                    io.own[write_side + 1u].view,
                    io.own[write_side + 2u].view,
                    io.own[write_side + 3u].view,
                    io.own[8].view};
                auto const written = bindings::write_set(this->device_, render_resource::gi_probe_io, render_resource::gi_probe_io.own_set, sets[which], views, {}, this->samplers_);
                if (!written) {
                    utility::log("probe cache: {}", written.error());
                }
            }
        };
        // The pool's per-set budget is DERIVED for the same reason the layout and the writes are: a literal
        // here is the drift class this sequence exists to remove, and this project's history has exactly that
        // bug (a pool sized for four descriptors per set while the layout asked for five). A pool size is a
        // capacity rather than an allocation, so deriving it costs nothing.
        uint32_t const descriptors_per_set = render_resource::descriptor_counts_for(render_resource::gi_probe_io, render_resource::gi_probe_io.own_set).total();
        if (!this->family_.ensure_all(this->device_, this->set_layout_, io.frame.image_count, sets_per_image, descriptors_per_set, fingerprints, write_sets)) {
            utility::log("probe cache: descriptor sets unavailable - the tracer keeps its environment fallback");
            return;
        }
        VkDescriptorSet const write_cache = this->family_.set(io.frame.image_index, 0);
        VkDescriptorSet const write_scratch = this->family_.set(io.frame.image_index, 1);
        if (write_cache == VK_NULL_HANDLE || write_scratch == VK_NULL_HANDLE) {
            return;
        }

        push_constants push = {};
        std::memcpy(&push, io.push.data(), sizeof(push));
        // If the global lighting has changed materially since the grid was filled, the grid is worthless - it
        // holds light for a sun that is not there any more - so it is CLEARED rather than faded out over
        // 1/rate frames. The direction arrives in the push block because the host owns it; the memory of what
        // the grid was filled under is the pass's, because it is the pass's cache.
        glm::vec3 const current_light_dir = glm::vec3(push.light_dir);
        bool const light_changed = this->light_dir_valid_ && glm::dot(this->light_dir_, current_light_dir) < light_reset_cosine;
        this->light_dir_ = current_light_dir;
        this->light_dir_valid_ = true;

        // The cache comes back from the sampler the tracer left it in, and goes to GENERAL for the whole
        // update: both sides of the ping-pong stay there, which is what makes the propagation's barriers
        // same-layout ones. All FOUR coefficients move together - a cell is its four images.
        std::array<VkImageMemoryBarrier2, 4> to_general = {};
        for (uint32_t c = 0; c < to_general.size(); ++c) {
            to_general[c] = vulkan::sampling_to_general_transition;
            to_general[c].image = io.own[c].image;
        }
        VkDependencyInfo const general_dependency = make_image_dependency_info(static_cast<uint32_t>(to_general.size()), to_general.data());
        vkCmdPipelineBarrier2(io.cmd, &general_dependency);

        if (light_changed) {
            // Every image is in GENERAL here (the cache by the barrier above, the other side and the geometry
            // by their first-use transitions), which is what a clear needs; TRANSFER_DST is in their usage for
            // this.
            VkClearColorValue const nothing = {};
            VkImageSubresourceRange const whole = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            for (uint32_t c = 0; c < to_general.size(); ++c) {
                vkCmdClearColorImage(io.cmd, io.own[c].image, VK_IMAGE_LAYOUT_GENERAL, &nothing, 1, &whole);
            }
            vkCmdClearColorImage(io.cmd, io.own[8].image, VK_IMAGE_LAYOUT_GENERAL, &nothing, 1, &whole);
            utility::log("probe cache: cleared - the global lighting changed materially");
        }

        // The pipeline is bound by the RUNNER, before this call, from the name the behaviour declares
        // ("gi_probe") - the mechanical part of "how this pass is called" belongs to the behaviour, and a
        // compute pass that had to bind its own pipeline would be doing the runner's job. What the pass does
        // below is push, bind its own set and dispatch.

        // The extent comes from the declaration's rule (`resource`, the probe grid): 32 cells on a side at a
        // workgroup of 4 is 8 groups, and the grid is cubic so one number covers all three axes.
        uint32_t const groups = (io.extent.width + group_size - 1u) / group_size;

        auto const storage_barrier = [&io](VkImage const image) {
            VkImageMemoryBarrier2 barrier = vulkan::compute_storage_transition;
            barrier.image = image;
            VkDependencyInfo const dependency = make_image_dependency_info(1, &barrier);
            vkCmdPipelineBarrier2(io.cmd, &dependency);
        };
        auto const dispatch = [&](VkDescriptorSet const set, float const mode) {
            push.params.w = mode;
            vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            // The shared scene set first, then this pass's own: a cell's ray needs the top level structure to
            // trace, and the shared hit shading needs the material table, the texture array and the light UBO
            // to shade what it finds.
            std::array<VkDescriptorSet, 2> const probe_sets = {io.shared.scene, set};
            vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, io.pipeline_layout, 0, static_cast<uint32_t>(probe_sets.size()), probe_sets.data(), 0, nullptr);
            vkCmdDispatch(io.cmd, groups, groups, groups);
        };
        // One side is four images (side * 4 + coefficient), so a barrier "for the side just written" is four
        // barriers: missing one would leave a coefficient of the next dispatch reading stale data, which looks
        // like the directional part lagging a frame behind rather than like an error.
        auto const side_barrier = [&storage_barrier, &io](uint32_t const first) {
            for (uint32_t c = 0; c < 4; ++c) {
                storage_barrier(io.own[first + c].image);
            }
        };

        // Inject: every cell fills itself from its OWN rays (see the shader); a cell whose frame has nothing to
        // trace against keeps what it holds.
        dispatch(write_cache, 0.0f);
        // Propagate: an even number of dispatches, so a frame's last one always lands back in the cache -
        // which is why the tracer samples one side instead of running a ping-pong of its own.
        for (uint32_t round = 0; round < this->rounds_; ++round) {
            side_barrier(0); // the cache was just written: the next dispatch reads it
            dispatch(write_scratch, 1.0f);
            side_barrier(4); // ... and the other side was: this one reads that
            dispatch(write_cache, 1.0f);
        }

        // Hand the cache's four coefficients back to the sampler the tracer reads them with next frame.
        std::array<VkImageMemoryBarrier2, 4> to_sampling = {};
        for (uint32_t c = 0; c < to_sampling.size(); ++c) {
            to_sampling[c] = vulkan::general_to_sampling_transition;
            to_sampling[c].image = io.own[c].image;
        }
        VkDependencyInfo const sampling_dependency = make_image_dependency_info(static_cast<uint32_t>(to_sampling.size()), to_sampling.data());
        vkCmdPipelineBarrier2(io.cmd, &sampling_dependency);

        // From here on the grid holds something, so the tracer may sample it (see cache_valid()).
        this->cache_valid_ = true;
    }

} // namespace vulkan::pass
