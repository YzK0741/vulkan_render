// The demo's implementation: the two callbacks the runtime asks for, and nothing else. Every branch below is a
// STAGE of this application's frame - the same names the runtime's frame loop uses for its stage structs - and what
// it does is what the runtime used to do inline: hand each pass its frame (built from the runtime's own data) and
// run the frame's ordering rules that belong to that stage.
//
// Moved out of `runtime.cpp` UNCHANGED in behaviour, and the order inside each branch is deliberate: the frame is
// set FIRST, then the ordering rules run, exactly where the two used to sit relative to `record_stage` - so the
// command stream is identical and the capture gate decides the move.

module;

#include <cstddef>
#include <string_view>

module vulkan.render_start_demo;

import utility;

namespace vulkan {

    std::size_t render_start_demo::attach(runtime& self) noexcept {
        this->passes_ = &self.frame_passes();
        // LOOKED UP BY THE NAME THE DECLARATION CARRIES, which is the only key a chain gives: the runtime holds no
        // reference to hand over, and a cast is what turns the declaration's owner into the type whose frame it
        // wants. A pass this build does not have (its declaration missing, or a variant of this app) stays null and
        // is simply never fed.
        this->cluster_ = this->find<pass::cluster_pass>("cluster");
        this->shadow_ = this->find<pass::shadow_pass>("shadow");
        this->scene_ = this->find<pass::scene_pass>("scene");
        this->transparent_ = this->find<pass::transparent_pass>("transparent");
        this->rt_shadow_ = this->find<pass::rt_shadow_pass>("rt_shadow");
        this->deferred_ = this->find<pass::deferred_pass>("deferred");
        this->taa_ = this->find<pass::taa_pass>("taa");
        this->gbuffer_debug_ = this->find<pass::gbuffer_debug_pass>("gbuffer-debug");
        this->ssgi_trace_ = this->find<pass::ssgi_trace_pass>("ssgi_trace");
        this->ssgi_spec_ = this->find<pass::ssgi_spec_pass>("ssgi_spec");
        this->ssgi_temporal_ = this->find<pass::ssgi_temporal_pass>("ssgi_temporal");
        this->ssgi_spatial_ = this->find<pass::ssgi_spatial_pass>("ssgi_spatial");
        this->composite_ = this->find<pass::post_composite_pass>("post_composite");
        this->fxaa_ = this->find<pass::fxaa_pass>("fxaa");

        std::size_t found = 0;
        found += this->cluster_ != nullptr ? 1u : 0u;
        found += this->shadow_ != nullptr ? 1u : 0u;
        found += this->scene_ != nullptr ? 1u : 0u;
        found += this->transparent_ != nullptr ? 1u : 0u;
        found += this->rt_shadow_ != nullptr ? 1u : 0u;
        found += this->deferred_ != nullptr ? 1u : 0u;
        found += this->taa_ != nullptr ? 1u : 0u;
        found += this->gbuffer_debug_ != nullptr ? 1u : 0u;
        found += this->ssgi_trace_ != nullptr ? 1u : 0u;
        found += this->ssgi_spec_ != nullptr ? 1u : 0u;
        found += this->ssgi_temporal_ != nullptr ? 1u : 0u;
        found += this->ssgi_spatial_ != nullptr ? 1u : 0u;
        found += this->composite_ != nullptr ? 1u : 0u;
        found += this->fxaa_ != nullptr ? 1u : 0u;
        if (found != 14) {
            // NOT a fatal error: this demo is one application's chain, and a build of it that lacks a pass (a
            // shader that did not compile is the usual reason - that pass's own create step says why) renders
            // without it. Saying so once at startup is what keeps "the pass did not run" from looking like a
            // rendering bug. NO FORMAT ARGUMENT, and that is not a style choice: `utility::log` with one crashes
            // THIS translation unit's code generation (clang 22.1.8, `EmitBuiltinNewDeleteCall` - a toolchain bug
            // this branch has now seen twice, see docs/pass_chain_plan.md).
            utility::log("render_start_demo: a pass this demo wires is missing from the chain - it is not fed a frame and will not record");
        }
        return found;
    }

    void render_start_demo::prepare(void* const owner, runtime::frame_services const& services, std::string_view const stage) {
        render_start_demo& self = *static_cast<render_start_demo*>(owner);
        // The stage names are the frame's own structure (the same names the runtime's stage structs carry), so this
        // switch is the frame ORDER written once, where the passes live.
        if (stage == "cluster") {
            if (self.cluster_ != nullptr) {
                self.cluster_->set_frame(services.make_cluster_frame(services.owner));
            }
        } else if (stage == "shadow") {
            if (self.shadow_ != nullptr) {
                self.shadow_->set_frame(services.make_shadow_frame(services.owner));
            }
        } else if (stage == "scene") {
            if (self.scene_ != nullptr) {
                self.scene_->set_frame(services.make_scene_frame(services.owner));
            }
        } else if (stage == "transparent") {
            if (self.transparent_ != nullptr) {
                self.transparent_->set_frame(services.make_transparent_frame(services.owner));
            }
        } else if (stage == "rt_shadow") {
            // THIS STAGE'S FRAME-ORDER DUTY: it may be the first sampler of the stored surface this frame, and
            // whoever samples it FIRST publishes the G-buffer instance's attachment writes. Gated on the same
            // predicate the runner gates the stage on, so the frame never touches them on a frame it does not run.
            if (services.feature_active != nullptr && services.feature_active(services.owner, "rt_shadow")) {
                static_cast<void>(services.ensure_gbuffer_targets_sampled(services.owner, services.cmd, services.image_index));
                static_cast<void>(services.ensure_gbuffer_depth_sampled(services.owner, services.cmd, services.image_index));
            }
        } else if (stage == "deferred") {
            if (self.deferred_ != nullptr) {
                self.deferred_->set_frame(services.make_deferred_frame(services.owner));
            }
            // ... and the same frame-order duty as the ray-traced shadow's above, for the same reason.
            static_cast<void>(services.ensure_gbuffer_targets_sampled(services.owner, services.cmd, services.image_index));
            static_cast<void>(services.ensure_gbuffer_depth_sampled(services.owner, services.cmd, services.image_index));
        } else if (stage == "taa" || stage == "gbuffer_debug") {
            // BOTH STAGES HAND THE STORED SURFACE TO SAMPLERS THIS FRAME, and both are gated exactly as the runner
            // gates the stage. The motion-vector flag is CLEARED rather than published: the GI chain later in the
            // frame is what will publish that image, and clearing it here is what stops this stage's own accessor
            // from claiming it (and, on the debug view's path, from letting the GI chain transition it twice).
            if (services.feature_active != nullptr && services.feature_active(services.owner, stage == "taa" ? "taa" : "gbuffer-debug")) {
                services.require_velocity_publish(services.owner, services.image_index);
                static_cast<void>(services.ensure_gbuffer_depth_sampled(services.owner, services.cmd, services.image_index));
            }
        } else if (stage == "gi_trace") {
            // The two writers of the raw trace. Their frames carry values that must be read BEFORE the half runs
            // (the tracer's `specular_next` decides who owes the denoiser the hand-off barrier), which is why the
            // two halves are prepared separately.
            if (self.ssgi_trace_ != nullptr) {
                self.ssgi_trace_->set_frame(services.make_ssgi_trace_frame(services.owner));
            }
            if (self.ssgi_spec_ != nullptr) {
                self.ssgi_spec_->set_frame(services.make_ssgi_spec_frame(services.owner));
            }
        } else if (stage == "gi_denoise") {
            if (self.ssgi_temporal_ != nullptr) {
                self.ssgi_temporal_->set_frame(services.make_ssgi_denoise_frame(services.owner));
            }
            if (self.ssgi_spatial_ != nullptr) {
                self.ssgi_spatial_->set_frame(services.make_ssgi_spatial_frame(services.owner));
            }
            // ---- THE FRAME'S RULE, BETWEEN THE CHAIN'S TWO HALVES ----
            // The temporal resolve is the first sampler of the G-buffer's depth (its depth guard) and of the
            // motion-vector target (its reprojection), and whether each still needs its "the G-buffer pass wrote me"
            // publication is the renderer's per-image bookkeeping. Both calls are idempotent, so on a frame another
            // stage already published them they record nothing at all.
            static_cast<void>(services.ensure_gbuffer_depth_sampled(services.owner, services.cmd, services.image_index));
            static_cast<void>(services.ensure_velocity_sampled(services.owner, services.cmd, services.image_index));
        } else if (stage == "post_composite") {
            if (self.composite_ != nullptr) {
                self.composite_->set_frame(services.make_composite_frame(services.owner));
            }
        } else if (stage == "fxaa") {
            if (self.fxaa_ != nullptr) {
                self.fxaa_->set_frame(services.make_fxaa_frame(services.owner));
            }
        }
        // A stage with no entry above is a stage whose pass wants no frame (the world-space probe cache, whose
        // declaration resolves every value it needs), which is why this is not an error.
    }

    void render_start_demo::collect(void* const owner, std::string_view const stage, runtime::frame_results& out) {
        render_start_demo& self = *static_cast<render_start_demo*>(owner);
        if (stage == "taa") {
            // The camera UBO's `prev_view_proj` is only advanced when the resolve actually wrote a history: a
            // resolve that bailed out (no descriptor set) must not claim one.
            out.taa_wrote_history = self.taa_ != nullptr && self.taa_->wrote_history();
        } else if (stage == "gi_denoise") {
            // The two answers the frame loop decides on: whether the spatial filter wrote the image the composite
            // samples (`gi_resolved`, which the composite reads as a frame constant while recording) and whether the
            // denoiser produced an accumulation this frame (the next frame's history flag is set from it).
            out.gi_resolved = self.ssgi_spatial_ != nullptr && self.ssgi_spatial_->resolved();
            out.gi_temporal_resolved = self.ssgi_temporal_ != nullptr && self.ssgi_temporal_->resolved();
        }
    }

} // namespace vulkan
