// The demo's implementation: the two callbacks the runtime asks for, and nothing else. Every branch below is a
// STAGE of this application's frame - the same names the runtime's frame loop uses for its stage structs - and what
// it does is what the runtime used to do inline: hand each pass its frame (built from the runtime's own data) and
// run the frame's ordering rules that belong to that stage.
//
// Moved out of `runtime.cpp` UNCHANGED in behaviour, and the order inside each branch is deliberate: the frame is
// set FIRST, then the ordering rules run, exactly where the two used to sit relative to `record_stage` - so the
// command stream is identical and the capture gate decides the move.

module;

#include <algorithm>
#include <cstddef>
#include <string_view>

module vulkan.render_start_demo;

import utility;

namespace vulkan {

    std::size_t render_start_demo::attach(runtime& self) noexcept {
        this->passes_ = &self.frame_passes();
        this->runtime_ = &self;
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
        this->gi_probe_ = this->find<pass::gi_probe_pass>("gi_probe");
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
        found += this->gi_probe_ != nullptr ? 1u : 0u;
        found += this->composite_ != nullptr ? 1u : 0u;
        found += this->fxaa_ != nullptr ? 1u : 0u;
        if (found != 15) {
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

    // =================================================================================================
    // THE APP'S KNOBS: the flag half goes to the runtime (its policy), the value half to the pass
    // =================================================================================================
    // The CLAMPS travel with the values (they are the same fact), which is why the argument for each of them now
    // lives in the pass that owns it rather than in the setter below.

    void render_start_demo::set_taa(bool const enabled, float const blend_static, float const blend_min) noexcept {
        bool const turned_on = this->runtime_ != nullptr && this->runtime_->set_taa_enabled(enabled);
        if (this->taa_ == nullptr) {
            return;
        }
        this->taa_->set_blend(blend_static, blend_min);
        if (turned_on) {
            // THE PASS's HALF OF THE OFF -> ON EDGE: whether each image's history holds anything is the pass's own
            // state, and the renderer's half (the matrix history and the jitter index) is what its setter just reset.
            this->taa_->reset_history();
        }
    }

    void render_start_demo::set_ssgi(bool const enabled, float const intensity, float const radius, uint32_t const rays, uint32_t const steps) noexcept {
        bool const turned_on = this->runtime_ != nullptr && this->runtime_->set_ssgi_enabled(enabled);
        if (this->ssgi_trace_ != nullptr) {
            this->ssgi_trace_->set_reach(intensity, radius, rays, steps);
        }
        if (turned_on && this->ssgi_spec_ != nullptr) {
            // ... and the same edge on the lobe's side: its two outputs are re-transitioned from UNDEFINED, which is
            // what "switched on" means for a pass that has not run yet in this generation.
            this->ssgi_spec_->reset_first_use();
        }
    }

    void render_start_demo::set_ssgi_spatial(float const sigma) noexcept {
        if (this->ssgi_spatial_ != nullptr) {
            this->ssgi_spatial_->set_sigma(sigma);
        }
    }

    void render_start_demo::set_ssgi_upsample(bool const enabled) noexcept {
        if (this->composite_ != nullptr) {
            this->composite_->set_gi_upsample(enabled);
        }
    }

    void render_start_demo::set_ssgi_bounce(float const gain) noexcept {
        if (this->ssgi_trace_ != nullptr) {
            this->ssgi_trace_->set_bounce(gain);
        }
    }

    void render_start_demo::set_ssgi_probes(bool const enabled, float const rate, uint32_t const rounds, float const gain) noexcept {
        if (this->runtime_ != nullptr) {
            this->runtime_->set_ssgi_probes_enabled(enabled);
        }
        if (this->gi_probe_ != nullptr) {
            this->gi_probe_->set_rate(rate);
            // The dispatch count is the pass's own update sequence, and the renderer's clamp (0..4 propagation
            // rounds) travels with it: each round is two ping-pong dispatches, and beyond four the trust has already
            // halved away.
            this->gi_probe_->set_rounds(std::clamp(rounds, 0u, 4u));
        }
        if (this->ssgi_trace_ != nullptr) {
            // The SIGN is the cache's direction A/B rather than a mistake: |gain| is the gain, and a negative value
            // looks the cache up along the opposite direction of the ray - the same cell, the other side. The
            // tracer's clamp keeps it in [-4, 4].
            this->ssgi_trace_->set_probe_gain(gain);
        }
    }

    void render_start_demo::set_ssgi_specular(bool const enabled, uint32_t const rays, float const radius) noexcept {
        if (this->runtime_ != nullptr) {
            this->runtime_->set_ssgi_specular_enabled(enabled);
        }
        if (this->ssgi_spec_ != nullptr) {
            this->ssgi_spec_->set_reach(radius, rays);
        }
    }

    void render_start_demo::set_gbuffer_channel(int const channel) noexcept {
        if (this->gbuffer_debug_ != nullptr) {
            this->gbuffer_debug_->set_channel(channel);
        }
    }

    void render_start_demo::set_unlit(bool const unlit) noexcept {
        if (this->deferred_ != nullptr) {
            this->deferred_->set_unlit(unlit);
        }
    }

    // =================================================================================================
    // THE FEATURE TABLE (moved out of runtime::active_features / feature_active / feature_available)
    // =================================================================================================
    // Every substitution below is the same one: `this-><runtime member>` became a FIELD of the facts the runtime
    // hands over, and `this->pass_ready("name")` became a question to the typed reference this demo found - so the
    // composition is unchanged and the two inputs it is made of now sit on their own sides of the seam.

    bool render_start_demo::feature_active(void* const owner, runtime::feature_facts const& facts, std::string_view const name) {
        render_start_demo& self = *static_cast<render_start_demo*>(owner);
        // the composed answers, each read by more than one branch below
        bool const gbuffer_debug = facts.gbuffer_debug && facts.gbuffer_pipeline && self.gbuffer_debug_ != nullptr && self.gbuffer_debug_->ready();
        bool const shaded_scene = !gbuffer_debug && self.deferred_ != nullptr && self.deferred_->ready() && facts.gbuffer_pipeline;
        // THE FLAT RENDER FLAG AND THE SSAO SWITCH LIVE IN THE LIGHTING PASS (they are its parameters), so the table
        // ASKS it - the same shape `ssgi_spatial` below uses for the temporal pass's own answer. One copy of each
        // value, and the pass that pushes them is the one that owns them.
        bool const unlit = self.deferred_ != nullptr && self.deferred_->unlit();

        // THE TWO GATES THAT USED TO BE THE FIRST LINE OF A RESOLVER. Both passes were "always active" before S3,
        // with the renderer's resolver returning false to skip them; the skip is the same, but the question now has
        // one name and one answer. What is NOT in these answers is the OTHER half of those old gates - whether this
        // frame's target generation exists: that is what the resource table already says, so a frame whose images
        // are not there fails the pass's own resolution. Two mechanisms, two questions.
        if (name == "scene") {
            return facts.gbuffer_pass;
        }
        if (name == "transparent") {
            return facts.transparent_pending;
        }
        if (name == "gbuffer-debug") {
            return gbuffer_debug;
        }
        if (name == "ssgi") {
            return facts.ssgi;
        }
        if (name == "ssgi_spatial") {
            // THE CHAIN'S LAST STAGE, and the one pass whose gate is not just "the chain is on": the filter must not
            // filter a STALE accumulation, so it runs only when THIS frame's temporal resolve recorded. The answer
            // comes from the pass that owns it (the temporal pass clears its flag when the host sets the frame, so
            // this cannot read an earlier frame's answer) - which is what replaced the frame loop's
            // `if (record_ssgi_denoise_pass(...))` around the filter's stage.
            //
            // NOTE WHICH FUNCTION THIS IS: the runner asks `feature_active`, NOT `feature_available` (which answers
            // "could this feature run this session"). The first version of this branch was added to
            // `feature_available` by mistake, and the symptom was precise: the filter was skipped on every frame
            // (`skipped_inactive 1`), so the composite sampled an image nothing had written.
            return facts.ssgi && self.ssgi_temporal_ != nullptr && self.ssgi_temporal_->resolved();
        }
        if (name == "ssgi_probes") {
            return facts.ssgi_probes;
        }
        if (name == "ssgi_specular") {
            // The lobe's own feature name: "the knob is on, the frame can shade a hit, and the pass built its
            // pipeline". It has to be this exact predicate because the tracer reads it too (`specular_next`, which
            // the runtime composes into the tracer's frame) to decide who owes the denoiser the hand-off barrier.
            return facts.ssgi_specular && facts.ssgi_hit_shading && facts.ssgi_traced && self.ssgi_spec_ != nullptr && self.ssgi_spec_->ready();
        }
        if (name == "taa") {
            return facts.taa && self.taa_ != nullptr && self.taa_->ready() && shaded_scene;
        }
        if (name == "fxaa") {
            return facts.fxaa;
        }
        if (name == "shadow") {
            // The shadow map is only read by the shading stages. The flat render mode samples nothing, so recording
            // the pass would be pure waste - it measured 0.22 ms of a 0.5 ms frame.
            return facts.shadow && self.shadow_ != nullptr && self.shadow_->ready() && !unlit;
        }
        if (name == "rt_shadow") {
            // THE PASS'S GATE, in full: the knob and the extension PLUS "this frame's structure is built for the
            // slot". The second half is a FACT rather than a resource-table entry because an acceleration structure
            // is not a `resolved_binding` - it has a device address and no view, buffer or image.
            return facts.rt_shadow && self.rt_shadow_ != nullptr && self.rt_shadow_->ready() && facts.structures_ready;
        }
        if (name == "clustered") {
            // Same argument as the shadow's: flat shading reads no light list, and with no active punctual light
            // there is nothing to sort in the first place.
            return facts.clustered && self.cluster_ != nullptr && self.cluster_->ready() && facts.punctual_lights > 0.5f && !unlit;
        }
        if (name == "ssao") {
            return self.deferred_ != nullptr && self.deferred_->ssao_enabled() && shaded_scene; // shader-side gate
        }
        if (name == "bloom") {
            return facts.bloom && self.composite_ != nullptr && self.composite_->ready() && !gbuffer_debug;
        }
        if (name == "deferred") {
            // THE LIGHTING STAGE'S OWN GATE, and it is deliberately the SAME predicate the frame loop's branch uses:
            // the runner asks this before resolving the pass, and the frame loop asks it before recording the stage,
            // so one answer means the two cannot disagree about whether the lighting runs this frame.
            return facts.deferred_lit;
        }
        if (name == "unlit") {
            return unlit;
        }
        return false;
    }

    bool render_start_demo::feature_available(void* const owner, runtime::feature_facts const& facts, std::string_view const name) {
        // "CAN this feature run at all this session", which the overlay's menu and `log_feature_status()` ask - the
        // different question from `feature_active` above, and the one whose absent branch was a bound bug (the SSAO
        // group was never offered because `deferred` answered false).
        render_start_demo& self = *static_cast<render_start_demo*>(owner);
        if (name == "gbuffer-debug") {
            return facts.gbuffer_pipeline && self.gbuffer_debug_ != nullptr && self.gbuffer_debug_->ready();
        }
        if (name == "deferred") {
            return self.deferred_ != nullptr && self.deferred_->ready();
        }
        if (name == "taa") {
            return self.taa_ != nullptr && self.taa_->ready();
        }
        if (name == "fxaa") {
            return self.fxaa_ != nullptr && self.fxaa_->ready();
        }
        if (name == "shadow") {
            return self.shadow_ != nullptr && self.shadow_->ready();
        }
        if (name == "clustered") {
            return self.cluster_ != nullptr && self.cluster_->ready();
        }
        if (name == "ssgi") {
            return self.ssgi_trace_ != nullptr && self.ssgi_trace_->ready();
        }
        if (name == "ssgi_spatial") {
            return self.ssgi_spatial_ != nullptr && self.ssgi_spatial_->ready();
        }
        return false;
    }

} // namespace vulkan
