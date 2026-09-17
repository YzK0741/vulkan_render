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
#include <array>
#include <cstddef>
#include <glm/glm.hpp>
#include <span>
#include <string_view>
#include <vulkan/vulkan.h>

module vulkan.render_start_demo;

import vulkan.constant_init;
import utility;

namespace vulkan {

    std::size_t render_start_demo::attach(runtime& self) noexcept {
        this->runtime_ = &self;
        // ---- THE CHAIN, CONSTRUCTED HERE ----
        // The same passes the renderer used to construct for itself, in the order their create step must run in (the
        // chain's order IS that order). THE G-BUFFER DEBUG VIEW COMES FIRST because it is a CREATE-ORDER constraint:
        // the passes that bind the G-buffer set ask the owner for that set's LAYOUT while they are being created,
        // and the first one to ask is what makes the runtime create it.
        this->chain_.emplace<pass::gbuffer_debug_pass>();
        this->chain_.emplace<pass::shadow_pass>();
        this->chain_.emplace<pass::scene_pass>();
        this->chain_.emplace<pass::transparent_pass>();
        this->chain_.emplace<pass::ssgi_trace_pass>();
        this->chain_.emplace<pass::megalights_trace_pass>();
        this->chain_.emplace<pass::megalights_temporal_pass>();
        this->chain_.emplace<pass::ssgi_spec_pass>();
        this->chain_.emplace<pass::ssgi_temporal_pass>();
        this->chain_.emplace<pass::ssgi_spatial_pass>();
        this->chain_.emplace<pass::gi_probe_pass>();
        this->chain_.emplace<pass::taa_pass>();
        this->chain_.emplace<pass::rt_shadow_pass>();
        this->chain_.emplace<pass::cluster_pass>();
        this->chain_.emplace<pass::deferred_pass>();
        this->chain_.emplace<pass::post_composite_pass>();
        // ... the bloom chain: FOUR instances of ONE class, one per level. The LEVEL is what differs - the target it
        // writes, the transition it declares, its extent and the `mode` lane of its push block - and the order IS
        // the chain: each level reads the one before it.
        this->chain_.emplace<pass::post_bloom_pass>(0u);
        this->chain_.emplace<pass::post_bloom_pass>(1u);
        this->chain_.emplace<pass::post_bloom_pass>(2u);
        this->chain_.emplace<pass::post_bloom_pass>(3u);
        this->chain_.emplace<pass::fxaa_pass>();
        this->passes_ = &this->chain_;
        // LOOKED UP BY THE NAME THE DECLARATION CARRIES, which is the only key a chain gives: a cast is what turns
        // the declaration's owner into the type whose frame it wants. A pass this build does not have (its
        // declaration missing, or a variant of this app) stays null and is simply never fed.
        this->cluster_ = this->find<pass::cluster_pass>("cluster");
        this->shadow_ = this->find<pass::shadow_pass>("shadow");
        this->scene_ = this->find<pass::scene_pass>("scene");
        this->transparent_ = this->find<pass::transparent_pass>("transparent");
        this->rt_shadow_ = this->find<pass::rt_shadow_pass>("rt_shadow");
        this->deferred_ = this->find<pass::deferred_pass>("deferred");
        this->taa_ = this->find<pass::taa_pass>("taa");
        this->gbuffer_debug_ = this->find<pass::gbuffer_debug_pass>("gbuffer-debug");
        this->ssgi_trace_ = this->find<pass::ssgi_trace_pass>("ssgi_trace");
        this->megalights_trace_ = this->find<pass::megalights_trace_pass>("megalights_trace");
        this->megalights_temporal_ = this->find<pass::megalights_temporal_pass>("megalights_temporal");
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
        // ---- AND HAND IT OVER ----
        // `set_pass_chain` binds this chain into the runtime's own frame structure (the stage sequence, the marks,
        // the renderer's work between the stages) and takes this demo's wiring for everything the runtime does not
        // know about those passes: their frames, the stage preambles, the results, the feature table.
        self.set_pass_chain(this->chain_, this->wiring());
        // ---- THE TWO HOOKS THE PASSES CARRY, INSTALLED ONCE ----
        // NEITHER IS A PER-FRAME VALUE, which is why neither belongs in a frame any more: the overlay's draw is a
        // property of this renderer and of WHICH pass is the frame's last writer (the composite and the FXAA pass
        // both hold it, and each frame's own decision says which of them uses it), and the reflection's recording
        // is this application's second signal through the temporal pass's one pipeline. `attach` is where both are
        // known, so they are set here and the frame loop never writes into a pass's frame again.
        if (this->composite_ != nullptr) {
            this->composite_->set_overlay(self.overlay_draw());
        }
        if (this->fxaa_ != nullptr) {
            this->fxaa_->set_overlay(self.overlay_draw());
        }
        if (this->ssgi_temporal_ != nullptr) {
            this->ssgi_temporal_->set_reflection_recorder(&render_start_demo::record_reflection, this);
        }
        return found;
    }

    void render_start_demo::prepare(void* const owner, runtime::frame_services const& services, std::string_view const stage) {
        render_start_demo& self = *static_cast<render_start_demo*>(owner);
        // THE FRAME'S TOOLKIT, cached for the callbacks that are carried by a PASS's frame and therefore cannot be
        // handed it (the reflection's recording - see the member's own note).
        self.services_ = services;
        // ... and the one fact the renderer's own policy reads about a pass it no longer holds: whether the lighting
        // stage is in the flat render mode, which is what makes the GI chain pointless on such a frame. Published on
        // every stage prepare rather than once, because it is one bool and the app can flip it between frames.
        if (self.runtime_ != nullptr && self.deferred_ != nullptr) {
            self.runtime_->set_scene_unlit(self.deferred_->unlit());
        }
        // The stage names are the frame's own structure (the same names the runtime's stage structs carry), so this
        // switch is the frame ORDER written once, where the passes live. WHAT IS NOT HERE ANY MORE: the frames.
        // Every pass builds its own before this runs (see frame_pass::prepare_frame), so what is left per stage is
        // only what is genuinely this owner's - the frame-ORDER duties (whose first sample publishes an image) and
        // the one answer no one else can give the tracer (whether the probe cache may be read).
        // THE THREE FRAMES THIS DEMO STILL HANDS OVER, and they are the three the runtime still builds (see
        // `frame_services`): each carries the renderer's own recording machinery - the per-slot secondary buffers
        // for the scene and the transparent pass, the per-cascade secondaries for the shadow - so the frame cannot
        // be composed by the pass alone yet. Every other frame is the pass's own.
        if (stage == "shadow") {
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
        } else if (stage == "megalights") {
            // THE SAME FRAME-ORDER DUTY as the ray-traced shadow's above, and it is the same constraint: this stage
            // runs between the G-buffer pass and the lighting stage, so it may be the FIRST sampler of the stored
            // surface this frame - the estimator reads the albedo, the normal, the material and the depth to evaluate
            // its sampled lights, and the G-buffer instance's attachment writes have to be published before those
            // reads. Gated on the same predicate the runner gates the stage on, so a frame that does not run it
            // touches nothing. (Measured: without this the pass dispatched against images still in their attachment
            // layout, which the validation layer reported as a descriptor/ layout mismatch and which left the
            // estimate at zero.)
            if (services.feature_active != nullptr && services.feature_active(services.owner, "megalights")) {
                static_cast<void>(services.ensure_gbuffer_targets_sampled(services.owner, services.cmd, services.image_index));
                static_cast<void>(services.ensure_gbuffer_depth_sampled(services.owner, services.cmd, services.image_index));
                // ... AND THE MOTION-VECTOR TARGET, which this stage's second pass is the first sampler of: the
                // resolve reprojects its history with it. This is the ordering rule the GI chain runs between its
                // two halves; here both passes are in one stage, so the rule runs before either records - correct
                // for the same reason, because nothing between them writes that target.
                static_cast<void>(services.ensure_velocity_sampled(services.owner, services.cmd, services.image_index));
            }
        } else if (stage == "deferred") {
            // ... and the same frame-order duty as the ray-traced shadow's above, for the same reason. Its frame is
            // the pass's own now (see pass::deferred_frame / frame_pass::prepare_frame).
            static_cast<void>(services.ensure_gbuffer_targets_sampled(services.owner, services.cmd, services.image_index));
            static_cast<void>(services.ensure_gbuffer_depth_sampled(services.owner, services.cmd, services.image_index));
        } else if (stage == "taa" || stage == "gbuffer_debug") {
            // BOTH STAGES HAND THE STORED SURFACE TO SAMPLERS THIS FRAME, and both are gated exactly as the runner
            // gates the stage. The motion-vector flag is CLEARED rather than published: the GI chain later in the
            // frame is what will publish that image, and clearing it here is what stops this stage's own accessor
            // from claiming it (and, on the debug view's path, from letting the GI chain transition it twice).
            if (services.feature_active != nullptr && services.feature_active(services.owner, stage == "taa" ? "taa" : "gbuffer-debug")) {
                if (stage == "taa") {
                    // PUBLISHED, NOT CLEARED, for the TAA stage: its resolve samples the motion vectors and the
                    // host owns that transition now (the pass used to record the barrier itself, unconditionally -
                    // see vulkan/pass/taa.cpp). `ensure_velocity_sampled` does it only while the G-buffer's flag
                    // is armed and consumes it, so whichever stage samples the velocity first publishes it and
                    // the later ones are no-ops: the property that broke when the stochastic punctual lighting
                    // chain started sampling it before this stage.
                    static_cast<void>(services.ensure_velocity_sampled(services.owner, services.cmd, services.image_index));
                } else {
                    // The debug view only needs the flag out of its own accessor's way - its pass reads no motion
                    // vector - so clearing stays right for it.
                    services.require_velocity_publish(services.owner, services.image_index);
                }
                static_cast<void>(services.ensure_gbuffer_depth_sampled(services.owner, services.cmd, services.image_index));
            }
        } else if (stage == "post_composite") {
            // No frame work left: the composite composes its own (which target it writes, who draws the overlay,
            // whether this frame's bloom sum exists), and the overlay hook was installed once in `attach`.
        } else if (stage == "fxaa") {
            // ... and neither has the FXAA pass, whose frame is the overlay hook it was given in `attach`.
        }
        // A stage with no entry above is a stage whose pass wants nothing from this owner: the world-space probe
        // cache's declaration resolves every value it needs, and the frames of every other stage are the passes'
        // own (see frame_pass::prepare_frame).
    }

    void render_start_demo::collect(void* const owner, std::string_view const stage, runtime::frame_results& out) {
        render_start_demo& self = *static_cast<render_start_demo*>(owner);
        if (stage == "megalights") {
            // THE ONE ANSWER THIS STAGE GIVES THE FRAME LOOP: whether the temporal resolve wrote its accumulation,
            // which is what sets this IMAGE's history flag for the next frame. The chain's other answer
            // (`megalights_resolved`) is the run report's, because the lighting stage has to act on it in the SAME
            // frame - a distinction the two names keep: this one is about the next frame, that one about this one.
            out.megalights_temporal_resolved = self.megalights_temporal_ != nullptr && self.megalights_temporal_->resolved();
            return;
        }
        if (stage == "taa") {
            // The camera UBO's `prev_view_proj` is only advanced when the resolve actually wrote a history: a
            // resolve that bailed out (no descriptor set) must not claim one.
            out.taa_wrote_history = self.taa_ != nullptr && self.taa_->wrote_history();
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

    void render_start_demo::set_megalights(bool const enabled, uint32_t const samples, float const min_weight, float const bias_floor, float const bias_grazing) noexcept {
        // The same split `set_ssgi` makes: the FLAG is the runtime's (it decides whether the deferred lighting
        // stage adds the punctual lights itself, so it is frame state the renderer publishes), the estimator's
        // NUMBERS are the pass's and are clamped there. The return value (the off -> on edge) is ignored: there is
        // no accumulation to restart until the temporal resolve lands.
        if (this->runtime_ != nullptr) {
            static_cast<void>(this->runtime_->set_megalights_enabled(enabled));
        }
        if (this->megalights_trace_ != nullptr) {
            this->megalights_trace_->set_estimator(samples, min_weight, bias_floor, bias_grazing);
        }
    }

    void render_start_demo::set_megalights_light_angle(float const radians) noexcept {
        // The estimator owns the angle, so this forwards like the other two setters do.
        if (this->megalights_trace_ != nullptr) {
            this->megalights_trace_->set_light_angle(radians);
        }
    }

    void render_start_demo::set_megalights_accumulation(float const depth_tolerance, float const max_frames, float const spatial_sigma) noexcept {
        // The policy is the PASS's (see megalights_temporal_pass::set_accumulation), so this forwards the way the
        // estimator's own setter does.
        if (this->megalights_temporal_ != nullptr) {
            this->megalights_temporal_->set_accumulation(depth_tolerance, max_frames);
            this->megalights_temporal_->set_spatial(spatial_sigma);
        }
    }

    void render_start_demo::set_megalights_history_tolerance(float const depth_tolerance) noexcept {
        // The pass owns the accumulated value, so the tolerance moves by re-stating the policy it is part of:
        // the two other lanes keep what the pass already holds (see megalights_temporal_pass).
        if (this->megalights_temporal_ != nullptr) {
            this->megalights_temporal_->set_accumulation(depth_tolerance, this->megalights_temporal_->max_frames());
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
        // ... and the renderer's own policy reads it (see set_scene_unlit): publish it here as well as per stage, so
        // a toggle that arrives between two frames is not one frame late.
        if (this->runtime_ != nullptr) {
            this->runtime_->set_scene_unlit(unlit);
        }
    }

    void render_start_demo::set_ssao(bool const enabled, float const radius, float const intensity, uint32_t const samples) noexcept {
        // CPU-side only (the same rule as the other render-mode knobs): the values are pushed with the lighting stage
        // each frame, so they are safe to change mid-run. THE VALUES AND THEIR CLAMPS ARE THE PASS'S (see
        // deferred_pass::set_ssao); the diagnostic below is about the SESSION rather than about the pass, which is
        // why it asks the runtime's registry and reports through the runtime's once-per-session logger.
        if (this->deferred_ != nullptr) {
            this->deferred_->set_ssao(enabled, radius, intensity, samples);
        }
        if (enabled && this->runtime_ != nullptr && !this->runtime_->feature_active("deferred")) {
            this->runtime_->warn_missing_feature("ssao", "screen-space AO has no effect: the G-buffer pass or its lighting stage was not created (see the startup log)");
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
        if (name == "megalights") {
            // THE STOCHASTIC PUNCTUAL LIGHTING PASS'S OWN GATE: the runtime's composed predicate (the knob, the
            // deferred shading path, and the flat-render-mode exclusion - the pass evaluates the BRDF from the
            // G-buffer, and the flat mode's lighting stage returns the stored albedo instead) AND the pass having
            // built its pipeline. The same predicate is what the frame loop asks before recording the stage, so
            // the runner and the loop cannot disagree about whether the punctual lights were handled this frame.
            return facts.megalights && self.megalights_trace_ != nullptr && self.megalights_trace_->ready();
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
        if (name == "megalights") {
            // BOTH passes, and not just the tracer, for the reason the `ssgi` answer above includes its whole
            // chain: what the lighting stage adds is the temporal resolve's ACCUMULATION, so a chain whose
            // resolve did not build has nothing to show and the overlay must not offer a switch that would do
            // nothing. (This branch was MISSING when the widgets were added, which is why they were invisible:
            // every `visible_when` on them was false.)
            return self.megalights_trace_ != nullptr && self.megalights_trace_->ready() && self.megalights_temporal_ != nullptr && self.megalights_temporal_->ready();
        }
        if (name == "ssgi_spatial") {
            return self.ssgi_spatial_ != nullptr && self.ssgi_spatial_->ready();
        }
        return false;
    }

    // =================================================================================================
    // THE REFLECTION: this app's second signal through the temporal pass's one pipeline
    // =================================================================================================
    // MOVED VERBATIM out of `runtime::ensure_ssgi_denoise_descriptors` + `runtime::record_reflection` +
    // `runtime::record_ssgi_resolve_pass`'s mode-1 path, with exactly two substitutions: the per-image views come
    // from the frame's RESOURCE TABLE (the families the runtime publishes, in the declaration's vocabulary) instead
    // of from the core's arrays, and the toolkit (device, samplers, frame, constants) comes from `frame_services`.
    // What did NOT move is mode 0: that is the temporal PASS's own recording, which the pass does itself.

    void render_start_demo::recreated(void* const owner) {
        // The reflection's sets name this generation's images, so they are stale the moment the swapchain is
        // rebuilt - the same duty the runner performs for every pass in a chain, for the one family this demo keeps
        // outside it.
        static_cast<render_start_demo*>(owner)->reflection_family_.retire_all();
    }

    void render_start_demo::ensure_reflection_descriptors(runtime::frame_services const& services) {
        if (this->ssgi_temporal_ == nullptr || services.table == nullptr || services.device == VK_NULL_HANDLE) {
            return;
        }
        VkDescriptorSetLayout const set_layout = this->ssgi_temporal_->set_layout();
        if (!this->ssgi_temporal_->pipeline_ready() || set_layout == VK_NULL_HANDLE) {
            return;
        }
        // THE VIEWS, ONE RUN PER BINDING, from the frame's table: this is what `resource_table::views_of` is for,
        // and it is why the demo does not need the renderer's image arrays. The order is the declaration's own (the
        // seven bindings of `ssgi_temporal_io`).
        std::span<VkImageView const> const spec_trace = services.table->views_of(render_resource::resource_id::gi_spec_trace, 0);
        std::span<VkImageView const> const spec_history = services.table->views_of(render_resource::resource_id::gi_spec_history, 0);
        std::span<VkImageView const> const velocity = services.table->views_of(render_resource::resource_id::velocity, 0);
        std::span<VkImageView const> const depth = services.table->views_of(render_resource::resource_id::gbuffer_depth, 0);
        std::span<VkImageView const> const spec_resolve = services.table->views_of(render_resource::resource_id::gi_spec_resolve, 0);
        std::span<VkImageView const> const normals = services.table->views_of(render_resource::resource_id::gbuffer_targets, 1);
        std::span<VkImageView const> const spec_reproject = services.table->views_of(render_resource::resource_id::gi_spec_reproject, 0);
        std::size_t const image_count = spec_trace.size();
        if (image_count == 0 || spec_history.size() != image_count || velocity.size() != image_count || depth.size() != image_count ||
            spec_resolve.size() != image_count || normals.size() != image_count || spec_reproject.size() != image_count) {
            return; // a family this frame does not have: the reflection is not resolved this frame
        }
        // THE COUNT COMES FROM THE DECLARATION, which is what makes it impossible for the pool and the layout to
        // disagree: this number and the layout the PASS generated in its create are both derived from
        // `ssgi_temporal_io`.
        uint32_t const descriptors_per_set = render_resource::descriptor_counts_for(render_resource::ssgi_temporal_io, render_resource::ssgi_temporal_io.own_set).total();
        // Bindings 2 and 3 (the surface's motion vectors and depth) are unused in mode 1 - its reprojection carries
        // its own depth - but every binding of the layout has to name a real view, so they carry the same ones the
        // diffuse set uses; binding 6 is the lobe's reprojection, which is what mode 1 actually reprojects by.
        std::array<VkImageView, 7> const signature = {spec_trace[0], spec_history[0], velocity[0], depth[0], spec_resolve[0], normals[0], spec_reproject[0]};
        auto const write_sets = [&services, spec_trace, spec_history, velocity, depth, spec_resolve, normals, spec_reproject](uint32_t const image_index,
                                                                                                                              std::span<VkDescriptorSet const> const sets) {
            std::array<VkImageView, 7> const views = {
                spec_trace[image_index], spec_history[image_index], velocity[image_index], depth[image_index], spec_resolve[image_index], normals[image_index], spec_reproject[image_index]};
            // The write is GENERATED from the pass's declaration, exactly as the pass's own family does it: the
            // binding numbers, the descriptor types, the layouts and the sampler are the declaration's, so this
            // family cannot drift from the layout it shares with the pass.
            auto const written = bindings::write_set(services.device, render_resource::ssgi_temporal_io, render_resource::ssgi_temporal_io.own_set, sets[0], views, {}, services.samplers);
            if (!written) {
                // NO FORMAT ARGUMENT: see the note on the log above (clang 22.1.8 crashes on a formatted one here).
                utility::log("render_start_demo: the reflection's descriptor set could not be written");
            }
        };
        if (!this->reflection_family_.ensure(services.device, set_layout, static_cast<uint32_t>(image_count), 1u, descriptors_per_set, signature, write_sets)) {
            utility::log("render_start_demo: the reflection's descriptor sets are unavailable - this frame's reflection is not resolved");
        }
    }

    void render_start_demo::record_reflection(void* const owner, VkCommandBuffer const command_buffer, [[maybe_unused]] bool const history_valid) {
        render_start_demo& self = *static_cast<render_start_demo*>(owner);
        runtime::frame_services const& services = self.services_;
        // The command buffer the pass handed us IS the frame's (services.cmd was built for the same frame), so the
        // recording below reads it from the toolkit rather than threading it through - and the parameter is kept
        // because the pass's callback signature says what it passes.
        static_cast<void>(command_buffer);
        bool spec_resolved = false;
        // THE SAME PREDICATE THE LOBE'S OWN FEATURE USES (`ssgi_specular`: the knob, hit shading, the traced path and
        // the pass's pipeline): the reflection is resolved exactly when the lobe ran, which is what keeps the
        // spatial filter's `spec_weight` lane and the accumulation in step. Asking the registry rather than the
        // pass's readiness is the difference this line exists for - a lobe that is OFF must not have its
        // accumulation advanced.
        bool const lobe_runs = self.runtime_ != nullptr && self.runtime_->feature_active("ssgi_specular");
        if (lobe_runs && services.table != nullptr) {
            uint32_t const image_index = services.frame.image_index;
            VkDescriptorSet const spec_set = self.reflection_family_.set(image_index, 0);
            pass::resolved_binding const resolve = services.table->find(render_resource::resource_id::gi_spec_resolve, 0, image_index);
            pass::resolved_binding const history = services.table->find(render_resource::resource_id::gi_spec_history, 0, image_index);
            if (spec_set != VK_NULL_HANDLE && resolve.image != VK_NULL_HANDLE && history.image != VK_NULL_HANDLE) {
                spec_resolved = self.record_resolve(services, spec_set, resolve.image, history.image, history_valid);
            }
        }
        // THE FRAME'S ANSWER: the spatial filter's `spec_weight` lane is read from it, and the filter resolves AFTER
        // this callback (the temporal pass calls it at the end of its own recording), so the value it copies out of
        // the frame's constants is this frame's. See frame_constants::gi_spec_resolved.
        if (self.runtime_ != nullptr) {
            self.runtime_->set_gi_spec_resolved(spec_resolved);
        }
    }

    bool render_start_demo::record_resolve(runtime::frame_services const& services, VkDescriptorSet const set, VkImage const resolve_image, VkImage const history_image,
                                           bool const history_valid) {
        // Everything here is mode 1's: the reflection's accumulation (mode 0 is the temporal pass's own recording).
        uint32_t const gi_width = std::max(1u, services.frame.extent.width / 2u);
        uint32_t const gi_height = std::max(1u, services.frame.extent.height / 2u);

        // Layouts, all before the dispatch: resolve -> GENERAL (storage write), with SHADER_READ as the old layout
        // once the image has been resolved before and UNDEFINED on its first frame (the resolve is READ across
        // frames - the tracer samples the previous frame's copy at a hit - so this write has to KEEP its contents).
        // history -> SHADER_READ, and only on its first use for this image: the previous frame's copy left it
        // readable, so a later frame needs no barrier at all.
        std::array<VkImageMemoryBarrier2, 2> barriers = {};
        uint32_t barrier_count = 0;
        barriers[barrier_count] = history_valid ? vulkan::sampling_to_general_transition : vulkan::undefined_to_general_transition;
        barriers[barrier_count].image = resolve_image;
        ++barrier_count;
        if (!history_valid) {
            barriers[barrier_count] = vulkan::undefined_to_sampling_transition;
            barriers[barrier_count].image = history_image;
            ++barrier_count;
        }
        VkDependencyInfo const dependency = make_image_dependency_info(barrier_count, barriers.data());
        vkCmdPipelineBarrier2(services.cmd, &dependency);

        // (Mode 0's two shared per-image transitions are NOT needed here: the reflection's own reprojection carries
        // the depth its guard needs, and the diffuse dispatch - which runs first in every frame that resolves both -
        // has already published the motion-vector target.)

        VkPipelineLayout const layout = this->ssgi_temporal_->pipeline_layout();
        vkCmdBindDescriptorSets(services.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdBindPipeline(services.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->ssgi_temporal_->pipeline());

        pass::ssgi_temporal_pass::push_constants const push = {
            .history_valid = history_valid ? 1.0f : 0.0f,
            // The two weights are the temporal PASS's constants: the reflection shares that pass's pipeline and
            // shader, so both lanes have to be the same number or the two signals would be denoised differently.
            .blend_static = pass::ssgi_temporal_pass::blend_static,
            .blend_min = pass::ssgi_temporal_pass::blend_min,
            .depth_scale = services.constants->proj[2][2],
            .depth_offset = services.constants->proj[3][2],
            // Which signal this dispatch resolves: 1.0 = the reflection (see the shader's `glossy`). One pipeline
            // serves both, each with a set and a history of its own.
            .mode = 1.0f,
            // The reflection keeps its own accumulation policy: the cold-start widening is the DIFFUSE
            // signal's (its history is reprojected from the point the ray found - see ssgi_temporal.comp).
            .cold_start = 0.0f,
            .unused2 = 0.0f,
            .gi_size = glm::vec4(static_cast<float>(gi_width), static_cast<float>(gi_height),
                                 static_cast<float>(services.frame.extent.width), static_cast<float>(services.frame.extent.height))};
        vkCmdPushConstants(services.cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        // THE SHADER'S OWN WORKGROUP SIZE, taken from the pass that owns the pipeline this dispatch goes through.
        constexpr uint32_t group_size = pass::ssgi_temporal_pass::group_size;
        vkCmdDispatch(services.cmd, (gi_width + group_size - 1) / group_size, (gi_height + group_size - 1) / group_size, 1);

        // ---- the resolved image becomes the next frame's history ---- (a copy rather than a ping-pong, exactly
        // like the TAA resolve: the resolve writes the image the composite reads, so the history has to be separate,
        // and copying into it keeps every descriptor set in the frame stable)
        std::array<VkImageMemoryBarrier2, 2> copy_barriers = {};
        copy_barriers[0] = vulkan::general_to_transfer_src_transition;
        copy_barriers[0].image = resolve_image;
        copy_barriers[1] = vulkan::sampling_to_transfer_dst_transition;
        copy_barriers[1].image = history_image;
        VkDependencyInfo const copy_dependency = make_image_dependency_info(static_cast<uint32_t>(copy_barriers.size()), copy_barriers.data());
        vkCmdPipelineBarrier2(services.cmd, &copy_dependency);

        VkImageCopy const region = {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffset = {0, 0, 0},
            .extent = {gi_width, gi_height, 1},
        };
        vkCmdCopyImage(services.cmd, resolve_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, history_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // Hand both on: the resolve to the composite, the history copy to the next frame's resolve (which
        // finds it in SHADER_READ_ONLY - the transition below already moved it out of TRANSFER_DST).
        std::array<VkImageMemoryBarrier2, 2> hand_back = {};
        hand_back[0] = vulkan::transfer_src_to_sampling_transition;
        hand_back[0].image = resolve_image;
        hand_back[1] = vulkan::transfer_dst_to_sampling_transition;
        hand_back[1].image = history_image;
        VkDependencyInfo const hand_back_dependency = make_image_dependency_info(static_cast<uint32_t>(hand_back.size()), hand_back.data());
        vkCmdPipelineBarrier2(services.cmd, &hand_back_dependency);
        return true;
    }

} // namespace vulkan
