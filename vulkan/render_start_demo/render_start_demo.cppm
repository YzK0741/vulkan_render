// module version: 0.1.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/render_start_demo/render_start_demo.cppm
 * @brief THE EXAMPLE: this repository's own pass chain, wired from OUTSIDE the renderer.
 * @defgroup vulkan_render_start_demo Render Start Demo
 *
 * WHY THIS MODULE EXISTS. `vulkan.runtime` is a frame loop: it owns the device, the pacing, the images and the
 * frame's constants, and it records whatever chain of passes it is given. Everything that is specific to THIS
 * application's renderer - which passes there are, in what order, what each one is handed per frame, and which
 * knob belongs to which one - used to live inside the runtime as sixteen typed members and eighty-nine call sites.
 * It lives here now, and the runtime knows the chain only by the names its declarations carry.
 *
 * WHAT IT IS: the app's wiring, and deliberately not a framework. It looks the passes up by declaration name, keeps
 * the typed references, and implements the two callbacks the runtime asks for (`prepare`, `collect`) by switching on
 * the STAGE name - which is the frame's own structure, so the seam needs no new vocabulary. A second application
 * that wants a different chain writes a module like this one; nothing in `vulkan.pass` or `vulkan.runtime` changes.
 *
 * WHERE IT IS HEADED (see docs/pass_chain_plan.md): the knobs and their setters move here next (today they are
 * still the runtime's public API and forward into the passes), then the three descriptor families the runtime
 * still builds from a pass's set layout, and finally the CONSTRUCTION - at which point the runtime is handed the
 * chain through `set_pass_chain` instead of owning it, and the transitional `runtime::passes()` accessor goes away.
 */

module;

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vulkan/vulkan.h>

export module vulkan.render_start_demo;

import vulkan.runtime;
import vulkan.pass;
import vulkan.pass.chain; // pass_chain: the chain the runtime owns its passes in, looked up by declaration name
import vulkan.pass.cluster;
import vulkan.pass.deferred;
import vulkan.pass.fxaa;
import vulkan.pass.gbuffer_debug;
import vulkan.pass.gi_probe;
import vulkan.pass.post;
import vulkan.pass.rt_shadow;
import vulkan.pass.scene;
import vulkan.pass.shadow;
import vulkan.pass.ssgi_spatial;
import vulkan.pass.ssgi_spec;
import vulkan.pass.ssgi_temporal;
import vulkan.pass.ssgi_trace;
import vulkan.pass.taa;
import vulkan.pass.transparent;

export namespace vulkan {

    /**
     * @brief the demo's wiring: this app's passes, found by declaration name, fed stage by stage
     *
     * @note the object must outlive the renderer's recording (the runtime holds pointers into it), which is what a
     *       local in the application's own frame scope gives it
     */
    class render_start_demo {
    public:
        /**
         * @brief look this renderer's passes up by the names their declarations carry, and remember the runtime
         * @return how many of the passes it knows about were found; a pass that is missing is simply not fed, and
         *         a pass that is not fed records nothing - which is the seam's own failure mode rather than a crash
         */
        std::size_t attach(runtime& self) noexcept;

        /// @brief the two callbacks the runtime asks for, with this demo as their context
        [[nodiscard]] runtime::chain_wiring wiring() noexcept {
            return runtime::chain_wiring{.owner = this, .prepare = &render_start_demo::prepare, .collect = &render_start_demo::collect};
        }

        // =============================================================================================
        // THE APP'S KNOBS for the passes this demo owns.
        //
        // EACH OF THESE IS THE SPLIT THE HANDOVER FORCES, and it is a real one: the renderer needs some flags for its
        // OWN policy (it jitters the projection for TAA, builds the structures for GI, picks the scene target), while
        // the VALUES are read by one pass each. So the demo forwards the flag half to the runtime and keeps the value
        // half on the pass - one owner per value, and the app calls one setter rather than two.
        // =============================================================================================
        /// @brief TAA on/off (`runtime::set_taa_enabled`) + the resolve's two blend weights (the pass's)
        void set_taa(bool enabled, float blend_static, float blend_min) noexcept;
        /// @brief GI on/off + the tracer's ray budget (intensity, reach as a fraction of the scene radius, rays, steps)
        void set_ssgi(bool enabled, float intensity, float radius, uint32_t rays, uint32_t steps) noexcept;
        /// @brief the spatial filter's width in GI texels (0 = a pass-through); the pass clamps it
        void set_ssgi_spatial(float sigma) noexcept;
        /// @brief the composite's joint-bilateral GI upsample switch
        void set_ssgi_upsample(bool enabled) noexcept;
        /// @brief the multi-bounce gain the tracer re-emits at a hit (the pass clamps it to [0, 1])
        void set_ssgi_bounce(float gain) noexcept;
        /// @brief the world-space cache: its flag (the runtime's), its rate and round count (the pass's) and the
        ///        tracer's gain over it
        void set_ssgi_probes(bool enabled, float rate, uint32_t rounds, float gain) noexcept;
        /// @brief the glossy lobe: its flag (the runtime's) and its own reach and ray count (the pass's)
        void set_ssgi_specular(bool enabled, uint32_t rays, float radius) noexcept;
        /// @brief which G-buffer channel the debug view shows (the pass's own parameter)
        void set_gbuffer_channel(int channel) noexcept;
        /// @brief the flat render mode, which the lighting stage's own parameter decides
        ///
        /// NOTE the whole setter moved here rather than its value half, and that is the honest shape: the render mode
        /// is a property of the LIGHTING STAGE (the shader returns the stored albedo), and the renderer's other
        /// features ask that pass for it - so there was never a second copy in the runtime to keep.
        void set_unlit(bool unlit) noexcept;

    private:
        /// give every pass of the named stage its frame, and run the frame's ordering rules for that stage
        static void prepare(void* owner, runtime::frame_services const& services, std::string_view stage);
        /// report the stage's results back (see runtime::frame_results)
        static void collect(void* owner, std::string_view stage, runtime::frame_results& out);

        /// the typed references, looked up once by `attach` (a pass whose declaration is missing stays null)
        template <typename PassT>
        [[nodiscard]] PassT* find(std::string_view const name) noexcept {
            return static_cast<PassT*>(this->passes_ != nullptr ? this->passes_->find(name) : nullptr);
        }

        pass::pass_chain* passes_ = nullptr;
        runtime* runtime_ = nullptr; // the flag halves of the knobs above are the runtime's policy
        pass::cluster_pass* cluster_ = nullptr;
        pass::shadow_pass* shadow_ = nullptr;
        pass::scene_pass* scene_ = nullptr;
        pass::transparent_pass* transparent_ = nullptr;
        pass::rt_shadow_pass* rt_shadow_ = nullptr;
        pass::deferred_pass* deferred_ = nullptr;
        pass::taa_pass* taa_ = nullptr;
        pass::gbuffer_debug_pass* gbuffer_debug_ = nullptr;
        pass::ssgi_trace_pass* ssgi_trace_ = nullptr;
        pass::ssgi_spec_pass* ssgi_spec_ = nullptr;
        pass::ssgi_temporal_pass* ssgi_temporal_ = nullptr;
        pass::ssgi_spatial_pass* ssgi_spatial_ = nullptr;
        pass::gi_probe_pass* gi_probe_ = nullptr;
        pass::post_composite_pass* composite_ = nullptr;
        pass::fxaa_pass* fxaa_ = nullptr;
    };

} // namespace vulkan
