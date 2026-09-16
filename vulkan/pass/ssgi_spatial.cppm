// module version: 0.4.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/ssgi_spatial.cppm
 * @brief The eighth real pass, and the LAST stage of the GI chain: the joint-bilateral spatial filter.
 * @defgroup vulkan_pass_ssgi_spatial SSGI Spatial Filter Pass
 *
 * WHAT IT OWNS: the frame's recording - the two barriers around its storage output, the two shared sets it
 * binds, the push block's values (composed by the renderer) and the dispatch - and the fact that the frame's GI
 * became usable, which is what the composite's weight is read from. It also owns its PIPELINE LAYOUT and its
 * COMPUTE PIPELINE, built at create time from its own declaration's push-block size and the two shared set
 * layouts its owner hands over (`pass_context::shared_set_layout`) - the same shape the tracer and the glossy
 * lobe settled on. The runtime used to build this pipeline (`runtime::make_ssgi_spatial_pipeline`) and hand it
 * over through `resolved_io::pipelines`; that entry point is gone, because a handle only this pass names is this
 * pass's to build and to release.
 *
 * WHAT IT DOES NOT OWN, in the same shape as the tracer and the glossy lobe: no descriptor set of its own. Every
 * binding it uses - the normal, the depth, the accumulation it reads and the image it writes - is in the shared
 * G-buffer set, whose contents its owner writes, so the only handles it needs are the IMAGE it transitions, and
 * those arrive through `pass_io::barrier_images` (the channel the tracer's extraction added).
 */

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <glm/glm.hpp>
#include <optional>
#include <span>
#include <string_view>
#include <vulkan/vulkan.h>

export module vulkan.pass.ssgi_spatial;

import vulkan.pass;
import vulkan.render_resource;
import vulkan.core.handles; // vk_pipeline: the RAII owner of the compute pipeline this pass builds

export namespace vulkan::pass {

    /**
     * @brief what the renderer hands the filter: whether this frame's rays were TRACED or marched
     *
     * ONE BOOLEAN, and it is the frame's rather than the pass's because it is not a quality knob: the traced path
     * REPLACES the probe's ambient (so the filter must subtract it) while the marched one ADDS to that ambient (so
     * it must not). The answer is the renderer's - the ray-tracing knob, the device's ray queries and whether the
     * acceleration structures exist - and the tracer is handed the same fact for its own push, so the two stages
     * cannot disagree about which oracle produced the accumulation they are looking at.
     */
    struct ssgi_spatial_frame {
        bool traced_oracle = false;
    };

    /**
     * @brief the joint-bilateral filter over the temporal accumulation: the chain's last stage
     *
     * It runs after the temporal resolve - filtering a stale accumulation would only make the staleness smoother
     * - and its output is what the composite samples, so `feature()` is the CHAIN's feature and the renderer sets
     * `gi_resolved` from whether this pass recorded.
     */
    class ssgi_spatial_pass final : public frame_pass {
    public:
        /// @brief the push block, which is also `ssgi_spatial.comp`'s
        struct push_constants {
            float depth_scale = 0.0f;   // proj[2][2]
            float depth_offset = 0.0f;  // proj[3][2]
            float sigma_spatial = 2.0f; // in GI texels; 0 = pass-through
            float sigma_depth = 0.02f;  // relative view-depth tolerance
            float normal_power = 16.0f; // exponent on the normal agreement term
            /// 1.0 = remove the probe's ambient from the filtered result (the traced path only: the marched one
            /// is an ADDITION to that ambient, so it must not remove anything)
            float subtract_ambient = 0.0f;
            /// 1.0 while this frame's lobe produced a reflection, 0.0 otherwise: the lane that keeps a STALE
            /// accumulation out of a frame whose reflection was not resolved
            float spec_weight = 0.0f;
            float unused2 = 0.0f;
            glm::vec4 gi_size = glm::vec4(0.0f); // xy = GI extent, zw = full-res extent
        };

        ssgi_spatial_pass() = default;
        ~ssgi_spatial_pass() override;

        [[nodiscard]] render_resource::pass_io const& io() const noexcept override;
        [[nodiscard]] vulkan::pass::behaviour const& behaviour() const noexcept override;
        [[nodiscard]] std::string_view feature() const noexcept override;
        void create(pass_context const& context) override;
        void on_swapchain_recreated(pass_host const& host) override;
        void record(resolved_io const& io) override;

        /// @brief whether the filtered image was written this frame, i.e. whether the GI may be composited
        [[nodiscard]] bool resolved() const noexcept;

        /// @brief the filter's width in GI texels (0 = a pass-through), with its own clamp
        ///
        /// THE PASS'S OWN PARAMETER, by the rule the framework settled on: one pass reads it, so the pass owns it
        /// and the renderer's `set_ssgi_spatial` forwards. The OTHER two criteria - the depth tolerance and the
        /// normal exponent - are NOT here: the composite's joint-bilateral upsample uses the same pair, so they are
        /// the frame's (`frame_constants::render_settings`) and this pass reads them from `io.constants`.
        void set_sigma(float sigma) noexcept;
        [[nodiscard]] float sigma() const noexcept;

        /// @brief this frame's answer about which oracle produced the accumulation (see ssgi_spatial_frame)
        void set_frame(ssgi_spatial_frame const& frame) noexcept;

        /// @brief whether the pass built what it records with (the renderer gates the GI chain on this)
        [[nodiscard]] bool pipeline_ready() const noexcept;
        /// @brief the framework's generic form of the same question, so an owner holding only a chain can ask it
        ///        (a pass with nothing of its own to build keeps the interface's `true`; see `frame_pass::ready`)
        [[nodiscard]] bool ready() const noexcept override {
            return this->pipeline_ready();
        }
        /// @brief the pipeline the runner binds before this pass records
        [[nodiscard]] VkPipeline pipeline() const noexcept override;
        [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept override;

    private:
        static constexpr std::string_view shader_name = "ssgi_spatial.comp.spv";
        static constexpr uint32_t group_size = 8; // `ssgi_spatial.comp`'s local_size_x/y
        static constexpr uint32_t barrier_output = 0;
        static_assert(barrier_output + 1 == render_resource::ssgi_spatial_barriers.size(),
                      "the filter's barrier slots must match the declaration it indexes");

        static constexpr std::array<std::string_view, 1> pipeline_names = {"ssgi_spatial"};
        inline static constexpr vulkan::pass::behaviour behaviour_ = {
            .kind = behaviour_kind::compute,
            .group_size_x = group_size,
            .group_size_y = group_size,
            .group_size_z = 1,
            .extent = extent_rule::half, // the GI chain's resolution
            .extent_of = resource_id::none,
            .pipelines = pipeline_names,
            .resync_viewport = false,
        };
        void release_owned() noexcept;

        bool resolved_ = false;
        /// the filter's own width, clamped where it is set (0 = pass-through, which the shader's own branch reads)
        float sigma_ = 2.0f;
        ssgi_spatial_frame frame_ = {};
        VkDevice device_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        std::optional<vk_pipeline> pipeline_ = std::nullopt;
    };

    static_assert(sizeof(ssgi_spatial_pass::push_constants) == render_resource::ssgi_spatial_io.push->size,
                  "the filter's declared push block must be the size of the struct the pass composes");

} // namespace vulkan::pass
