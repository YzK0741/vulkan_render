// module version: 0.1.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/deferred.cppm
 * @brief The eleventh real pass: the deferred lighting stage, which shades every pixel from the G-buffer.
 * @defgroup vulkan_pass_deferred Deferred Lighting Pass
 *
 * WHAT IT OWNS: its pipeline layout and its pipeline (built at create time from the two shared set layouts its
 * owner hands over and from its own shader, with the ADDITIVE blend the stage needs), the frame's recording - the
 * scene-colour dependency barrier, the LOAD instance over the frame's scene target, the two shared sets, the push
 * block and the draw - and its own frame data (the callback that performs the two per-image input transitions).
 *
 * THE TARGET IS THE FRAME'S, NOT THE DECLARATION'S: the host resolves `scene_color` while the TAA resolve is on and
 * `hdr` when it is off (the accessor the renderer always used), because those two paths light different images. A
 * `render_target` names one resource today, so the declaration names the TAA path's and the resolver hands over the
 * frame's; the deviation is recorded in `render_resource::deferred_io` and in docs/pass_chain_plan.md.
 *
 * WHAT IT DOES NOT OWN: the descriptor families. The scene set and the G-buffer set are shared resources whose
 * OWNERS write them, so this pass receives them resolved and binds them - and a frame with no G-buffer set does not
 * come here at all: that fallback is the renderer's, because "the owner has no set" is a failure of its pool.
 */

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <glm/glm.hpp>
#include <optional>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

export module vulkan.pass.deferred;

import vulkan.pass;
import vulkan.render_resource;
import vulkan.core.handles; // vk_pipeline: the RAII owner of the pipeline this pass builds

export namespace vulkan::pass {

    /// @brief what the renderer hands the lighting stage: the two per-image transitions its inputs need
    struct deferred_frame {
        /// the renderer's two shared per-image transitions (the three G-buffer targets and the G-buffer depth):
        /// their "was it written this frame" flags belong to the pass that WROTE those images, so this pass cannot
        /// own them - the same shape as the temporal resolve's `ensure_inputs`
        void (*ensure_inputs)(void* owner, VkCommandBuffer command_buffer, uint32_t image_index) = nullptr;
        void* owner = nullptr;
    };

    /**
     * @brief the deferred lighting stage: shade every pixel from the stored surface, ADDING to the emissive
     *
     * It runs between the G-buffer pass (whose surface it reads) and the transparent pass (which composites over
     * the shaded image); the debug view stands in for it rather than running alongside it.
     */
    class deferred_pass final : public frame_pass {
    public:
        /// @brief the push block, which is also `deferred.frag`'s
        struct push_constants {
            glm::mat4 inv_view_proj = glm::mat4(1.0f);           // clip -> world, reconstructed per pixel
            glm::vec4 ssao = glm::vec4(0.5f, 0.0f, 8.0f, 0.02f); // radius, intensity (0 = off), samples, bias
            float unlit = 0.0f;                                  // 1.0 = write the stored albedo, unshaded
            /// 1.0 = the traced GI chain is replacing BOTH ambient terms this frame (the diffuse one always, the
            /// specular one when the glossy lobe runs), so SSAO must not scale them: the chain's subtraction
            /// takes back the UN-occluded ambient, and an SSAO-darkened one left an `ambient * (ssao - 1)` term
            /// behind. Measured on the reference scene: -1.90 of mean green with 37.6% of pixels differing and
            /// 13.7% off by more than 4/255, on a frame where SSAO is supposed to do NOTHING because the rays ARE
            /// the occlusion (see shaders/deferred.frag and shaders/ssgi_spatial.comp). 0.0 everywhere else, which
            /// is what keeps the marched and the GI-off frames byte-identical.
            float gi_replaces_ambient = 0.0f;
        };

        deferred_pass() = default;
        ~deferred_pass() override;

        [[nodiscard]] render_resource::pass_io const& io() const noexcept override;
        [[nodiscard]] vulkan::pass::behaviour const& behaviour() const noexcept override;
        [[nodiscard]] std::string_view feature() const noexcept override;
        void create(pass_context const& context) override;
        void on_swapchain_recreated(pass_host const& host) override;
        void record(resolved_io const& io) override;

        /// @brief whether the pass built what it records with (the renderer gates the deferred path on this)
        [[nodiscard]] bool pipeline_ready() const noexcept;
        /// @brief the pipeline the runner binds before this pass records
        [[nodiscard]] VkPipeline pipeline() const noexcept override;
        [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept override;

        void set_frame(deferred_frame const& frame) noexcept;

    private:
        static constexpr std::string_view vertex_shader_name = "post.vert.spv"; // the synthetic fullscreen triangle
        static constexpr std::string_view fragment_shader_name = "deferred.frag.spv";

        static constexpr std::array<std::string_view, 1> pipeline_names = {"deferred"};
        inline static constexpr vulkan::pass::behaviour behaviour_ = {
            .kind = behaviour_kind::fullscreen,
            .extent = extent_rule::full, // the lighting runs at the frame's resolution
            .extent_of = resource_id::none,
            .pipelines = pipeline_names,
            .resync_viewport = true, // the runner sets the viewport and scissor from io.extent
        };
        void release_owned() noexcept;

        VkDevice device_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        std::optional<vk_pipeline> pipeline_ = std::nullopt;
        deferred_frame frame_ = {};
    };

    static_assert(sizeof(deferred_pass::push_constants) == render_resource::deferred_io.push->size,
                  "the lighting stage's declared push block must be the size of the struct the renderer composes");

} // namespace vulkan::pass