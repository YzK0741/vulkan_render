// module version: 0.1.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/compute_skin.cppm
 * @brief Compute skinning for the traced features: it writes the vertices the VERTEX shader would compute into
 *        the buffer a bottom level structure is built from, so a traced shadow follows the pose instead of
 *        standing in the bind pose.
 * @defgroup vulkan_pass_compute_skin Compute-Skinning Job
 *
 * WHY THIS IS A JOB AND NOT A `frame_pass`, exactly as `vulkan.pass.mask_bake_job` is: its work is a LIST of
 * dispatches the renderer derives from the caster set it is walking (`rt_caster_levels`), and the SAME list is
 * recorded twice with different meanings - once on the frame the structures are created (the build below reads
 * those vertices) and once per frame after (the structure is REFITTED, because only the bytes change). A frame
 * pass would have to pretend the first of those is a frame. What it shares with a pass is the ownership rule and
 * the construction path: `create` is handed the same `pass_context` a pass gets, and the job owns its pipeline
 * layout, its pipeline, the per-slot descriptor sets it writes and the ordering barrier its own write demands.
 *
 * THE SETS ARE PER FRAME SLOT, and that is load-bearing rather than tidy: binding 9 is the slot's own per-joint
 * matrix buffer (the animation writes the slot it paced), each set is written ONCE and never updated - a set
 * updated while a recording command buffer holds it invalidates that buffer, and this job runs before the frame
 * writes the scene set's binding 16. The renderer allocates them (the pool is the core's) and MOVES them in; the
 * job writes binding 9 from each slot's buffer and frees them with its own destructor.
 */

module;

#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>

export module vulkan.pass.compute_skin;

import vulkan.pass;
import vulkan.core.handles; // vk_pipeline / vk_descriptor_set: the RAII owners of what this job builds

export namespace vulkan::pass {

    /// @brief the push block, which is also `compute_skin.comp`'s
    ///
    /// Two device addresses as two 32-bit halves each, then four uints: 16 + 16 = 32 bytes. The strides are the
    /// shader's precondition, not a heuristic - it reads the joints at byte 32 and the weights at byte 48 of the
    /// engine's 64-byte interleaved vertex, so a caster whose vertices are packed differently is refused by the
    /// renderer rather than skinned with the wrong words.
    struct compute_skin_push_constants {
        glm::uvec2 source_vertices = glm::uvec2(0u); // the primitive's bind-pose vertices, low and high
        glm::uvec2 destination = glm::uvec2(0u);     // the skinned buffer this job fills
        uint32_t source_stride = 0;                  // 64: the engine's interleaved vertex
        uint32_t destination_stride = 0;             // 32: position, normal, uv
        uint32_t vertex_count = 0;
        uint32_t skin_base = 0; // this primitive's joint block in skins.matrices
    };

    static_assert(sizeof(compute_skin_push_constants) == 32, "the job's push block must stay the size the shader declares");

    /// @brief one skinned caster's dispatch: the two buffers it works between, and where its joints start
    struct compute_skin_request {
        VkDeviceAddress source_vertices = 0; // the primitive's bind-pose vertices
        VkDeviceAddress destination = 0;     // the skinned buffer this job fills (0 = not a skinned caster)
        uint32_t source_stride = 0;          // 64
        uint32_t destination_stride = 0;     // 32
        uint32_t vertex_count = 0;
        uint32_t skin_base = 0;
    };

    /**
     * @brief the compute-skinning job: it owns its pipeline layout, its pipeline, its per-slot sets
     *
     * `record` records ONE dispatch per request and then the memory barrier the acceleration structure build or
     * refit needs before it reads what those dispatches wrote (a compute SHADER_WRITE is not visible to a build
     * without it, and the symptom would be a shadow one frame behind - which reads as animation lag). It answers
     * whether it recorded anything, which is what the renderer gates the per-frame refit on.
     */
    class compute_skin_job {
    public:
        compute_skin_job() = default;
        ~compute_skin_job();

        compute_skin_job(compute_skin_job const&) = delete;
        compute_skin_job& operator=(compute_skin_job const&) = delete;
        compute_skin_job(compute_skin_job&&) = default;
        compute_skin_job& operator=(compute_skin_job&&) = delete;

        /**
         * @brief build the pipeline and write the per-slot sets' binding 9
         * @param context the same create-time context a pass gets (device, shared set layouts, shader lookup)
         * @param sets the per-slot sets, allocated by the renderer from the scene layout (the pool is the core's)
         * @param skin_buffers one handle per set, in the same order: that slot's per-joint matrix buffer
         */
        [[nodiscard]] std::expected<void, std::string> create(pass_context const& context, std::vector<vk_descriptor_set> sets, std::span<VkBuffer const> skin_buffers);
        /**
         * @brief record one dispatch per request, then the build-ordering barrier; whether anything was recorded
         * @param slot the FRAME SLOT whose per-joint buffer this recording reads (the renderer paces it; the job
         *        cannot know it, and picking the wrong one would skin against another frame's animation)
         */
        [[nodiscard]] bool record(VkCommandBuffer command_buffer, uint32_t slot, std::span<compute_skin_request const> requests) const noexcept;
        /// @brief whether the job built what it records with (the renderer's gate for skinning at all)
        [[nodiscard]] bool ready() const noexcept;

        [[nodiscard]] VkPipeline pipeline() const noexcept;
        [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept;
        /// @brief the set for a frame slot (what the job binds), or none
        [[nodiscard]] VkDescriptorSet set(uint32_t slot) const noexcept;

    private:
        static constexpr std::string_view shader_name = "compute_skin.comp.spv";
        static constexpr uint32_t group_size = 64; // `compute_skin.comp`'s local_size_x
        void release_owned() noexcept;

        VkDevice device_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        std::optional<vk_pipeline> pipeline_ = std::nullopt;
        std::vector<vk_descriptor_set> sets_ = {};
    };

} // namespace vulkan::pass
