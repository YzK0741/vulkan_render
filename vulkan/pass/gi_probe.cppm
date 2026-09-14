// module version: 0.1.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/gi_probe.cppm
 * @brief The FIRST REAL PASS: the world-space radiance probe cache, and what building it found out.
 * @defgroup vulkan_pass_gi_probe World-Space Probe Cache Pass
 *
 * WHAT MOVED HERE, and why it is the probe cache that moved first: of every pass in this renderer it is the
 * one whose I/O was already written down twice - once as the hand-written binding loop in `pipelines.cppm`,
 * once as the parallel view arrays and ternaries in `runtime::ensure_gi_probe_descriptors` - and it is the
 * one the capture gate can actually decide, because `sponza_gi` runs with the cache ON. So the sequence that
 * generated its layout and its writes from `render_resource::gi_probe_io` ends here, with the pass that
 * declares that I/O owning everything derived from it: the set layout, the two-set ping-pong family, the
 * dispatch sequence, the barriers, the clear and the push.
 *
 * FOUR THINGS THE FRAMEWORK HAD TO LEARN FROM IT, each one recorded in `vulkan.pass` where it lives:
 *
 *  * A PASS THAT OWNS A FAMILY MUST BE ABLE TO ENSURE IT, which is why `record` is no longer const. The
 *    runner still hands over everything the pass needs; what changed is that ensuring a family is a mutation
 *    (it re-points when the swapchain's views changed, and it retires rather than destroys its pool).
 *  * A DESCRIPTOR TAKES A VIEW AND A BARRIER TAKES AN IMAGE, so `resolved_binding` carries both. Without the
 *    image handle the nine same-layout storage barriers this pass records are unwritable from a pass.
 *  * THE PUSH BLOCK IS THE PASS'S SHAPE AND THE RENDERER'S VALUES, so the host composes the bytes and the
 *    pass reads them as the struct it declared (`push_constants` below) - see the design doc's section 8.
 *  * THE PING-PONG IS THE PASS'S OWN FACT. The declaration describes ONE set: elements 0..3 are "the side
 *    being read" and 4..7 "the side being written", while the pass's second set is the same declaration with
 *    the two halves EXCHANGED. No declaration could express that and no host should have to know it, which is
 *    exactly why the family belongs here rather than in the runtime.
 *
 * WHAT THE PASS DOES NOT OWN, deliberately: the pipelines and their layout (`vulkan.runtime` still builds and
 * destroys them, and the pass names what it records with), and the IMAGES (they are `vulkan.core`'s, created
 * and destroyed with the target generation - the pass is handed their handles per frame and holds none).
 */

module;

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string_view>
#include <vulkan/vulkan.h>

export module vulkan.pass.gi_probe;

import vulkan.pass;
import vulkan.render_resource;
import vulkan.render_resource.shared;
import vulkan.bindings;
import vulkan.constant_init;

export namespace vulkan::pass {

    /**
     * @brief the world-space probe cache: inject every cell from its own rays, then propagate
     *
     * One shader, one pipeline, two modes (`params.w`), and a two-set ping-pong that makes the propagation's
     * barriers same-layout ones for the whole update. The cache is anchored to the SCENE (the shadow fit's
     * cube, so one set of numbers means the same thing on a 1.6-unit model and on Sponza's 18.5) and survives
     * the camera turning away, which is the whole reason it exists: the screen-space chain cannot answer for a
     * hit it cannot see.
     */
    class gi_probe_pass final : public frame_pass {
    public:
        /**
         * @brief the pass's per-frame input block, which is also the shader's push block
         *
         * The first three members are the shader's block, in its order, and the order is load-bearing: an
         * 8-byte member in the MIDDLE would pad this struct while the shader's block stays packed, which
         * shifts every lane after it and silently turns the mode lane into half of a device address read as a
         * float. `instance_table` is last among the three for exactly that reason.
         *
         * `light_dir` is appended AFTER them and the shader does not read it: it is the one input the pass
         * needs and the host owns - the global light direction the cache was filled under, which decides
         * whether the whole grid is stale (a slow EMA would otherwise hold light for a sun that has moved for
         * ~1/rate frames instead of starting over). It rides in the push block because that is the pass's
         * per-frame input channel; it lands past every lane the shader reads, so the shader's block is
         * untouched.
         */
        struct push_constants {
            glm::vec4 grid_min_cell = glm::vec4(0.0f); // xyz = cell (0,0,0)'s corner, w = cell size
            /// x = the injection rate, w = the mode (0 = inject, 1 = propagate); y and z are unused
            glm::vec4 params = glm::vec4(0.0f);
            glm::uvec2 instance_table = glm::uvec2(0u); // the instance table's device address, two halves
            glm::vec4 light_dir = glm::vec4(0.0f);      // xyz = the light the cache was filled under (NOT read by the shader)
        };

        gi_probe_pass() = default;
        ~gi_probe_pass() override;

        [[nodiscard]] render_resource::pass_io const& io() const noexcept override;
        // The return type is QUALIFIED because the member function below hides the type's name inside this
        // class, and inside another module an elaborated `struct behaviour` would read as a new declaration
        // of it (the framework's own class does not have this problem: it declares the type and the function
        // in the same scope).
        [[nodiscard]] vulkan::pass::behaviour const& behaviour() const noexcept override;
        [[nodiscard]] std::string_view feature() const noexcept override;
        void create(pass_host const& host) override;
        void on_swapchain_recreated(pass_host const& host) override;
        void record(resolved_io const& io) override;

        /**
         * @brief the set layout this pass built from its declaration
         *
         * The renderer builds the pass's pipeline layout from it, which is the one ordering constraint the
         * pass owning its layout introduces: `create` must run before the pipeline is built.
         */
        [[nodiscard]] VkDescriptorSetLayout set_layout() const noexcept;

        /// how many times the grid is propagated per frame ([render] ssgi_probe_rounds; 0 = injection only)
        void set_rounds(uint32_t rounds) noexcept;

        /**
         * @brief whether the grid holds anything the tracer may sample
         *
         * False until the update has run once IN THIS TARGET GENERATION: the grid images are created with the
         * swapchain, so a new generation is UNDEFINED and the tracer's gain is forced to 0 until then.
         */
        [[nodiscard]] bool cache_valid() const noexcept;

    private:
        /// the pass's own bindings, which the declaration numbers contiguously from zero (0..3 read, 4..7
        /// write, 8 the per-cell geometry)
        static constexpr uint32_t own_binding_count = 9;
        /// two sets per swapchain image, and they are the whole ping-pong
        static constexpr uint32_t sets_per_image = 2;
        /// how far the light direction may drift before the grid is thrown away (about 25 degrees)
        static constexpr float light_reset_cosine = 0.9f;
        /// the shader's `local_size_x/y/z` (shaders/gi_probe.comp)
        static constexpr uint32_t group_size = 4;

        /// the pipeline name the renderer resolves; the pass never builds one
        static constexpr std::array<std::string_view, 1> pipeline_names = {"gi_probe"};
        inline static constexpr vulkan::pass::behaviour behaviour_ = {
            .kind = behaviour_kind::compute,
            .group_size_x = group_size,
            .group_size_y = group_size,
            .group_size_z = group_size,
            .extent = extent_rule::resource, // the grid is 32^3, not the frame
            .extent_of = resource_id::probe_grid,
            .pipelines = pipeline_names,
            .resync_viewport = false,
        };

        VkDevice device_ = VK_NULL_HANDLE;
        render_resource::shared::sampler_set samplers_ = {};
        VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
        bindings::image_set_family family_ = {};
        /// what the host pushed this pass's values into, and what the pass owns
        uint32_t rounds_ = 2;
        bool cache_valid_ = false;
        /// the light the cache currently holds light for, and whether it holds any
        glm::vec3 light_dir_ = glm::vec3(0.0f);
        bool light_dir_valid_ = false;
    };

    /// THE DECLARATION'S NUMBER AND THE PASS'S STRUCT CANNOT DRIFT: the declaration's `push` block is what the
    /// pipeline layout's range is built from and what the host composes, so a member added here without the
    /// number being updated would push a block the layout does not cover - which is a validation error at
    /// submit, or worse, a silently truncated block.
    static_assert(sizeof(gi_probe_pass::push_constants) == render_resource::gi_probe_io.push->size,
                  "the probe cache's declared push block must be the size of the struct the pass pushes");

} // namespace vulkan::pass
