module;

#include <vulkan/vulkan.h> // VkCommandBuffer / VkPipelineLayout handle typedefs only

export module vulkan.render_environment;

export import std;

/**
 * @defgroup vulkan_render_environment Render Environment
 * @ingroup vulkan_runtime
 * @brief per-recording-session render state: the session's command buffer, which pipeline is
 * bound and how to bind others.
 *
 * One instance exists per parallel recording worker (each sub_render_task builds its own, per
 * frame), so the command buffer and the "currently bound pipeline" state are always
 * thread-local - workers never share them. The environment carries the session's command
 * buffer, available pipelines and its default, and lets a primitive's draw() bind the pipeline
 * it wants through an injected callback; the real vkCmdBindPipeline lives in that callback
 * (owned by the runtime, which captures its pipeline cache). The environment itself holds no
 * Vulkan objects - only the handle typedefs it needs to forward them.
 *
 * Bindings are deduplicated: bind_default() / bind_pipeline() no-op when the requested
 * pipeline is already the bound one, so consecutive draws sharing a pipeline do not re-bind
 * (the same saving the old per-pipeline grouping gave, now per session and per draw).
 */
namespace vulkan {
    /**
     * @ingroup vulkan_render_environment
     * @brief per-recording-session render state handed to primitive::draw().
     *
     * Members are set by the recording site (the runtime) before the session's leaves draw:
     * - command_buffer: the buffer being recorded into (each session owns its own, so draw()
     *   needs no separate command-buffer parameter)
     * - available: the scene's pipeline names (pointer to the runtime's stable name table)
     * - default_name: this pass's default pipeline ("pbr" for the main pass, the shadow
     *   pipeline for the shadow pass) - the fallback a default-semantics primitive wants
     * - bind: injected binding action; takes the pipeline name and records the real
     *   vkCmdBindPipeline (+ any per-bind dynamic state) on this session's command buffer
     * - layout: the shared scene pipeline layout (every pipeline shares it; push constants
     *   record against it, independent of which pipeline is bound)
     * - bound: the pipeline name currently bound in this session (empty = nothing bound yet)
     *
     * A "default-semantics" primitive (normal / instanced / static draw) does:
     * @code
     * if (!env.in_default_pipeline()) { env.bind_default(); }
     * // ... bind geometry, push constants via env.layout(), draw on env.command_buffer ...
     * @endcode
     * A custom primitive stores its pipeline name and does:
     * @code
     * env.bind_pipeline(this->pipeline_name);
     * @endcode
     */
    export struct render_environment {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE; // session's recording target
        // the scene's pipeline names: pointer to the runtime's stable name table (a std::deque -
        // appends never invalidate existing elements, so the pointed-to entries stay valid for the
        // session's whole lifetime even if setup-time registration runs on another thread)
        std::deque<std::string> const* available = nullptr;
        std::string_view default_name = {};                                     // this pass's default
        std::function<void(VkCommandBuffer, std::string_view)> bind = {};       // injected binder
        std::function<void(VkCommandBuffer, VkBool32)> set_depth_write_fn = {}; // injected depth-write setter
        VkPipelineLayout layout = VK_NULL_HANDLE;                               // shared scene layout
        std::string_view bound = {};                                            // currently bound name
        // depth-write state actually recorded so far. Starts "unknown" (nothing recorded yet):
        // the first set_depth_write() must ALWAYS emit vkCmdSetDepthWriteEnable even when the
        // requested state matches the pipeline default - a dynamic state that is never set is
        // invalid (VUID). known == true once any set has been recorded.
        VkBool32 depth_write_recorded = VK_TRUE;
        bool depth_write_known = false;

        /** @brief whether the session's default pipeline is the one currently bound */
        [[nodiscard]] bool in_default_pipeline() const noexcept {
            return this->bound == this->default_name;
        }

        /** @brief bind the session's default pipeline when it is not already bound */
        void bind_default() {
            if (!this->in_default_pipeline()) {
                this->bind(this->command_buffer, this->default_name);
                this->bound = this->default_name;
            }
        }

        /** @brief bind @p name when it is not the currently bound pipeline */
        void bind_pipeline(std::string_view const name) {
            if (this->bound != name) {
                this->bind(this->command_buffer, name);
                this->bound = name;
            }
        }

        /**
         * @brief record the depth-write state when it differs from what is already recorded
         * @param enabled true = depth writes on (opaque passes); false = off (transparent
         *        draws, which must not occlude later back-to-front geometry)
         */
        void set_depth_write(bool const enabled) {
            VkBool32 const want = enabled ? VK_TRUE : VK_FALSE;
            if (!this->depth_write_known || this->depth_write_recorded != want) {
                this->set_depth_write_fn(this->command_buffer, want);
                this->depth_write_recorded = want;
                this->depth_write_known = true;
            }
        }
    };
} // namespace vulkan
