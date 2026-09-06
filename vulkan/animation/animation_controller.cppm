module;

#include <glm/glm.hpp>

export module vulkan.animation;

import std;
import gltf_loader;
import utility.thread_pool; // per-frame sampling fan-out (a small 2-6 thread pool, not full core count)
import vulkan.runtime;
import vulkan.runtime.scene_tree;

/**
 * @file animation_controller.cppm
 * @defgroup vulkan_animation Vulkan Animation Controller
 * @brief bridge between glTF animation/skin/morph data (gltf_loader, pure CPU) and the
 *        scene runtime (vulkan.runtime): plays the file's keyframe animations by sampling
 *        pure CPU, writing the evaluated T/R/S back into scene node locals, and rebuilding
 *        the per-frame skin matrices + morph weights into the runtime's per-slot buffers.
 *
 * The controller is the demo/application-facing playback layer, NOT part of vulkan.runtime:
 * the runtime stays scene-format agnostic (see import_scene's docs); this module is the only
 * place that knows how glTF keyframes/skins/morph targets map onto the runtime scene.
 *
 * Contract summary (mirrors make_primitive / import_scene / set_ibl):
 *   - init() registers materials/geometry state and writes every scene set's shared buffers,
 *     so call it before the first frame, or only while the runtime is idle.
 *   - update(dt) writes the paced frame slot's skin/morph buffers and scene node locals, so
 *     call it after pace_and_acquire() and before begin_recording() (after the slot's timeline wait).
 */
namespace vulkan {
    /**
     * @ingroup vulkan_animation
     * @brief one glTF animation + its playback state, applied to the runtime scene tree
     */
    export class animation_controller {
    public:
        /**
         * @ingroup vulkan_animation
         * @brief build the playback table and resolve the skin/morph rigs against the runtime
         *        scene: collect the playable (channel-bearing) animations and TRS base poses from
         *        the loader, map asset node indices onto live scene nodes, bake the morph deltas
         *        with their default weights into every frame slot's morph buffer and upload the
         *        identity skin block into every slot's skin buffer. Skinned/morphable primitives
         *        get their push.skin_base / push.morph_* fields set here.
         * @param scenes the loaded glTF data (loader node pool + animations + skins + meshes)
         * @param runtime the scene runtime whose tree the controller drives
         * @param import_shift translation the import applied to every scene ROOT node's local
         *        (animated roots must re-apply it, like import_scene did)
         * @note call before the first frame, or only while the runtime is idle (no frame in
         *       flight) - this writes scene buffers/descriptors like make_primitive() does.
         */
        void init(gltf::scenes const& scenes, vulkan::runtime& runtime, glm::vec3 const& import_shift);

        // ---- playback table / gui binding ----

        /** @brief number of channel-bearing animations (the combo lists these) */
        [[nodiscard]] std::size_t playable_count() const noexcept;
        /** @brief display name of playable @p index ("<unnamed>" when the glTF has none) */
        [[nodiscard]] std::string_view playable_name(std::size_t index) const noexcept;
        /** @brief longest playable duration (fixed slider range, like the old demo combo) */
        [[nodiscard]] float playable_max_duration() const noexcept;
        /** @brief true when a playable animation is selected (auto-picks the first on init) */
        [[nodiscard]] bool has_active() const noexcept;
        /** @brief index of the active playable in the playable list */
        [[nodiscard]] std::size_t current() const noexcept;
        /** @brief switch to playable @p index: reset every animated node to its base pose (so
         *         nodes the previous animation moved but the new one does not return), then set
         *         time to zero. Keeps the playing flag as-is.
         */
        void select(std::size_t index);
        /** @brief pause/resume the clock (sampling keeps running while paused) */
        void set_playing(bool playing) noexcept;
        /** @brief whether the clock advances each update() */
        [[nodiscard]] bool is_playing() const noexcept;
        /** @brief scrub to @p t seconds and pause (mirrors the gui time-slider behavior: the
         *         next update() samples the new time without the clock fighting the drag) */
        void set_time(float t);
        /** @brief current playback time in seconds */
        [[nodiscard]] float current_time() const noexcept;
        /** @brief loop length of the active animation in seconds */
        [[nodiscard]] float loop_duration() const noexcept;

        /**
         * @ingroup vulkan_animation
         * @brief advance and apply one frame: sample the active animation at the (possibly
         *        advanced) time, write each animated node's T/R/S local (scene roots keep the
         *        import shift) and mark the scene changed, write the active frame slot's morph
         *        weights, then rebuild + upload the skin matrices into the active slot.
         * @param dt_seconds clock advance when playing (e.g. frame_clock::delta_seconds())
         * @note call after runtime.pace_and_acquire() and before runtime.begin_recording(): the runtime's
         *       per-slot buffers may only be written once pace_and_acquire() paced the slot.
         */
        void update(float dt_seconds);

        // ---- read-only bridge for demo-side consumers (e.g. glTF camera seeding) ----

        /** @brief loader nodes by asset node index (kept alive by @p scenes) */
        [[nodiscard]] std::unordered_map<std::size_t, gltf::node const*> const& get_loader_nodes() const noexcept;
        /** @brief whether an asset node index occurs in the runtime scene tree */
        [[nodiscard]] bool has_runtime_node(std::size_t source) const noexcept;

        // ---- per-second diagnostics (demo log lines) ----

        /** @brief display name of the active animation ("" when none) */
        [[nodiscard]] std::string_view active_name() const noexcept;
        /** @brief name of the reported animated node ("" when none) */
        [[nodiscard]] std::string_view get_debug_node_name() const noexcept;
        /** @brief translation of the reported animated node this frame */
        [[nodiscard]] glm::vec3 get_debug_translation() const noexcept;
        /** @brief true when the first skin rig's last joint world was resolved this frame */
        [[nodiscard]] bool is_skin_debug_valid() const noexcept;
        /** @brief world x-axis of the first skin rig's LAST joint (rotation debug) */
        [[nodiscard]] glm::vec3 get_skin_debug_translation() const noexcept;
        /** @brief display name of the first active skin rig ("" when none) */
        [[nodiscard]] std::string_view get_skin_debug_name() const noexcept;

    private:
        struct anim_target {
            vulkan::scene_tree::scene_node* node = nullptr;
            bool scene_root = false;
        };
        struct skin_rig {
            gltf::skin const* skin = nullptr; // loader skin: joints (asset node indices) + IBM
            std::size_t mesh_source = 0;      // asset node index of the skinned mesh node
            uint32_t block_base = 0;          // block start in the skin buffer (after identity)
        };
        struct morph_rig {
            vulkan::primitive* prim = nullptr;
            uint32_t vertex_count = 0;
            uint32_t target_count = 0;
            uint32_t morph_base = 0; // float index into the morph buffer
            std::size_t source = 0;  // owning loader node (weights animation target)
        };

        vulkan::runtime* runtime = nullptr;
        glm::vec3 import_shift{};
        // small sampling pool (2-6 threads, "lite" fan-out for heavy per-source sampling) +
        // the worker count it was created with (thread_pool exposes no getter for it)
        std::unique_ptr<utility::thread_pool> pool = nullptr;
        unsigned pool_threads = 0;
        // source keys in stable order for parallel sampling (the source set is fixed after
        // init(); sample_keys mirrors source_nodes's keys so update() can slice them)
        std::vector<std::size_t> sample_keys = {};
        std::vector<gltf::animation const*> playable = {};
        std::unordered_map<std::size_t, std::vector<anim_target>> source_nodes = {};
        std::unordered_map<std::size_t, gltf::node_pose> base_poses = {};
        std::unordered_map<std::size_t, gltf::node const*> loader_nodes = {};
        gltf::animation const* active = nullptr;
        std::size_t current_index = 0;
        float time = 0.0f;
        float duration = 1.0f;
        float max_duration = 1.0f;
        bool playing = true;
        std::size_t debug_source = std::numeric_limits<std::size_t>::max();
        glm::vec3 debug_translation{};
        std::string debug_node_name = {};
        std::vector<skin_rig> skin_rigs = {};
        std::vector<morph_rig> morph_rigs = {};
        bool skin_debug_valid = false;
        glm::vec3 skin_debug_translation{};
        std::string skin_debug_name = {};

        // the node reported per second: prefer a translation channel target, fall back to the
        // first channel target present in the tree
        std::size_t pick_debug_source(gltf::animation const& animation) const;
        void refresh_debug_name();
    };
} // namespace vulkan
