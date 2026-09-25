/**
 * @file mmd_motion.cppm
 * @brief MMD VMD motion parser: a .vmd becomes named bone/morph tracks the animation controller
 *        samples on demand.
 *
 * A VMD cannot be expressed as a glTF animation without losing information.  It addresses bones
 * by name, and it eases every channel with its own cubic bezier whose two interior control
 * points are stored per segment; glTF's CUBICSPLINE carries Hermite in/out tangents, which is a
 * different parameterisation, so the curves would have to be flattened into per-frame linear
 * keys.  That costs the animator's easing, multiplies the asset size (a single 168-bone,
 * 3955-frame motion bakes to roughly 19 MB of binary keys), and welds one motion to one
 * skeleton.  Keeping the original keys and sampling them here avoids all three: the same parsed
 * motion can be retargeted onto any rig, and nothing has to be re-exported to change a model.
 *
 * Names are kept as the raw Shift-JIS bytes the file stores.  Converting them would need a
 * Shift-JIS codec (there is no portable one in the standard library), and guessing an encoding
 * silently is worse than preserving bytes: a retarget table compares these bytes verbatim.
 * @ref escape_mmd_name renders them for logs without assuming a codepage.
 */
module;
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

export module vulkan.animation.mmd_motion;

import vstd;
import utility;

namespace vulkan::animation {

    /** @ingroup vulkan_animation
     *  @brief one channel's easing: a cubic bezier from (0,0) to (1,1) with interior control
     *         points (x1,y1) and (x2,y2), stored normalized to [0,1] */
    export struct mmd_bezier {
        float x1 = 20.0f / 127.0f;
        float y1 = 20.0f / 127.0f;
        float x2 = 107.0f / 127.0f;
        float y2 = 107.0f / 127.0f;

        /** @brief the curve MMD writes for "linear" - still a bezier, not the identity */
        static constexpr mmd_bezier linear() noexcept {
            return {};
        }

        /** @brief map normalized segment time @p t through the curve (0 -> 0, 1 -> 1) */
        [[nodiscard]] float evaluate(float t) const noexcept {
            if (t <= 0.0f) {
                return 0.0f;
            }
            if (t >= 1.0f) {
                return 1.0f;
            }
            // Bx is monotone because x1 and x2 are confined to [0,1] (MMD stores 0..127), so plain
            // bisection converges; 32 steps put the parameter far below float precision.
            float lo = 0.0f;
            float hi = 1.0f;
            for (int i = 0; i < 32; ++i) {
                float const s = 0.5f * (lo + hi);
                if (x_at(s) < t) {
                    lo = s;
                } else {
                    hi = s;
                }
            }
            return y_at(0.5f * (lo + hi));
        }

    private:
        [[nodiscard]] float x_at(float s) const noexcept {
            float const u = 1.0f - s;
            return 3.0f * u * u * s * x1 + 3.0f * u * s * s * x2 + s * s * s;
        }
        [[nodiscard]] float y_at(float s) const noexcept {
            float const u = 1.0f - s;
            return 3.0f * u * u * s * y1 + 3.0f * u * s * s * y2 + s * s * s;
        }
    };

    /** @brief one bone keyframe: frame number, pose, and the easing applied towards the *next* key */
    export struct mmd_bone_key {
        std::uint32_t frame = 0;
        glm::vec3 translation = glm::vec3(0.0f);
        glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        mmd_bezier ease_x = {};
        mmd_bezier ease_y = {};
        mmd_bezier ease_z = {};
        mmd_bezier ease_r = {};

        /** @brief the raw physics-toggle bytes (interpolation block bytes 2 and 3), kept verbatim
         *
         *  They sit in the slots a uniform channel stride would hand to Z's and R's first control
         *  point, and the available references disagree on which value means "physics on", so the
         *  pair is preserved rather than interpreted.
         */
        std::uint16_t physics_flags = 0;
    };

    /** @brief every key of one bone, ordered by frame */
    export struct mmd_bone_track {
        std::string name = {};
        std::vector<mmd_bone_key> keys = {};
    };

    /** @brief one morph (blendshape) keyframe - morph channels carry no easing curve */
    export struct mmd_morph_key {
        std::uint32_t frame = 0;
        float weight = 0.0f;
    };

    /** @brief every key of one morph, ordered by frame */
    export struct mmd_morph_track {
        std::string name = {};
        std::vector<mmd_morph_key> keys = {};
    };

    /** @brief one IK enable/disable event; @p enabled false leaves the chain on its FK keys */
    export struct mmd_ik_event {
        std::string name = {};
        std::uint32_t frame = 0;
        bool enabled = false;
    };

    /** @brief a sampled pose in MMD space (local to the bone's parent) */
    export struct mmd_pose {
        glm::vec3 translation = glm::vec3(0.0f);
        glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    };

    /**
     * @ingroup vulkan_animation
     * @brief a parsed VMD motion: named tracks, sampled on demand over MMD's 30 fps timeline
     */
    export struct mmd_motion {
        static constexpr float frames_per_second = 30.0f;

        std::string model_name = {};              // the model name the motion was authored against
        std::vector<mmd_bone_track> bones = {};   // in file order; keys sorted by frame
        std::vector<mmd_morph_track> morphs = {}; // in file order; keys sorted by frame
        std::vector<mmd_ik_event> ik = {};        // in file order
        std::uint32_t last_frame = 0;             // highest frame any track reaches

        /** @brief bytes stepped over while locating the IK section (0 for a file that walks cleanly) */
        std::size_t ik_offset_gap = 0;
        /** @brief bytes left over after the IK section; nonzero means the file is malformed */
        std::size_t unparsed_bytes = 0;

        /** @brief timeline length in seconds at @ref frames_per_second */
        [[nodiscard]] float duration_seconds() const noexcept {
            return static_cast<float>(last_frame) / frames_per_second;
        }

        /** @brief the track named @p name, or nullptr (MMD bone names are byte-exact) */
        [[nodiscard]] mmd_bone_track const* bone(std::string_view name) const noexcept;

        /**
         * @brief sample @p name at @p frame; fractional frames interpolate with each channel's easing
         * @param out receives the pose only when the call succeeds
         * @return false when there is no such track, leaving @p out untouched
         */
        [[nodiscard]] bool sample_bone(std::string_view name, float frame, mmd_pose& out) const noexcept;

        /** @brief sample a morph weight at @p frame (linear; 0 when there is no such morph) */
        [[nodiscard]] float sample_morph(std::string_view name, float frame) const noexcept;
    };

    /**
     * @brief parse @p bytes as a VMD 0002 file
     * @return the motion, or nullopt when the file is truncated, malformed, or not version 0002
     */
    export std::optional<mmd_motion> parse_mmd_motion(std::vector<std::uint8_t> const& bytes);

    /** @brief read @p path and parse it (nullopt when the file cannot be read or parsed) */
    export std::optional<mmd_motion> load_mmd_motion(std::string const& path);

    /**
     * @brief render a raw Shift-JIS name as ASCII, escaping every non-ASCII byte as @c \\xNN
     *
     * Names stay undecoded on purpose, so this is how they are logged without inventing a codepage.
     * The result is also a usable C string literal body: a hex escape followed by a literal hex
     * digit would otherwise merge into one escape (0x89 followed by 'E' is the single value 0x89E),
     * so such a pair is separated by an empty literal, and a quote is escaped rather than passed
     * through.
     */
    export std::string escape_mmd_name(std::string_view raw);

    /**
     * @brief one MMD standard bone name paired with the joint name a skeleton should carry
     *
     * @p mmd_name holds the raw Shift-JIS bytes a VMD stores, so a table built from it is
     * byte-exact instead of depending on a codepage.  @p joint_name is ASCII because glTF node
     * names live in a UTF-8 JSON string, which cannot carry those bytes.
     */
    export struct mmd_bone_alias {
        std::string_view mmd_name = {};
        std::string_view joint_name = {};
    };

    /** @brief the MMD standard humanoid subset this engine knows how to drive */
    export std::vector<mmd_bone_alias> const& mmd_bone_aliases();

    /**
     * @ingroup vulkan_animation
     * @brief which skeleton joint each bone of a parsed motion drives
     *
     * A VMD addresses bones by name, so the mapping is resolved once against a skeleton's joint
     * names (@ref build_mmd_retarget) and then read per frame.
     */
    export struct mmd_retarget {
        /** per motion bone, the joint it drives, or -1 when the bone has no counterpart */
        std::vector<std::int32_t> joint_of_bone = {};
        /** the motion bones that resolved to nothing, for reporting coverage */
        std::vector<std::size_t> unresolved_bones = {};
        std::size_t mapped = 0;
        std::size_t unmapped = 0;

        /** @brief the joint for @p bone_index, or -1 when it is unmapped or out of range */
        [[nodiscard]] std::int32_t joint_of(std::size_t bone_index) const noexcept;
    };

    /**
     * @brief resolve every bone of @p motion against @p joint_names
     *
     * Matching goes through @ref mmd_bone_aliases, so a skeleton only has to name its joints after
     * that table (mmd_head, mmd_wrist_l, ...).  Bones with no alias - the cloth, hair, accessory
     * and finger chains a motion may also carry - stay unmapped and are reported rather than
     * guessed at.
     */
    export mmd_retarget build_mmd_retarget(mmd_motion const& motion,
                                           std::vector<std::string> const& joint_names);

} // namespace vulkan::animation
