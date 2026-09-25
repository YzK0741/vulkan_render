/**
 * @file mmd_motion.cpp
 * @brief VMD parser implementation for vulkan.animation.mmd_motion
 *
 * Layout per the OpenMMD VMD specification (only 16 of a bone key's 64 interpolation bytes are
 * used) plus babylon-mmd's field map (src/Loader/Parser/vmdObject.ts) for the [4][4][4] indexing
 * and the trailing property-keyframe section.  Both were checked against a real 20 MB motion:
 * with those offsets every one of its 193525 keys decodes to MMD's default linear curve
 * (20,20,107,107) on all four channels, and the file parses to exactly its last byte.
 */
module;
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

module vulkan.animation.mmd_motion;

import vstd;
import utility;

namespace vulkan::animation {
    namespace {

        constexpr std::size_t vmd_header_bytes = 30;
        constexpr std::size_t vmd_model_name_bytes = 20;
        constexpr std::size_t vmd_name_bytes = 15;
        constexpr std::size_t vmd_bone_record_bytes = 111;
        constexpr std::size_t vmd_morph_record_bytes = 23;
        constexpr std::size_t vmd_camera_record_bytes = 61;
        constexpr std::size_t vmd_light_record_bytes = 28;
        constexpr std::size_t vmd_self_shadow_record_bytes = 9;
        constexpr std::size_t vmd_ik_name_bytes = 20;
        constexpr std::size_t vmd_ik_state_bytes = vmd_ik_name_bytes + 1;
        constexpr std::size_t vmd_property_header_bytes = 9; // frame u32 + visibility u8 + ik state count u32
        constexpr std::string_view vmd_signature = "Vocaloid Motion Data 0002";

        /**
         * Byte offsets of (x1, y1, x2, y2) inside a bone key's 64-byte interpolation block, per channel
         * (0 = X, 1 = Y, 2 = Z, 3 = R).
         *
         * The block is a [4][4][4] array that stores each value several times over, and the four channels
         * are NOT at a uniform stride: bytes 2 and 3 carry the physics-toggle flags, so Z's x1 and R's x1
         * live at 17 and 18 instead.  Reading every channel at the uniform stride (i.e. 2 and 3) still
         * yields a well-formed monotone curve, so the error is invisible in the output - with MMD's
         * default linear keys it silently reports (0,20,107,107) for Z and R instead of (20,20,107,107).
         */
        constexpr std::size_t vmd_bezier_x1[4] = {0, 1, 17, 18};
        constexpr std::size_t vmd_bezier_y1[4] = {4, 5, 6, 7};
        constexpr std::size_t vmd_bezier_x2[4] = {8, 9, 10, 11};
        constexpr std::size_t vmd_bezier_y2[4] = {12, 13, 14, 15};

        [[nodiscard]] std::uint32_t read_u32(std::uint8_t const* p) noexcept {
            return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8u) |
                   (static_cast<std::uint32_t>(p[2]) << 16u) | (static_cast<std::uint32_t>(p[3]) << 24u);
        }

        [[nodiscard]] float read_f32(std::uint8_t const* p) noexcept {
            std::uint32_t const bits = read_u32(p);
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }

        /** VMD pads fixed-width name fields with NULs; the Shift-JIS bytes themselves are preserved. */
        [[nodiscard]] std::string read_name(std::uint8_t const* p, std::size_t width) {
            std::size_t n = 0;
            while (n < width && p[n] != 0u) {
                ++n;
            }
            return std::string(reinterpret_cast<char const*>(p), n);
        }

        /** @brief one channel's control points, read at the offsets the format actually uses */
        [[nodiscard]] mmd_bezier read_bezier(std::uint8_t const* block, std::size_t channel) noexcept {
            return mmd_bezier{static_cast<float>(block[vmd_bezier_x1[channel]]) / 127.0f,
                              static_cast<float>(block[vmd_bezier_y1[channel]]) / 127.0f,
                              static_cast<float>(block[vmd_bezier_x2[channel]]) / 127.0f,
                              static_cast<float>(block[vmd_bezier_y2[channel]]) / 127.0f};
        }

        /** @brief index of the first key at or after @p frame, assuming @p frames is sorted ascending */
        template <class key>
        [[nodiscard]] std::size_t upper_index(std::vector<key> const& keys, float frame) noexcept {
            std::size_t hi = 1;
            while (hi < keys.size() && static_cast<float>(keys[hi].frame) < frame) {
                ++hi;
            }
            return hi;
        }

    } // namespace

    mmd_bone_track const* mmd_motion::bone(std::string_view name) const noexcept {
        for (auto const& track : bones) {
            if (track.name == name) {
                return &track;
            }
        }
        return nullptr;
    }

    bool mmd_motion::sample_bone(std::string_view name, float frame, mmd_pose& out) const noexcept {
        mmd_bone_track const* const track = bone(name);
        if (track == nullptr || track->keys.empty()) {
            return false;
        }
        auto const& keys = track->keys;
        if (frame <= static_cast<float>(keys.front().frame)) {
            out.translation = keys.front().translation;
            out.rotation = keys.front().rotation;
            return true;
        }
        if (frame >= static_cast<float>(keys.back().frame)) {
            out.translation = keys.back().translation;
            out.rotation = keys.back().rotation;
            return true;
        }
        std::size_t const hi = upper_index(keys, frame);
        mmd_bone_key const& a = keys[hi - 1];
        mmd_bone_key const& b = keys[hi];
        float const span = static_cast<float>(b.frame) - static_cast<float>(a.frame);
        float const s = span > 0.0f ? (frame - static_cast<float>(a.frame)) / span : 0.0f;
        // Each channel is eased separately: MMD stores one bezier per axis, so the three position
        // components do not share a fraction, and the quaternion has a fourth curve of its own.
        out.translation = glm::vec3(glm::mix(a.translation.x, b.translation.x, a.ease_x.evaluate(s)),
                                    glm::mix(a.translation.y, b.translation.y, a.ease_y.evaluate(s)),
                                    glm::mix(a.translation.z, b.translation.z, a.ease_z.evaluate(s)));
        out.rotation = glm::slerp(a.rotation, b.rotation, a.ease_r.evaluate(s));
        return true;
    }

    float mmd_motion::sample_morph(std::string_view name, float frame) const noexcept {
        for (auto const& track : morphs) {
            if (track.name != name) {
                continue;
            }
            auto const& keys = track.keys;
            if (keys.empty()) {
                return 0.0f;
            }
            if (frame <= static_cast<float>(keys.front().frame)) {
                return keys.front().weight;
            }
            if (frame >= static_cast<float>(keys.back().frame)) {
                return keys.back().weight;
            }
            std::size_t const hi = upper_index(keys, frame);
            mmd_morph_key const& a = keys[hi - 1];
            mmd_morph_key const& b = keys[hi];
            float const span = static_cast<float>(b.frame) - static_cast<float>(a.frame);
            float const s = span > 0.0f ? (frame - static_cast<float>(a.frame)) / span : 0.0f;
            return glm::mix(a.weight, b.weight, s); // morph channels carry no easing curve
        }
        return 0.0f;
    }

    std::string escape_mmd_name(std::string_view raw) {
        static constexpr char digits[] = "0123456789abcdef";
        std::string out;
        out.reserve(raw.size());
        for (char const c : raw) {
            auto const byte = static_cast<unsigned char>(c);
            if (byte >= 0x20u && byte < 0x7fu && byte != '\\') {
                out.push_back(static_cast<char>(byte));
            } else {
                out.push_back('\\');
                out.push_back('x');
                out.push_back(digits[(byte >> 4u) & 0xfu]);
                out.push_back(digits[byte & 0xfu]);
            }
        }
        return out;
    }

    std::optional<mmd_motion> parse_mmd_motion(std::vector<std::uint8_t> const& bytes) {
        std::uint8_t const* const d = bytes.data();
        std::size_t const size = bytes.size();
        if (size < vmd_header_bytes + vmd_model_name_bytes) {
            return std::nullopt;
        }
        if (std::memcmp(d, vmd_signature.data(), vmd_signature.size()) != 0) {
            return std::nullopt;
        }

        mmd_motion motion;
        motion.model_name = read_name(d + vmd_header_bytes, vmd_model_name_bytes);
        std::size_t o = vmd_header_bytes + vmd_model_name_bytes;

        // --- bone keyframes ---------------------------------------------------------------
        if (o + 4 > size) {
            return std::nullopt;
        }
        std::uint32_t const bone_count = read_u32(d + o);
        o += 4;
        if (bone_count > (size - o) / vmd_bone_record_bytes) {
            return std::nullopt;
        }
        std::unordered_map<std::string, std::size_t> bone_track;
        for (std::uint32_t i = 0; i < bone_count; ++i) {
            std::uint8_t const* const rec = d + o;
            std::string name = read_name(rec, vmd_name_bytes);
            mmd_bone_key key;
            key.frame = read_u32(rec + 15);
            key.translation = glm::vec3(read_f32(rec + 19), read_f32(rec + 23), read_f32(rec + 27));
            // the file stores the quaternion as x, y, z, w; glm::quat's constructor takes w first
            key.rotation = glm::quat(read_f32(rec + 43), read_f32(rec + 31), read_f32(rec + 35),
                                     read_f32(rec + 39));
            key.ease_x = read_bezier(rec + 47, 0);
            key.ease_y = read_bezier(rec + 47, 1);
            key.ease_z = read_bezier(rec + 47, 2);
            key.ease_r = read_bezier(rec + 47, 3);
            key.physics_flags = static_cast<std::uint16_t>(rec[47 + 2]) |
                                (static_cast<std::uint16_t>(rec[47 + 3]) << 8u);
            auto const [it, inserted] = bone_track.try_emplace(name, motion.bones.size());
            if (inserted) {
                motion.bones.push_back(mmd_bone_track{std::move(name), {}});
            }
            motion.bones[it->second].keys.push_back(key);
            o += vmd_bone_record_bytes;
        }

        // --- morph keyframes --------------------------------------------------------------
        if (o + 4 > size) {
            return std::nullopt;
        }
        std::uint32_t const morph_count = read_u32(d + o);
        o += 4;
        if (morph_count > (size - o) / vmd_morph_record_bytes) {
            return std::nullopt;
        }
        std::unordered_map<std::string, std::size_t> morph_track;
        for (std::uint32_t i = 0; i < morph_count; ++i) {
            std::uint8_t const* const rec = d + o;
            std::string name = read_name(rec, vmd_name_bytes);
            mmd_morph_key key;
            key.frame = read_u32(rec + 15);
            key.weight = read_f32(rec + 19);
            auto const [it, inserted] = morph_track.try_emplace(name, motion.morphs.size());
            if (inserted) {
                motion.morphs.push_back(mmd_morph_track{std::move(name), {}});
            }
            motion.morphs[it->second].keys.push_back(key);
            o += vmd_morph_record_bytes;
        }

        // --- camera, light and self-shadow sections: sized but not consumed ----------------
        auto const skip_section = [&](std::size_t record_bytes) {
            if (o + 4 > size) {
                return false;
            }
            std::uint32_t const count = read_u32(d + o);
            o += 4;
            if (count > (size - o) / record_bytes) {
                return false;
            }
            o += static_cast<std::size_t>(count) * record_bytes;
            return true;
        };
        if (!skip_section(vmd_camera_record_bytes) || !skip_section(vmd_light_record_bytes) ||
            !skip_section(vmd_self_shadow_record_bytes)) {
            return std::nullopt;
        }

        // --- property keyframes (visibility + IK on/off states) ---------------------------
        // This section is not a bare IK list: every entry is a property keyframe carrying a
        // visibility flag and its own list of IK states.  Treating it as "count then 21-byte IK
        // records" swallows the 9-byte header and then reports a perfectly plausible record holding
        // an impossible frame number - 27266 in a motion that only reaches frame 3954.
        if (o + 4 <= size) {
            std::uint32_t const property_count = read_u32(d + o);
            o += 4;
            for (std::uint32_t i = 0; i < property_count; ++i) {
                if (o + vmd_property_header_bytes > size) {
                    return std::nullopt;
                }
                std::uint32_t const frame = read_u32(d + o);
                std::uint32_t const ik_state_count = read_u32(d + o + 5); // byte 4 is visibility
                o += vmd_property_header_bytes;
                if (ik_state_count > (size - o) / vmd_ik_state_bytes) {
                    return std::nullopt;
                }
                for (std::uint32_t j = 0; j < ik_state_count; ++j) {
                    mmd_ik_event event;
                    event.name = read_name(d + o, vmd_ik_name_bytes);
                    event.frame = frame;
                    event.enabled = d[o + vmd_ik_name_bytes] != 0u;
                    motion.ik.push_back(std::move(event));
                    o += vmd_ik_state_bytes;
                }
            }
        }
        motion.unparsed_bytes = size - o;

        // Keyframes are not guaranteed to be grouped or ordered in the file, so group them above and
        // order them here; the last frame any track reaches is the timeline length.
        std::uint32_t last = 0;
        for (auto& track : motion.bones) {
            std::sort(track.keys.begin(), track.keys.end(),
                      [](mmd_bone_key const& a, mmd_bone_key const& b) { return a.frame < b.frame; });
            if (!track.keys.empty() && track.keys.back().frame > last) {
                last = track.keys.back().frame;
            }
        }
        for (auto& track : motion.morphs) {
            std::sort(track.keys.begin(), track.keys.end(),
                      [](mmd_morph_key const& a, mmd_morph_key const& b) { return a.frame < b.frame; });
            if (!track.keys.empty() && track.keys.back().frame > last) {
                last = track.keys.back().frame;
            }
        }
        motion.last_frame = last;
        return motion;
    }

    std::optional<mmd_motion> load_mmd_motion(std::string const& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return std::nullopt;
        }
        std::vector<std::uint8_t> const bytes((std::istreambuf_iterator<char>(in)),
                                              std::istreambuf_iterator<char>());
        if (bytes.empty()) {
            return std::nullopt;
        }
        return parse_mmd_motion(bytes);
    }

} // namespace vulkan::animation
