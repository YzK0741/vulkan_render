module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

module vulkan.animation;

import std;
import utility;

namespace vulkan::animation {
    // clip_duration / display_name are module-private helpers declared in the interface unit
    // (animation_controller.cppm); the non-template members below use them from here.

    // ---- sampling (format-neutral keyframe evaluation; glTF rules) ----

    channel_sample sample_channel(sampler const& sampler, channel_path const path, float const t) {
        channel_sample out = {};
        std::size_t const keys = sampler.times.size();
        if (keys == 0) {
            return out; // no keyframes: nothing to sample
        }
        // values per keyframe: the sampler records it (morph-weights channels vary per mesh);
        // fall back to the path rule when a sampler carries no per_key shape
        std::size_t comps = sampler.per_key;
        if (comps == 0) {
            comps = path == channel_path::rotation ? 4 : 3;
        }
        bool const cubic = sampler.interp == interpolation::cubic_spline;
        std::size_t const stored_per_key = comps * (cubic ? 3 : 1);
        if (sampler.values.size() < keys * stored_per_key) {
            return out; // value count does not match the key count: broken sampler
        }
        out.valid = true;

        // read one key's block: 'offset' selects the value triplet (0 for linear, comps for the
        // middle value triplet of a cubic block) or a tangent (comps / 2 * comps of a cubic block)
        auto const read_block = [&](std::size_t const key, std::size_t const offset, std::vector<float>& block) {
            block.resize(comps);
            std::size_t const base = key * stored_per_key + offset;
            for (std::size_t c = 0; c < comps; ++c) {
                block[c] = sampler.values[base + c];
            }
        };
        auto const assign = [&](std::vector<float> const& block) {
            if (path == channel_path::rotation) {
                out.quat = glm::quat(block[3], block[0], block[1], block[2]); // glm ctor order (w, x, y, z)
            } else if (path == channel_path::weights) {
                out.scalars = block; // one value per morph target
            } else {
                out.vec3 = glm::vec3(block[0], block[1], block[2]);
            }
        };

        // clamp t into the keyframe range, then find the left key: times[key] <= t < times[key + 1]
        float const time = std::clamp(t, sampler.times.front(), sampler.times.back());
        std::size_t key = 0;
        while (key + 1 < keys && sampler.times[key + 1] <= time) {
            ++key;
        }
        auto const hold_key = [&] {
            std::vector<float> value;
            read_block(key, cubic ? comps : 0, value);
            assign(value);
        };

        // STEP interpolation and the range end hold the left key's value
        if (sampler.interp == interpolation::step || key + 1 >= keys) {
            hold_key();
            return out;
        }

        float const dt = sampler.times[key + 1] - sampler.times[key];
        if (dt <= 0.0f) { // duplicate timestamps (invalid per the spec): hold the key's value
            hold_key();
            return out;
        }
        float const u = (time - sampler.times[key]) / dt;

        std::vector<float> a;
        std::vector<float> b;
        read_block(key, cubic ? comps : 0, a);
        read_block(key + 1, cubic ? comps : 0, b);

        if (cubic) {
            // Hermite spline over the segment; tangents are scaled by the segment duration
            std::vector<float> out_tangent;
            std::vector<float> in_tangent;
            read_block(key, 2 * comps, out_tangent);
            read_block(key + 1, 0, in_tangent);
            float const h00 = 2.0f * u * u * u - 3.0f * u * u + 1.0f;
            float const h10 = u * u * u - 2.0f * u * u + u;
            float const h01 = -2.0f * u * u * u + 3.0f * u * u;
            float const h11 = u * u * u - u * u;
            std::vector<float> value(comps);
            for (std::size_t c = 0; c < comps; ++c) {
                value[c] = h00 * a[c] + h10 * dt * out_tangent[c] + h01 * b[c] + h11 * dt * in_tangent[c];
            }
            if (path == channel_path::rotation) {
                // component-wise spline over the quaternion, then normalize (per the spec)
                glm::quat const q(value[3], value[0], value[1], value[2]);
                float const norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
                out.quat = norm > 0.0f ? glm::normalize(q) : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            } else {
                assign(value);
            }
            return out;
        }

        // LINEAR
        if (path == channel_path::rotation) {
            glm::quat const q0(a[3], a[0], a[1], a[2]);
            glm::quat q1(b[3], b[0], b[1], b[2]);
            if (glm::dot(q0, q1) < 0.0f) {
                q1 = glm::quat(-q1.w, -q1.x, -q1.y, -q1.z); // shortest arc: flip one endpoint
            }
            out.quat = glm::normalize(glm::slerp(q0, q1, u));
        } else {
            std::vector<float> value(comps);
            for (std::size_t c = 0; c < comps; ++c) {
                value[c] = a[c] + (b[c] - a[c]) * u;
            }
            assign(value);
        }
        return out;
    }

    node_pose sample_node(clip const& clip, std::size_t const target_node, node_pose const& base, float const t) {
        node_pose pose = base;
        for (channel const& channel : clip.channels) {
            if (channel.target_node != target_node || channel.sampler >= clip.samplers.size()) {
                continue;
            }
            channel_sample const sample = sample_channel(clip.samplers[channel.sampler], channel.path, t);
            if (!sample.valid) {
                continue;
            }
            switch (channel.path) {
            case channel_path::translation:
                pose.translation = sample.vec3;
                pose.any_transform = true;
                break;
            case channel_path::rotation:
                pose.rotation = sample.quat;
                pose.any_transform = true;
                break;
            case channel_path::scale:
                pose.scale = sample.vec3;
                pose.any_transform = true;
                break;
            case channel_path::weights:
                pose.weights = sample.scalars; // active morph weights for the node's mesh
                break;
            }
            pose.any_channel = true;
        }
        return pose;
    }

    // ---- init is a template member defined in the interface unit (animation_controller.cppm)
    //      so any importer can instantiate it with its concrete animation source ----

    // ---- playback table / gui binding ----

    std::size_t controller::playable_count() const noexcept {
        return this->playable.size();
    }

    std::string_view controller::playable_name(std::size_t const index) const noexcept {
        if (index >= this->playable.size()) {
            return {};
        }
        return display_name(this->playable[index].name);
    }

    float controller::playable_max_duration() const noexcept {
        return this->max_duration;
    }

    bool controller::has_active() const noexcept {
        return this->active != nullptr;
    }

    std::size_t controller::current() const noexcept {
        return this->current_index;
    }

    void controller::select(std::size_t const index) {
        if (index >= this->playable.size()) {
            return;
        }
        // reset every animated node to its base pose first, so nodes the PREVIOUS clip moved
        // but the new one does not sample return to rest (same compose rule as update())
        for (auto const& [source, targets] : this->source_nodes) {
            auto const base_it = this->base_poses.find(source);
            node_pose const base = base_it == this->base_poses.end() ? node_pose{} : base_it->second;
            glm::mat4 const trs = glm::translate(glm::mat4(1.0f), base.translation) * glm::mat4_cast(base.rotation) * glm::scale(glm::mat4(1.0f), base.scale);
            for (node_target const& target : targets) {
                target.node->local = target.scene_root ? glm::translate(glm::mat4(1.0f), this->import_shift) * trs : trs;
            }
        }
        this->host.scene_changed();
        this->active = &this->playable[index];
        this->current_index = index;
        this->time = 0.0f;
        this->duration = clip_duration(*this->active);
        this->debug_source = this->pick_debug_source(*this->active);
        this->refresh_debug_name();
    }

    void controller::set_playing(bool const playing) noexcept {
        this->playing = playing;
    }

    bool controller::is_playing() const noexcept {
        return this->playing;
    }

    void controller::set_time(float const t) {
        // scrub: clamp and pause so the clock does not fight the drag (gui slider semantics)
        this->time = std::clamp(t, 0.0f, this->duration);
        this->playing = false;
    }

    float controller::current_time() const noexcept {
        return this->time;
    }

    float controller::loop_duration() const noexcept {
        return this->duration;
    }

    // ---- per-frame drive (after pace_and_acquire(), before begin_recording()) ----

    // sample one loader source into its runtime nodes + the active slot's morph weights;
    // returns whether any node local moved (morph-only writes are not "changed": they do not
    // invalidate the culling BVH)
    bool controller::sample_source(std::size_t const source, std::vector<node_target> const& targets) {
        auto const base_it = this->base_poses.find(source);
        node_pose const base = base_it == this->base_poses.end() ? node_pose{} : base_it->second;
        node_pose const pose = sample_node(*this->active, source, base, this->time);
        if (!pose.weights.empty()) {
            float* const active_scratch = this->host.morph_scratch_active();
            if (active_scratch != nullptr) {
                for (morph_rig const& rig : this->morph_rigs) {
                    if (rig.source == source && static_cast<std::size_t>(rig.target_count) == pose.weights.size()) {
                        std::size_t const weight_offset = static_cast<std::size_t>(rig.morph_base) + static_cast<std::size_t>(rig.vertex_count) * static_cast<std::size_t>(rig.target_count) * 6u;
                        for (std::size_t t = 0; t < pose.weights.size(); ++t) {
                            active_scratch[weight_offset + t] = pose.weights[t];
                        }
                    }
                }
            }
        }
        if (!pose.any_transform) {
            return false; // weights-only channels don't move the node's local transform
        }
        glm::mat4 const trs = glm::translate(glm::mat4(1.0f), pose.translation) * glm::mat4_cast(pose.rotation) * glm::scale(glm::mat4(1.0f), pose.scale);
        for (node_target const& target : targets) {
            target.node->local = target.scene_root ? glm::translate(glm::mat4(1.0f), this->import_shift) * trs : trs;
        }
        if (source == this->debug_source) {
            this->debug_translation = pose.translation;
        }
        return true;
    }

    void controller::update(float const dt_seconds) {
        if (this->host.scene == nullptr) {
            return;
        }

        // 1. playback: advance the clock (when playing) and sample the active clip into the
        //    scene node locals + the active slot's morph weights. Each source samples
        //    independently (channels are keyed by target node; only that source's own runtime
        //    nodes are written), so heavy animations fan the per-source sampling out over the
        //    runtime's shared task pool while light ones stay on this thread.
        if (this->active != nullptr) {
            if (this->playing) {
                this->time += dt_seconds;
                if (this->time >= this->duration) {
                    this->time = std::fmod(this->time, this->duration);
                }
            }

            bool changed = false;
            if (this->parallel_sampling && !this->sample_keys.empty()) {
                // slice the stable source list over the runtime's shared task pool; each worker
                // owns a contiguous range and reports whether it moved any node. Different
                // sources touch different runtime nodes / morph offsets, so no shared state is
                // written concurrently (debug_translation has one writer: its own source's slice).
                // The tasks capture only `this` (+ the slice bounds): self-contained, so the
                // list can be handed to the host's run_tasks and executed on the host pool.
                std::size_t const total = this->sample_keys.size();
                unsigned const workers = std::max(1, this->host.task_worker_count());
                std::vector<std::atomic<bool>> slice_changed(workers);
                std::vector<std::function<void()>> tasks;
                tasks.reserve(workers);
                for (unsigned w = 0; w < workers; ++w) {
                    std::size_t const begin = total * w / workers;
                    std::size_t const end = total * (w + 1) / workers;
                    if (begin >= end) {
                        continue;
                    }
                    tasks.emplace_back([this, begin, end, &slice_changed, w] {
                        bool any = false;
                        for (std::size_t i = begin; i < end; ++i) {
                            std::size_t const source = this->sample_keys[i];
                            auto const targets_it = this->source_nodes.find(source);
                            if (targets_it != this->source_nodes.end() && this->sample_source(source, targets_it->second)) {
                                any = true;
                            }
                        }
                        slice_changed[static_cast<std::size_t>(w)].store(any);
                    });
                }
                // forward the task list to the host's pool and wait for this stage's
                // group: run_tasks is synchronous, so sampling finishes before update() returns
                this->host.run_tasks(tasks);
                for (std::atomic<bool> const& c : slice_changed) {
                    changed = changed || c.load();
                }
            } else {
                for (auto const& [source, targets] : this->source_nodes) {
                    changed = this->sample_source(source, targets) || changed;
                }
            }
            if (changed) {
                this->host.scene_changed(); // node.locals edited -> culling BVH must track them
            }
        }

        // 2. skin matrices: the joint worlds follow the locals above, so rebuild every frame
        //    [identity block | per-rig joint blocks] into the active slot's skin buffer
        if (!this->skin_rigs.empty()) {
            std::unordered_map<std::size_t, glm::mat4> skin_worlds;
            // collect the world matrix of every node the skin rigs need (mesh nodes + joints):
            // one O(1) set test per visited node instead of scanning rig x joint pairs per node
            auto const collect_worlds = [this, &skin_worlds](auto&& self, vulkan::scene_tree::scene_node& node, glm::mat4 const& parent_world) -> void {
                glm::mat4 const world = parent_world * node.local;
                if (this->skin_sources.contains(node.source_index)) {
                    skin_worlds.try_emplace(node.source_index, world);
                }
                for (vulkan::scene_tree::scene_node& child : node.children) {
                    self(self, child, world);
                }
            };
            for (vulkan::scene_tree::scene_node& root : this->host.scene->roots) {
                collect_worlds(collect_worlds, root, glm::mat4(1.0f));
            }
            std::vector<glm::mat4> matrices;
            matrices.reserve(4 + (this->skin_rigs.size() * 8));
            matrices.insert(matrices.end(), {glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f)});
            for (skin_rig const& rig : this->skin_rigs) {
                auto const mesh_it = skin_worlds.find(rig.mesh_source);
                glm::mat4 const mesh_world_inv = glm::inverse(mesh_it == skin_worlds.end() ? glm::mat4(1.0f) : mesh_it->second);
                for (std::size_t j = 0; j < rig.s.joints.size(); ++j) {
                    auto const joint_it = skin_worlds.find(rig.s.joints[j]);
                    glm::mat4 const joint_world = joint_it == skin_worlds.end() ? glm::mat4(1.0f) : joint_it->second;
                    matrices.push_back(mesh_world_inv * joint_world * rig.s.inverse_bind[j]);
                }
            }
            this->host.set_skin_matrices_active(matrices);
            // per-second report data: first rig's LAST joint world x-axis (rotations change it,
            // unlike the joint's position which stays fixed under rotation-only animations)
            this->skin_debug_valid = false;
            if (!this->skin_rigs.front().s.joints.empty()) {
                std::size_t const last_joint = this->skin_rigs.front().s.joints.back();
                auto const joint_it = skin_worlds.find(last_joint);
                if (joint_it != skin_worlds.end()) {
                    this->skin_debug_valid = true;
                    this->skin_debug_translation = glm::vec3(joint_it->second[0]); // world x axis
                }
            }
        }
    }

    // ---- read-only bridge ----

    bool controller::has_runtime_node(std::size_t const source) const noexcept {
        return this->source_nodes.contains(source);
    }

    // ---- diagnostics ----

    std::string_view controller::active_name() const noexcept {
        return this->active == nullptr ? std::string_view{} : display_name(this->active->name);
    }

    std::string_view controller::get_debug_node_name() const noexcept {
        return this->debug_node_name;
    }

    glm::vec3 controller::get_debug_translation() const noexcept {
        return this->debug_translation;
    }

    bool controller::is_skin_debug_valid() const noexcept {
        return this->skin_debug_valid;
    }

    glm::vec3 controller::get_skin_debug_translation() const noexcept {
        return this->skin_debug_translation;
    }

    std::string_view controller::get_skin_debug_name() const noexcept {
        return this->skin_debug_name;
    }

    // ---- internal helpers ----

    // the node reported per second: prefer a translation channel target (its value is visible
    // in the log), fall back to the first channel target present in the tree
    std::size_t controller::pick_debug_source(clip const& clip) const {
        std::size_t fallback = std::numeric_limits<std::size_t>::max();
        for (channel const& channel : clip.channels) {
            if (!this->source_nodes.contains(channel.target_node)) {
                continue;
            }
            if (channel.path == channel_path::translation) {
                return channel.target_node;
            }
            if (fallback == std::numeric_limits<std::size_t>::max()) {
                fallback = channel.target_node;
            }
        }
        return fallback;
    }

    void controller::refresh_debug_name() {
        this->debug_node_name.clear();
        if (this->debug_source == std::numeric_limits<std::size_t>::max()) {
            return;
        }
        auto const it = this->source_nodes.find(this->debug_source);
        if (it != this->source_nodes.end() && !it->second.empty()) {
            this->debug_node_name = it->second.front().node->name;
        }
    }
} // namespace vulkan::animation
