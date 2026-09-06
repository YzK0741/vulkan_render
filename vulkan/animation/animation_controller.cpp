module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

module vulkan.animation;

import utility;

namespace vulkan {
    namespace {
        // Loop length of an animation: the latest keyframe time across its samplers. The 1.0s
        // floor is only a fallback for animations with no usable keyframes (avoids a zero-length
        // loop / fmod by zero) - a real animation shorter than 1s must keep its own duration
        // (e.g. Fox's Walk is 0.708s; flooring it to 1.0 would freeze the last 0.29s of every
        // loop on the end pose before wrapping).
        float animation_duration(gltf::animation const& animation) {
            float duration = 0.0f;
            for (gltf::animation_sampler const& sampler : animation.samplers) {
                if (!sampler.times.empty()) {
                    duration = std::max(duration, sampler.times.back());
                }
            }
            return duration > 0.0f ? duration : 1.0f;
        }

        std::string_view display_name(std::string_view const name) {
            return name.empty() ? std::string_view("<unnamed>") : name;
        }
    } // namespace

    // ---- init: playback table + rig resolution + static buffer bakes ----

    void animation_controller::init(gltf::scenes const& scenes, vulkan::runtime& runtime, glm::vec3 const& import_shift) {
        this->runtime = &runtime;
        this->import_shift = import_shift;

        // live-tree lookup: asset node index -> runtime scene nodes + root flag (import applied
        // the shift to root locals only, so animated roots must re-apply it)
        auto const collect = [this](auto&& self, vulkan::scene_tree::scene_node& node, bool const scene_root) -> void {
            this->source_nodes[node.source_index].push_back(anim_target{&node, scene_root});
            for (vulkan::scene_tree::scene_node& child : node.children) {
                self(self, child, false);
            }
        };
        for (vulkan::scene_tree::scene_node& root : runtime.get_scene().roots) {
            collect(collect, root, true);
        }

        // TRS base pose + loader node per asset node index (the loader tree stays alive)
        for (gltf::scene const& loader_scene : scenes.scene) {
            for (gltf::node const& loader_node : loader_scene.nodes) {
                this->base_poses.try_emplace(loader_node.source_index,
                                             gltf::node_pose{.translation = loader_node.translation,
                                                             .rotation = loader_node.rotation,
                                                             .scale = loader_node.scale});
                this->loader_nodes.try_emplace(loader_node.source_index, &loader_node);
            }
        }

        // playable table: channel-bearing animations, in glTF order; auto-pick the first
        for (gltf::animation const& candidate : scenes.animations) {
            if (!candidate.channels.empty()) {
                this->playable.push_back(&candidate);
            }
        }
        this->max_duration = 1.0f;
        for (gltf::animation const* playable : this->playable) {
            this->max_duration = std::max(this->max_duration, animation_duration(*playable));
        }
        if (!this->playable.empty()) {
            this->active = this->playable[0];
            this->current_index = 0;
            this->time = 0.0f;
            this->duration = animation_duration(*this->active);
            this->debug_source = this->pick_debug_source(*this->active);
            this->refresh_debug_name();
            utility::log("animation: playing '{}' ({} channels, {:.2f}s loop)", display_name(this->active->name), this->active->channels.size(), this->duration);
        }

        // Sampling fan-out pool ("lite" version, not full core count): only when the animation
        // is heavy enough that per-source sampling (each source scans all channels) is worth
        // splitting across a few workers. 2-6 threads, sized to the machine; light animations
        // (a handful of channels) stay on the caller thread - the pool sync would cost more
        // than the work. utility::thread_pool has no thread-count getter, so keep ours.
        {
            std::size_t max_channels = 0;
            for (gltf::animation const* playable : this->playable) {
                max_channels = std::max(max_channels, playable->channels.size());
            }
            if (max_channels >= 32 && this->source_nodes.size() >= 64) {
                unsigned const hw = std::thread::hardware_concurrency();
                this->pool_threads = std::clamp(hw / 4u, 2u, 6u);
                this->pool = std::make_unique<utility::thread_pool>(static_cast<int>(this->pool_threads));
                // stable source list for slicing update()'s sampling across the workers
                // (source_nodes is fixed after init; select() only swaps the active animation)
                this->sample_keys.reserve(this->source_nodes.size());
                for (auto const& [source, targets] : this->source_nodes) {
                    this->sample_keys.push_back(source);
                }
                utility::log("animation: sampling pool started ({} threads, {} channels / {} sources)", this->pool_threads, max_channels, this->sample_keys.size());
            }
        }

        // ---- skin rigs: resolve each exported skin that drives an imported mesh ----
        if (!scenes.skins.empty()) {
            uint32_t next_block = 4; // identity block occupies indices 0-3
            for (std::size_t skin_id = 0; skin_id < scenes.skins.size(); ++skin_id) {
                gltf::skin const& loader_skin = scenes.skins[skin_id];
                // the first loader node referencing this skin that is present in the tree
                std::size_t mesh_source = std::numeric_limits<std::size_t>::max();
                for (auto const& [source, loader_node] : this->loader_nodes) {
                    if (loader_node->skin_index && *loader_node->skin_index == skin_id && this->source_nodes.contains(source)) {
                        mesh_source = source;
                        break;
                    }
                }
                if (mesh_source == std::numeric_limits<std::size_t>::max()) {
                    continue; // the skin is not used by the imported scene
                }
                bool const all_joints_present = std::ranges::all_of(loader_skin.joints, [this](std::size_t const joint) { return this->source_nodes.contains(joint); });
                if (!all_joints_present) {
                    utility::log("skinning: skin '{}' skipped (joint(s) missing from the imported scene)", display_name(loader_skin.name));
                    continue;
                }
                if (static_cast<uint32_t>(loader_skin.joints.size()) > vulkan::scene_skin_capacity - next_block) {
                    utility::log("skinning: skin '{}' skipped ({} joints, skin matrix buffer capacity {} exceeded)", display_name(loader_skin.name), loader_skin.joints.size(), vulkan::scene_skin_capacity);
                    continue;
                }
                uint32_t const block_base = next_block;
                next_block += static_cast<uint32_t>(loader_skin.joints.size());
                // point every primitive leaf of the skinned node at the block: the node's own
                // leaf plus extra-primitive child leaves (import adds them under the node with
                // source_index 0); real child nodes keep skin_base 0
                vulkan::scene_tree::scene_node* const mesh_node = this->source_nodes.at(mesh_source).front().node;
                auto const assign_block = [block_base, mesh_source](auto&& self, vulkan::scene_tree::scene_node& node) -> void {
                    if (node.primitive_leaf != nullptr && (node.source_index == 0 || node.source_index == mesh_source)) {
                        static_cast<vulkan::primitive*>(node.primitive_leaf.get())->push.skin_base = block_base;
                    }
                    for (vulkan::scene_tree::scene_node& child : node.children) {
                        self(self, child);
                    }
                };
                assign_block(assign_block, *mesh_node);
                this->skin_rigs.push_back(skin_rig{&loader_skin, mesh_source, block_base});
            }
            // wanted set for the per-frame world collection: every accepted rig's mesh node +
            // every joint it references (deduplicated; fixed after this init pass)
            for (skin_rig const& rig : this->skin_rigs) {
                this->skin_sources.insert(rig.mesh_source);
                for (std::size_t const joint : rig.skin->joints) {
                    this->skin_sources.insert(joint);
                }
            }
            if (!skin_rigs.empty()) {
                utility::log("skinning: {} skin rig(s) active ({} joint matrix block(s) + identity block)", this->skin_rigs.size(), next_block - 4);
                this->skin_debug_name = std::string(display_name(this->skin_rigs.front().skin->name));
            }
        }
        // identity block for unskinned draws: upload once into EVERY slot's skin buffer
        {
            constexpr std::array<glm::mat4, 4> identity_block = {glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f)};
            for (uint32_t slot = 0; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
                runtime.set_skin_matrices(identity_block, slot);
            }
        }

        // ---- morph rigs: bake deltas + default weights into every slot's morph buffer ----
        float* const morph_scratch_mem = static_cast<float*>(runtime.morph_scratch(0));
        if (morph_scratch_mem != nullptr) {
            auto const read_delta_vec3 = [](std::map<std::string, gltf::vertex_portion> const& attrs, std::string_view const name, std::size_t const i) -> glm::vec3 {
                auto const it = attrs.find(std::string(name));
                if (it == attrs.end() || it->second.component != gltf::component_type::float_t) {
                    return glm::vec3(0.0f); // missing/unsupported delta -> no displacement
                }
                return reinterpret_cast<glm::vec3 const*>(it->second.data.data())[i];
            };
            // collect leaves per effective source (a "/prim" extra leaf inherits its parent's source)
            std::unordered_map<std::size_t, std::vector<vulkan::primitive*>> source_leaves;
            auto const collect_leaves = [&source_leaves](auto&& self, vulkan::scene_tree::scene_node& node, std::size_t const parent_source) -> void {
                bool const is_extra = node.name.ends_with("/prim");
                std::size_t const source = is_extra ? parent_source : node.source_index;
                if (node.primitive_leaf != nullptr) {
                    source_leaves[source].push_back(static_cast<vulkan::primitive*>(node.primitive_leaf.get()));
                }
                for (vulkan::scene_tree::scene_node& child : node.children) {
                    self(self, child, source);
                }
            };
            for (vulkan::scene_tree::scene_node& root : runtime.get_scene().roots) {
                collect_leaves(collect_leaves, root, 0);
            }
            std::size_t total_floats = 0;
            for (auto& [source, leaves] : source_leaves) {
                auto const loader_it = this->loader_nodes.find(source);
                if (loader_it == this->loader_nodes.end()) {
                    continue;
                }
                gltf::node const& loader_node = *loader_it->second;
                std::vector<gltf::primitive const*> loader_prims;
                for (gltf::mesh const& mesh : loader_node.meshes) {
                    for (gltf::primitive const& prim : mesh.primitives) {
                        loader_prims.push_back(&prim);
                    }
                }
                // default weights: node.weights override, else the mesh defaults, else zeros
                std::vector<float> default_weights;
                if (loader_node.weights) {
                    default_weights = *loader_node.weights;
                } else if (!loader_node.meshes.empty()) {
                    default_weights = loader_node.meshes[0].weights;
                }
                for (std::size_t i = 0; i < leaves.size() && i < loader_prims.size(); ++i) {
                    gltf::primitive const& loader_prim = *loader_prims[i];
                    if (loader_prim.targets.empty()) {
                        continue;
                    }
                    auto const pos_portion = loader_prim.vertex.find("POSITION");
                    if (pos_portion == loader_prim.vertex.end()) {
                        continue;
                    }
                    uint32_t const verts = leaves[i]->vertex_count;
                    uint32_t const target_count = static_cast<uint32_t>(loader_prim.targets.size());
                    if (pos_portion->second.data.size() / sizeof(glm::vec3) != verts) {
                        utility::log("morph: skipping primitive (vertex count mismatch with its POSITION data)");
                        continue;
                    }
                    std::size_t const delta_floats = static_cast<std::size_t>(verts) * target_count * 6u;
                    if (total_floats + delta_floats + target_count > vulkan::scene_morph_capacity) {
                        utility::log("morph: scene morph buffer capacity exceeded, remaining primitives skipped");
                        break;
                    }
                    float* dst = morph_scratch_mem + total_floats;
                    for (uint32_t v = 0; v < verts; ++v) {
                        for (uint32_t t = 0; t < target_count; ++t) {
                            glm::vec3 const dpos = read_delta_vec3(loader_prim.targets[t].attributes, "POSITION", v);
                            glm::vec3 const dnrm = read_delta_vec3(loader_prim.targets[t].attributes, "NORMAL", v);
                            *dst++ = dpos.x;
                            *dst++ = dpos.y;
                            *dst++ = dpos.z;
                            *dst++ = dnrm.x;
                            *dst++ = dnrm.y;
                            *dst++ = dnrm.z;
                        }
                    }
                    for (uint32_t t = 0; t < target_count; ++t) {
                        *dst++ = t < default_weights.size() ? default_weights[t] : 0.0f;
                    }
                    this->morph_rigs.push_back(morph_rig{leaves[i], verts, target_count, static_cast<uint32_t>(total_floats), source});
                    leaves[i]->push.morph_base = static_cast<uint32_t>(total_floats);
                    leaves[i]->push.morph_targets = target_count;
                    leaves[i]->push.morph_vertices = verts;
                    total_floats += delta_floats + target_count;
                }
            }
            if (!this->morph_rigs.empty()) {
                utility::log("morph: baked {} morphable primitive(s) into the scene morph buffer ({} floats)", this->morph_rigs.size(), total_floats);
                // duplicate the baked blocks (contiguous [0, total_floats)) into every other
                // frame slot's morph buffer: deltas are static, only the per-frame weight
                // rewrites target the active slot's buffer
                for (uint32_t slot = 1; slot < vulkan::core::MAX_FRAMES_IN_FLIGHT; ++slot) {
                    float* const other = static_cast<float*>(runtime.morph_scratch(slot));
                    if (other != nullptr) {
                        std::memcpy(other, morph_scratch_mem, total_floats * sizeof(float));
                    }
                }
            }
        }
    }

    // ---- playback table / gui binding ----

    std::size_t animation_controller::playable_count() const noexcept {
        return this->playable.size();
    }

    std::string_view animation_controller::playable_name(std::size_t const index) const noexcept {
        if (index >= this->playable.size()) {
            return {};
        }
        return display_name(this->playable[index]->name);
    }

    float animation_controller::playable_max_duration() const noexcept {
        return this->max_duration;
    }

    bool animation_controller::has_active() const noexcept {
        return this->active != nullptr;
    }

    std::size_t animation_controller::current() const noexcept {
        return this->current_index;
    }

    void animation_controller::select(std::size_t const index) {
        if (index >= this->playable.size()) {
            return;
        }
        // reset every animated node to its base pose first, so nodes the PREVIOUS animation
        // moved but the new one does not sample return to rest (same compose rule as update())
        for (auto const& [source, targets] : this->source_nodes) {
            auto const base_it = this->base_poses.find(source);
            gltf::node_pose const base = base_it == this->base_poses.end() ? gltf::node_pose{} : base_it->second;
            glm::mat4 const trs = glm::translate(glm::mat4(1.0f), base.translation) * glm::mat4_cast(base.rotation) * glm::scale(glm::mat4(1.0f), base.scale);
            for (anim_target const& target : targets) {
                target.node->local = target.scene_root ? glm::translate(glm::mat4(1.0f), this->import_shift) * trs : trs;
            }
        }
        this->runtime->scene_changed();
        this->active = this->playable[index];
        this->current_index = index;
        this->time = 0.0f;
        this->duration = animation_duration(*this->active);
        this->debug_source = this->pick_debug_source(*this->active);
        this->refresh_debug_name();
    }

    void animation_controller::set_playing(bool const playing) noexcept {
        this->playing = playing;
    }

    bool animation_controller::is_playing() const noexcept {
        return this->playing;
    }

    void animation_controller::set_time(float const t) {
        // scrub: clamp and pause so the clock does not fight the drag (gui slider semantics)
        this->time = std::clamp(t, 0.0f, this->duration);
        this->playing = false;
    }

    float animation_controller::current_time() const noexcept {
        return this->time;
    }

    float animation_controller::loop_duration() const noexcept {
        return this->duration;
    }

    // ---- per-frame drive (after pace_and_acquire(), before begin_recording()) ----

    void animation_controller::update(float const dt_seconds) {
        if (this->runtime == nullptr) {
            return;
        }

        // 1. playback: advance the clock (when playing) and sample the active animation into
        //    the scene node locals + the active slot's morph weights. Each source samples
        //    independently (channels are keyed by target node; only that source's own runtime
        //    nodes are written), so heavy animations fan the per-source sampling out over the
        //    small pool while light ones stay on this thread.
        if (this->active != nullptr) {
            if (this->playing) {
                this->time += dt_seconds;
                if (this->time >= this->duration) {
                    this->time = std::fmod(this->time, this->duration);
                }
            }
            // sample one loader source into its runtime nodes + the active slot's morph
            // weights; returns whether any node local moved (morph-only writes are not
            // "changed": they do not invalidate the culling BVH)
            auto const sample_source = [this](std::size_t const source, std::vector<anim_target> const& targets) -> bool {
                auto const base_it = this->base_poses.find(source);
                gltf::node_pose const base = base_it == this->base_poses.end() ? gltf::node_pose{} : base_it->second;
                gltf::node_pose const pose = gltf::sample_node(*this->active, source, base, this->time);
                if (!pose.weights.empty()) {
                    float* const active_scratch = static_cast<float*>(this->runtime->morph_scratch());
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
                for (anim_target const& target : targets) {
                    target.node->local = target.scene_root ? glm::translate(glm::mat4(1.0f), this->import_shift) * trs : trs;
                }
                if (source == this->debug_source) {
                    this->debug_translation = pose.translation;
                }
                return true;
            };

            bool changed = false;
            if (this->pool != nullptr && !this->sample_keys.empty()) {
                // slice the stable source list over the pool; each worker owns a contiguous
                // range and reports whether it moved any node. Different sources touch
                // different runtime nodes / morph offsets, so no shared state is written
                // concurrently (debug_translation has one writer: its own source's slice).
                std::size_t const total = this->sample_keys.size();
                unsigned const workers = this->pool_threads;
                std::vector<std::atomic<bool>> slice_changed(workers);
                for (unsigned w = 0; w < workers; ++w) {
                    std::size_t const begin = total * w / workers;
                    std::size_t const end = total * (w + 1) / workers;
                    if (begin >= end) {
                        continue;
                    }
                    this->pool->post([this, &sample_source, begin, end, &slice_changed, w] {
                        bool any = false;
                        for (std::size_t i = begin; i < end; ++i) {
                            std::size_t const source = this->sample_keys[i];
                            auto const targets_it = this->source_nodes.find(source);
                            if (targets_it != this->source_nodes.end() && sample_source(source, targets_it->second)) {
                                any = true;
                            }
                        }
                        slice_changed[static_cast<std::size_t>(w)].store(any);
                    });
                }
                this->pool->wait_until_free();
                for (std::atomic<bool> const& c : slice_changed) {
                    changed = changed || c.load();
                }
            } else {
                for (auto const& [source, targets] : this->source_nodes) {
                    changed = sample_source(source, targets) || changed;
                }
            }
            if (changed) {
                this->runtime->scene_changed(); // node.locals edited -> culling BVH must track them
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
            for (vulkan::scene_tree::scene_node& root : this->runtime->get_scene().roots) {
                collect_worlds(collect_worlds, root, glm::mat4(1.0f));
            }
            std::vector<glm::mat4> matrices;
            matrices.reserve(4 + (this->skin_rigs.size() * 8));
            matrices.insert(matrices.end(), {glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f)});
            for (skin_rig const& rig : this->skin_rigs) {
                auto const mesh_it = skin_worlds.find(rig.mesh_source);
                glm::mat4 const mesh_world_inv = glm::inverse(mesh_it == skin_worlds.end() ? glm::mat4(1.0f) : mesh_it->second);
                for (std::size_t j = 0; j < rig.skin->joints.size(); ++j) {
                    auto const joint_it = skin_worlds.find(rig.skin->joints[j]);
                    glm::mat4 const joint_world = joint_it == skin_worlds.end() ? glm::mat4(1.0f) : joint_it->second;
                    matrices.push_back(mesh_world_inv * joint_world * rig.skin->inverse_bind_matrices[j]);
                }
            }
            this->runtime->set_skin_matrices(matrices);
            // per-second report data: first rig's LAST joint world x-axis (rotations change it,
            // unlike the joint's position which stays fixed under rotation-only animations)
            this->skin_debug_valid = false;
            if (!this->skin_rigs.front().skin->joints.empty()) {
                std::size_t const last_joint = this->skin_rigs.front().skin->joints.back();
                auto const joint_it = skin_worlds.find(last_joint);
                if (joint_it != skin_worlds.end()) {
                    this->skin_debug_valid = true;
                    this->skin_debug_translation = glm::vec3(joint_it->second[0]); // world x axis
                }
            }
        }
    }

    // ---- read-only bridge (camera seeding etc.) ----

    std::unordered_map<std::size_t, gltf::node const*> const& animation_controller::get_loader_nodes() const noexcept {
        return this->loader_nodes;
    }

    bool animation_controller::has_runtime_node(std::size_t const source) const noexcept {
        return this->source_nodes.contains(source);
    }

    // ---- diagnostics ----

    std::string_view animation_controller::active_name() const noexcept {
        return this->active == nullptr ? std::string_view{} : display_name(this->active->name);
    }

    std::string_view animation_controller::get_debug_node_name() const noexcept {
        return this->debug_node_name;
    }

    glm::vec3 animation_controller::get_debug_translation() const noexcept {
        return this->debug_translation;
    }

    bool animation_controller::is_skin_debug_valid() const noexcept {
        return this->skin_debug_valid;
    }

    glm::vec3 animation_controller::get_skin_debug_translation() const noexcept {
        return this->skin_debug_translation;
    }

    std::string_view animation_controller::get_skin_debug_name() const noexcept {
        return this->skin_debug_name;
    }

    // ---- internal helpers ----

    // the node reported per second: prefer a translation channel target (its value is visible
    // in the log), fall back to the first channel target present in the tree
    std::size_t animation_controller::pick_debug_source(gltf::animation const& animation) const {
        std::size_t fallback = std::numeric_limits<std::size_t>::max();
        for (gltf::animation_channel const& channel : animation.channels) {
            if (!this->source_nodes.contains(channel.target_node)) {
                continue;
            }
            if (channel.path == gltf::animation_path::translation) {
                return channel.target_node;
            }
            if (fallback == std::numeric_limits<std::size_t>::max()) {
                fallback = channel.target_node;
            }
        }
        return fallback;
    }

    void animation_controller::refresh_debug_name() {
        this->debug_node_name.clear();
        if (this->debug_source == std::numeric_limits<std::size_t>::max()) {
            return;
        }
        auto const it = this->source_nodes.find(this->debug_source);
        if (it != this->source_nodes.end() && !it->second.empty()) {
            this->debug_node_name = it->second.front().node->name;
        }
    }
} // namespace vulkan
