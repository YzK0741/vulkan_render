// THE STRUCTURE PHASE'S IMPLEMENTATION: see the header for why this is a value the renderer owns rather than a
// pass. The two functions below are the ones that used to be `runtime::record_acceleration_structures` and
// `runtime::record_top_level_structure`, moved with their reasoning; the only changes are that the knobs and the
// two jobs arrive as `build_inputs` and that the failures are RETURNED rather than logged and swallowed, because
// what a failure means for a knob is the renderer's decision.
module;

#include <chrono>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

module vulkan.ray_tracing;

import utility;

namespace vulkan::ray_tracing {

    structure_set::structure_set(core& device_root) noexcept
        : device_(&device_root) {
    }

    bool structure_set::attempted() const noexcept {
        return this->attempted_;
    }

    bool structure_set::ready() const noexcept {
        return this->bottom_.has_value() && this->top_.has_value();
    }

    VkAccelerationStructureKHR structure_set::handle(uint32_t const frame_slot) const noexcept {
        return this->top_.has_value() ? this->top_->handle(frame_slot) : VK_NULL_HANDLE;
    }

    VkBuffer structure_set::instance_table(uint32_t const frame_slot) const noexcept {
        return this->top_.has_value() ? this->top_->instance_table(frame_slot) : VK_NULL_HANDLE;
    }

    std::span<caster_level const> structure_set::casters() const noexcept {
        return this->casters_;
    }

    void structure_set::abandon() noexcept {
        this->bottom_.reset();
        this->top_.reset();
        // The expansion buffers and the caster map go with the structures they belong to: a stale mapping would
        // have the instance list read geometry no structure was built from.
        this->mask_buffers_.clear();
        this->skin_buffers_.clear();
        this->skin_levels_.clear();
        this->casters_.clear();
    }

    acceleration_structure::geometry_source structure_set::caster_geometry(primitive const& caster,
                                                                           VkDeviceAddress const source_vertex_address,
                                                                           VkDeviceAddress const source_index_address,
                                                                           VkDeviceAddress const mask_address,
                                                                           uint32_t const mask_stride,
                                                                           VkDeviceAddress const skin_address,
                                                                           uint32_t const skin_stride) const noexcept {
        // The three cases, in the order they take precedence: a MASK bake replaces the geometry entirely (an
        // expanded, NON-INDEXED triangle list), a skinned caster keeps the primitive's index buffer because its
        // vertex ORDER is unchanged, and everything else is the primitive's own memory.
        if (mask_address != 0) {
            return acceleration_structure::geometry_source{.vertex_address = mask_address,
                                                           .vertex_stride = mask_stride,
                                                           .vertex_count = caster.index_count,
                                                           .index_address = 0,
                                                           .index_type = caster.index_type,
                                                           .index_count = caster.index_count};
        }
        if (skin_address != 0) {
            return acceleration_structure::geometry_source{.vertex_address = skin_address,
                                                           .vertex_stride = skin_stride,
                                                           .vertex_count = caster.vertex_count,
                                                           .index_address = source_index_address,
                                                           .index_type = caster.index_type,
                                                           .index_count = caster.index_count};
        }
        return acceleration_structure::geometry_source{.vertex_address = source_vertex_address,
                                                       .vertex_stride = caster.vertex_stride,
                                                       .vertex_count = caster.vertex_count,
                                                       .index_address = source_index_address,
                                                       .index_type = caster.index_type,
                                                       .index_count = caster.index_count};
    }

    std::expected<void, failure> structure_set::build(VkCommandBuffer const command_buffer, build_inputs const& inputs) {
        // BUILT ONCE, SUCCESS OR FAILURE: a device that refused the build is not asked again, and no log repeats.
        if (this->attempted_) {
            return {};
        }
        this->attempted_ = true;

        core& vk = *this->device_;
        auto const start = std::chrono::steady_clock::now();
        this->bottom_.emplace(vk);
        // The top level structure is per FRAME SLOT (see its class docs): with frames in flight one buffer would
        // be rewritten by the frame being recorded while the previous one still reads it.
        this->top_.emplace(vk, vulkan::core::MAX_FRAMES_IN_FLIGHT);
        auto& structures = *this->bottom_;

        uint32_t skipped_no_address = 0;
        uint32_t skipped_no_stride = 0;
        // The mask bake: how many casters had an alphaMode MASK baked into their geometry, and how many could not
        // be (an allocation failure falls back to the documented solid behaviour rather than failing the build).
        uint32_t mask_baked = 0;
        uint32_t skipped_mask_buffers = 0;
        bool mask_bakes_recorded = false;
        // ... and the same two counters for the skinned casters (see the SKINNED branch below).
        uint32_t skinned_baked = 0;
        uint32_t skipped_skin_buffers = 0;
        for (primitive const* caster : inputs.casters) {
            if (caster == nullptr) {
                continue;
            }
            VkBufferDeviceAddressInfo vertex_address_info = {};
            vertex_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            VkBufferDeviceAddressInfo index_address_info = vertex_address_info;

            auto const* const vertex_detail = caster->vertex_detail;
            auto const* const index_detail = caster->index_detail;
            if (vertex_detail == nullptr || index_detail == nullptr || vertex_detail->buffer == VK_NULL_HANDLE || index_detail->buffer == VK_NULL_HANDLE) {
                ++skipped_no_address;
                continue;
            }
            // A buffer only has a device address when it was created with SHADER_DEVICE_ADDRESS_BIT, which the
            // primitive uploads add when this device has the extensions - so this is a check on a device that has
            // ray queries but whose buffers were uploaded before the flag... which cannot happen: the buffers are
            // uploaded with the bits whenever the device supports them, regardless of the config. Kept as a guard
            // because the alternative is a validation error per frame instead of one line in the log.
            if (caster->vertex_stride == 0) {
                ++skipped_no_stride;
                continue;
            }
            vertex_address_info.buffer = vertex_detail->buffer;
            index_address_info.buffer = index_detail->buffer;
            VkDeviceAddress const source_vertex_address = vkGetBufferDeviceAddress(vk.device, &vertex_address_info);
            VkDeviceAddress const source_index_address = vkGetBufferDeviceAddress(vk.device, &index_address_info);

            // alphaMode MASK: bake the material's holes into an EXPANDED copy of this caster's vertices and build
            // the structure from that. An inline ray query has no any-hit stage, so a traversal cannot run the
            // material's discard - this bake is where the mask is applied instead, and it is startup work because
            // the structures are built once and a MASK material is a property of the file (see
            // shaders/mask_bake.comp for the rule and for what the mechanism cannot represent).
            //
            // WHAT IS THE POLICY HERE and what is the JOB's: which casters carry a MASK material is this loop's,
            // and the allocation of the expanded buffer each one is baked into is this object's (the buffer
            // outlives the loop: the build below reads it, and hit shading reads its vertices through the instance
            // table for as long as the structures live). The pipeline, its layout, the set it binds and the
            // dispatch are the JOB's (vulkan.pass.mask_bake_job), reached through the hook.
            VkDeviceAddress mask_address = 0;
            uint32_t mask_stride = 0;
            if (inputs.mask_bake && inputs.hooks.mask_ready != nullptr && inputs.hooks.mask_ready(inputs.hooks.owner)) {
                // material_record::flags bit 4 is alphaMode MASK (see vulkan/primitive.cppm; the bits are literals
                // in register_material, so they are literals here too).
                uint32_t const material_index = caster->push.material_index.value;
                material_record const* const material = material_index < inputs.materials.size() ? &inputs.materials[material_index] : nullptr;
                if (material != nullptr && (material->flags & 16u) != 0u && caster->index_count >= 3u) {
                    // Three vertices per triangle, 32 bytes each: position(3) + normal(3) + uv(2), which is what
                    // the hit shading reads (offsets 0, 3 and 6). GPU-only and never mapped - the bake fills it and
                    // the build reads it.
                    constexpr uint32_t mask_vertex_stride = 32u;
                    uint64_t const expanded_bytes = static_cast<uint64_t>(caster->index_count) * mask_vertex_stride;
                    vk_buffer expanded = vk.vma.create_buffer(nullptr, expanded_bytes, buffer_type::storage_gpu_only, acceleration_structure::build_input_usage);
                    auto const* const expanded_detail = expanded.valid() ? vk.vma.get_buffer_detail(expanded.handle()) : nullptr;
                    if (expanded_detail != nullptr) {
                        VkBufferDeviceAddressInfo const expanded_info = {
                            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = expanded_detail->buffer};
                        mask_address = vkGetBufferDeviceAddress(vk.device, &expanded_info);
                        mask_stride = mask_vertex_stride;
                        inputs.hooks.record_mask_bake(inputs.hooks.owner,
                                                      command_buffer,
                                                      pass::mask_bake_request{
                                                          .source_vertices = source_vertex_address,
                                                          .source_indices = source_index_address,
                                                          .destination = mask_address,
                                                          .source_stride = caster->vertex_stride,
                                                          .destination_stride = mask_vertex_stride,
                                                          .index_type = static_cast<uint32_t>(caster->index_type),
                                                          .triangle_count = caster->index_count / 3u,
                                                          .material_index = material_index,
                                                      });
                        mask_bakes_recorded = true;
                        ++mask_baked;
                        // The buffer outlives this loop: the build below reads it, and a hit's shading reads its
                        // vertices through the instance table for as long as the structures live.
                        this->mask_buffers_.push_back(std::move(expanded));
                    } else {
                        ++skipped_mask_buffers;
                    }
                }
            }

            // SKINNED: the job writes this caster's deformed vertices into a buffer of its own, the structure is
            // built from that buffer, and every frame after it is REFITTED - which is legal because the vertex
            // order, the index buffer and the triangle count are all the primitive's own: only the bytes change.
            // `skin_base != 0` is the test for "skinned", because index 0 is the identity block every unskinned
            // draw uses (see set_skin_matrices). The stride test is the shader's precondition, not a heuristic:
            // shaders/compute_skin.comp reads the joints at byte 32 and the weights at byte 48 of the engine's
            // 64-byte interleaved vertex, so a caster whose vertices are packed differently is REFUSED (it keeps
            // its bind pose and is counted in the log) rather than skinned with the wrong words.
            constexpr uint32_t skin_source_stride_expected = 64u;
            VkDeviceAddress skin_address = 0;
            uint32_t skin_stride = 0;
            uint32_t skin_source_stride = 0;
            uint32_t skin_vertex_count = 0;
            uint32_t skin_base = 0;
            if (mask_address == 0 && inputs.skin_bake && inputs.hooks.skin_ready != nullptr && inputs.hooks.skin_ready(inputs.hooks.owner) &&
                caster->push.skin_base != 0 && caster->vertex_count != 0 && caster->vertex_stride == skin_source_stride_expected) {
                constexpr uint32_t skin_vertex_stride = 32u; // position, normal, uv - what hit shading reads
                uint64_t const skinned_bytes = static_cast<uint64_t>(caster->vertex_count) * skin_vertex_stride;
                vk_buffer skinned_vertices = vk.vma.create_buffer(nullptr, skinned_bytes, buffer_type::storage_gpu_only, acceleration_structure::build_input_usage);
                auto const* const skinned_detail = skinned_vertices.valid() ? vk.vma.get_buffer_detail(skinned_vertices.handle()) : nullptr;
                if (skinned_detail != nullptr) {
                    VkBufferDeviceAddressInfo const skinned_info = {
                        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = skinned_detail->buffer};
                    skin_address = vkGetBufferDeviceAddress(vk.device, &skinned_info);
                    skin_stride = skin_vertex_stride;
                    skin_source_stride = caster->vertex_stride;
                    skin_vertex_count = caster->vertex_count;
                    skin_base = caster->push.skin_base;
                    this->skin_buffers_.push_back(std::move(skinned_vertices));
                    ++skinned_baked;
                } else {
                    ++skipped_skin_buffers;
                }
            }

            acceleration_structure::geometry_source const source =
                this->caster_geometry(*caster, source_vertex_address, source_index_address, mask_address, mask_stride, skin_address, skin_stride);
            // A skinned structure is built ALLOW_UPDATE so the per-frame refit is legal; everything else is built
            // once and never touched again.
            auto const added = structures.add(source, skin_address != 0);
            if (!added) {
                this->abandon();
                return std::unexpected(failure{.message = added.error()});
            }
            // Remember which caster got which index: the per-frame instance list walks THIS, so a caster that was
            // skipped above is skipped there too and the two walks cannot disagree. The mask and skin addresses
            // ride along, because that list is what a hit's shading reads the geometry through - a baked or
            // skinned caster must be read from the copy it was built from.
            this->casters_.emplace_back(caster_level{.caster = caster,
                                                     .blas_index = added.value(),
                                                     .mask_stride = mask_stride,
                                                     .mask_vertex_address = mask_address,
                                                     .skin_source_address = source_vertex_address,
                                                     .skin_destination_address = skin_address,
                                                     .skin_source_stride = skin_source_stride,
                                                     .skin_destination_stride = skin_stride,
                                                     .skin_vertex_count = skin_vertex_count,
                                                     .skin_base = skin_base});
        }

        // The skinned casters' first skinning pass, recorded here because the BUILD below has to read skinned
        // vertices - and every frame after this one re-skins and REFITS in update(). The refit is not recorded
        // here: this is the frame the structures are created, and a refit against a structure that does not exist
        // yet is illegal.
        if (skinned_baked != 0 && inputs.hooks.record_skin != nullptr) {
            for (auto const& built : this->casters_) {
                if (built.skin_destination_address != 0) {
                    this->skin_levels_.push_back(built.blas_index);
                }
            }
            static_cast<void>(inputs.hooks.record_skin(inputs.hooks.owner, command_buffer, this->casters_));
        }

        // Every bake wrote a buffer the build below reads: one barrier covers them all, because every dispatch is
        // recorded before the first build (add() only sizes and allocates; record_build() records). A compute
        // WRITE is not visible to an acceleration structure build without it, and the symptom would be a structure
        // built from an empty buffer - i.e. geometry that stops casting.
        if (mask_bakes_recorded) {
            VkMemoryBarrier2 bake_order = {};
            bake_order.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            bake_order.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bake_order.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            bake_order.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            bake_order.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            VkDependencyInfo const bake_dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                      .pNext = nullptr,
                                                      .dependencyFlags = 0,
                                                      .memoryBarrierCount = 1,
                                                      .pMemoryBarriers = &bake_order,
                                                      .bufferMemoryBarrierCount = 0,
                                                      .pBufferMemoryBarriers = nullptr,
                                                      .imageMemoryBarrierCount = 0,
                                                      .pImageMemoryBarriers = nullptr};
            vkCmdPipelineBarrier2(command_buffer, &bake_dependency);
        }

        if (auto const built = structures.record_build(command_buffer); !built) {
            // The two structures go, the COPIES stay: this is the same asymmetry the renderer had, and it is kept
            // deliberately - a failed record of a build does not invalidate the buffers the map points at, and
            // dropping them would be a second, unrelated change of behaviour.
            this->bottom_.reset();
            this->top_.reset();
            return std::unexpected(failure{.message = built.error()});
        }

        acceleration_structure::build_stats const& stats = structures.last_stats();
        double const host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        utility::log("ray-traced shadows: built {} bottom level structures ({} triangles, {:.1f} MiB + {:.1f} MiB scratch) in {:.1f} ms",
                     stats.geometry_count,
                     stats.triangle_count,
                     static_cast<double>(stats.structure_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(stats.scratch_bytes) / (1024.0 * 1024.0),
                     host_ms);
        if (skipped_no_stride != 0) {
            utility::log("  {} casters skipped (no vertex stride recorded - a primitive not created by make_primitive)", skipped_no_stride);
        }
        if (skipped_no_address != 0) {
            utility::log("  {} casters skipped (no vertex/index buffer)", skipped_no_address);
        }
        if (mask_baked != 0 || skipped_mask_buffers != 0) {
            // The measurement this feature is read with: how much geometry the mask actually removed is a property
            // of the asset (a two-quad MASK plane whose pattern is in the middle keeps every triangle; a vase of
            // flowers loses 40% of them).
            utility::log("ray-traced shadows: {} MASK casters baked into their structures ({} could not be - those are still cut per hit by the any-hit stage)", mask_baked, skipped_mask_buffers);
        }
        if (skinned_baked != 0 || skipped_skin_buffers != 0) {
            // The skinned casters are re-skinned and REFITTED every frame (see update), so this count is also the
            // number of structures a frame's refit touches.
            utility::log("ray-traced shadows: {} skinned casters re-skinned and REFITTED from their deformed vertices every frame ({} could not be - those keep their bind pose)", skinned_baked, skipped_skin_buffers);
        }
        return {};
    }

    std::expected<void, failure> structure_set::update(VkCommandBuffer const command_buffer, uint32_t const frame_slot, build_inputs const& inputs) {
        if (!this->ready()) {
            return {}; // nothing was built (or the build failed): there is nothing to refit or to instance
        }
        core& vk = *this->device_;
        auto& levels = *this->bottom_;
        auto& top = *this->top_;

        // The skinned casters are deformed and their structures REFITTED here, before the instance list is walked
        // (the addresses do not change, so the order does not matter to correctness - but the refit has to be
        // recorded before this frame writes the scene set's binding 16, the ordering the mask bake's own-set
        // comment explains).
        if (inputs.skin_bake && !this->skin_levels_.empty() && inputs.hooks.skin_ready != nullptr && inputs.hooks.skin_ready(inputs.hooks.owner)) {
            if (inputs.hooks.record_skin(inputs.hooks.owner, command_buffer, this->casters_)) {
                if (auto const updated = levels.record_update(command_buffer, this->skin_levels_); !updated) {
                    // Once, and off: a failure here would otherwise log every frame, and a refit is not something
                    // to keep attempting against structures the device refused. The knob is the CALLER's, so the
                    // decision travels back with the failure.
                    return std::unexpected(failure{.message = updated.error(), .disable_skin_bake = true});
                }
            }
        }

        if (auto const begun = top.begin(frame_slot); !begun) {
            return std::unexpected(failure{.message = begun.error()});
        }
        // The instance list is the caster set the shadow pass draws, with the world matrix the raster passes use
        // for each caster - the same matrix shadow_geometry_signature() hashes, which is why an animated or moved
        // caster is reflected here for free.
        for (auto const& built : this->casters_) {
            primitive const* const caster = built.caster;
            // The addresses a hit-shading path reads the hit triangle from: the same buffers, and the same
            // vkGetBufferDeviceAddress calls, the bottom level build already used for this caster - so the triangle
            // a shader fetches with them IS the triangle the ray hit. They are the buffers' base addresses (the
            // build applies no offset), which is also what makes them legal as a buffer reference: a buffer's
            // address is aligned, an offset into one need not be.
            //
            // A baked or skinned caster is read from the copy its structure was built from: the mask bake's
            // expanded, non-indexed one (zero index address = a flat vertex list), or the skinned one, which keeps
            // the primitive's own index buffer because its vertex ORDER is unchanged.
            VkDeviceAddress vertex_address = built.mask_vertex_address != 0 ? built.mask_vertex_address : built.skin_destination_address;
            VkDeviceAddress index_address = 0;
            uint32_t vertex_stride = built.mask_vertex_address != 0 ? built.mask_stride : built.skin_destination_stride;
            if (vertex_address == 0) {
                VkBufferDeviceAddressInfo const vertex_address_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = caster->vertex_detail->buffer};
                VkBufferDeviceAddressInfo const index_address_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = caster->index_detail->buffer};
                vertex_address = vkGetBufferDeviceAddress(vk.device, &vertex_address_info);
                index_address = vkGetBufferDeviceAddress(vk.device, &index_address_info);
                vertex_stride = caster->vertex_stride;
            } else if (built.skin_destination_address != 0) {
                VkBufferDeviceAddressInfo const index_address_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = caster->index_detail->buffer};
                index_address = vkGetBufferDeviceAddress(vk.device, &index_address_info);
            }
            acceleration_structure::instance_source const instance = {
                .transform = caster->push.model,
                .blas_index = built.blas_index,
                .record = {.vertex_address = vertex_address,
                           .index_address = index_address,
                           .model = caster->push.model,
                           .vertex_stride = vertex_stride,
                           .index_type = static_cast<uint32_t>(caster->index_type),
                           .material_index = caster->push.material_index.value,
                           .primitive_index = built.blas_index},
            };
            if (auto const added = top.add(levels, instance); !added) {
                return std::unexpected(failure{.message = added.error()});
            }
        }

        // The top level reads the BOTTOM levels, and on the frame that creates them the two builds are in the same
        // command buffer with nothing between them: without this barrier the driver is free to run the second
        // build's reads against writes the first one has not published. It costs a no-op on every later frame
        // (nothing wrote a bottom level in this buffer), which is cheaper than a flag that would have to track
        // "which frame built them".
        VkMemoryBarrier2 build_order = {};
        build_order.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        build_order.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        build_order.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        build_order.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        build_order.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo const build_order_info = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                   .pNext = nullptr,
                                                   .dependencyFlags = 0,
                                                   .memoryBarrierCount = 1,
                                                   .pMemoryBarriers = &build_order,
                                                   .bufferMemoryBarrierCount = 0,
                                                   .pBufferMemoryBarriers = nullptr,
                                                   .imageMemoryBarrierCount = 0,
                                                   .pImageMemoryBarriers = nullptr};
        vkCmdPipelineBarrier2(command_buffer, &build_order_info);

        if (auto const built = top.record_build(command_buffer); !built) {
            return std::unexpected(failure{.message = built.error()});
        }

        if (!this->top_logged_) {
            this->top_logged_ = true;
            // The class measured the host cost of the build itself (see build_stats); reporting that rather than a
            // second timer around it keeps one definition of "what the build costs".
            utility::log("ray-traced shadows: {} instances in the top level structure, one instance table entry each ({:.3f} ms host per frame)",
                         top.instance_count(frame_slot),
                         top.last_stats().build_ms);
        }
        return {};
    }

} // namespace vulkan::ray_tracing
