// THE STRUCTURE PHASE'S IMPLEMENTATION: see the header for why this is a value the renderer owns rather than a
// pass. The two functions below are the ones that used to be `runtime::record_acceleration_structures` and
// `runtime::record_top_level_structure`, moved with their reasoning; the only changes are that the knobs and the
// two jobs arrive as `build_inputs` and that the failures are RETURNED rather than logged and swallowed, because
// what a failure means for a knob is the renderer's decision.
module;

#include <chrono>
#include <cstdint>
#include <cstring>
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

    VkDeviceSize structure_set::structure_size(uint32_t const frame_slot) const noexcept {
        // The top level is `top_`'s object (see the member block in the header): this is the forwarding half, for
        // the descriptor heap, whose acceleration-structure descriptor is an address range that must carry a REAL
        // size (see docs/descriptor_heap_migration.md - the heap's payload union has no AS member).
        return this->top_.has_value() ? this->top_->structure_size(frame_slot) : 0;
    }

    VkDeviceSize structure_set::instance_table_size(uint32_t const frame_slot) const noexcept {
        // ... and the same for the instance table at binding 17, which the set path may write with VK_WHOLE_SIZE
        // and a heap range may not.
        return this->top_.has_value() ? this->top_->instance_table_size(frame_slot) : 0;
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

    void structure_set::release_micromaps() noexcept {
        if (this->device_ != nullptr) {
            auto const destroy = reinterpret_cast<PFN_vkDestroyMicromapEXT>(vkGetDeviceProcAddr(this->device_->device, "vkDestroyMicromapEXT"));
            if (destroy != nullptr) {
                for (micromap_resource const& resource : this->micromaps_) {
                    if (resource.micromap != VK_NULL_HANDLE) {
                        destroy(this->device_->device, resource.micromap, nullptr);
                    }
                }
            }
        }
        this->micromaps_.clear();
    }

    structure_set::~structure_set() {
        this->release_micromaps();
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
        // The micromaps go too, and they are the ONE thing here that does not free itself: VkMicromapEXT has no
        // RAII wrapper in this project, so the handle is destroyed explicitly before the buffers that back it.
        this->release_micromaps();
    }

    /**
     * @brief create and build ONE opacity micromap for a MASK caster's triangles
     *
     * WHAT IS IN IT, for this first step: every micro-triangle is written as UNKNOWN. A 4-state micromap at
     * subdivision level 0 has one micro-triangle per triangle (two bits, so one byte of the packed array per
     * triangle, which this writes with a 4-byte stride to keep every dataOffset 4-byte aligned), and an unknown
     * micro-triangle is what makes the traversal invoke the any-hit shader - so a micromap that says "unknown"
     * everywhere leaves the image EXACTLY as it was and can be attached, built and debugged on its own. That is
     * the point of this step: create, size, allocate, build and synchronize the object, with an acceptance that
     * cannot be confused with a rendering change.
     *
     * WHY THE LAYOUT IS EXPLICIT: `triangleArray` carries a per-triangle {dataOffset, subdivisionLevel, format}
     * record, which is the one layout this pass controls completely (the alternative, usage counts alone, fixes
     * the packing) and the one the spec documents byte for byte: 4-byte offset, then 2-byte level, then 2-byte
     * format.
     *
     * @param vk the device (and its allocator)
     * @param triangle_count the caster's triangles; 0 makes this a no-op that returns nothing
     * @return the resource, or nullopt when the device does not publish the entry points or an allocation fails
     */
    std::optional<structure_set::micromap_resource> make_micromap(core& vk, uint32_t const triangle_count) {
        if (triangle_count == 0) {
            return std::nullopt;
        }
        auto const get_sizes = reinterpret_cast<PFN_vkGetMicromapBuildSizesEXT>(vkGetDeviceProcAddr(vk.device, "vkGetMicromapBuildSizesEXT"));
        auto const create = reinterpret_cast<PFN_vkCreateMicromapEXT>(vkGetDeviceProcAddr(vk.device, "vkCreateMicromapEXT"));
        auto const destroy = reinterpret_cast<PFN_vkDestroyMicromapEXT>(vkGetDeviceProcAddr(vk.device, "vkDestroyMicromapEXT"));
        if (get_sizes == nullptr || create == nullptr || destroy == nullptr) {
            return std::nullopt;
        }

        // The two attribute records, both 4-byte-per-triangle so the dataOffsets are aligned: 0x03 is the 4-state
        // "unknown" pair (see the spec's Ray Opacity Micromap table), one micro-triangle per triangle.
        constexpr uint32_t data_stride = 4u;
        constexpr unsigned char unknown_state = 0x03u;
        std::vector<unsigned char> const data(static_cast<std::size_t>(triangle_count) * data_stride, unknown_state);
        std::vector<VkMicromapTriangleEXT> triangles = [triangle_count] {
            std::vector<VkMicromapTriangleEXT> records(static_cast<std::size_t>(triangle_count));
            for (uint32_t i = 0; i < triangle_count; ++i) {
                records[i] = VkMicromapTriangleEXT{.dataOffset = i * data_stride,
                                                   .subdivisionLevel = 0u,
                                                   .format = static_cast<uint16_t>(VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT)};
            }
            return records;
        }();
        std::vector<uint32_t> indices(static_cast<std::size_t>(triangle_count), 0u); // identity is not needed: one micromap triangle each

        structure_set::micromap_resource out;
        out.usage = VkMicromapUsageEXT{.count = triangle_count, .subdivisionLevel = 0u, .format = VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT};
        out.triangle_array_stride = sizeof(VkMicromapTriangleEXT);
        out.index_stride = sizeof(uint32_t);
        out.triangle_count = triangle_count;

        // The inputs and the scratch: `data` and `triangleArray` must carry MICROMAP_BUILD_INPUT_READ_ONLY and a
        // device address, the scratch must be STORAGE, and the micromap's own memory must carry MICROMAP_STORAGE.
        //
        // THE ADDRESS ALIGNMENT IS 256 BYTES and it is a requirement on the ADDRESS rather than on the buffer
        // (VUID-vkCmdBuildMicromapsEXT-pInfos-07515, which validation reported the first time this ran), so the
        // buffers are allocated with a quarter-kilobyte of slack and the payload is written at the first
        // 256-aligned address inside them. The allocator's addresses are not aligned to anything in particular,
        // so there is no "create it aligned" flag that could have done this for us.
        constexpr VkDeviceSize micromap_address_alignment = 256u;
        VkBufferUsageFlags const input_usage = VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        // A `vk_buffer` holds a VMA handle, so the VkBuffer behind it (and the address of that) comes from
        // get_buffer_detail() - the same shape the mask bake's expanded buffer uses. Every buffer here carries
        // SHADER_DEVICE_ADDRESS_BIT, which is what makes its address queryable at all.
        auto const buffer_of = [&vk](vk_buffer const& buffer) -> VkBuffer {
            auto const* const detail = buffer.valid() ? vk.vma.get_buffer_detail(buffer.handle()) : nullptr;
            return detail != nullptr ? detail->buffer : VK_NULL_HANDLE;
        };
        auto const address_of = [&vk, &buffer_of](vk_buffer const& buffer) -> VkDeviceAddress {
            VkBuffer const handle = buffer_of(buffer);
            if (handle == VK_NULL_HANDLE) {
                return 0;
            }
            VkBufferDeviceAddressInfo const info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = handle};
            return vkGetBufferDeviceAddress(vk.device, &info);
        };
        auto const create_setup_buffer = [&vk, &address_of](std::vector<unsigned char> const& bytes) -> std::pair<vk_buffer, VkDeviceAddress> {
            vk_buffer buffer = vk.vma.create_buffer(nullptr, bytes.size() + micromap_address_alignment, buffer_type::storage_coherent, input_usage);
            VkDeviceAddress const base = address_of(buffer);
            auto const* const detail = buffer.valid() ? vk.vma.get_buffer_detail(buffer.handle()) : nullptr;
            if (base == 0 || detail == nullptr || detail->allocation_info.pMappedData == nullptr) {
                return {std::move(buffer), 0};
            }
            VkDeviceSize const offset = (micromap_address_alignment - (base % micromap_address_alignment)) % micromap_address_alignment;
            std::memcpy(static_cast<unsigned char*>(detail->allocation_info.pMappedData) + offset, bytes.data(), bytes.size());
            return {std::move(buffer), base + offset};
        };

        std::vector<unsigned char> const data_bytes(data);
        auto [data_buffer, data_address] = create_setup_buffer(data_bytes);
        out.data = std::move(data_buffer);
        out.data_address = data_address;
        std::vector<unsigned char> triangle_bytes(triangles.size() * sizeof(VkMicromapTriangleEXT));
        std::memcpy(triangle_bytes.data(), triangles.data(), triangle_bytes.size());
        auto [triangle_buffer, triangle_address] = create_setup_buffer(triangle_bytes);
        out.triangles = std::move(triangle_buffer);
        out.triangles_address = triangle_address;
        std::vector<unsigned char> index_bytes(indices.size() * sizeof(uint32_t));
        std::memcpy(index_bytes.data(), indices.data(), index_bytes.size());
        auto [index_buffer, index_address] = create_setup_buffer(index_bytes);
        out.indices = std::move(index_buffer);
        out.indices_address = index_address;
        if (!out.data.valid() || !out.triangles.valid() || !out.indices.valid() || out.data_address == 0 || out.triangles_address == 0 || out.indices_address == 0) {
            return std::nullopt;
        }
        if (out.data_address == 0 || out.triangles_address == 0 || out.indices_address == 0) {
            return std::nullopt;
        }

        VkMicromapBuildInfoEXT info = {};
        info.sType = VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT;
        info.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
        info.flags = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT;
        info.mode = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
        info.usageCountsCount = 1;
        info.pUsageCounts = &out.usage;
        info.data.deviceAddress = out.data_address;
        info.triangleArray.deviceAddress = out.triangles_address;
        info.triangleArrayStride = out.triangle_array_stride;

        VkMicromapBuildSizesInfoEXT sizes = {};
        sizes.sType = VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT;
        get_sizes(vk.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &sizes);
        if (sizes.micromapSize == 0) {
            return std::nullopt;
        }
        out.storage = vk.vma.create_buffer(nullptr, sizes.micromapSize, buffer_type::storage_gpu_only, VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT);
        if (sizes.buildScratchSize != 0) {
            out.scratch = vk.vma.create_buffer(nullptr, sizes.buildScratchSize, buffer_type::storage_gpu_only, 0);
            out.scratch_address = address_of(out.scratch);
            if (out.scratch_address == 0) {
                return std::nullopt;
            }
        }
        VkDeviceAddress const storage_address = address_of(out.storage);
        if (storage_address == 0) {
            return std::nullopt;
        }

        VkMicromapCreateInfoEXT create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT;
        create_info.buffer = buffer_of(out.storage);
        create_info.offset = 0;
        create_info.size = sizes.micromapSize;
        create_info.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
        create_info.deviceAddress = 0;
        if (create(vk.device, &create_info, nullptr, &out.micromap) != VK_SUCCESS || out.micromap == VK_NULL_HANDLE) {
            return std::nullopt;
        }
        return out;
    }

    acceleration_structure::geometry_source structure_set::caster_geometry(primitive const& caster,
                                                                           VkDeviceAddress const source_vertex_address,
                                                                           VkDeviceAddress const source_index_address,
                                                                           VkDeviceAddress const mask_address,
                                                                           uint32_t const mask_stride,
                                                                           VkDeviceAddress const skin_address,
                                                                           uint32_t const skin_stride,
                                                                           micromap_resource const* const micromap) const noexcept {
        // The three cases, in the order they take precedence: a MASK bake replaces the geometry entirely (an
        // expanded, NON-INDEXED triangle list), a skinned caster keeps the primitive's index buffer because its
        // vertex ORDER is unchanged, and everything else is the primitive's own memory.
        acceleration_structure::geometry_source source = {};
        if (mask_address != 0) {
            source = acceleration_structure::geometry_source{.vertex_address = mask_address,
                                                             .vertex_stride = mask_stride,
                                                             .vertex_count = caster.index_count,
                                                             .index_address = 0,
                                                             .index_type = caster.index_type,
                                                             .index_count = caster.index_count};
        } else if (skin_address != 0) {
            source = acceleration_structure::geometry_source{.vertex_address = skin_address,
                                                             .vertex_stride = skin_stride,
                                                             .vertex_count = caster.vertex_count,
                                                             .index_address = source_index_address,
                                                             .index_type = caster.index_type,
                                                             .index_count = caster.index_count};
        } else {
            source = acceleration_structure::geometry_source{.vertex_address = source_vertex_address,
                                                             .vertex_stride = caster.vertex_stride,
                                                             .vertex_count = caster.vertex_count,
                                                             .index_address = source_index_address,
                                                             .index_type = caster.index_type,
                                                             .index_count = caster.index_count};
        }
        // THE MICROMAP RIDES ALONG WHATEVER GEOMETRY WAS CHOSEN, because it is indexed by the triangle rather than
        // by a vertex: the geometry's primitive `i` is the same triangle the micromap's element `i` describes, so
        // it applies to the baked, the skinned and the plain case alike. That is also why the micromap is built
        // from the CASTER's triangle count rather than from either of those buffers' vertex count.
        if (micromap != nullptr) {
            source.opacity_micromap = micromap->micromap;
            source.opacity_index_address = micromap->indices_address;
            source.opacity_index_stride = micromap->index_stride;
            source.opacity_index_type = VK_INDEX_TYPE_UINT32;
            source.opacity_usage = micromap->usage;
        }
        return source;
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
        // ... and the micromaps: how many were created and built, how many triangles they describe, and how many
        // casters could not have one (an allocation or a missing entry point, which is again a log line rather
        // than a failed build).
        uint32_t micromap_triangles = 0;
        uint32_t skipped_micromaps = 0;
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
            // ---- THE OPACITY MICROMAP for this caster, created HERE and independently of the bake above ----
            //
            // The bake is off by default and the micromap is the mechanism that replaces it, so this must not sit
            // behind the same gate: it is created for every caster that carries an alphaMode MASK material, which
            // is the same test the bake's rule makes ("material_record::flags bit 4").
            //
            // MEASURED, AND IT IS A NEGATIVE RESULT WORTH KEEPING: the attachment is legal and validation is
            // silent, but on this device (NVIDIA RTX 4060, 591.59.0.0) the micromap does NOT change traversal.
            // Two arms, same config and pose, on the AlphaBlendModeTest asset (3 MASK materials, 6 triangles):
            //   - all-UNKNOWN 4-state (what this writes) against no micromap at all: byte-identical, which is
            //     expected by construction - an unknown micro-triangle is the state that asks the any-hit shader;
            //   - all-TRANSPARENT 2-state against the same, WITH THE ANY-HIT NEUTRALISED so that only the micromap
            //     could decide: byte-identical as well (123.11 mean both ways, mean|d| = 0.0000). A transparent
            //     micro-triangle must skip the hit entirely, so this says the micromap is not being CONSULTED
            //     rather than that its content is wrong - the other 2-state value is opaque, and an all-opaque
            //     micromap would have reverted the any-hit's cut, which is not what happens either.
            // BOTH WERE THEN FALSIFIED BY MEASUREMENT, one arm each and each byte-identical to the same arm with no
            // micromap at all: dropping gl_RayFlagsTerminateOnFirstHitEXT (123.11 both ways, mean|d| = 0.0000) and
            // adding gl_RayFlagsForceOpacityMicromap2StateEXT (mean|d| = 0.0000 again), validation silent in both.
            // What is left to try, in the order worth trying: gl_RayFlagsForceOpacityMicromap2StateEXT (the flag
            // the spec provides for exactly this mechanism); dropping gl_RayFlagsTerminateOnFirstHitEXT for one
            // arm, since a traversal that may stop at the first hit can take a path that never asks about opacity;
            // and a scene with a larger MASK footprint, because six triangles from three small quads is a weak
            // instrument. Until one of those shows an effect, what makes MASK surfaces correct is the any-hit
            // stage's own cut (shaders/rt_shadow.rahit, 3.0/255 from the raster shadow over the pixels it changes
            // against 136.6/255 without it), and this micromap is architecture that is in place and verified to be
            // LEGAL rather than a working feature. It costs one build per MASK caster at load and is inert after.
            uint32_t micromap_index = caster_level::micromap_none;
            if (caster->index_count >= 3u) {
                uint32_t const material_index = caster->push.material_index.value;
                material_record const* const material = material_index < inputs.materials.size() ? &inputs.materials[material_index] : nullptr;
                if (material != nullptr && (material->flags & 16u) != 0u) {
                    if (auto resource = make_micromap(vk, caster->index_count / 3u); resource.has_value()) {
                        micromap_triangles += resource->triangle_count;
                        micromap_index = static_cast<uint32_t>(this->micromaps_.size());
                        this->micromaps_.push_back(std::move(*resource));
                    } else {
                        ++skipped_micromaps;
                    }
                }
            }

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
                this->caster_geometry(*caster,
                                      source_vertex_address,
                                      source_index_address,
                                      mask_address,
                                      mask_stride,
                                      skin_address,
                                      skin_stride,
                                      micromap_index != caster_level::micromap_none ? &this->micromaps_[micromap_index] : nullptr);
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
                                                     .skin_base = skin_base,
                                                     .micromap_index = micromap_index});
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

        // ---- THE MICROMAP BUILDS, recorded here because the structure builds below READ them ----
        //
        // The two barriers are the ones the spec names for exactly this pair of operations, and they are not
        // symmetric: the micromap's INPUT buffers were written by the HOST (they are host-visible and coherent, and
        // filled at setup), so they need HOST_WRITE -> MICROMAP_BUILD/SHADER_READ; the micromap itself is written
        // by MICROMAP_BUILD/MICROMAP_WRITE and read by ACCELERATION_STRUCTURE_BUILD/MICROMAP_READ, which is what
        // makes the attachment in the next step legal. Getting the first one wrong reads a micromap built from
        // memory the host had not published; getting the second wrong reads a micromap that is still being built.
        if (!this->micromaps_.empty()) {
            auto const build_micromaps = reinterpret_cast<PFN_vkCmdBuildMicromapsEXT>(vkGetDeviceProcAddr(vk.device, "vkCmdBuildMicromapsEXT"));
            if (build_micromaps != nullptr) {
                std::vector<VkMicromapBuildInfoEXT> infos(this->micromaps_.size());
                for (std::size_t i = 0; i < this->micromaps_.size(); ++i) {
                    micromap_resource const& resource = this->micromaps_[i];
                    VkMicromapBuildInfoEXT& info = infos[i];
                    info = VkMicromapBuildInfoEXT{};
                    info.sType = VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT;
                    info.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
                    info.flags = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT;
                    info.mode = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
                    info.dstMicromap = resource.micromap;
                    info.usageCountsCount = 1;
                    info.pUsageCounts = &this->micromaps_[i].usage;
                    info.data.deviceAddress = resource.data_address;
                    info.triangleArray.deviceAddress = resource.triangles_address;
                    info.triangleArrayStride = resource.triangle_array_stride;
                    info.scratchData.deviceAddress = resource.scratch_address;
                }
                VkMemoryBarrier2 const inputs_ready = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                                       .pNext = nullptr,
                                                       .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                                                       .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
                                                       .dstStageMask = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT,
                                                       .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT};
                VkDependencyInfo const before = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                 .pNext = nullptr,
                                                 .dependencyFlags = 0,
                                                 .memoryBarrierCount = 1,
                                                 .pMemoryBarriers = &inputs_ready,
                                                 .bufferMemoryBarrierCount = 0,
                                                 .pBufferMemoryBarriers = nullptr,
                                                 .imageMemoryBarrierCount = 0,
                                                 .pImageMemoryBarriers = nullptr};
                vkCmdPipelineBarrier2(command_buffer, &before);
                build_micromaps(command_buffer, static_cast<uint32_t>(infos.size()), infos.data());
                VkMemoryBarrier2 const micromaps_ready = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                                          .pNext = nullptr,
                                                          .srcStageMask = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT,
                                                          .srcAccessMask = VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT,
                                                          .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                                          .dstAccessMask = VK_ACCESS_2_MICROMAP_READ_BIT_EXT};
                VkDependencyInfo const after = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                .pNext = nullptr,
                                                .dependencyFlags = 0,
                                                .memoryBarrierCount = 1,
                                                .pMemoryBarriers = &micromaps_ready,
                                                .bufferMemoryBarrierCount = 0,
                                                .pBufferMemoryBarriers = nullptr,
                                                .imageMemoryBarrierCount = 0,
                                                .pImageMemoryBarriers = nullptr};
                vkCmdPipelineBarrier2(command_buffer, &after);
                utility::log("ray-traced shadows: built {} opacity micromaps ({} triangles, subdivision level 0, 4-state, every micro-triangle UNKNOWN, {} casters skipped - so this step cannot change a pixel)",
                             this->micromaps_.size(),
                             micromap_triangles,
                             skipped_micromaps);
            }
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
        // recorded before this frame writes the scene block's binding 16, the ordering the mask bake's own-set
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
