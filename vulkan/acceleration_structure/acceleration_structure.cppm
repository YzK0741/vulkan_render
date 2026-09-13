// ============================================================================
// module: vulkan.acceleration_structure
// module version: 0.2.0  (independent of the app version in CMakeLists project(VERSION))
//
// Ray-tracing acceleration structures: the bottom level structures of the
// scene's shadow casters, built from the geometry buffers the raster passes
// already hold (no copy, no re-upload), and the host-side bookkeeping a later
// top level structure and its instance table are built on.
//
// Depends on vulkan.core (device, queue, VMA).
//
// evolve: bump MAJOR on breaking interface changes, MINOR on additive features,
//         PATCH on internal fixes - independently of the rest of the project.
// ============================================================================
module;

#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

export module vulkan.acceleration_structure;

export import vstd;
export import vulkan.core;

/**
 * @file vulkan/acceleration_structure/acceleration_structure.cppm
 * @defgroup vulkan_acceleration_structure Ray-Tracing Acceleration Structures
 * @brief Bottom level acceleration structures for the renderer's own geometry buffers.
 *
 * The engine's geometry already lives in device-local vertex and index buffers, and an acceleration
 * structure build reads exactly those - so a BLAS is a *view* of the renderer's geometry rather than a
 * second copy of it. That is the whole design: `geometry_source` carries two DEVICE ADDRESSES and a
 * triangle count, and nothing in this module knows what a primitive, a material or a draw call is.
 *
 * Two things are deliberately not here yet, and both are additive rather than structural changes:
 *  - COMPACTION. `VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR` plus a size query and a
 *    copy typically halves the memory, at the price of a second command and a query reset; measuring
 *    the uncompacted cost first (Sponza: 17.8 MiB for 262k triangles) is what makes that decision
 *    reviewable rather than assumed. It is also a BOTTOM level concern here: the top level is rebuilt
 *    every frame, and a compacted structure is rebuilt in place rather than re-compacted.
 *  - the shading a ray-traced hit needs. The instance table IS filled (see instance_record), but no
 *    shader reads it yet: the first consumer is a shadow ray, which only asks "did anything block me".
 *
 * What a hit can and cannot be told, today, is worth stating where it is decided:
 *  - every geometry is built OPAQUE. Inline ray queries have no any-hit shader, so an alphaMode MASK
 *    surface is SOLID to a ray while the raster shadow pass cuts its holes - a known difference, not a
 *    bug, until there is a ray-tracing PIPELINE with an any-hit stage.
 *  - a skinned or morphed mesh is built from its SOURCE vertex buffer, which holds the bind pose: the
 *    deformation happens in the vertex shader and never reaches this memory. Such a primitive casts
 *    its bind-pose shadow until a compute skinning pass exists to write deformed vertices somewhere a
 *    build can read.
 */
namespace vulkan::acceleration_structure {
    /**
     * @ingroup vulkan_acceleration_structure
     * @brief the usage bits a buffer must carry to be an acceleration-structure build input
     * @note the second bit needs VK_KHR_acceleration_structure, so this must only be OR'd into a
     *       buffer's usage when the device has it (see vma_allocator::create_buffer's extra_usage and
     *       core::ray_query_available). The first bit needs only the core 1.2 bufferDeviceAddress
     *       feature, which this engine enables by policy.
     */
    export constexpr VkBufferUsageFlags build_input_usage =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    /**
     * @ingroup vulkan_acceleration_structure
     * @brief one indexed triangle geometry to build a bottom level structure from
     * @note the addresses come from `vkGetBufferDeviceAddress` on the renderer's own vertex and index
     *       buffers; `index_address` may be 0 for a non-indexed geometry, which the build then reads as
     *       a flat vertex list
     */
    export struct geometry_source {
        VkDeviceAddress vertex_address = 0; // first vertex, already offset into the buffer
        uint32_t vertex_stride = 0;         // bytes per vertex (the engine's interleaved layout)
        uint32_t vertex_count = 0;
        VkDeviceAddress index_address = 0; // first index, already offset into the buffer
        VkIndexType index_type = VK_INDEX_TYPE_UINT32;
        uint32_t index_count = 0; // triangles = index_count / 3
    };

    /**
     * @ingroup vulkan_acceleration_structure
     * @brief what a build cost, for the startup log
     * @note `build_ms` is HOST time inside record_build() - creating buffers and recording the command
     *       - which is what a load-time step can be judged by. The GPU time of the build itself is the
     *       frame's, and belongs in the pass timings like any other pass.
     */
    export struct build_stats {
        uint32_t geometry_count = 0;
        uint64_t triangle_count = 0;
        uint64_t structure_bytes = 0;
        uint64_t scratch_bytes = 0; // the scratch BUFFER, i.e. the aligned per-geometry ranges in it
        double build_ms = 0.0;
    };

    /**
     * @ingroup vulkan_acceleration_structure
     * @brief one entry of the instance table: what a shader needs to resolve a hit back to a surface
     * @note filled now - the build already walks the same instance list - even though the first ray
     *       that lands (a shadow ray) only asks "did anything block me": shading at a hit needs the
     *       triangle's vertex data and the material, and reconstructing that list later would mean
     *       walking the scene a second time for information this pass has in hand.
     * @note std430 layout: two 8-byte addresses then four 4-byte fields = 32 bytes, no padding
     */
    export struct instance_record {
        VkDeviceAddress vertex_address = 0;
        VkDeviceAddress index_address = 0;
        uint32_t vertex_stride = 0;
        uint32_t index_type = 0;     // VkIndexType, for the shader that indexes with it
        uint32_t material_index = 0; // into the scene's material table (set 0 binding 5)
        uint32_t primitive_index = 0;
    };

    /**
     * @ingroup vulkan_acceleration_structure
     * @brief one instance of the top level structure: which bottom level, where it sits, what it is
     */
    export struct instance_source {
        glm::mat4 transform = glm::mat4(1.0f); // the world matrix the raster passes draw this instance with
        uint32_t blas_index = 0;               // index into the bottom_level_structures it was added to
        instance_record record = {};           // what the shader sees through instanceCustomIndex
    };

    /**
     * @ingroup vulkan_acceleration_structure
     * @brief the scene's bottom level structures, built in one command
     * @note the builds are batched into ONE `vkCmdBuildAccelerationStructuresKHR` call, which is why
     *       the scratch space is one buffer with a per-geometry aligned range: the driver executes the
     *       builds of a single call in order, and giving each its own range is what keeps that
     *       ordering out of the correctness argument entirely.
     */
    export class bottom_level_structures {
        struct entry {
            VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
            vk_buffer storage = {};          // the memory the structure lives in (RAII)
            VkDeviceSize scratch_offset = 0; // into the shared scratch buffer, already aligned
            VkDeviceSize scratch_size = 0;
            VkAccelerationStructureBuildRangeInfoKHR range = {};             // primitiveCount etc, for the build
            std::vector<VkAccelerationStructureGeometryKHR> geometries = {}; // one per entry, kept alive
        };

        core* vk = nullptr; // non-const: VMA's detail lookups and buffer creation are not const
        std::vector<entry> entries = {};
        /// The extension entry points, resolved per device in the constructor. Declared incomplete here
        /// and defined in the .cpp, because a function pointer table is implementation detail - and a
        /// unique_ptr so the header does not have to name the four PFN types either.
        struct entry_points;
        std::unique_ptr<entry_points> functions = {};
        vk_buffer scratch = {};
        VkDeviceAddress scratch_address = 0;
        VkDeviceSize scratch_size = 0;
        build_stats stats = {};
        /// the build infos handed to the command, assembled once per record_build (they point into
        /// `entries`, so they cannot outlive a resize of it)
        std::vector<VkAccelerationStructureBuildGeometryInfoKHR> build_infos = {};
        std::vector<VkAccelerationStructureBuildRangeInfoKHR const*> range_ptrs = {};

    public:
        explicit bottom_level_structures(core& device);
        bottom_level_structures(bottom_level_structures const&) = delete;
        bottom_level_structures& operator=(bottom_level_structures const&) = delete;
        bottom_level_structures(bottom_level_structures&&) = delete;
        bottom_level_structures& operator=(bottom_level_structures&&) = delete;
        ~bottom_level_structures();

        /**
         * @ingroup vulkan_acceleration_structure
         * @brief create the structure and its storage for one geometry
         * @param source the geometry's addresses and triangle count
         * @return the index this geometry got, or an error message on failure
         * @note this is HOST work (a size query plus an allocation); the GPU build happens in
         *       record_build(). A source with no triangles is skipped and returns the index it WOULD
         *       have had, so the caller's indices stay aligned with its own array.
         */
        std::expected<uint32_t, std::string> add(geometry_source const& source);

        /**
         * @ingroup vulkan_acceleration_structure
         * @brief allocate the scratch and record every build into @p command_buffer
         * @return success, or an error message
         * @note must be recorded OUTSIDE a rendering instance (it is a transfer/compute-class command)
         * @note call once: a second call would rebuild into the same structures with no scratch space
         *       left (the buffer is sized by the first call's requirements)
         */
        std::expected<void, std::string> record_build(VkCommandBuffer command_buffer);

        /** @brief how many structures were added */
        [[nodiscard]] uint32_t size() const noexcept {
            return static_cast<uint32_t>(this->entries.size());
        }

        /** @brief the structure at @p index, or VK_NULL_HANDLE when it was skipped */
        [[nodiscard]] VkAccelerationStructureKHR handle(uint32_t index) const noexcept {
            return index < this->entries.size() ? this->entries[index].handle : VK_NULL_HANDLE;
        }

        /** @brief what the last build cost (see build_stats) */
        [[nodiscard]] build_stats const& last_stats() const noexcept {
            return this->stats;
        }
    };

    /**
     * @ingroup vulkan_acceleration_structure
     * @brief the scene's top level structure, rebuilt from the instance list once per frame
     *
     * @details the instances are host-visible arrays the caller fills through add(), and the structure
     *          itself is built into a command buffer. One set of resources PER FRAME SLOT, because with
     *          more than one frame in flight a single buffer would be rewritten by the frame being
     *          recorded while the previous one is still reading it - the same per-slot rule the
     *          engine's camera and material buffers follow.
     *
     * The build is MODE_BUILD every frame rather than MODE_UPDATE: an update can only change
     * transforms, requires the same instance count and ALLOW_UPDATE on the original build, and the
     * instance list here is culled per frame - so the cheaper update path is exactly the one that would
     * need the most bookkeeping to stay legal. It is PREFER_FAST_BUILD rather than PREFER_FAST_TRACE
     * for the same reason: this structure is built 60+ times a second and traversed a few times per
     * pixel, while the bottom levels are built once and traversed constantly.
     */
    export class top_level_structure {
        struct slot {
            vk_buffer instances = {}; // host-visible VkAccelerationStructureInstanceKHR[capacity]
            VkBuffer instances_buffer = VK_NULL_HANDLE;
            void* instances_mapped = nullptr;
            vk_buffer records = {}; // host-visible instance_record[capacity] (the instance table)
            VkBuffer records_buffer = VK_NULL_HANDLE;
            void* records_mapped = nullptr;
            uint32_t capacity = 0;  // instances the buffers above can hold
            uint32_t count = 0;     // instances added this frame
            vk_buffer storage = {}; // the structure's own memory, sized for `capacity`
            VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
            VkDeviceSize scratch_size = 0; // what the build of `count` instances needs
            vk_buffer scratch = {};        // the build's scratch memory, kept once sized
        };

        core* vk = nullptr; // non-const: VMA's detail lookups and buffer creation are not const
        struct entry_points;
        std::unique_ptr<entry_points> functions = {};
        std::vector<slot> slots = {};
        uint32_t current_slot = 0;
        build_stats stats = {};

    public:
        explicit top_level_structure(core& device, uint32_t frame_slot_count);
        top_level_structure(top_level_structure const&) = delete;
        top_level_structure& operator=(top_level_structure const&) = delete;
        top_level_structure(top_level_structure&&) = delete;
        top_level_structure& operator=(top_level_structure&&) = delete;
        ~top_level_structure();

        /**
         * @ingroup vulkan_acceleration_structure
         * @brief start a frame's instance list in @p frame_slot (dropping whatever it held)
         * @param frame_slot the slot the frame being recorded belongs to
         * @return success, or an error message when the slot's buffers cannot be sized
         * @note the slot is free to write because the runtime waits for it before recording (see
         *       runtime::pace_and_acquire), the same reason every other per-slot buffer is.
         */
        std::expected<void, std::string> begin(uint32_t frame_slot);

        /**
         * @ingroup vulkan_acceleration_structure
         * @brief append one instance to the current slot's list
         * @note @p source.blas_index must be an index of the bottom_level_structures the reference is
         *       taken from; a null handle there (a geometry that was skipped) skips the instance, so the
         *       caller's arrays stay aligned
         */
        std::expected<void, std::string> add(bottom_level_structures const& levels, instance_source const& source);

        /**
         * @ingroup vulkan_acceleration_structure
         * @brief record the build of the current slot's list into @p command_buffer
         * @param command_buffer a buffer being recorded outside a rendering instance
         * @return success, or an error message
         */
        std::expected<void, std::string> record_build(VkCommandBuffer command_buffer);

        /** @brief the structure the slot's frame must bind, or VK_NULL_HANDLE when it is empty */
        [[nodiscard]] VkAccelerationStructureKHR handle(uint32_t frame_slot) const noexcept {
            return frame_slot < this->slots.size() ? this->slots[frame_slot].handle : VK_NULL_HANDLE;
        }

        /** @brief the slot's instance table (instance_record[count]); the shading-at-a-hit step binds it */
        [[nodiscard]] VkBuffer instance_table(uint32_t frame_slot) const noexcept {
            return frame_slot < this->slots.size() ? this->slots[frame_slot].records_buffer : VK_NULL_HANDLE;
        }

        /** @brief how many instances the slot's last build held */
        [[nodiscard]] uint32_t instance_count(uint32_t frame_slot) const noexcept {
            return frame_slot < this->slots.size() ? this->slots[frame_slot].count : 0;
        }

        /** @brief what the last recorded build cost (geometry_count is the instance count) */
        [[nodiscard]] build_stats const& last_stats() const noexcept {
            return this->stats;
        }
    };
} // namespace vulkan::acceleration_structure
