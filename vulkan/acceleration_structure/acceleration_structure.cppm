// ============================================================================
// module: vulkan.acceleration_structure
// module version: 0.1.0  (independent of the app version in CMakeLists project(VERSION))
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
 *  - the TOP LEVEL structure and its instance table. They belong to the frame (the instance set is
 *    culled per frame, and a per-frame-slot buffer has to survive two frames in flight), so they are
 *    built where the frame's other per-slot resources are - see vulkan.runtime.
 *  - compaction. `VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR` plus a size query and a
 *    copy typically halves the memory, at the price of a second command and a query reset; measuring
 *    the uncompacted cost first is what makes that decision reviewable rather than assumed.
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
} // namespace vulkan::acceleration_structure
