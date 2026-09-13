module;

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vulkan/vulkan.h>

module vulkan.acceleration_structure;

import utility;

namespace vulkan::acceleration_structure {
    namespace {
        /// round @p value up to the next multiple of @p alignment (a power of two, as Vulkan requires)
        constexpr VkDeviceSize align_up(VkDeviceSize const value, VkDeviceSize const alignment) noexcept {
            return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
        }

        /**
         * @brief the four acceleration-structure entry points, resolved per device
         *
         * They are NOT in the SDK's vulkan-1 import library - checked, not assumed: that lib exports
         * `vkGetBufferDeviceAddress` (core 1.2) and no `vk*AccelerationStructure*` symbol at all, so a
         * direct call is an undefined symbol at LINK time on this toolchain. Resolving through
         * vkGetDeviceProcAddr is the documented way to reach an extension entry point and the only one
         * that works here.
         *
         * A missing pointer is not fatal: the builder reports it as an error and the caller keeps its
         * raster path, which is the bargain the whole ray-tracing feature makes.
         */
    } // namespace

    // The nested type declared in the interface, defined here: a function-pointer table is an
    // implementation detail (the header only needs to know it exists, so it holds a unique_ptr).
    struct bottom_level_structures::entry_points {
        PFN_vkCreateAccelerationStructureKHR create = nullptr;
        PFN_vkDestroyAccelerationStructureKHR destroy = nullptr;
        PFN_vkGetAccelerationStructureBuildSizesKHR get_build_sizes = nullptr;
        PFN_vkCmdBuildAccelerationStructuresKHR cmd_build = nullptr;
        PFN_vkGetAccelerationStructureDeviceAddressKHR get_device_address = nullptr;

        [[nodiscard]] bool loaded() const noexcept {
            return this->create != nullptr && this->destroy != nullptr && this->get_build_sizes != nullptr && this->cmd_build != nullptr && this->get_device_address != nullptr;
        }

        [[nodiscard]] bool load(VkDevice const device) noexcept {
            this->create = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
            this->destroy = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
            this->get_build_sizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
            this->cmd_build = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
            this->get_device_address = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
            return this->loaded();
        }
    };

    bottom_level_structures::bottom_level_structures(core& device)
        : vk(&device)
        , functions(std::make_unique<entry_points>()) {
        if (!this->functions->load(device.device)) {
            utility::log("acceleration structures: the loader does not expose the vk*AccelerationStructure* entry points "
                         "(vkGetDeviceProcAddr returned null) - ray-traced shadows stay off");
        }
    }

    bottom_level_structures::~bottom_level_structures() {
        // The structures are destroyed before their storage buffers are released (the vk_buffer owners
        // below free the memory): vkDestroyAccelerationStructureKHR only drops the handle, but a
        // structure whose memory is gone is not something to leave to member-destruction order.
        for (entry const& item : this->entries) {
            if (item.handle != VK_NULL_HANDLE && this->vk != nullptr && this->functions != nullptr && this->functions->loaded()) {
                this->functions->destroy(this->vk->device, item.handle, nullptr);
            }
        }
    }

    std::expected<uint32_t, std::string> bottom_level_structures::add(geometry_source const& source) {
        core& vk = *this->vk;
        // Every geometry gets an entry, even one with nothing to build: the caller's index into this
        // list is the caller's index into its own geometry array, and skipping one silently would
        // shift every later index by one.
        entry item = {};
        uint32_t const triangle_count = source.index_count / 3u;
        this->stats.triangle_count += triangle_count;
        if (triangle_count == 0) {
            this->entries.push_back(std::move(item));
            return static_cast<uint32_t>(this->entries.size() - 1);
        }

        VkAccelerationStructureGeometryTrianglesDataKHR triangles = {};
        triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; // position at offset 0 of the interleaved vertex
        triangles.vertexData.deviceAddress = source.vertex_address;
        triangles.vertexStride = source.vertex_stride;
        triangles.maxVertex = source.vertex_count == 0 ? 0u : source.vertex_count - 1u;
        triangles.indexType = source.index_type;
        triangles.indexData.deviceAddress = source.index_address;

        VkAccelerationStructureGeometryKHR geometry = {};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        // OPAQUE for every geometry, including alphaMode MASK ones: an inline ray query has no any-hit
        // shader to run the material's discard in (see the module docs).
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geometry.geometry.triangles = triangles;
        item.geometries.push_back(geometry);

        VkAccelerationStructureBuildGeometryInfoKHR size_info = {};
        size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        size_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        size_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        size_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        size_info.geometryCount = static_cast<uint32_t>(item.geometries.size());
        size_info.pGeometries = item.geometries.data();

        VkAccelerationStructureBuildSizesInfoKHR sizes = {};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        this->functions->get_build_sizes(vk.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &size_info, &triangle_count, &sizes);
        if (sizes.accelerationStructureSize == 0) {
            return std::unexpected(std::string("acceleration structure: the device reported a zero-sized bottom level structure"));
        }

        item.storage = vk.vma.create_buffer(nullptr, sizes.accelerationStructureSize, buffer_type::acceleration_structure_storage);
        if (!item.storage.valid()) {
            return std::unexpected(std::string("acceleration structure: the bottom level storage allocation failed"));
        }
        auto const* const storage_detail = vk.vma.get_buffer_detail(item.storage.handle());
        if (storage_detail == nullptr) {
            return std::unexpected(std::string("acceleration structure: the bottom level storage has no VMA detail"));
        }

        VkAccelerationStructureCreateInfoKHR create = {};
        create.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create.buffer = storage_detail->buffer;
        create.offset = 0;
        create.size = sizes.accelerationStructureSize;
        create.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        if (this->functions->create(vk.device, &create, nullptr, &item.handle) != VK_SUCCESS) {
            return std::unexpected(std::string("acceleration structure: vkCreateAccelerationStructureKHR failed"));
        }

        // The scratch range is aligned per geometry (see record_build), so the sizes accumulate here
        // and only the offsets are decided then - the buffer itself is allocated once, in record_build.
        item.scratch_size = sizes.buildScratchSize;
        item.range.primitiveCount = triangle_count;

        this->stats.geometry_count += 1;
        this->stats.structure_bytes += sizes.accelerationStructureSize;
        this->stats.scratch_bytes += sizes.buildScratchSize;
        this->entries.push_back(std::move(item));
        return static_cast<uint32_t>(this->entries.size() - 1);
    }

    std::expected<void, std::string> bottom_level_structures::record_build(VkCommandBuffer const command_buffer) {
        core& vk = *this->vk;
        auto const start = std::chrono::steady_clock::now();

        if (this->entries.empty()) {
            return {}; // nothing was added: an empty command is not an error, it is an empty scene
        }

        // One scratch buffer for every build, each geometry's range aligned to what the device
        // requires of a SCRATCH ADDRESS (not of an offset - the requirement is on the address the
        // build is handed, which is why the base address is taken into account and why the buffer
        // carries one alignment worth of slack).
        VkDeviceSize const alignment = std::max<VkDeviceSize>(vk.acceleration_structure_properties.minAccelerationStructureScratchOffsetAlignment, 1);
        VkDeviceSize total = 0;
        for (entry& item : this->entries) {
            total += item.scratch_size + alignment;
        }
        this->scratch = vk.vma.create_buffer(nullptr, total, buffer_type::acceleration_structure_scratch);
        if (!this->scratch.valid()) {
            return std::unexpected(std::string("acceleration structure: the scratch allocation failed"));
        }
        auto const* const scratch_detail = vk.vma.get_buffer_detail(this->scratch.handle());
        if (scratch_detail == nullptr) {
            return std::unexpected(std::string("acceleration structure: the scratch buffer has no VMA detail"));
        }
        VkBufferDeviceAddressInfo const scratch_address_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = scratch_detail->buffer};
        VkDeviceAddress const scratch_base = vkGetBufferDeviceAddress(vk.device, &scratch_address_info);

        this->build_infos.clear();
        this->range_ptrs.clear();
        this->build_infos.reserve(this->entries.size());
        this->range_ptrs.reserve(this->entries.size());
        VkDeviceSize cursor = 0;
        for (entry& item : this->entries) {
            if (item.handle == VK_NULL_HANDLE) {
                continue; // a geometry with no triangles: no build, no scratch range
            }
            VkDeviceAddress const address = align_up(scratch_base + cursor, alignment);
            item.scratch_offset = address - scratch_base;
            cursor = item.scratch_offset + item.scratch_size;

            VkAccelerationStructureBuildGeometryInfoKHR info = {};
            info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            info.dstAccelerationStructure = item.handle;
            info.geometryCount = static_cast<uint32_t>(item.geometries.size());
            info.pGeometries = item.geometries.data();
            info.scratchData.deviceAddress = address;
            this->build_infos.push_back(info);
            this->range_ptrs.push_back(&item.range);
        }

        if (!this->build_infos.empty()) {
            // ONE call for every structure: the pieces of a single vkCmdBuildAccelerationStructuresKHR
            // are executed in order, so a geometry's build is complete before the next one starts.
            this->functions->cmd_build(command_buffer, static_cast<uint32_t>(this->build_infos.size()), this->build_infos.data(), this->range_ptrs.data());
        }

        this->scratch_address = scratch_base;
        this->scratch_size = total;
        this->stats.scratch_bytes = total;
        this->stats.build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return {};
    }

    // ---- top level structure ----

    namespace {
        /// the world matrix the raster passes draw with, as the 3x4 ROW-major transform the instance
        /// wants. glm is column-major (m[column][row]) and VkTransformMatrixKHR is row-major
        /// (matrix[row][column]), so the indices swap - which is the one place a transposed instance
        /// would silently mirror the whole scene.
        VkTransformMatrixKHR to_instance_transform(glm::mat4 const& matrix) noexcept {
            VkTransformMatrixKHR out = {};
            for (uint32_t row = 0; row < 3; ++row) {
                for (uint32_t column = 0; column < 4; ++column) {
                    out.matrix[row][column] = matrix[column][row];
                }
            }
            return out;
        }
    } // namespace

    struct top_level_structure::entry_points {
        PFN_vkCreateAccelerationStructureKHR create = nullptr;
        PFN_vkDestroyAccelerationStructureKHR destroy = nullptr;
        PFN_vkGetAccelerationStructureBuildSizesKHR get_build_sizes = nullptr;
        PFN_vkCmdBuildAccelerationStructuresKHR cmd_build = nullptr;
        PFN_vkGetAccelerationStructureDeviceAddressKHR get_device_address = nullptr;

        [[nodiscard]] bool loaded() const noexcept {
            return this->create != nullptr && this->destroy != nullptr && this->get_build_sizes != nullptr && this->cmd_build != nullptr && this->get_device_address != nullptr;
        }

        [[nodiscard]] bool load(VkDevice const device) noexcept {
            this->create = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
            this->destroy = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
            this->get_build_sizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
            this->cmd_build = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
            this->get_device_address = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
            return this->loaded();
        }
    };

    top_level_structure::top_level_structure(core& device, uint32_t const frame_slot_count)
        : vk(&device)
        , functions(std::make_unique<entry_points>())
        , slots(frame_slot_count) {
        if (!this->functions->load(device.device)) {
            utility::log("acceleration structures: the loader does not expose the vk*AccelerationStructure* entry points "
                         "(vkGetDeviceProcAddr returned null) - ray-traced shadows stay off");
        }
    }

    top_level_structure::~top_level_structure() {
        for (slot const& item : this->slots) {
            if (item.handle != VK_NULL_HANDLE && this->vk != nullptr && this->functions != nullptr && this->functions->loaded()) {
                this->functions->destroy(this->vk->device, item.handle, nullptr);
            }
        }
    }

    std::expected<void, std::string> top_level_structure::begin(uint32_t const frame_slot) {
        if (!this->functions->loaded()) {
            return std::unexpected(std::string("acceleration structures: the top level entry points were not resolved"));
        }
        if (frame_slot >= this->slots.size()) {
            return std::unexpected(std::string("acceleration structures: frame slot out of range"));
        }
        this->current_slot = frame_slot;
        this->slots[frame_slot].count = 0;
        return {};
    }

    std::expected<void, std::string> top_level_structure::add(bottom_level_structures const& levels, instance_source const& source) {
        core& vk = *this->vk;
        if (this->current_slot >= this->slots.size()) {
            return std::unexpected(std::string("acceleration structures: no frame slot is being built"));
        }
        VkAccelerationStructureKHR const blas = levels.handle(source.blas_index);
        if (blas == VK_NULL_HANDLE) {
            return {}; // a geometry with no triangles: no instance, and the caller's indices stay put
        }
        slot& target = this->slots[this->current_slot];

        // Grow the per-slot arrays when this frame's list outgrew them. Doubling keeps the reallocation
        // rare (it destroys and recreates the structure, so it is not something to do every frame), and
        // the capacity - not the count - is what the structure is sized for, which is legal: the size
        // query's instance count is an upper bound for the build.
        if (target.count >= target.capacity) {
            uint32_t const wanted = std::max(target.capacity * 2u, 256u);
            if (wanted > vk.acceleration_structure_properties.maxInstanceCount) {
                return std::unexpected(std::string("acceleration structures: the scene has more instances than the device allows in one top level structure"));
            }
            target.instances = vk.vma.create_buffer(nullptr, static_cast<uint64_t>(wanted) * sizeof(VkAccelerationStructureInstanceKHR), buffer_type::storage_coherent, build_input_usage);
            target.records = vk.vma.create_buffer(nullptr, static_cast<uint64_t>(wanted) * sizeof(instance_record), buffer_type::storage_coherent, build_input_usage);
            if (!target.instances.valid() || !target.records.valid()) {
                return std::unexpected(std::string("acceleration structures: the instance buffers could not be allocated"));
            }
            auto const* const instance_detail = vk.vma.get_buffer_detail(target.instances.handle());
            auto const* const record_detail = vk.vma.get_buffer_detail(target.records.handle());
            if (instance_detail == nullptr || record_detail == nullptr) {
                return std::unexpected(std::string("acceleration structures: the instance buffers have no VMA detail"));
            }
            target.instances_buffer = instance_detail->buffer;
            target.instances_mapped = instance_detail->allocation_info.pMappedData;
            target.records_buffer = record_detail->buffer;
            target.records_mapped = record_detail->allocation_info.pMappedData;
            if (target.instances_mapped == nullptr || target.records_mapped == nullptr) {
                return std::unexpected(std::string("acceleration structures: the instance buffers are not mapped"));
            }

            // The structure the new capacity needs. The old handle goes first: a structure must not
            // outlive the memory it was created in, and that memory is about to be released.
            if (target.handle != VK_NULL_HANDLE) {
                this->functions->destroy(vk.device, target.handle, nullptr);
                target.handle = VK_NULL_HANDLE;
            }
            VkAccelerationStructureBuildGeometryInfoKHR size_info = {};
            size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            size_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            size_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
            size_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            size_info.geometryCount = 1;
            VkAccelerationStructureGeometryKHR geometry = {};
            geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            geometry.geometry.instances.arrayOfPointers = VK_FALSE;
            geometry.geometry.instances.data.deviceAddress = 0; // the size query does not read the data
            size_info.pGeometries = &geometry;

            VkAccelerationStructureBuildSizesInfoKHR sizes = {};
            sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            this->functions->get_build_sizes(vk.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &size_info, &wanted, &sizes);
            if (sizes.accelerationStructureSize == 0) {
                return std::unexpected(std::string("acceleration structures: the device reported a zero-sized top level structure"));
            }
            target.storage = vk.vma.create_buffer(nullptr, sizes.accelerationStructureSize, buffer_type::acceleration_structure_storage);
            if (!target.storage.valid()) {
                return std::unexpected(std::string("acceleration structures: the top level storage allocation failed"));
            }
            auto const* const storage_detail = vk.vma.get_buffer_detail(target.storage.handle());
            if (storage_detail == nullptr) {
                return std::unexpected(std::string("acceleration structures: the top level storage has no VMA detail"));
            }
            VkAccelerationStructureCreateInfoKHR create = {};
            create.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            create.buffer = storage_detail->buffer;
            create.offset = 0;
            create.size = sizes.accelerationStructureSize;
            create.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            if (this->functions->create(vk.device, &create, nullptr, &target.handle) != VK_SUCCESS) {
                return std::unexpected(std::string("acceleration structures: the top level structure could not be created"));
            }
            target.capacity = wanted;
            target.scratch_size = sizes.buildScratchSize;
            this->stats.structure_bytes += sizes.accelerationStructureSize;
        }

        // The instance itself: a device address for the geometry, the world transform, and the record
        // the shader will see through instanceCustomIndex (which is why the index IS the table slot).
        VkAccelerationStructureDeviceAddressInfoKHR const address_info = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR, .pNext = nullptr, .accelerationStructure = blas};
        VkAccelerationStructureInstanceKHR instance = {};
        instance.transform = to_instance_transform(source.transform);
        instance.instanceCustomIndex = target.count;         // == the slot in the instance table
        instance.mask = 0xFF;                                // visible to every ray: nothing here is ray-type specific
        instance.instanceShaderBindingTableRecordOffset = 0; // unused by a ray query (no SBT exists)
        // FACING_CULL_DISABLE: a shadow ray must be blocked by a surface it approaches from behind, and
        // the raster shadow pass has the same property for the casters whose pipeline disables culling.
        // Leaving culling on would make every plane and every open mesh leak light.
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = this->functions->get_device_address(vk.device, &address_info);

        std::memcpy(static_cast<unsigned char*>(target.instances_mapped) + static_cast<std::size_t>(target.count) * sizeof(VkAccelerationStructureInstanceKHR),
                    &instance,
                    sizeof(instance));
        std::memcpy(static_cast<unsigned char*>(target.records_mapped) + static_cast<std::size_t>(target.count) * sizeof(instance_record),
                    &source.record,
                    sizeof(source.record));
        target.count += 1;
        return {};
    }

    std::expected<void, std::string> top_level_structure::record_build(VkCommandBuffer const command_buffer) {
        core& vk = *this->vk;
        auto const start = std::chrono::steady_clock::now();
        slot& target = this->slots[this->current_slot];
        if (target.count == 0 || target.handle == VK_NULL_HANDLE) {
            return {}; // an empty scene has an empty top level structure, and nothing to trace against
        }

        VkDeviceAddress const instances_address = [&] {
            VkBufferDeviceAddressInfo const info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = target.instances_buffer};
            return vkGetBufferDeviceAddress(vk.device, &info);
        }();

        VkDeviceSize const alignment = std::max<VkDeviceSize>(vk.acceleration_structure_properties.minAccelerationStructureScratchOffsetAlignment, 1);
        target.scratch_size = 0;
        VkAccelerationStructureGeometryKHR geometry = {};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geometry.geometry.instances.arrayOfPointers = VK_FALSE;
        geometry.geometry.instances.data.deviceAddress = instances_address;

        VkAccelerationStructureBuildGeometryInfoKHR info = {};
        info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        info.dstAccelerationStructure = target.handle;
        info.geometryCount = 1;
        info.pGeometries = &geometry;

        VkAccelerationStructureBuildSizesInfoKHR sizes = {};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        this->functions->get_build_sizes(vk.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &target.count, &sizes);

        // One scratch buffer per slot, sized for the count actually being built. It is allocated on the
        // FIRST build of a slot and kept: the count is culled per frame and drifts, but a buffer sized
        // for the largest count seen is what a build of any smaller count needs.
        if (!target.scratch.valid() || target.scratch_size < sizes.buildScratchSize) {
            target.scratch = vk.vma.create_buffer(nullptr, sizes.buildScratchSize + alignment, buffer_type::acceleration_structure_scratch);
            if (!target.scratch.valid()) {
                return std::unexpected(std::string("acceleration structures: the top level scratch allocation failed"));
            }
        }
        target.scratch_size = sizes.buildScratchSize + alignment;
        auto const* const scratch_detail = vk.vma.get_buffer_detail(target.scratch.handle());
        if (scratch_detail == nullptr) {
            return std::unexpected(std::string("acceleration structures: the top level scratch has no VMA detail"));
        }
        VkBufferDeviceAddressInfo const scratch_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = scratch_detail->buffer};
        info.scratchData.deviceAddress = align_up(vkGetBufferDeviceAddress(vk.device, &scratch_info), alignment);

        VkAccelerationStructureBuildRangeInfoKHR range = {};
        range.primitiveCount = target.count;
        VkAccelerationStructureBuildRangeInfoKHR const* ranges[1] = {&range};
        this->functions->cmd_build(command_buffer, 1, &info, ranges);

        this->stats.geometry_count = target.count;
        this->stats.scratch_bytes = target.scratch_size;
        this->stats.build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return {};
    }
} // namespace vulkan::acceleration_structure
