module;

#include <chrono>
#include <cstdint>
#include <expected>
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

        [[nodiscard]] bool loaded() const noexcept {
            return this->create != nullptr && this->destroy != nullptr && this->get_build_sizes != nullptr && this->cmd_build != nullptr;
        }

        [[nodiscard]] bool load(VkDevice const device) noexcept {
            this->create = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
            this->destroy = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
            this->get_build_sizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
            this->cmd_build = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
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
} // namespace vulkan::acceleration_structure
