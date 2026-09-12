module;

#include <array>
#include <cstring>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

module vulkan.readback;

import utility;

namespace vulkan {
    readback::readback(core& device)
        : vk(&device) {
    }

    readback::~readback() {
        // The staging buffer is about to be released, so any copy still referencing it must finish
        // first: wait() is what guarantees the GPU is done before the allocation goes away.
        this->wait();
        if (this->fence != VK_NULL_HANDLE && this->vk != nullptr) {
            vkDestroyFence(this->vk->device, this->fence, nullptr);
            this->fence = VK_NULL_HANDLE;
        }
        // `staging` releases itself (RAII), and with it the allocation the copies were writing into
    }

    void readback::wait() {
        if (!this->fence_pending || this->fence == VK_NULL_HANDLE || this->vk == nullptr) {
            return;
        }
        vkWaitForFences(this->vk->device, 1, &this->fence, VK_TRUE, UINT64_MAX);
        vkResetFences(this->vk->device, 1, &this->fence);
        this->fence_pending = false;
    }

    std::optional<readback::staged_target> readback::stage_for_copy(VkDeviceSize const size) {
        core& vk = *this->vk;
        if (size == 0) {
            return std::nullopt;
        }
        if (this->staging.valid() && this->staging_size >= size && this->staging_mapped != nullptr) {
            return staged_target{.buffer = this->staging_buffer, .mapped = this->staging_mapped, .size = static_cast<std::size_t>(size)};
        }
        // Growing replaces the buffer, so the copy that used the old one has to have completed:
        // assigning the new owner would release an allocation the GPU may still be writing into.
        this->wait();
        this->staging = vk.vma.create_buffer(nullptr, size, buffer_type::readback_coherent);
        this->staging_buffer = VK_NULL_HANDLE;
        this->staging_mapped = nullptr;
        this->staging_size = 0;
        if (!this->staging.valid()) {
            utility::log("readback: staging buffer creation failed ({} bytes)", size);
            return std::nullopt;
        }
        auto const* const detail = vk.vma.get_buffer_detail(this->staging.handle());
        if (detail == nullptr) {
            this->staging.reset();
            return std::nullopt;
        }
        this->staging_buffer = detail->buffer;
        this->staging_mapped = detail->allocation_info.pMappedData;
        this->staging_size = size;
        return staged_target{.buffer = this->staging_buffer, .mapped = this->staging_mapped, .size = static_cast<std::size_t>(size)};
    }

    std::expected<std::vector<unsigned char>, std::string> readback::read(VkBuffer const source, VkDeviceSize const size, VkDeviceSize const offset) {
        core& vk = *this->vk;
        this->last_read_size = 0;
        if (source == VK_NULL_HANDLE || size == 0) {
            return std::unexpected(std::string("readback: nothing to read (null buffer or zero size)"));
        }
        if (this->fence == VK_NULL_HANDLE) {
            VkFenceCreateInfo const fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = nullptr, .flags = 0};
            if (vkCreateFence(vk.device, &fence_info, nullptr, &this->fence) != VK_SUCCESS) {
                return std::unexpected(std::string("readback: fence creation failed"));
            }
        }
        auto const target = this->stage_for_copy(size);
        if (!target) {
            return std::unexpected(std::string("readback: staging buffer unavailable"));
        }

        // one command buffer per call: the core's pool hands them out and this wrapper frees the
        // buffer back to it on destruction, so nothing accumulates across calls
        vk_command_buffer const commands = vk.make_command_buffer();
        VkCommandBufferBeginInfo const begin_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                                     .pNext = nullptr,
                                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                                     .pInheritanceInfo = nullptr};
        if (vkBeginCommandBuffer(*commands, &begin_info) != VK_SUCCESS) {
            return std::unexpected(std::string("readback: vkBeginCommandBuffer failed"));
        }

        // The transfer needs the source's writes visible and finished, and it must not start while an
        // earlier transfer/stage is still writing. A buffer barrier (not an image one) is what carries
        // srcAccess/dstAccess for buffers; ALL_COMMANDS as the source is deliberate: the caller may
        // have written the buffer from any stage, and this is a one-shot path where a conservative
        // source costs nothing measurable.
        std::array<VkBufferMemoryBarrier2, 1> barriers = {};
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barriers[0].pNext = nullptr;
        barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barriers[0].srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = source;
        barriers[0].offset = offset;
        barriers[0].size = size;
        VkDependencyInfo const dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                             .pNext = nullptr,
                                             .dependencyFlags = 0,
                                             .memoryBarrierCount = 0,
                                             .pMemoryBarriers = nullptr,
                                             .bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
                                             .pBufferMemoryBarriers = barriers.data(),
                                             .imageMemoryBarrierCount = 0,
                                             .pImageMemoryBarriers = nullptr};
        vkCmdPipelineBarrier2(*commands, &dependency);

        VkBufferCopy const region = {.srcOffset = offset, .dstOffset = 0, .size = size};
        vkCmdCopyBuffer(*commands, source, target->buffer, 1, &region);
        if (vkEndCommandBuffer(*commands) != VK_SUCCESS) {
            return std::unexpected(std::string("readback: vkEndCommandBuffer failed"));
        }

        // the fence has just been reset by wait()/creation, so it is unsignaled as submit requires
        VkSubmitInfo const submit_info = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                          .pNext = nullptr,
                                          .waitSemaphoreCount = 0,
                                          .pWaitSemaphores = nullptr,
                                          .pWaitDstStageMask = nullptr,
                                          .commandBufferCount = 1,
                                          .pCommandBuffers = &*commands,
                                          .signalSemaphoreCount = 0,
                                          .pSignalSemaphores = nullptr};
        if (vkQueueSubmit(vk.graphics_queue, 1, &submit_info, this->fence) != VK_SUCCESS) {
            return std::unexpected(std::string("readback: vkQueueSubmit failed"));
        }
        this->fence_pending = true;

        vkWaitForFences(vk.device, 1, &this->fence, VK_TRUE, UINT64_MAX);
        vkResetFences(vk.device, 1, &this->fence);
        this->fence_pending = false;

        // The staging type is HOST_VISIBLE | HOST_COHERENT by contract (see the vma module), so the CPU
        // can normally read the mapping directly. The invalidate is the belt to that suspenders: a
        // device that served the allocation from a non-coherent host-visible type would otherwise hand
        // back stale bytes. It costs one property lookup per read, not per byte.
        if (auto const* const detail = vk.vma.get_buffer_detail(this->staging.handle()); detail != nullptr) {
            vk.vma.invalidate_if_not_coherent(detail->allocation, detail->allocation_info.memoryType, 0, size);
        }

        std::vector<unsigned char> out(static_cast<std::size_t>(size));
        std::memcpy(out.data(), target->mapped, static_cast<std::size_t>(size));
        this->last_read_size = out.size();
        return out;
    }
} // namespace vulkan
