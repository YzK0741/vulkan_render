// The compute-skinning job's implementation: the pipeline it owns, the per-slot sets it writes once, and the
// per-caster dispatch plus the build-ordering barrier. Moved out of `runtime::make_compute_skin_pipeline` and
// `runtime::record_compute_skin_pass` UNCHANGED in behaviour - the same scene layout for the sets, the same
// binding 9, the same 32-byte push block, the same 64-wide workgroup, the same one dispatch per skinned caster
// and the same single memory barrier at the end - so the A/B against the parent commit decides it.

module;

#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

module vulkan.pass.compute_skin;

import vulkan.pipelines; // build_compute_skin: the compute pipeline this job owns
import utility;

namespace vulkan::pass {

    compute_skin_job::~compute_skin_job() {
        this->release_owned();
    }

    void compute_skin_job::release_owned() noexcept {
        this->pipeline_.reset();
        if (this->pipeline_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->device_, this->pipeline_layout_, nullptr);
            this->pipeline_layout_ = VK_NULL_HANDLE;
        }
        for (vk_descriptor_set& set : this->sets_) {
            set.release(); // the pool is the core's and outlives this job (see the header)
        }
        this->sets_.clear();
    }

    bool compute_skin_job::ready() const noexcept {
        return this->pipeline_.has_value() && !this->sets_.empty();
    }

    VkPipeline compute_skin_job::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout compute_skin_job::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    VkDescriptorSet compute_skin_job::set(uint32_t const slot) const noexcept {
        return slot < this->sets_.size() ? this->sets_[slot].get() : VK_NULL_HANDLE;
    }

    std::expected<void, std::string> compute_skin_job::create(pass_context const& context, std::vector<vk_descriptor_set> sets, std::span<VkBuffer const> skin_buffers) {
        this->sets_ = std::move(sets);
        if (context.device == VK_NULL_HANDLE) {
            return std::unexpected(std::string("compute skin: no device"));
        }
        if (this->device_ != VK_NULL_HANDLE && this->device_ != context.device) {
            this->release_owned();
        }
        this->device_ = context.device;
        std::span<unsigned char const> const spirv = context.shader != nullptr ? context.shader(context.owner, shader_name) : std::span<unsigned char const>{};
        if (spirv.empty()) {
            return std::unexpected(std::string("compute skin: the owner has no ") + std::string(shader_name));
        }
        // The scene set ALONE, because the job reads exactly one thing from it: the per-joint matrices at
        // binding 9. The vertices come through push-constant device addresses, like every other traced stage.
        VkDescriptorSetLayout const scene_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 0) : VK_NULL_HANDLE;
        if (scene_layout == VK_NULL_HANDLE) {
            return std::unexpected(std::string("compute skin: the owner has no scene set layout"));
        }
        auto built = pipelines::build_compute_skin(context.device, scene_layout, static_cast<uint32_t>(sizeof(compute_skin_push_constants)), spirv);
        if (!built) {
            return std::unexpected(std::move(built.error()));
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->trace);

        // One set per frame slot, with only binding 9 written: the slot's OWN per-joint matrices. The animation
        // rewrites that BUFFER every frame, not the descriptor, so the sets are written once here and stay
        // valid - which matters twice over, because a set updated while a recording command buffer holds it
        // invalidates that buffer (the trap the mask bake's set documents) and one set would point at the wrong
        // slot's matrices for half the frames.
        if (this->sets_.size() != skin_buffers.size()) {
            return std::unexpected(std::string("compute skin: the per-slot skin matrix buffers are not created"));
        }
        for (std::size_t slot = 0; slot < this->sets_.size(); ++slot) {
            if (this->sets_[slot].get() == VK_NULL_HANDLE || skin_buffers[slot] == VK_NULL_HANDLE) {
                return std::unexpected(std::string("compute skin: a per-slot descriptor set or skin matrix buffer is missing"));
            }
            VkDescriptorBufferInfo const skins_info = {.buffer = skin_buffers[slot], .offset = 0, .range = VK_WHOLE_SIZE};
            VkWriteDescriptorSet write = {};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = this->sets_[slot].get();
            write.dstBinding = 9; // SkinMatrices, the same binding shaders/pbr.vert reads
            write.dstArrayElement = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &skins_info;
            vkUpdateDescriptorSets(this->device_, 1, &write, 0, nullptr);
        }
        utility::log("SUCCESS: compute skinning pipeline created (skinned casters can be refitted per frame)");
        return {};
    }

    bool compute_skin_job::record(VkCommandBuffer const command_buffer, uint32_t const slot, std::span<compute_skin_request const> const requests) const noexcept {
        if (!this->ready() || requests.empty()) {
            return false;
        }
        VkDescriptorSet const set_handle = this->set(slot);
        if (set_handle == VK_NULL_HANDLE) {
            return false;
        }
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->pipeline());
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->pipeline_layout_, 0, 1, &set_handle, 0, nullptr);

        bool recorded = false;
        for (compute_skin_request const& request : requests) {
            if (request.destination == 0 || request.vertex_count == 0) {
                continue; // not a skinned caster: its geometry is what the build read, unchanged
            }
            compute_skin_push_constants const push = {
                .source_vertices = glm::uvec2(static_cast<uint32_t>(request.source_vertices & 0xFFFFFFFFu), static_cast<uint32_t>(request.source_vertices >> 32u)),
                .destination = glm::uvec2(static_cast<uint32_t>(request.destination & 0xFFFFFFFFu), static_cast<uint32_t>(request.destination >> 32u)),
                .source_stride = request.source_stride,
                .destination_stride = request.destination_stride,
                .vertex_count = request.vertex_count,
                .skin_base = request.skin_base,
            };
            vkCmdPushConstants(command_buffer, this->pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(command_buffer, (push.vertex_count + group_size - 1u) / group_size, 1, 1);
            recorded = true;
        }
        if (!recorded) {
            return false;
        }

        // What follows reads what these dispatches wrote: the BUILD on the frame the structures are created, and
        // the REFIT on every frame after. A compute write is not visible to the acceleration structure build
        // stage without this barrier, and the symptom would be a structure built or refitted against the
        // previous frame's vertices - a shadow one frame behind, which reads as animation lag.
        VkMemoryBarrier2 skin_order = {};
        skin_order.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        skin_order.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        skin_order.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        skin_order.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        skin_order.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo const skin_dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                  .pNext = nullptr,
                                                  .dependencyFlags = 0,
                                                  .memoryBarrierCount = 1,
                                                  .pMemoryBarriers = &skin_order,
                                                  .bufferMemoryBarrierCount = 0,
                                                  .pBufferMemoryBarriers = nullptr,
                                                  .imageMemoryBarrierCount = 0,
                                                  .pImageMemoryBarriers = nullptr};
        vkCmdPipelineBarrier2(command_buffer, &skin_dependency);
        return true;
    }

} // namespace vulkan::pass
