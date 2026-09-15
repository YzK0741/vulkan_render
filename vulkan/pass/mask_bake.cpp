// The alphaMode MASK bake's implementation: the pipeline it owns, the set it writes once, and the dispatch per
// caster. Moved out of `runtime::make_mask_bake_pipeline` and the mask branch of
// `runtime::record_acceleration_structures` UNCHANGED in behaviour - the same scene layout for the set, the same
// two bindings (1 = the bindless texture array, 5 = the material table), the same 48-byte push block, the same
// 64-wide workgroup and the same one dispatch per caster - so the A/B against the parent commit decides it.

module;

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <utility>
#include <vulkan/vulkan.h>

module vulkan.pass.mask_bake;

import vulkan.pipelines; // build_mask_bake: the compute pipeline this job owns
import utility;

namespace vulkan::pass {

    mask_bake_job::~mask_bake_job() {
        this->release_owned();
    }

    void mask_bake_job::release_owned() noexcept {
        this->pipeline_.reset();
        if (this->pipeline_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->device_, this->pipeline_layout_, nullptr);
            this->pipeline_layout_ = VK_NULL_HANDLE;
        }
        this->set_.release(); // the pool is the core's and outlives this job (see the header)
    }

    bool mask_bake_job::ready() const noexcept {
        return this->pipeline_.has_value() && this->set_.get() != VK_NULL_HANDLE;
    }

    VkPipeline mask_bake_job::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout mask_bake_job::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    std::expected<void, std::string> mask_bake_job::create(pass_context const& context, vk_descriptor_set set, mask_bake_inputs const& inputs) {
        this->set_ = std::move(set);
        if (context.device == VK_NULL_HANDLE) {
            return std::unexpected(std::string("mask bake: no device"));
        }
        if (this->device_ != VK_NULL_HANDLE && this->device_ != context.device) {
            this->release_owned();
        }
        this->device_ = context.device;
        std::span<unsigned char const> const spirv = context.shader != nullptr ? context.shader(context.owner, shader_name) : std::span<unsigned char const>{};
        if (spirv.empty()) {
            return std::unexpected(std::string("mask bake: the owner has no ") + std::string(shader_name));
        }
        // The scene set ALONE, because everything the bake reads is in it: the material table (the alpha
        // texture's index, the base colour factor's alpha, the cutoff) and the bindless texture array.
        VkDescriptorSetLayout const scene_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 0) : VK_NULL_HANDLE;
        if (scene_layout == VK_NULL_HANDLE) {
            return std::unexpected(std::string("mask bake: the owner has no scene set layout"));
        }
        auto built = pipelines::build_mask_bake(context.device, scene_layout, static_cast<uint32_t>(sizeof(mask_bake_push_constants)), spirv);
        if (!built) {
            return std::unexpected(std::move(built.error()));
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->trace);

        // The two bindings the bake reads, written ONCE here: binding 1 is the bindless texture array and
        // binding 5 the material table, exactly as the raster path declares them.
        if (this->set_.get() == VK_NULL_HANDLE || inputs.textures == VK_NULL_HANDLE || inputs.texture_sampler == VK_NULL_HANDLE || inputs.material_table == VK_NULL_HANDLE) {
            return std::unexpected(std::string("mask bake: the material table, the texture array or the set is not ready"));
        }
        VkDescriptorImageInfo const textures_info = {.sampler = inputs.texture_sampler, .imageView = inputs.textures, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorBufferInfo const materials_info = {.buffer = inputs.material_table, .offset = 0, .range = VK_WHOLE_SIZE};
        std::array<VkWriteDescriptorSet, 2> writes = {};
        for (uint32_t b = 0; b < writes.size(); ++b) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = this->set_.get();
            writes[b].dstBinding = b == 0u ? 1u : 5u; // the texture array, then the material table
            writes[b].dstArrayElement = 0;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = b == 0u ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pImageInfo = b == 0u ? &textures_info : nullptr;
            writes[b].pBufferInfo = b == 0u ? nullptr : &materials_info;
        }
        vkUpdateDescriptorSets(this->device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return {};
    }

    void mask_bake_job::record(VkCommandBuffer const command_buffer, mask_bake_request const& request) const noexcept {
        if (!this->ready() || request.triangle_count == 0) {
            return;
        }
        auto const halves = [](VkDeviceAddress const address) {
            return glm::uvec2(static_cast<uint32_t>(address & 0xFFFFFFFFu), static_cast<uint32_t>(address >> 32u));
        };
        mask_bake_push_constants bake = {};
        bake.source_vertices = halves(request.source_vertices);
        bake.source_indices = halves(request.source_indices);
        bake.destination = halves(request.destination);
        bake.source_stride = request.source_stride;
        bake.destination_stride = request.destination_stride;
        bake.index_type = request.index_type;
        bake.triangle_count = request.triangle_count;
        bake.material_index = request.material_index;
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->pipeline());
        // The bake's own set - NOT the frame's scene set, which this same command buffer will have updated by
        // the end of the frame (binding 16). See the file's header for what that cost when it was not split.
        VkDescriptorSet const bake_set = this->set_.get();
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->pipeline_layout_, 0, 1, &bake_set, 0, nullptr);
        vkCmdPushConstants(command_buffer, this->pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bake), &bake);
        vkCmdDispatch(command_buffer, (bake.triangle_count + group_size - 1u) / group_size, 1, 1);
    }

} // namespace vulkan::pass
