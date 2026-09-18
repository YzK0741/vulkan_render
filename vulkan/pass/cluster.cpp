// The clustered-light sort's implementation: the shared scene set, the 1D dispatch over the cluster grid and
// the two buffer barriers that publish its writes. Moved out of `runtime::record_cluster_pass` and
// `core::make_cluster_pipeline` UNCHANGED in behaviour - the same single set bound at index 0, the same 64-wide
// workgroup, the same `(count + 63) / 64` group count and the same two COMPUTE_SHADER -> FRAGMENT_SHADER buffer
// barriers over the same two buffers - so the A/B against the parent commit decides it.
//
// It is the first compute pipeline this branch has taken out of `vulkan.core`: `core::make_cluster_pipeline`
// built it against the core's own scene pipeline layout, which a pass cannot own. The replacement builds the
// same shape in `vulkan.pipelines` (one set layout, no push range - `light_cluster.comp` takes no constants),
// so what changes is WHO owns the layout, not what the driver is asked for.

module;

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

module vulkan.pass.cluster;

import vulkan.pipelines; // build_cluster: the compute pipeline this pass owns
import utility;

namespace vulkan::pass {

    cluster_pass::~cluster_pass() {
        this->release_owned();
    }

    void cluster_pass::release_owned() noexcept {
        this->pipeline_.reset();
        if (this->pipeline_layout_ != VK_NULL_HANDLE && this->device_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(this->device_, this->pipeline_layout_, nullptr);
            this->pipeline_layout_ = VK_NULL_HANDLE;
        }
    }

    render_resource::pass_io const& cluster_pass::io() const noexcept {
        return render_resource::cluster_io;
    }

    vulkan::pass::behaviour const& cluster_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view cluster_pass::feature() const noexcept {
        // The renderer's registry answers whether there is anything to sort (the knob, a punctual light, a
        // shading stage that reads a light list) and whether this pass built its pipeline.
        return "clustered";
    }

    bool cluster_pass::pipeline_ready() const noexcept {
        return this->pipeline_.has_value();
    }

    VkPipeline cluster_pass::pipeline() const noexcept {
        return this->pipeline_.has_value() ? this->pipeline_->get_pipeline() : VK_NULL_HANDLE;
    }

    VkPipelineLayout cluster_pass::pipeline_layout() const noexcept {
        return this->pipeline_layout_;
    }

    void cluster_pass::set_frame(cluster_frame const& frame) noexcept {
        this->frame_ = frame;
    }

    void cluster_pass::prepare_frame(frame_facts const& facts) noexcept {
        // ONE number, and the host is the only one that can compute it: the grid the light culling produced,
        // which is the same `cluster_grid` the light UBO carries (see `runtime::make_frame_facts`).
        this->set_frame(cluster_frame{.cluster_count = facts.cluster_count});
    }

    void cluster_pass::create(pass_context const& context) {
        if (context.device == VK_NULL_HANDLE) {
            return;
        }
        if (this->device_ != VK_NULL_HANDLE && this->device_ != context.device) {
            this->release_owned();
        }
        this->device_ = context.device;
        if (this->pipeline_.has_value()) {
            return; // already built for this device
        }
        std::span<unsigned char const> const spirv = context.shader != nullptr ? context.shader(context.owner, shader_name) : std::span<unsigned char const>{};
        if (spirv.empty()) {
            utility::log("clustered lights disabled: the owner has no {}", shader_name);
            return;
        }
        VkDescriptorSetLayout const scene_layout = context.shared_set_layout != nullptr ? context.shared_set_layout(context.owner, 0) : VK_NULL_HANDLE;
        if (scene_layout == VK_NULL_HANDLE) {
            utility::log("clustered lights disabled: the owner has no scene set layout");
            return;
        }
        auto built = pipelines::build_cluster(context.device, scene_layout, spirv);
        if (!built) {
            utility::log("clustered lights disabled: {}", built.error());
            this->release_owned();
            return;
        }
        this->pipeline_layout_ = built->pipeline_layout;
        this->pipeline_ = std::move(built->trace);
        utility::log("SUCCESS: clustered light pipeline created (the frame's punctual lights are binned per cluster)");
    }

    void cluster_pass::on_swapchain_recreated(pass_host const&) {
        // Nothing to reset: the grid's DIMENSIONS are the renderer's frame data (it recomputes them from the new
        // extent and hands them over every frame), and this pass owns no per-generation handle.
    }

    void cluster_pass::record(resolved_io const& io) {
        if (!this->pipeline_.has_value() || io.pipelines.empty() || io.pipelines[0] == VK_NULL_HANDLE || io.pipeline_layout == VK_NULL_HANDLE ||
            io.shared.scene == VK_NULL_HANDLE || io.barrier_buffers.size() < render_resource::cluster_barriers.size() || this->frame_.cluster_count == 0) {
            return; // the runner resolves all of this or skips the pass (see runtime::resolve_cluster_pass)
        }
        VkBuffer const counts = io.barrier_buffers[barrier_counts].buffer;
        VkBuffer const indices = io.barrier_buffers[barrier_indices].buffer;
        if (counts == VK_NULL_HANDLE || indices == VK_NULL_HANDLE) {
            return;
        }

        // The dispatch reads the frame's OWN scene set (the paced slot's camera and light UBOs) and writes the
        // same slot's cluster buffers: a compute stage is not part of a rendering instance, so this records
        // before vkCmdBeginRendering. The HEAP path (see pipelines::build_cluster) would drop this bind and let
        // the frame's pushed index pick the slot instead - and it cannot come before the rest of the frame does.
        VkDescriptorSet const scene_set = io.shared.scene;
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, io.pipeline_layout, 0, 1, &scene_set, 0, nullptr);
        vkCmdBindPipeline(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, io.pipelines[0]);
        vkCmdDispatch(io.cmd, (this->frame_.cluster_count + group_size - 1u) / group_size, 1, 1);

        // Hand the two buffers to the fragment stages that read them later in this submission (forward shading
        // inside the main instance, and the deferred lighting pass): a compute SHADER_WRITE is not visible to a
        // later SHADER_READ without this barrier. One barrier per buffer, because VkBufferMemoryBarrier2 covers a
        // single buffer - and the two handles come from the DECLARATION, which is what makes this the writer's
        // own ordering rather than something the host has to remember on the pass's behalf.
        std::array<VkBufferMemoryBarrier2, 2> barriers = {};
        barriers[0].buffer = counts;
        barriers[1].buffer = indices;
        for (VkBufferMemoryBarrier2& barrier : barriers) {
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            // the shader reads counts/indices as storage buffers, not as sampled images
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.offset = 0;
            barrier.size = VK_WHOLE_SIZE;
        }
        VkDependencyInfo dependency = {};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
        dependency.pBufferMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(io.cmd, &dependency);
    }

} // namespace vulkan::pass
