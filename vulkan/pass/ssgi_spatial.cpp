// The spatial filter's implementation: the two barriers around its storage output, the two shared sets and the
// dispatch. Moved out of `runtime::record_ssgi_spatial_pass` UNCHANGED in behaviour - the same barrier pair, the
// same bind order (scene set, then G-buffer set), the same 48-byte push and the same half-resolution dispatch -
// so the capture gate decides the move on the four GI scenarios.

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <vulkan/vulkan.h>

module vulkan.pass.ssgi_spatial;

import vulkan.render_resource;
import vulkan.constant_init;

namespace vulkan::pass {

    render_resource::pass_io const& ssgi_spatial_pass::io() const noexcept {
        return render_resource::ssgi_spatial_io;
    }

    vulkan::pass::behaviour const& ssgi_spatial_pass::behaviour() const noexcept {
        return behaviour_;
    }

    std::string_view ssgi_spatial_pass::feature() const noexcept {
        // The CHAIN's feature: the filter is the last stage, and a frame whose chain is off records nothing.
        return "ssgi";
    }

    void ssgi_spatial_pass::create(pass_context const&) {
        // Nothing to build: no set layout (every binding it uses is in the shared G-buffer set, ensured by that
        // set's owner) and no pipeline (the renderer builds and hands it over). See the header.
    }

    void ssgi_spatial_pass::on_swapchain_recreated(pass_host const&) {
        this->resolved_ = false;
    }

    bool ssgi_spatial_pass::resolved() const noexcept {
        return this->resolved_;
    }

    void ssgi_spatial_pass::record(resolved_io const& io) {
        this->resolved_ = false;
        if (io.barrier_images.size() < render_resource::ssgi_spatial_barriers.size() || io.pipelines.empty() || io.pipelines[0] == VK_NULL_HANDLE ||
            io.pipeline_layout == VK_NULL_HANDLE || io.shared.scene == VK_NULL_HANDLE || io.shared.gbuffer == VK_NULL_HANDLE || io.push.size() < sizeof(push_constants) ||
            io.extent.width == 0 || io.extent.height == 0) {
            return; // the runner resolves all of this or skips the pass (see runtime::resolve_ssgi_spatial)
        }
        VkImage const output = io.barrier_images[barrier_output].image;

        // The output is a storage image whose contents are fully overwritten this frame: UNDEFINED -> GENERAL
        // here, and GENERAL -> SHADER_READ at the end for the composite. The INPUT needs no barrier - the
        // temporal resolve handed it to SHADER_READ through a transition that names COMPUTE as well as FRAGMENT,
        // which is the read this dispatch does.
        VkImageMemoryBarrier2 to_general = vulkan::undefined_to_general_transition;
        to_general.image = output;
        VkDependencyInfo const general_dependency = make_image_dependency_info(1, &to_general);
        vkCmdPipelineBarrier2(io.cmd, &general_dependency);

        // Two sets, then the pipeline: the shared scene set and the shared G-buffer set, in that order.
        std::array<VkDescriptorSet, 2> const sets = {io.shared.scene, io.shared.gbuffer};
        vkCmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, io.pipeline_layout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);

        push_constants push = {};
        std::memcpy(&push, io.push.data(), sizeof(push));
        vkCmdPushConstants(io.cmd, io.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(io.cmd, (io.extent.width + group_size - 1u) / group_size, (io.extent.height + group_size - 1u) / group_size, 1);

        // Hand the filtered image to the composite. This is also where the frame's GI becomes usable: the
        // composite's weight is read from `gi_resolved`, and only this pass writes the image that weight applies
        // to - which is why the renderer sets it from this pass's own answer.
        VkImageMemoryBarrier2 to_sampling = vulkan::general_to_sampling_transition;
        to_sampling.image = output;
        VkDependencyInfo const sampling_dependency = make_image_dependency_info(1, &to_sampling);
        vkCmdPipelineBarrier2(io.cmd, &sampling_dependency);

        this->resolved_ = true;
    }

} // namespace vulkan::pass
