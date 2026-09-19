// ============================================================================
// module: vulkan.runtime:readback  - the screenshot read-back
//
// THE SCREENSHOT PATH: acquire_current_frame_image waits for the GPU and hands back pixels,
// record_screenshot_copy is the copy recorded INSIDE the frame command buffer before the present
// transition, and consume_screenshot_request is the one-shot flag the caller reads.
//
// Imports are NOT transitive: this partition imports what its own code calls, and repeats the pmr
// keep-alive that must run before any pmr container in this TU.
// ============================================================================
module;

#include <GLFW/glfw3.h>
#include <algorithm> // std::min in the resource publication
#include <bit>       // std::bit_cast for the caster world-matrix hash
#include <chrono>
#include <cstring> // std::memcpy, for composing a pass's push block
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <span>   // std::as_bytes for the init_utils calls (the bytes behind a UBO or a zeroed table)
#include <thread> // std::this_thread::yield in the frame limiter
#include <vulkan/vulkan.h>

module vulkan.runtime:readback;

import :declarations;

import vulkan.profiling;
import vulkan.pipelines;
import vulkan.bindings;
import vulkan.render_resource;
import vulkan.render_resource.shared;

import utility;
import vulkan.constant_init;
import vulkan.init_utils;      // the resource-creation patterns the init/ensure functions below repeat
import vulkan.frame_constants; // one frame's shared constants (see update_frame_constants)
import vulkan.core.pipeline;   // vulkan::make_pipeline for the post-process pipeline

// Route std::pmr allocations through mimalloc for this TU (utility.better_pmr). Idempotent:
// init_pmr() returns the same process-wide singleton no matter which TU calls it first, so
// main.cpp's keep-alive and this one coexist safely. The reference itself is never read; it
// only forces the (dynamic) initialization before any pmr container in this TU is constructed.

namespace vulkan {
    std::expected<runtime::frame_image, std::string> runtime::acquire_current_frame_image() {
        core& vk = this->vulkan_core;
        if (vk.swap_chain == VK_NULL_HANDLE || vk.swap_chain_images.empty()) {
            return std::unexpected(std::string("screenshot: no swapchain image available"));
        }

        VkFormat const format = vk.swap_chain_image_format;
        bool const bgra = format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_B8G8R8A8_UNORM;
        bool const rgba = format == VK_FORMAT_R8G8B8A8_SRGB || format == VK_FORMAT_R8G8B8A8_UNORM;
        if (!bgra && !rgba) {
            return std::unexpected(std::string("screenshot: unsupported swapchain format (need 8-bit RGBA/BGRA)"));
        }

        // The pixels come from the staging buffer record_screenshot_copy() filled while the frame was
        // being recorded (see the class docs): by the time the caller asks, the frame has been
        // submitted, so one wait for the GPU is all that is left - and the staging's owner is
        // vulkan.readback, which is also what sized the buffer and handed out the mapping.
        if (this->screenshot_staging_mapped == nullptr || this->screenshot_readback_extent.width == 0) {
            return std::unexpected(std::string("screenshot: no captured frame (the read-back copy was never recorded)"));
        }
        VkExtent2D const extent = this->screenshot_readback_extent;
        VkDeviceSize const buffer_size = static_cast<VkDeviceSize>(extent.width) * static_cast<VkDeviceSize>(extent.height) * 4u;
        vk.wait_idle();

        frame_image result = {};
        result.width = extent.width;
        result.height = extent.height;
        result.rgba.resize(static_cast<std::size_t>(buffer_size));
        auto const* source = static_cast<unsigned char const*>(this->screenshot_staging_mapped);
        if (bgra) {
            // the swapchain is BGRA (sRGB); the PNG writer wants RGBA
            for (std::size_t i = 0; i < result.rgba.size(); i += 4) {
                result.rgba[i + 0] = source[i + 2]; // R
                result.rgba[i + 1] = source[i + 1]; // G
                result.rgba[i + 2] = source[i + 0]; // B
                result.rgba[i + 3] = source[i + 3]; // A
            }
        } else {
            std::memcpy(result.rgba.data(), source, static_cast<std::size_t>(buffer_size));
        }
        return result;
    }

    void runtime::record_screenshot_copy(VkCommandBuffer const command_buffer) {
        core& vk = this->vulkan_core;
        VkExtent2D const extent = vk.swap_chain_extent;
        if (extent.width == 0 || extent.height == 0 || this->current_image_index >= vk.swap_chain_images.size()) {
            return; // nothing sensible to copy (the frame will be skipped anyway)
        }
        if (!vk.swapchain_transfer_src_supported) {
            // The swapchain images lack VK_IMAGE_USAGE_TRANSFER_SRC_BIT, so vkCmdCopyImageToBuffer
            // from one of them would violate VUID-vkCmdCopyImageToBuffer-srcImage-00186. The surface
            // cannot do screenshots at all: say it once, drop the request (a permanent condition -
            // retrying every frame would only spam), and let main see "nothing captured".
            if (!this->screenshot_unsupported_logged) {
                this->screenshot_unsupported_logged = true;
                utility::log("screenshot: unsupported (swapchain has no TRANSFER_SRC usage) - F12 disabled");
            }
            this->screenshot_requested = false;
            return;
        }
        // The staging buffer and its mapping are vulkan.readback's; only the IMAGE side is this function's
        // business (the layout transitions, the region, the format the caller will unpack).
        auto const staged = this->readback_staging.stage_for_copy(static_cast<VkDeviceSize>(extent.width) * static_cast<VkDeviceSize>(extent.height) * 4u);
        if (!staged) {
            utility::log("screenshot: read-back staging buffer unavailable");
            return;
        }
        this->screenshot_staging_mapped = staged->mapped;
        this->screenshot_readback_extent = extent;

        // The swapchain image is in COLOR_ATTACHMENT_OPTIMAL here (the composite pass just wrote it, and the
        // overlay with it): COLOR_ATTACHMENT -> TRANSFER_SRC -> copy -> back to COLOR_ATTACHMENT, so
        // end_recording's present_transition still sees the layout it expects.
        std::array<VkImageMemoryBarrier2, 1> barriers = {vulkan::color_attachment_to_transfer_transition};
        barriers[0].image = vk.swap_chain_images[this->current_image_index];
        VkDependencyInfo dependency_info = make_image_dependency_info(1, barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency_info);

        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(command_buffer, vk.swap_chain_images[this->current_image_index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staged->buffer, 1, &region);

        barriers[0] = vulkan::transfer_to_color_attachment_transition;
        barriers[0].image = vk.swap_chain_images[this->current_image_index];
        dependency_info = make_image_dependency_info(1, barriers.data());
        vkCmdPipelineBarrier2(command_buffer, &dependency_info);

        this->screenshot_pending = true;
    }

    bool runtime::consume_screenshot_request() noexcept {
        // Could the requested frame be captured? Single-shot by design: the flag is cleared HERE, on
        // success and on failure alike. A failing read-back (unsupported swapchain format, missing
        // read-back buffer) used to leave the flag set, so main's loop called
        // acquire_current_frame_image() - which begins with vkDeviceWaitIdle - and logged an error
        // every single frame until exit. A dropped capture is the correct outcome; one F12 is one
        // attempt.
        bool const captured = this->screenshot_pending;
        this->screenshot_pending = false;
        return captured;
    }
} // namespace vulkan