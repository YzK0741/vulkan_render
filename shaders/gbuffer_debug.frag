#version 450

/**
 * @file shaders/gbuffer_debug.frag
 * @brief Debug view of the G-buffer: shows one target channel at a time on the screen.
 * @ingroup shaders
 *
 * The G-buffer is the one part of the renderer whose contents cannot be judged from a normal
 * screenshot - a wrong normal encoding, a swapped roughness/metallic channel or a wrong material
 * index all look like "the lighting is a bit off" once the lighting pass consumed them. So the
 * deferred path ships with a way to look at the raw data: this fullscreen pass reads the three
 * targets (+ the pass's depth) and writes one selected channel into the HDR scene target, which
 * then goes through the ordinary post chain (exposure / tonemapping / FXAA) like any other frame.
 *
 * Push constant: the channel selector plus the two projection terms needed to linearize depth.
 *
 * @note pairing: this pass is only recorded when the runtime draws the G-buffer instead of the
 *       shaded scene (runtime::set_gbuffer_debug), and it replaces the lighting pass in that mode -
 *       milestone M2 adds the real deferred lighting stage next to it.
 */

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D gbuffer_albedo;  // RGBA8: albedo.rgb + metallic
layout(set = 0, binding = 1) uniform sampler2D gbuffer_normal;  // RGBA16F: normal.xyz + roughness
layout(set = 0, binding = 2) uniform sampler2D gbuffer_material; // RGBA8: id lo/hi + ao + flags
layout(set = 0, binding = 3) uniform sampler2D gbuffer_depth;   // the pass's single-sampled depth

layout(push_constant) uniform GbufferDebugPush {
    float channel; // 0 albedo, 1 normal, 2 roughness, 3 metallic, 4 ao, 5 material id, 6 depth, 7 flags
    float proj_22; // projection[2][2]: the depth-linearization terms (see view_depth)
    float proj_32; // projection[3][2]
    float unused;
} pc;

/**
 * @brief view-space distance of a depth-buffer sample
 * @param depth the stored depth in [0,1] (Vulkan NDC range)
 * @return positive view-space distance
 * @note for the standard perspective projection the CPU builds, the stored depth is
 *       z_ndc = (proj_22 * z_view + proj_32) / -z_view, which inverts to z_view = proj_32 / (z_ndc + proj_22).
 *       Passing the two matrix terms avoids having the shader reconstruct them from near/far and
 *       stays correct if the runtime changes the depth range convention.
 */
float view_depth(float depth) {
    return pc.proj_32 / (depth + pc.proj_22);
}

/**
 * @brief a stable pseudo-random color per material id (neighbouring ids must not look alike)
 * @param id 16-bit material index
 * @return an RGB color with full-range channels
 */
vec3 material_id_color(uint id) {
    return vec3(float((id * 37u) % 251u), float((id * 91u) % 251u), float((id * 53u) % 251u)) / 250.0;
}

/**
 * @brief show the selected G-buffer channel
 * @note the view runs through the ordinary post chain (exposure -> ACES -> display encode), so the
 *       displayed brightness is not the stored value: it is the stored value with the renderer's
 *       tonemapping on top. The point of the view is structure - where a channel is right, wrong or
 *       missing - not reading numbers back off the screen.
 */
void main() {
    const uint channel = uint(pc.channel + 0.5);
    const vec4 albedo_metallic = texture(gbuffer_albedo, v_uv);
    const vec4 normal_roughness = texture(gbuffer_normal, v_uv);
    const vec4 material = texture(gbuffer_material, v_uv);
    // The cleared depth is the far plane: those pixels hold no geometry, so every channel shows
    // them black. Without this the normal channel would paint them mid-grey (the remap of a zero
    // normal), which reads like a surface that is lit strangely rather than like empty space.
    const bool background = texture(gbuffer_depth, v_uv).r >= 1.0;

    vec3 color;
    if (channel == 6u) {
        // near = white, far = black; the cleared background is the far plane
        const float far_plane = pc.proj_32 / (1.0 + pc.proj_22);
        color = vec3(1.0 - clamp(view_depth(texture(gbuffer_depth, v_uv).r) / far_plane, 0.0, 1.0));
    } else if (background) {
        color = vec3(0.0);
    } else if (channel == 0u) {
        color = albedo_metallic.rgb;
    } else if (channel == 1u) {
        color = normal_roughness.xyz * 0.5 + 0.5; // world normal -> visible color
    } else if (channel == 2u) {
        color = vec3(normal_roughness.w);
    } else if (channel == 3u) {
        color = vec3(albedo_metallic.a);
    } else if (channel == 4u) {
        color = vec3(material.b);
    } else if (channel == 5u) {
        const uint id = uint(round(material.r * 255.0)) | (uint(round(material.g * 255.0)) << 8u);
        color = material_id_color(id);
    } else {
        color = vec3(material.a); // raw material flag byte
    }
    out_color = vec4(color, 1.0);
}
