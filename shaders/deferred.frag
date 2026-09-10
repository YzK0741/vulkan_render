#version 450
#extension GL_EXT_nonuniform_qualifier : enable

/**
 * @file shaders/deferred.frag
 * @brief Deferred lighting: shade every pixel once, from the G-buffer, in screen space.
 * @ingroup shaders
 *
 * The second half of the deferred path. It reads the three G-buffer targets plus the pass's 1x depth
 * image, rebuilds the world position from that depth, and evaluates the same lighting the forward
 * path evaluates per fragment - literally the same code: shaders/shading.glsl owns the sun, the
 * shadow test, the punctual lights, the split-sum IBL and the BRDF presets, and this stage only
 * supplies a shade_input assembled from G-buffer texels instead of from interpolated vertex data.
 * That is what makes the forward and the deferred image comparable rather than merely similar.
 *
 * Inputs (its own descriptor set, which the runtime binds as set 1 - the scene set stays set 0):
 * - binding 0: albedo.rgb + metallic (RGBA8)
 * - binding 1: world normal.xyz + roughness (RGBA16F)
 * - binding 2: material id low/high byte + ambient occlusion + material flags (RGBA8)
 * - binding 3: the G-buffer pass's depth (D32_SFLOAT, sampled)
 *
 * Push constant: the inverse view-projection, which turns (uv, depth) back into a world position.
 * Reconstructing instead of storing a world position costs one 4x4 multiply per pixel and saves
 * 8-12 bytes per pixel of G-buffer memory; the depth is exact, so the reconstruction is exact too.
 *
 * Output: LINEAR HDR radiance, added to the HDR target (the pipeline blends with ONE/ONE). What is
 * already in that target when this pass runs:
 * - the emissive term, which the G-buffer pass added additively (gbuffer.frag): emissive is
 *   lighting-independent, so it does not belong to this stage's work, and
 * - nothing else - the HDR target is cleared (to zero) before the G-buffer pass, so a pixel with no
 *   geometry holds exactly 0. This stage is therefore free to write the SKY there, which is why the
 *   deferred path draws no background pass at all: a background is not a surface and has no place in
 *   a G-buffer. The sky evaluation is the same function the forward skybox pass uses
 *   (shaders/sky.glsl), so both paths produce the same background from the same code.
 */

#include "shading.glsl"
#include "sky.glsl"

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 1, binding = 0) uniform sampler2D gbuffer_albedo;   // RGBA8: albedo.rgb + metallic
layout(set = 1, binding = 1) uniform sampler2D gbuffer_normal;   // RGBA16F: normal.xyz + roughness
layout(set = 1, binding = 2) uniform sampler2D gbuffer_material; // RGBA8: id lo/hi + ao + flags
layout(set = 1, binding = 3) uniform sampler2D gbuffer_depth;    // the pass's single-sampled depth

layout(push_constant) uniform DeferredPush {
    mat4 inv_view_proj; // clip (NDC xyz, w = 1) -> world position
} pc;

/**
 * @brief rebuild the world-space position of the pixel from the stored depth
 * @param uv the pixel's texture coordinate (v_uv)
 * @param depth the G-buffer depth in [0,1]
 * @return world-space position
 * @note the engine's projection is a RH_ZO matrix with its Y row flipped for Vulkan
 *       (make_orbit_camera_ubo), and post.vert's v_uv already maps (0,0) to the top-left texel, so
 *       the NDC of a pixel is (2*uv - 1) in both axes: no extra flip belongs here - the inverse
 *       matrix already carries the projection's own conventions.
 */
vec3 world_position_from_depth(vec2 uv, float depth) {
    const vec4 world = pc.inv_view_proj * vec4(uv * 2.0 - 1.0, depth, 1.0);
    return world.xyz / world.w;
}

/**
 * @brief shade the pixel described by the G-buffer
 */
void main() {
    const float depth = texture(gbuffer_depth, v_uv).r;
    if (depth >= 1.0) {
        // No geometry here: this pixel shows the sky. The far-plane point along the pixel's ray gives
        // the view direction (the reconstructed position is exact even at depth 1), and the sky
        // function is the forward skybox pass's - so the two paths cannot disagree about the
        // background. The value is ADDED, like the lighting below, onto an HDR target this stage
        // knows to be zero there.
        const vec3 far_point = world_position_from_depth(v_uv, 1.0);
        out_color = vec4(sky_color(normalize(far_point - camera.camera_pos)), 1.0);
        return;
    }

    const vec4 albedo_metallic = texture(gbuffer_albedo, v_uv);
    const vec4 normal_roughness = texture(gbuffer_normal, v_uv);
    const vec4 material = texture(gbuffer_material, v_uv);

    shade_input si;
    si.world_pos = world_position_from_depth(v_uv, depth);
    si.normal = normal_roughness.xyz;
    si.albedo = albedo_metallic.rgb;
    si.metallic = albedo_metallic.a;
    si.roughness = normal_roughness.w;
    si.ao = material.b;
    // emissive is NOT re-evaluated here: the G-buffer pass already added it into the HDR target,
    // because it needs the material's emissive texture and the UVs - neither of which the G-buffer
    // stores (see gbuffer.frag). Adding it again would double it.
    si.emissive = vec3(0.0);

    out_color = vec4(shade_surface(si), 1.0);
}
