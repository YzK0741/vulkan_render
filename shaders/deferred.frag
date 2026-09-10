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
 * It also owns the screen-space ambient occlusion (M6): the occlusion is computed from the very depth
 * and normals this stage already reads, folded into the shade_input's ao, and therefore scales the
 * IBL ambient exactly like a material's baked AO map does - no extra render target, no extra pass.
 *
 * Inputs (its own descriptor set, which the runtime binds as set 1 - the scene set stays set 0):
 * - binding 0: albedo.rgb + metallic (RGBA8)
 * - binding 1: world normal.xyz + roughness (RGBA16F)
 * - binding 2: material id low/high byte + ambient occlusion + material flags (RGBA8)
 * - binding 3: the G-buffer pass's depth (D32_SFLOAT, sampled)
 *
 * Push constant: the inverse view-projection, which turns (uv, depth) back into a world position, and
 * the SSAO parameters. Reconstructing the position instead of storing it costs one 4x4 multiply per
 * pixel and saves 8-12 bytes per pixel of G-buffer memory; the depth is exact, so the reconstruction
 * is exact too.
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
    // Screen-space ambient occlusion (M6): x = world-space radius, y = intensity (0 = off),
    // z = sample count, w = depth bias. The runtime fills it per frame; an intensity <= 0 makes the
    // term return exactly 1.0, which is what keeps an SSAO-off frame bit for bit the pre-M6 frame.
    vec4 ssao;
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

/// @brief AO sample budget: the shader loop bound (the runtime's ssao_samples is clamped to it)
const int MAX_SSAO_SAMPLES = 16;

/**
 * @brief screen-space ambient occlusion of the pixel (hemisphere samples against the depth buffer)
 * @param uv the pixel's texture coordinate (v_uv)
 * @param depth the G-buffer depth at @p uv
 * @param world_normal the G-buffer world normal at @p uv
 * @return 1.0 = fully unoccluded (also when SSAO is off), 0.0 = fully occluded
 *
 * The classic screen-space AO: a hemisphere of sample points around the pixel's view-space position,
 * each projected back to screen and compared against the depth buffer. A sample counts as occluded
 * when the surface stored at its screen position is CLOSER to the camera than the sample point and
 * inside the radius - view-space z is negative in front of the camera, so "closer" means a greater z.
 * The kernel is a golden-angle hemisphere SPIRAL (uniform in the projected disc, no literal table)
 * rotated per pixel by a hash of gl_FragCoord: a fixed kernel would show the sample pattern as
 * spiral banding.
 *
 * This is deliberately the affordable end of screen-space AO. It is not a horizon-based GTAO (which
 * estimates the true horizon angle per slice and is the modern engine default) and it carries the
 * usual screen-space limitations: geometry off screen occludes nothing, and the radius is in world
 * units, so its apparent strength depends on the view distance. It scales the IBL ambient only -
 * `shade_surface()` applies `s.ao` to the diffuse and specular ambient and never to the direct sun.
 */
float ssao_occlusion(vec2 uv, float depth, vec3 world_normal) {
    const float radius = pc.ssao.x;
    const float intensity = pc.ssao.y;
    const int samples = int(pc.ssao.z + 0.5);
    if (radius <= 0.0 || intensity <= 0.0 || samples <= 0) {
        return 1.0; // off: an exact 1.0 leaves the shaded result untouched
    }
    const vec3 world_pos = world_position_from_depth(uv, depth);
    const vec3 view_pos = (camera.view * vec4(world_pos, 1.0)).xyz;
    // the G-buffer normal is world space; the hemisphere must be built around its VIEW-space form
    const vec3 view_normal = normalize((camera.view * vec4(world_normal, 0.0)).xyz);

    // per-pixel rotation: hash the pixel coordinate into an angle (the classic fract(sin(dot(..))))
    const float hash = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    const float angle = hash * 6.2831853;
    const vec3 random_vec = vec3(cos(angle), sin(angle), 0.0);
    const vec3 tangent = normalize(random_vec - view_normal * dot(random_vec, view_normal));
    const vec3 bitangent = cross(view_normal, tangent);
    const mat3 tbn = mat3(tangent, bitangent, view_normal);

    float occlusion = 0.0;
    for (int i = 0; i < MAX_SSAO_SAMPLES; ++i) {
        if (i >= samples) {
            break;
        }
        // golden-angle spiral in the hemisphere: cos/sin over the projected disc, z = the height
        const float t = (float(i) + 0.5) / float(samples);
        const float disc_radius = sqrt(t);
        const float phi = float(i) * 2.39996323;
        const vec3 kernel = tbn * vec3(cos(phi) * disc_radius, sin(phi) * disc_radius, sqrt(max(1.0 - t, 0.0)));
        // samples closer to the pixel count more: scale the kernel towards the origin
        const vec3 sample_pos = view_pos + kernel * (radius * (0.2 + 0.8 * t));

        const vec4 clip = camera.proj * vec4(sample_pos, 1.0);
        const vec2 sample_uv = (clip.xy / clip.w) * 0.5 + 0.5;
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue; // outside the frame: no depth to compare against (the screen-space limit)
        }
        const float sample_depth = texture(gbuffer_depth, sample_uv).r;
        if (sample_depth >= 1.0) {
            continue; // sky there: not an occluder
        }
        const vec3 scene_pos = world_position_from_depth(sample_uv, sample_depth);
        const float scene_z = (camera.view * vec4(scene_pos, 1.0)).z;
        // occluded when the stored surface sits IN FRONT of the sample point (greater z = closer) ...
        // ... and the range check fades the contact out over the radius instead of cutting it hard
        const float range = clamp(radius / max(abs(view_pos.z - scene_z), 1e-5), 0.0, 1.0);
        occlusion += (scene_z >= sample_pos.z + pc.ssao.w) ? range : 0.0;
    }
    occlusion /= float(samples);
    return clamp(1.0 - occlusion * intensity, 0.0, 1.0);
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
    // ambient occlusion: the material's baked AO map times the screen-space term (M6). With SSAO
    // off ssao_occlusion() returns exactly 1.0, so this is the pre-M6 value bit for bit.
    si.ao = material.b * ssao_occlusion(v_uv, depth, si.normal);
    // emissive is NOT re-evaluated here: the G-buffer pass already added it into the HDR target,
    // because it needs the material's emissive texture and the UVs - neither of which the G-buffer
    // stores (see gbuffer.frag). Adding it again would double it.
    si.emissive = vec3(0.0);

    out_color = vec4(shade_surface(si), 1.0);
}
