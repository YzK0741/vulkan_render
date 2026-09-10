#version 450
#extension GL_EXT_nonuniform_qualifier : enable

/**
 * @file shaders/gbuffer.frag
 * @brief G-buffer write stage: stores the surface of an opaque fragment instead of shading it.
 * @ingroup shaders
 *
 * The deferred path's first half. Instead of evaluating lights (what pbr.frag does), this stage
 * only asks what the fragment is made of and writes that into the G-buffer targets, so the lighting
 * pass can shade every pixel once, from screen space, and screen-space effects (SSAO, SSR, TAA
 * velocity) get the data they need. Both stages gather that surface through the same
 * shaders/surface.glsl, so the deferred and forward views of one object cannot drift apart.
 *
 * Targets (attachment order = core::gbuffer_formats, and the location order below):
 * - location 0 (RGBA8_UNORM): albedo.rgb + metallic - base color is linear and stored unencoded,
 *   and 8-bit metallic is plenty for a metallic-roughness workflow (glTF allows 8-bit inputs);
 * - location 1 (RGBA16F): world normal.xyz + roughness - the normal is stored as three floats
 *   rather than octahedral-encoded: 4 bytes per pixel buys the absence of a whole class of
 *   precision bugs while the layout is young;
 * - location 2 (RGBA8_UNORM): material_id low byte + high byte + ambient occlusion + material flags.
 *   The id is 16 bits split over two 8-bit channels (each an exact k/255 value, so it round-trips
 *   exactly through the UNORM target and a NEAREST fetch): the lighting pass needs it to look up
 *   the material record (emissive, shading model, flags) without the G-buffer having to carry
 *   textures of its own.
 *
 * Depth is written by this pass (the pipeline's own single-sampled depth image), so the lighting
 * pass reconstructs the world position from it instead of storing one.
 *
 * @note alphaMode BLEND materials never reach this pass: the runtime keeps them in the forward
 *       transparent pass (a G-buffer cannot blend a surface into existence). alphaMode MASK
 *       materials DO render here and gather_surface() discards their cut-out texels.
 * @note skybox / background pixels keep the cleared G-buffer values (all zero) - material id 0 is
 *       the reserved default material, and the lighting pass treats a zero normal as "no geometry".
 */

#include "surface.glsl"

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;

layout(location = 0) out vec4 out_albedo_metallic;  // rgb albedo, a metallic
layout(location = 1) out vec4 out_normal_roughness; // xyz world normal, w roughness
layout(location = 2) out vec4 out_material;         // r/g material id, b ao, a flags

/**
 * @brief write the fragment's surface into the three G-buffer targets
 * @note no lighting, no tonemapping, no output alpha semantics: everything the surface carries that
 *       is not lighting-independent is stored as-is, and the lighting pass decides what to do with it
 */
void main() {
    const surface_sample s = gather_surface(v_world_pos, v_normal, v_uv);

    // the world normal needs no encoding in an RGBA16F target (see the file docs)
    out_albedo_metallic = vec4(s.albedo, s.metallic);
    out_normal_roughness = vec4(s.normal, s.roughness);

    // 16-bit material id split over two UNORM channels: k / 255 is exactly representable, so
    // round(255 * stored) returns the byte that was written even after the format's conversion
    const uint id = push.material_index;
    out_material = vec4(
        float(id & 0xFFu) / 255.0,
        float((id >> 8u) & 0xFFu) / 255.0,
        s.ao,
        float(s.flags & 0xFFu) / 255.0);
}
