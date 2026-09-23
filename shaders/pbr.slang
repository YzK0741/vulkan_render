// shaders/pbr.slang - the SLANG source of the FORWARD PBR fragment stage.
//
// PRODUCES `pbr.frag.spv`, the same name the GLSL version produced, so the host side is untouched. Ported
// from shaders/pbr.frag (glsl.old/ once the migration finishes): the body is the GLSL body - gather the
// surface, fill a shade_input, call shade_surface - and what changes is only what a different language
// requires. In THIS stage that is exactly three things:
//
//   - the entry point's varyings and its output carry `[[vk::location(n)]]` / `SV_Target` instead of
//     `layout(location = n)`;
//   - `gl_FragCoord.xy` becomes the `SV_Position` parameter's `.xy`, which is the same value (pixel
//     centres, y down) - and it is why shading.glsl takes the pixel as a PARAMETER rather than reading
//     gl_FragCoord itself: the same shade path runs in compute stages on the deferred side, where that
//     builtin does not exist;
//   - the heap is reached through the shared accessors, which is the shim's whole job.
//
// THIS IS THE STAGE THAT PROVES THE WHOLE SHADING PATH COMPILES UNDER SLANG. gbuffer.slang (the previous
// port) exercises gather_surface and the material table; this one calls shade_surface, i.e. the sun and the
// cascaded shadow maps, the punctual and clustered light loops, the split-sum IBL with env_cube /
// irradiance / BRDF LUT, and the selectable BRDF and cel-shading branches - every fetch helper in
// heap_access.slang, in the order a lit pixel reaches them.
//
// THE INCLUDE ORDER IS PART OF THE CONTRACT (see docs/slang_migration.md section 8): the shim first, then
// surface.glsl, which declares the push-constant block and the heap_frame_slot / heap_image_index aliases
// the slot macros in heap_slot_constants.glsl expand against.

#include "heap_access.slang"
#include "surface.glsl"
#include "shading.glsl"

/// @brief shade one fragment: gather the surface, light it, apply the glTF output alpha
/// @note glTF alpha semantics: only alphaMode BLEND materials carry real coverage in the output alpha.
///       OPAQUE and MASK outputs must write alpha = 1 (their base color alpha is ignored by the spec),
///       otherwise the always-on blending below would make e.g. an albedo texture with an alpha channel
///       unexpectedly translucent. BLEND keeps base_color.a.
[shader("fragment")]
float4 main(
    [[vk::location(0)]] float3 v_world_pos : TEXCOORD0,
    [[vk::location(1)]] float3 v_normal : TEXCOORD1,
    [[vk::location(2)]] float2 v_uv : TEXCOORD2,
    float4 sv_position : SV_Position) : SV_Target
{
    const surface_sample s = gather_surface(v_world_pos, v_normal, v_uv);

    shade_input si;
    si.pixel = int2(sv_position.xy); // gl_FragCoord.xy: the cluster grid's tile coordinate (see shade_input)
    si.world_pos = v_world_pos;
    si.normal = s.normal;
    si.albedo = s.albedo;
    si.emissive = s.emissive;
    si.metallic = s.metallic;
    si.roughness = s.roughness;
    si.ao = s.ao;
    // The forward path adds the diffuse ambient itself; the deferred path leaves it to the resolve.

    const float out_alpha = ((s.flags & 32u) != 0u) ? s.alpha : 1.0;
    return float4(shade_surface(si), out_alpha);
}
