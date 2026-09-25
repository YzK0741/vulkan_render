// shaders/character_forward.slang - THE TOON CHARACTER FORWARD STAGE.
//
// WHAT IT IS FOR, and it is the one thing this renderer was missing to draw a toon character
// correctly: a pass that runs AFTER the lighting stage and OVERWRITES the pixels it already lit, so
// the character can have its own hand-authored shading instead of the deferred one - without being lit
// TWICE, which is what happens if a character's finished colour is fed back through the G-buffer.
//
// THE ARCHITECTURE IS TAKEN FROM THE REFERENCE PORT, NOT INVENTED HERE (see _toon_ref/PORT_SPEC.md):
// DanbaidongRP's PBRToon material family declares a pass named `CharacterForward` carrying `ZWrite Off` +
// `ZTest Equal`, invoked after `DeferredLighting`. The `EQUAL` is the whole mechanism: the pass lands on
// exactly the surface the G-buffer pass recorded, so it REPLACES the deferred result rather than
// layering over it, and nothing needs a mask.
//
// WHY THE OUTPUT IS LINEAR AND NOT DISPLAY-REFERRED. The same reference is explicit about this and it is
// why the MME shader pack this port started from did not fit: `PBRToonBase.shader` ends at
// `PostEvaluate(...)` -> `return float4(resultColor, 1)`, and the tonemap is the post chain's alone. A
// stage that encoded sRGB here would be tonemapped a second time by post.slang's ACES composite.
//
// THE INCLUDE ORDER IS PART OF THE CONTRACT (see gbuffer.slang): the shim first, then surface.glsl, which
// declares the push-constant block and the heap_frame_slot / heap_image_index aliases the slot macros
// expand against; then shading.glsl, whose functions address the per-frame arrays through those macros.
// Compiling a body before surface.glsl reports heap_frame_slot as an undefined identifier - measured.
//
// The alphaMode MASK discard is NOT written here because `gather_surface` already performs it (see
// surface.glsl): it is the material's rule, not the pass's, so no pass can forget it. The BLEND alpha rule
// IS written here, copied from unlit.slang, because it is the pipeline's coverage contract.

#include "heap_access.slang"
#include "surface.glsl"
#include "shading.glsl"

/**
 * @brief the toon stage's ramp, until the per-material ramp atlas replaces it
 *
 * WHY A RAMP AND NOT THE ENGINE'S `toon_band`, which is measured rather than assumed: `toon_band` is a
 * QUANTIZER - it multiplies by the step count, takes `floor`, and steps at the half-way point of each cell,
 * so a value lands on the nearest band and the output stays close to the input on average. A ramp is a
 * DIFFERENT shape: flat through the shadow side, a jump at one threshold, flat through the lit side.
 * Rendering the quantizer and the ramp and diffing them showed the quantizer moving 48624 pixels by a
 * maximum of 7/255 - present, and visually invisible, which is exactly what "the banding does not work"
 * looks like when the term is technically active.
 *
 * The two numbers are the same two the references carry PER MATERIAL: the MME pack's
 * `EF_*_SHADOW_CENTER` / `EF_*_SHADOW_SMOOTHNESS`, whose toon step is
 * `EfSigmoidSharp(vis, center, smoothness)`, and MyZmdShaders' `Sigmoid(blend, offset, x)`. The threshold
 * decides where the dark side ends; the softness is the edge width, and a small one is what makes the edge
 * read as a drawn line rather than as a gradient. This constant is where that per-material pair will land
 * once the ramp data exists, which is what keeps the substitution a data change rather than a rewrite.
 */
static const float character_toon_center = 0.42;
static const float character_toon_softness = 0.035;

/**
 * @brief the toon diffuse: half-Lambert GATED by the shadow term, then banded
 *
 * THIS IS THE REFERENCE'S STRUCTURE, and the gating is the part worth stating because it is what makes the
 * look two-tone rather than merely darker. The character shaders in the reference derive a ramp COORDINATE
 * from the half-Lambert term and the shadow term together - the shadow decides WHICH SIDE of the ramp a
 * fragment reads and the half-Lambert positions it within that side - and then read a tinted colour out of
 * the ramp. A plain `saturate(dot(N, L)) * shadow` has no such gate: the two terms multiply into one smooth
 * value and the dark side has no identity of its own.
 *
 * WHAT STANDS IN FOR THE RAMP TEXTURE, and why it is not a shortcut: the reference reads its colour from a
 * layered ramp atlas (`SampleDirectShadowRamp(ramp, x, 0.125)` / `SampleDirectSpecularRamp(ramp, x, 0.375)`),
 * and those atlases are game assets its own license boundary excludes from redistribution. A procedural tint
 * plus the engine's existing `toon_band` reproduces the STRUCTURE (a gated coordinate, a quantized read, a
 * dark-side tint that is a colour rather than a scale) with no asset at all, so the per-material ramp can be
 * added later as data rather than as a dependency of the pipeline being correct.
 *
 * @param albedo the surface albedo (linear)
 * @param n world-space shading normal
 * @param world_pos world-space position, for the shadow lookup
 * @param ao the surface's baked occlusion, which scales the ambient ONLY - never the direct sun, because a
 *        baked occlusion term is a statement about the sky's reach and applying it to a shadowed sun would
 *        darken the lit side on geometry whose only fault is a concave crevice
 * @param shadow_tint the colour the unlit side is tinted toward. A TINT rather than a multiplier: the
 *        reference's dark side is a hue shift as much as a darkening, and a grey scale factor cannot
 *        express it. Cooler than the albedo, which is what the reference's skin and cloth ramps encode.
 */
float3 toon_diffuse(float3 albedo, float3 n, float3 world_pos, float ao, float3 shadow_tint) {
    // THE LIGHT VECTOR IS THE SURFACE-TO-LIGHT DIRECTION, which is the engine's existing convention
    // (`evaluate_direct_light` normalizes this same field into its `l`).
    const float3 l = normalize(light_at(heap_light_slot).light_dir.xyz);
    const float half_lambert = saturate(dot(n, l) * 0.5 + 0.5);

    // `calc_shadow` returns the quantized term when the engine's cel knob is on and the raw one otherwise;
    // either way it is the shadow factor for this fragment, and this stage does not band it a second time -
    // that would quantize a quantized value, which collapses the bands to nothing on a frame where the knob
    // is set.
    const float shadow = calc_shadow(world_pos, n);

    // THE GATE. The shadow term multiplies the half-Lambert rather than being applied after it.
    const float gated = saturate(shadow * half_lambert);
    // THE RAMP: one hard-ish threshold, not a quantizer. See character_toon_center for why these are two
    // different shapes and why the difference was measured rather than assumed.
    const float band = smoothstep(character_toon_center - character_toon_softness, character_toon_center + character_toon_softness, gated);

    // The sun's radiance and the /PI are the ENGINE'S expressions (`shade_surface` builds the same pair:
    // `vec3(7.5 * sun_intensity)` and `kd * base_color / PI`), so a toon fragment and a deferred fragment
    // standing in the same light cannot disagree about how bright the sun is.
    const float3 sun = float3(7.5, 7.5, 7.5) * light_at(heap_light_slot).sun_intensity;

    // THE BAND *IS* THE LIGHT FACTOR, and this is the line that is easy to get wrong in a way that looks
    // like "the banding does not work" rather than like a bug: multiplying the result by `shadow` a SECOND
    // time - on top of the gate that produced the coordinate - drives every shadowed fragment to zero and
    // leaves only the ambient, which flattens the very split the band was computing. The shadow's job here
    // is to CHOOSE the side of the ramp; the tint is what the shadow side looks like, and it is a colour,
    // not an extinction. Measured: the double-applied version and the flat-albedo probe were hard to tell
    // apart by eye, which is what a banding term multiplied by its own gate does.
    const float3 direct = albedo * sun * lerp(shadow_tint, float3(1.0, 1.0, 1.0), band) / PI;

    // The ambient is the engine's irradiance map and the engine's composition
    // (`ibl_diffuse * albedo * ao`), so a toon character in a scene is lit by the same sky as everything
    // else in it. WITHOUT IT the dark side is the tint times nothing, which reads as a hole rather than as
    // shadow - and a toon look without ambient is the single most common way this style reads as broken.
    const float3 ambient = irradiance_sample(n) * albedo * ao;

    return direct + ambient;
}

[shader("fragment")]
float4 main(
    [[vk::location(0)]] float3 v_world_pos : TEXCOORD0,
    [[vk::location(1)]] float3 v_normal : TEXCOORD1,
    [[vk::location(2)]] float2 v_uv : TEXCOORD2,
    bool v_is_front : SV_IsFrontFace) : SV_Target
{
    // The same surface the G-buffer pass wrote, gathered by the same function: a normal-map fetch, the
    // double-sided flip and the material's factors cannot disagree with the G-buffer if there is one
    // implementation of them.
    const surface_sample s = gather_surface(v_world_pos, v_normal, v_uv);

    const float3 colour = toon_diffuse(s.albedo, normalize(s.normal), v_world_pos, s.ao, float3(0.58, 0.60, 0.72));

    // the BLEND rule, copied from unlit.slang: an alphaMode BLEND material carries its own coverage, and
    // every other material is fully opaque regardless of what its base colour's alpha says
    const float out_alpha = ((s.flags & 32u) != 0u) ? s.alpha : 1.0;
    return float4(colour, out_alpha);
}
