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

#include "toon_params.slang"

/**
 * @brief the toon stage's VIEW-ANGLE rim - the second half of both references' rim light
 *
 * BOTH references carry TWO rim terms, and they are different mechanisms rather than two tunings of one:
 * a view-angle term here, and a SCREEN-SPACE DEPTH term that compares this fragment's depth against the
 * depth at a screen offset (the MME pack's `Draw*ScreenRim` passes sampling the viewport map's depth
 * channel; the reference's `GetCharacterDirectRimLightArea` sampling `_CameraDepthTexture`). This is the
 * first one, and it is here first because it needs NOTHING the stage does not already have - a normal, the
 * view vector and the light vector.
 *
 * THE DEPTH TERM CANNOT FOLLOW WITHOUT A NEW RESOURCE, and the reason is a Vulkan rule rather than an
 * omission: this pass holds the depth image as its DEPTH ATTACHMENT (that is what makes `EQUAL` possible),
 * and an image cannot be a depth attachment and a sampled texture in the same instance. The reference has a
 * `CopyDepthPass` producing `_CameraDepthTexture` for exactly this. That copy is the next stage's work.
 *
 * `rim_power` comes from the family's table now rather than from a constant here, because the references
 * carry it PER FAMILY (`EF_*_RIM_AREA` / `EF_*_RIM_STRENGTH` / `EF_*_RIM_COLOR` exist once per family, and
 * hair's rim colour is its `MATERIALEDGECOLOR`). `rim_light_end` stays a constant and is the reference's
 * light-side mask: the rim is drawn on the LIT side only, because a rim on the shadow side reads as a second
 * light source rather than as a contour.
 */
static const float character_rim_light_end = 0.35;

/**
 * @brief the V band a RAMP ATLAS's diffuse ramp lives at
 *
 * The reference samples ONE ramp texture twice, at two V bands - the diffuse ramp at 0.125 and the specular
 * ramp at 0.375 - which is why a single map serves two lookups. The band is the ASSET's layout rather than this
 * stage's choice, so it is named rather than inlined: a ramp authored with its bands elsewhere would need this
 * number, and finding it here is faster than finding it in a shader body.
 */
static const float character_ramp_band_v = 0.125;

/**
 * @brief the neutral ramp's HALF edge width, and the other half of a shared contract
 *
 * The ramp this stage reads when a material HAS one is not the game's and not any family's: it is a NEUTRAL
 * UNIT STEP baked by the host - zero through the shadow side, one through the lit side, a step at x = 0.5 with
 * this half width (see `bake_diffuse_ramp` in main.cpp). ONE redistributable asset therefore serves every
 * family, and where along it this material's terminator sits comes from the family's own `center` / `softness`
 * rather than from the pixels.
 *
 * THE REMAP BELOW INVERTS THE BAKE EXACTLY WHEN THESE TWO NUMBERS MATCH, and that is the whole point rather
 * than a nicety: with them equal, reading the texture at `0.5 + (gated - center) * (w / softness)` produces the
 * same value `smoothstep(center - softness, center + softness, gated)` would have, so the two branches of
 * `toon_diffuse` disagree only by the sampler's 8-bit quantization. Drift between the numbers breaks that
 * quietly - the ramp still looks like a ramp, it is just read at the wrong width - so
 * `tests/test_toon_material_sidecar.cpp` reads this file and main.cpp and fails when they differ.
 */
static const float character_ramp_half_width = 0.035;

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
 * and those atlases are game assets its own license boundary excludes from redistribution. What replaced them
 * is a NEUTRAL UNIT STEP baked by the host (`bake_diffuse_ramp`) and read through the material's own sidecar
 * lane - the same lane, the same texture heap slot and the same read as the game's atlas had - with the
 * family's `center` / `softness` deciding where the step is read (see `character_ramp_half_width`). So the ramp
 * data path is exercised by an asset this repository can carry, and the shading still varies per material
 * family rather than collapsing to one look.
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
float3 toon_diffuse(float3 albedo, float3 n, float3 world_pos, float ao, toon_params params, uint ramp_index) {
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

    // ---- THE LIGHT TINT: the material's OWN RAMP when it has one, the procedural threshold otherwise ----
    //
    // THE RAMP CARRIES THE SHAPE AND THE FAMILY CARRIES THE MAPPING. Both branches build the same number -
    // `smoothstep(center - softness, center + softness, gated)` - and they differ only in WHERE that step comes
    // from: the procedural branch evaluates the Hermite curve here, and the texture branch READS a neutral unit
    // step out of the material's own heap lane at a coordinate that inverts it. That inversion is the reason
    // the baked asset can be shared by every family without the family table having to exist twice - the
    // pixels hold a step at 0.5 and the family's `center` / `softness` are what move it.
    //
    // THE TWO ARE ALTERNATIVES, NOT LAYERS, and that is unchanged from when the ramp was the game's own atlas:
    // a ramp IS a baked quantizer, so running the procedural threshold over a ramp read would quantize an
    // already-quantized value and flatten the step back out. The texture branch replaces the procedural one
    // rather than refining it.
    //
    // A LANE OF 0 MEANS "DO NOT READ" (it is the white fallback - see material_record::toon_indices), so a
    // material with no ramp, or one whose artist switched the ramp off, takes the procedural path. That is what
    // keeps the family table's thresholds meaningful: they are what a character looks like BEFORE its ramps
    // arrive, not a second opinion about them.
    float3 light_tint;
    if (ramp_index != 0u) {
        // THE REMAP, and each factor is doing a different job: `- center` slides the neutral step to this
        // family's threshold, and `* (character_ramp_half_width / softness)` rescales the coordinate so the
        // family's edge width lands on the ramp's own. `saturate` gives the flat shadow and lit sides, which is
        // what a ramp has and a bare `smoothstep` does not.
        const float ramp_x = saturate(0.5 + (gated - params.center) * (character_ramp_half_width / max(params.softness, 1e-4)));
        // `.r` because the baked ramp holds the STEP in all three channels and no colour - the dark side's
        // colour is `params.shadow_tint` below, so one shared asset can serve a family whose shadow is warm and
        // one whose shadow is cool, which a baked-in tint could not.
        const float ramp_step = heap_texel(heap_at<Texture2D<float4>>(heap_slots_textures + ramp_index), heap_sampler_texture, float2(ramp_x, character_ramp_band_v)).r;
        light_tint = lerp(params.shadow_tint, float3(1.0, 1.0, 1.0), ramp_step);
    } else {
        light_tint = lerp(params.shadow_tint, float3(1.0, 1.0, 1.0), smoothstep(params.center - params.softness, params.center + params.softness, gated));
    }

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
    const float3 direct = albedo * sun * light_tint / PI;

    // The ambient is the engine's irradiance map and the engine's composition
    // (`ibl_diffuse * albedo * ao`), so a toon character in a scene is lit by the same sky as everything
    // else in it. WITHOUT IT the dark side is the tint times nothing, which reads as a hole rather than as
    // shadow - and a toon look without ambient is the single most common way this style reads as broken.
    const float3 ambient = irradiance_sample(n) * albedo * ao;

    // ---- the view-angle rim ----
    // The view vector, the engine's own expression for it (`shade_surface` builds the same one).
    const float3 v = normalize(camera_at(heap_camera_slot).camera_pos - world_pos);
    const float no_v = saturate(dot(n, v));
    // Fresnel-shaped: strongest where the surface turns away from the camera, which is the silhouette.
    const float fresnel = pow(1.0 - no_v, params.rim_power);
    // THE LIGHT-SIDE MASK. Without it the rim doubles on the shadow side and reads as a second light source
    // rather than a contour - which is the single most common way a rim light looks wrong. The references
    // both gate it (`EF_*_RIM_AREA` with a NoL term; `GetRimColor`'s `frontRim`/`backRim` split by the
    // sign of `dot(normalVS, lightDirVS)`).
    const float light_side = smoothstep(0.0, character_rim_light_end, dot(n, l));
    // The MME pack lightens the albedo for the rim rather than adding a flat colour
    // (`albedoRimColor = saturate(albedo + 0.3)`), so a dark material's rim is still recognisably ITS
    // colour instead of a white outline pasted on.
    const float3 rim = params.rim_colour * fresnel * light_side * params.rim_strength * saturate(albedo + 0.3);

    // ---- the toon SPECULAR: the reference's second ramp band ----
    // The reference samples ONE ramp atlas twice, at two V bands: the diffuse at v=0.125 and the highlight at
    // v=0.375. The diffuse band is indexed by the shadow-gated half-Lambert; this one by the HALF-VECTOR
    // angle, which is what makes a highlight follow the camera instead of the light's terminator.
    const float3 h = normalize(l + v);
    const float no_h = saturate(dot(n, h));
    const float spec_band = smoothstep(params.spec_center - params.spec_softness, params.spec_center + params.spec_softness, no_h);
    // GATED BY THE SHADOW AND THE LIT SIDE, and both gates are the references': a highlight that survives
    // into the shadow reads as a glow rather than as a reflection, and the MME pack's skin ships
    // `lightSideMask = smoothstep(0, EF_SKIN_SPECULAR_LIGHT_END, noL) * diffuseWeight` for exactly this.
    const float3 specular = sun * spec_band * params.spec_strength * shadow * light_side / PI;

    return direct + ambient + rim + specular * albedo;
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

    // THE PER-FAMILY RAMP, selected by the family the LOADER classified this material's name into (see
    // gltf_loader's toon_family_of) and carried through the material record. An unclassified material gets
    // `toon_params_for(0)`, which is the family-independent path.
    const toon_params params = toon_params_for(s.toon_family);
    const float3 colour = toon_diffuse(s.albedo, normalize(s.normal), v_world_pos, s.ao, params, s.toon_indices.x);

    // the BLEND rule, copied from unlit.slang: an alphaMode BLEND material carries its own coverage, and
    // every other material is fully opaque regardless of what its base colour's alpha says
    const float out_alpha = ((s.flags & 32u) != 0u) ? s.alpha : 1.0;
    return float4(colour, out_alpha);
}