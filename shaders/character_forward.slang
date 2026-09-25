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
 * @brief the toon stage's ramp, PER MATERIAL FAMILY
 *
 * WHY A RAMP AND NOT THE ENGINE'S `toon_band`, which is measured rather than assumed: `toon_band` is a
 * QUANTIZER - it multiplies by the step count, takes `floor`, and steps at the half-way point of each cell,
 * so a value lands on the nearest band and the output stays close to the input on average. A ramp is a
 * DIFFERENT shape: flat through the shadow side, a jump at one threshold, flat through the lit side.
 * Rendering the quantizer and the ramp and diffing them showed the quantizer moving 48624 pixels by a
 * maximum of 7/255 - present, and visually invisible, which is exactly what "the banding does not work"
 * looks like when the term is technically active.
 *
 * WHY PER FAMILY, and where the numbers come from. The references carry this pair PER MATERIAL, and the
 * differences between the families are the visible ones: hair's shadow edge is the hardest and sits highest
 * (a strand's terminator is a line), skin's is the softest and lowest (a soft terminator across a curved
 * cheek), and cloth and the eye sit between. `EF_*_SHADOW_CENTER` / `EF_*_SHADOW_SMOOTHNESS` in the MME
 * pack and `Sigmoid(blend, offset, x)` in MyZmdShaders are the same two parameters, and each family's entry
 * file gives them different values.
 *
 * THE VALUES BELOW ARE THE REFERENCES' RELATIVE ORDER, NOT THEIR FIGURES - they cannot be their figures,
 * because the reference's ramps are TEXTURES whose own contrast is part of the number and those textures are
 * assets this port cannot carry (see the note on the family textures below). What is reproduced here is the
 * STRUCTURE the order expresses: which family gets the hard edge, which gets the soft one, and which side
 * of the albedo each one's tint moves toward. They are named as the place the measured values land.
 *
 * THE FAMILY IDS MUST MATCH `gltf::toon_family` (gltf_loader.cppm) - they are the same enum, written out
 * here because a shader cannot include a C++ enum and the numbers ARE the contract.
 */
#define VR_FAMILY_BASE 1u
#define VR_FAMILY_SKIN 2u
#define VR_FAMILY_FACE 3u
#define VR_FAMILY_HAIR 4u
#define VR_FAMILY_EYE 5u
#define VR_FAMILY_CLOTH 6u

/// @brief one family's toon parameters: its diffuse ramp, its specular ramp, and its rim
struct toon_params {
    // ---- the DIFFUSE ramp ----
    float center;       // the threshold: where the dark side ends
    float softness;     // its edge width
    float3 shadow_tint; // the colour the dark side is tinted toward
    // ---- the SPECULAR ramp: the reference's SECOND ramp band ----
    // The reference samples ONE ramp atlas twice, at two different V bands: `SampleDirectShadowRamp(ramp, x,
    // 0.125)` for the diffuse and `SampleDirectSpecularRamp(ramp, x, 0.375)` for the highlight. Two bands,
    // two mechanisms: the diffuse band is indexed by the shadow-gated half-Lambert, the specular band by the
    // half-vector angle. Kept as a separate pair of numbers here for the same reason.
    float spec_center;   // the half-vector angle at which the highlight appears
    float spec_softness; // its edge width - small is the drawn highlight blob, large is a sheen
    float spec_strength; // 0 disables it, which is what the eye family needs (its highlight is a matcap)
    // ---- the RIM ----
    float rim_power;   // how tightly the rim hugs the silhouette
    float rim_strength;
    float3 rim_colour;
};

/// @brief the parameters for @p family, falling back to the family-independent ones for `none` and anything
///        this shader does not know - so an unclassified material is shaded exactly as it was before
toon_params toon_params_for(uint family) {
    // THE DEFAULT IS THE FAMILY-INDEPENDENT PATH (family 0 == none), and it is also the fallback for a family
    // this shader has no entry for. Keeping it first and returning it unchanged is what makes "an
    // unclassified material did not move" a property of the code rather than a promise - and it is why
    // EVERY field below has a defined default rather than the table only filling the fields it cares about.
    //
    // `spec_strength = 0` IS PART OF THAT CONTRACT, not a placeholder: the specular band is a term this
    // stage did not have before, so a non-zero default would grow a highlight on every material whose name
    // claimed no family - and the claim "the family table cannot change a frame with no character in it"
    // would become false the moment the table gained a term. The families OPT IN to it; `none` does not.
    // The rim fields, by contrast, default to exactly the constants they replaced (power 3.0, strength 0.9,
    // the same colour), so the rim is unchanged for `none` and only varies where a family overrides it.
    toon_params p = {0.42, 0.035, float3(0.58, 0.60, 0.72), 0.80f, 0.06f, 0.0f, 3.0f, 0.9f, float3(1.0, 0.94, 0.90)};
    if (family == VR_FAMILY_HAIR) {
        // the hardest diffuse edge and the highest threshold: a strand's terminator is a drawn line. The
        // widest, strongest highlight - the reference gives hair THREE specular models and a dedicated LUT
        // because the highlight is what identifies hair, and a broad band is the toon reduction of that.
        p.center = 0.52;
        p.softness = 0.015;
        p.shadow_tint = float3(0.42, 0.42, 0.50);
        p.spec_center = 0.72;
        p.spec_softness = 0.22;
        p.spec_strength = 0.55;
        p.rim_power = 2.2;
        p.rim_strength = 1.1;
    } else if (family == VR_FAMILY_SKIN) {
        // the softest diffuse edge and a red-shifted dark side: the MME pack's skin ships `EF_SKIN_SSS_COLOR`
        // as a warm colour, and its ramp's dark side is warm for the same reason. Skin's highlight is small
        // and weak - the reference's is a GGX lobe with `EF_SKIN_SPECULAR_STRENGTH 0.65` gated to the lit
        // side, which as a toon band is a tight, low-contrast spot.
        p.center = 0.36;
        p.softness = 0.07;
        p.shadow_tint = float3(0.80, 0.66, 0.63);
        p.spec_center = 0.86;
        p.spec_softness = 0.10;
        p.spec_strength = 0.22;
        p.rim_power = 3.4;
        p.rim_strength = 0.8;
        p.rim_colour = float3(1.0, 0.86, 0.82);
    } else if (family == VR_FAMILY_FACE) {
        // near skin's: the face is shaded by its SDF in the full reference, and this fallback should not be
        // softer than skin's before that data exists. Almost no specular: the reference's face highlight is a
        // SEPARATE lip-specular mask, not a lobe on the whole surface.
        p.center = 0.40;
        p.softness = 0.06;
        p.shadow_tint = float3(0.82, 0.70, 0.67);
        p.spec_center = 0.88;
        p.spec_softness = 0.08;
        p.spec_strength = 0.12;
        p.rim_power = 3.4;
        p.rim_strength = 0.7;
        p.rim_colour = float3(1.0, 0.88, 0.84);
    } else if (family == VR_FAMILY_CLOTH) {
        // a hard diffuse edge and a cool dark side: cloth's shadow reads as a fold, not a gradient. A BROAD,
        // dim highlight - the reference's cloth carries three lobes (direct, broad, anisotropic) and their
        // sum is what a wide low band stands in for.
        p.center = 0.46;
        p.softness = 0.025;
        p.shadow_tint = float3(0.50, 0.52, 0.64);
        p.spec_center = 0.76;
        p.spec_softness = 0.24;
        p.spec_strength = 0.30;
        p.rim_power = 2.8;
        p.rim_strength = 1.0;
    } else if (family == VR_FAMILY_EYE) {
        // the softest diffuse of all: an eye is a sphere whose highlight moves, and a hard terminator on it
        // reads as a painted dot rather than as a lit surface. NO specular - the reference's eye highlight is
        // a matcap and a separate overlay material, so a lobe here would be a second, wrong highlight.
        p.center = 0.30;
        p.softness = 0.20;
        p.shadow_tint = float3(0.74, 0.74, 0.78);
        p.spec_strength = 0.0;
        p.rim_power = 4.0;
        p.rim_strength = 0.5;
        p.rim_colour = float3(0.92, 0.94, 1.0);
    }
    return p;
}

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
float3 toon_diffuse(float3 albedo, float3 n, float3 world_pos, float ao, toon_params params) {
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
    const float band = smoothstep(params.center - params.softness, params.center + params.softness, gated);

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
    const float3 direct = albedo * sun * lerp(params.shadow_tint, float3(1.0, 1.0, 1.0), band) / PI;

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
    const float3 colour = toon_diffuse(s.albedo, normalize(s.normal), v_world_pos, s.ao, params);

    // the BLEND rule, copied from unlit.slang: an alphaMode BLEND material carries its own coverage, and
    // every other material is fully opaque regardless of what its base colour's alpha says
    const float out_alpha = ((s.flags & 32u) != 0u) ? s.alpha : 1.0;
    return float4(colour, out_alpha);
}
