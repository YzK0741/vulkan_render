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

/// @brief the V band the reference's SPECULAR ramp lives at, in the same atlas as the diffuse band above
///
/// The reference samples one atlas twice, and this is the second sample. The two bands are different MECHANISMS
/// rather than the same curve at two offsets - the diffuse band is indexed by the shadow-gated half-Lambert and
/// this one by the half-vector angle - so they get a constant each instead of one constant and an assumption.
static const float character_ramp_spec_band_v = 0.375;

/// @brief the MATCAP's strength - the reference's `EF_EYE_IRIS_MATCAP05_STRENGTH`
///
/// The reference carries it per MATERIAL (`EF_EYE_IRIS_MATCAP05_STRENGTH = 0.5666667`, recorded from the material
/// the pack was extracted from) rather than per family, and the family table here has no matcap strength because
/// until this lane existed nothing read one. It is a named constant rather than an inline literal for the reason
/// the bands above are: a number whose provenance is a specific material should say so where it is written.
static const float character_matcap_strength = 0.5666667;

/**
 * @brief the MATCAP's UV - the sphere map, from the view-space normal
 *
 * THE THIRD VARIABLE, after the shading term (ramps) and the material's own albedo (the shadow LUT): a matcap is
 * indexed by WHERE THE SURFACE POINTS RELATIVE TO THE CAMERA, which is why a matcap image is a picture of a ball
 * and why it is the one toon map that does not depend on the light at all.
 *
 * THE REFERENCE'S OWN REMAP, `normalVS.xy * float2(0.5, -0.5) + 0.5`, INCLUDING THE FLIPPED Y - both
 * `EfEyeIrisApplyMatcap05` and `EfClothSampleMatcap` carry it, and the latter's comment is explicit that it is
 * there because the texture's V axis is opposite the convention the map was authored in.
 *
 * THE VIEW BASIS IS BUILT FROM THE VIEW VECTOR rather than read from a view matrix, and the whole difference is
 * a ROLL about the view axis: `right` and `up_ortho` are any stable pair orthogonal to `v`, so a matcap read
 * this way agrees with the reference's up to a rotation of the ball - which a matcap cannot care about, because
 * a ball has no distinguished up. What it buys is not needing a second per-frame matrix in this stage for one
 * lookup.
 *
 * @param n world-space shading normal
 * @param v the unit surface-to-camera vector, which the caller has already computed
 */
float2 toon_matcap_uv(const float3 n, const float3 v) {
    const float3 up = abs(v.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 right = normalize(cross(up, v));
    const float3 up_ortho = cross(v, right);
    return saturate(float2(dot(n, right), -dot(n, up_ortho)) * 0.5 + 0.5);
}

/// THE HEAD FRAME the face SDF shades against (scene block slot 749). Layout matches `head_ubo` in
/// vulkan/primitive.cppm: three `vec4`s rather than three `vec3`s, so the host's struct and this one agree
/// without either side needing padding or `alignas`. Declared BY THIS STAGE, exactly as `LightUBO` and
/// `CameraUBO` are declared by the stages that read them - the heap accessor is a macro and expands here.
struct HeadUBO {
    float4 front; // the direction the face looks
    float4 right; // its right
    float4 up;    // its up
};

/// @brief the reference's `EF_FACE_SDF_GOO_SHARP` - the Sigmoid's SLOPE, neutral 0.5 (range 0.05..5.0)
static const float character_sdf_sharp = 0.5;
/// @brief the reference's `EF_FACE_SDF_GOO_CENTER` - a constant added to the per-pixel angle threshold
static const float character_sdf_center_offset = 0.1;
/// @brief the reference's `EF_FACE_SDF_GOO_BASE` - the Sigmoid's base, hardcoded in its named twin too
static const float character_sdf_base = 100000.0;

/**
 * @brief the FACE's shadow terminator, from its SDF lightmap - the reference's `EF_FACE_SDF_GOO_ANGLE` path
 *
 * WHY A FACE NEEDS A DIFFERENT MECHANISM FROM EVERY OTHER SURFACE, and it is the whole reason this lane exists:
 * a low-poly anime face is a handful of triangles whose normals are a poor description of its form - the cheeks
 * and the nose are painted into the texture rather than modelled - so shading it by `dot(N, L)` puts the
 * terminator wherever the tessellation happens to put it. The reference therefore shades the face as a 2D SHAPE:
 * a distance field authored per face, thresholded against an angle derived from the light's direction IN THE
 * HEAD'S OWN FRAME. That is why the light is projected onto the head plane below and the surface normal is
 * never used - the only place the reference touches `normalWS` in its whole SDF path is a debug escape hatch.
 *
 * THE HEAD FRAME IS READ, NOT ASSUMED. `head_frame_at(heap_slots_scene_head + heap_frame_slot)` below is written
 * once per frame by the host from the HEAD BONE's world matrix (the `-row1`/`row3` extraction `EfFaceGetHeadBasis`
 * performs, reproduced in gltf_loader's `head_basis_from_axes`), and it falls back to glTF's own basis -
 * `front = (0,0,1)`, `right = (-1,0,0)`, `up = (0,1,0)` - for a model whose skeleton has no usable head bone.
 * `zhuangfy_scalar.glb` is that case: 7 nodes, 0 skins, a static pose split by body part, so the fallback is not
 * an approximation for it - a static model's head does not turn, and a fixed basis and a bone matrix give the
 * same three vectors. A model that DOES have one publishes its own, so a skinned character's terminator follows
 * its head as it turns.
 *
 * THE FRONT AXIS POINTS WHERE THE FACE LOOKS, which is glTF's convention rather than the reference's: it is `+Z`
 * for every model here, measured on the face primitive's mean bind normal - `(+0.10 +0.31 +0.95)` on
 * `lizhiyan_toon.glb`, `(-0.18 +0.41 +0.89)` on `zhuangfy_toon.glb`, `(0 -0.38 +0.93)` on the game's own
 * `actor_zhuangfy.glb`. `EfFaceGetHeadBasis`'s `-row3` pointed the other way for ITS assets, and following it
 * here is not a cosmetic difference: it puts the sun 0.83 of the way round the head for a light the face is
 * looking almost straight into, and 94.1% of the texels the face's own UVs sample then read as shadow - the whole
 * face wears the family's shadow tint and this function shapes nothing. See `gltf::head_basis` for the numbers.
 *
 * THE MIRROR IS THE PART WORTH STATING, because it looks like a trick and is the reference's design: the atlas
 * holds ONE face, and a light from the other side reads it MIRRORED (`1 - uv.x`). Measured on the real asset
 * (`T_actor_common_female_face_02_SDF`): the image correlates -0.89 with its own horizontal flip against +0.62
 * with its vertical flip, i.e. it is strongly antisymmetric about u = 0.5, which is exactly the structure that
 * makes the mirrored read land on a meaningful counterpart.
 *
 * THE SIGMOID IS THE REFERENCE'S, INLINED AS IT INLINES IT: `1 / (1 + base^(-3 * sharp * (x - center)))`, with
 * `x` the R/G average of the mirrored sample, `center` the per-pixel light angle plus a constant, and a `saturate`
 * on the way out. The angle is `|atan2(side, front)| / PI`, so 0 is a light straight ahead and 1 is a light
 * directly behind - and `center` is therefore where the terminator sits for THIS light direction.
 *
 * WHAT IT DOES NOT DO, said plainly: it never reads the surface normal - that is the point of the lane - so a
 * material with no SDF map, or one whose artist switched `_UseSDFLightmap` off, keeps the geometric terminator
 * instead. The caller's `sdf_lane != 0u` test is the whole of that decision, and both cases collapse to it.
 *
 * @param uv the surface's texture coordinate, which is the SDF atlas's coordinate too
 * @param lane the material's SDF heap lane; the caller must not call this with 0
 * @param light_ws the unit SURFACE-TO-LIGHT vector
 * @return the light/dark weight, 0 = shadow side, 1 = lit side
 */
float toon_sdf_mask(const float2 uv, const uint lane, const float3 light_ws) {
    // THE HEAD FRAME, READ FROM THE PER-FRAME BLOCK rather than written as constants here. `HeadUBO` is declared
    // by this stage exactly as `LightUBO` and `CameraUBO` are declared by the stages that read them - a macro is
    // text, so the accessor above expands at this point and finds the declaration below.
    //
    // The values are the SAME as the reference's fallback constants for a model with no usable skeleton (the
    // application resolves it, and this repository's model has none), so moving them out of here changed no
    // pixel - which is the capture gate's job to confirm rather than this comment's to assert.
    const float3 head_front = normalize(head_frame_at(heap_slots_scene_head + heap_frame_slot).front.xyz);
    const float3 head_right = normalize(head_frame_at(heap_slots_scene_head + heap_frame_slot).right.xyz);
    const float3 head_up = normalize(head_frame_at(heap_slots_scene_head + heap_frame_slot).up.xyz);
    // THE LIGHT IS PROJECTED ONTO THE HEAD PLANE. A light straight down the head's up axis has no stable
    // left/right side, and the reference bails out of that case (`return float4(1,1,0,1)`); this returns "lit",
    // because a magenta debug frame is not an option in a shipped stage and 1.0 is the value the ramp treats as
    // unshadowed.
    const float3 projected = light_ws - dot(light_ws, head_up) * head_up;
    const float projected_len_sq = dot(projected, projected);
    if (projected_len_sq < 1e-8) {
        return 1.0;
    }
    const float3 light_dir = projected * rsqrt(projected_len_sq);
    const float side = dot(light_dir, head_right);
    const float mirror = step(0.0, side);
    const float2 sdf_uv = float2(lerp(1.0 - uv.x, uv.x, mirror), uv.y);
    const float4 sdf = heap_texel(heap_at<Texture2D<float4>>(heap_slots_textures + lane), heap_sampler_texture, sdf_uv);
    // R AND G AVERAGED, which is what the reference's production path does. Its debug branches read the
    // channels separately (front light -> G, back light -> R); the goo path this ports averages them, and B and
    // A are never read outside a raw dump.
    const float average = saturate(0.5 * (sdf.r + sdf.g));
    const float angle01 = saturate(abs(atan2(side, dot(light_dir, head_front))) * (1.0 / 3.14159265));
    const float center = angle01 + character_sdf_center_offset;
    const float exponent = -3.0 * character_sdf_sharp * (average - center);
    return saturate(1.0 / max(1.0 + pow(character_sdf_base, exponent), 1e-4));
}

/**
 * @brief the neutral ramp's HALF edge width, and the other half of a shared contract
 *
 * The ramp this stage reads when a material HAS one is not the game's and not any family's: it is a NEUTRAL
 * UNIT STEP baked by the host - zero through the shadow side, one through the lit side, a step at x = 0.5 with
 * this half width (see `bake_unit_ramp` in main.cpp). ONE redistributable asset therefore serves every
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

/// @brief the SHADOW LUT's edge: how many 32x32 tiles across its 1024x32 strip
///
/// The reference's skin and cloth LUTs are `1024x32` strips holding 32 horizontal `32x32` tiles - a `32^3`
/// colour cube flattened - and this is that number. It is shared with the host's bake (`baked_lut_tiles` in
/// main.cpp) because the tiling below and the bake are INVERSE operations: a bake laid out for a different
/// tile count reads back as a different colour with no other symptom, so
/// `tests/test_toon_material_sidecar.cpp` compares the two.
static const float character_shadow_lut_tiles = 32.0;

/**
 * @brief the material's SHADOW COLOUR for its own albedo - the reference's `_ShadowLutTex` lane
 *
 * THIS IS THE ONE LANE THAT IS NOT A RAMP, and the difference is what it is for. A ramp answers "how deep is
 * the shadow here"; this answers "what is THIS MATERIAL's colour when it is in shadow", indexed by the
 * material's own albedo. `EfFaceSampleSkinLut` / `EfClothSampleLut` in the reference do exactly this, and the
 * reason skin ships one is that a skin tone's shadow is a function of the skin tone - a single per-family
 * constant cannot express it, which is why the family table wants a LUT here and a threshold there.
 *
 * THE TILING IS THE REFERENCE'S, deliberately and to the constant: a `32^3` cube in a `1024x32` strip, the
 * tile chosen by the X channel, the in-tile position by Y and Z, and a bilinear hop between the two adjacent
 * tiles because X is a continuous coordinate rather than an index. Its numbers are `1/32` and `31/32` plus the
 * half-texel offsets, and they are written below as the arithmetic they are so the next reader does not have
 * to recover them from decimals.
 *
 * WHERE IT DEVIATES, and it is one thing: the reference indexes its LUT by the DISPLAY-SPACE (sRGB) albedo,
 * because the atlas it reads was authored in the space the artist painted in. This one is indexed by the
 * LINEAR albedo, because the LUT here is OUR bake and this stage works in linear - so the domain is ours to
 * choose, and choosing the working space removes two colour conversions whose only possible contribution
 * would be to disagree with the sampler's own.
 *
 * THE CONTENT OF THE BAKED LUT IS THE IDENTITY, which makes this lane READ WITHOUT CHANGING THE LOOK - the
 * same property the ramps have - and it is not a placeholder: what the lane is for is honouring an AUTHORED
 * cube, and an artist's intent is not something a neutral bake can invent. `albedo` in gives `albedo` out,
 * `params.shadow_tint` still rides on top, and the caller then evaluates the reference's own expression
 * `lerp(lutDark, authoredBright, diffuseWeight)` with `diffuseWeight` being this stage's ramp step.
 *
 * @param albedo the surface albedo (LINEAR, matching `surface_sample::albedo`)
 * @param lane the material's shadow-LUT heap lane; 0 means "do not read" and the caller must not call this
 */
float3 toon_shadow_lut(const float3 albedo, const uint lane) {
    const float3 cube = saturate(albedo);
    // The tile the X channel lands in, and how far between it and the next one - the reference's
    // `lutFloorX` / `lutTileLerp` pair.
    const float tile_f = cube.x * (character_shadow_lut_tiles - 1.0);
    const float tile = floor(tile_f);
    const float tile_lerp = tile_f - tile;
    // The in-tile coordinate: X from the Y channel, Y from the Z channel and FLIPPED, with the half-texel
    // offsets that put a sample at a texel CENTRE rather than at its corner - which is what makes the bake
    // and this read invert each other instead of being off by half a texel everywhere.
    const float in_tile_x = cube.y * ((character_shadow_lut_tiles - 1.0) / (character_shadow_lut_tiles * character_shadow_lut_tiles)) + (0.5 / (character_shadow_lut_tiles * character_shadow_lut_tiles));
    const float in_tile_y = 1.0 - (cube.z * ((character_shadow_lut_tiles - 1.0) / character_shadow_lut_tiles) + (0.5 / character_shadow_lut_tiles));
    const float u = tile / character_shadow_lut_tiles + in_tile_x;
    const float3 low = heap_texel(heap_at<Texture2D<float4>>(heap_slots_textures + lane), heap_sampler_texture, float2(u, in_tile_y)).rgb;
    const float3 high = heap_texel(heap_at<Texture2D<float4>>(heap_slots_textures + lane), heap_sampler_texture, float2(u + 1.0 / character_shadow_lut_tiles, in_tile_y)).rgb;
    return lerp(low, high, tile_lerp);
}

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
 * is a NEUTRAL UNIT STEP baked by the host (`bake_unit_ramp`) and read through the material's own sidecar
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
 * @param toon_indices the material's FOUR sidecar LANES (see `vulkan::toon_slot`): x is the diffuse ramp, y the
 *        shadow LUT, z the specular ramp, w the matcap. A lane of 0 is "the artist switched this off, or the
 *        model does not have it", and every read below is gated on that rather than on a separate flag - which
 *        is what makes the `_Use` flag in the sidecar the only switch there is.
 */
float3 toon_diffuse(float3 albedo, float3 n, float3 world_pos, float ao, toon_params params, uint4 toon_indices, const float2 uv, const uint sdf_lane) {
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

    // ---- THE FACE SDF LANE: the terminator as a SHAPE rather than as geometry (see toon_sdf_mask) ----
    //
    // THE MASK REPLACES THE COORDINATE, NOT THE COLOUR. It answers "which side of the ramp is this fragment on"
    // and the ramp below still decides what the two sides look like - which is the reference's own composition:
    // its SDF mask becomes the U coordinate of an RD ramp lookup and that lookup's alpha becomes the light/dark
    // weight. So this is a SUBSTITUTION, not a second term to blend with the gated half-Lambert, and a material
    // without the lane keeps the coordinate it always had.
    //
    // A LANE OF 0 MEANS "DO NOT READ", as in every other lane, and it is the ONLY test needed: the sidecar's
    // `_UseSDFLightmap` and the model's lack of an SDF both collapse to it at import (see
    // core::heap_slots::sdf_lanes), so this stage cannot tell them apart and does not have to.
    const float shaped = sdf_lane != 0u ? toon_sdf_mask(uv, sdf_lane, l) : gated;

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
    // THE BRANCH PRODUCES A STEP, NOT A COLOUR, and that is what lets a SECOND lane join it below: the shadow
    // LUT supplies the dark side's colour and needs the same weight this produces. `light_tint` is derived from
    // it immediately afterwards, so the no-LUT arithmetic is character-for-character what it was.
    float ramp_step;
    if (toon_indices.x != 0u) {
        // THE REMAP, and each factor is doing a different job: `- center` slides the neutral step to this
        // family's threshold, and `* (character_ramp_half_width / softness)` rescales the coordinate so the
        // family's edge width lands on the ramp's own. `saturate` gives the flat shadow and lit sides, which is
        // what a ramp has and a bare `smoothstep` does not.
        const float ramp_x = saturate(0.5 + (shaped - params.center) * (character_ramp_half_width / max(params.softness, 1e-4)));
        // `.r` because the baked ramp holds the STEP in all three channels and no colour - the dark side's
        // colour is `params.shadow_tint` below, so one shared asset can serve a family whose shadow is warm and
        // one whose shadow is cool, which a baked-in tint could not.
        ramp_step = heap_texel(heap_at<Texture2D<float4>>(heap_slots_textures + toon_indices.x), heap_sampler_texture, float2(ramp_x, character_ramp_band_v)).r;
    } else {
        ramp_step = smoothstep(params.center - params.softness, params.center + params.softness, shaped);
    }
    const float3 light_tint = lerp(params.shadow_tint, float3(1.0, 1.0, 1.0), ramp_step);

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
    // ---- THE SHADOW LUT LANE, which replaces the dark side's COLOUR and not its weight ----
    //
    // THE REFERENCE'S OWN EXPRESSION, `lerp(lutDark, authoredBright, diffuseWeight)`: the shadowed albedo is
    // looked up from the lit one, and the ramp step is the weight between them. The no-LUT path is the same
    // expression with the lookup replaced by the family's constant tint - `lerp(albedo * tint, albedo, step)`
    // is `albedo * lerp(tint, white, step)` - so the two branches differ in WHERE the dark side's colour comes
    // from and in nothing else. That is why this is a branch rather than a second term: a LUT blended ON TOP of
    // the tint would be a third opinion about a colour that already has two sources.
    float3 direct;
    if (toon_indices.y != 0u) {
        // `* params.shadow_tint` because our bake holds the IDENTITY cube rather than the game's palette, so
        // the family's tint is still what says how dark and how warm the shadow side is; an authored cube would
        // carry that itself and this multiply would become the identity it already is for the baked one.
        const float3 lut_dark = toon_shadow_lut(albedo, toon_indices.y) * params.shadow_tint;
        direct = sun * lerp(lut_dark, albedo, ramp_step) / PI;
    } else {
        direct = albedo * sun * light_tint / PI;
    }

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
    // ---- THE SECOND RAMP LANE, read exactly like the first ----
    //
    // THE SAME NEUTRAL STEP AND THE SAME REMAP, which is the point of the bake being neutral: `_DiffRampMap` and
    // `_SpecRampMap` are two LANES of one sidecar, they are read with the same inversion, and the only thing
    // that differs between them is which pair of family numbers moves the step. A lane of 0 means this material
    // has no specular ramp or its artist switched `_UseSpecRampMap` off, and it takes the procedural threshold
    // - the same contract as the diffuse lane, checked the same way.
    //
    // THE TWO BRANCHES ARE THE SAME FUNCTION HERE TOO, and that is deliberate rather than a wasted read: the
    // bake is `smoothstep(0.5 - w, 0.5 + w, x)` and the remap inverts it, so a material whose lane is live gets
    // the number `smoothstep(spec_center - spec_softness, spec_center + spec_softness, no_h)` back - through a
    // texture fetch and its 8-bit quantization instead of in registers. What the lane buys is not a different
    // look at these parameters; it is that the ARTIST'S asset is what decides the look, and a hand-authored
    // ramp (a highlight with a falloff, a shaped spike) would be honoured without this stage changing at all.
    float spec_band;
    if (toon_indices.z != 0u) {
        const float spec_x = saturate(0.5 + (no_h - params.spec_center) * (character_ramp_half_width / max(params.spec_softness, 1e-4)));
        spec_band = heap_texel(heap_at<Texture2D<float4>>(heap_slots_textures + toon_indices.z), heap_sampler_texture, float2(spec_x, character_ramp_spec_band_v)).r;
    } else {
        spec_band = smoothstep(params.spec_center - params.spec_softness, params.spec_center + params.spec_softness, no_h);
    }
    // GATED BY THE SHADOW AND THE LIT SIDE, and both gates are the references': a highlight that survives
    // into the shadow reads as a glow rather than as a reflection, and the MME pack's skin ships
    // `lightSideMask = smoothstep(0, EF_SKIN_SPECULAR_LIGHT_END, noL) * diffuseWeight` for exactly this.
    const float3 specular = sun * spec_band * params.spec_strength * shadow * light_side / PI;

    const float3 toon_colour = direct + ambient + rim + specular * albedo;

    // ---- THE MATCAP LANE: applied to the FINISHED colour, which is where the reference applies it ----
    //
    // `EfEyeIrisApplyMatcap05` returns `max(color, 0) * (1 + max(matcap, 0) * strength)`: a matcap MULTIPLIES the
    // shaded result rather than being another light, because it stands for the environment the surface is sitting
    // in rather than for a second sun. The `max(..., 0)` is the reference's and is kept - a matcap's negative
    // texels are an artifact of its encoding, not a request to darken the surface.
    //
    // A LANE OF 0 IS "DO NOT READ", as in every other lane, and the BAKED CONTENT IS BLACK (see main.cpp's
    // bake_matcap for why black is this lane's neutral value rather than a placeholder). Black makes the factor
    // exactly `1.0`, so this multiply is bit-exact for every material whose artist did not supply a matcap.
    if (toon_indices.w != 0u) {
        const float3 matcap = heap_texel(heap_at<Texture2D<float4>>(heap_slots_textures + toon_indices.w), heap_sampler_texture, toon_matcap_uv(n, v)).rgb;
        return toon_colour * (1.0 + max(matcap, 0.0) * character_matcap_strength);
    }
    return toon_colour;
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
    const float3 colour = toon_diffuse(s.albedo, normalize(s.normal), v_world_pos, s.ao, params, s.toon_indices, v_uv, sdf_lane_at(heap_slots_sdf_lanes, push.material_index));

    // the BLEND rule, copied from unlit.slang: an alphaMode BLEND material carries its own coverage, and
    // every other material is fully opaque regardless of what its base colour's alpha says
    const float out_alpha = ((s.flags & 32u) != 0u) ? s.alpha : 1.0;
    return float4(colour, out_alpha);
}