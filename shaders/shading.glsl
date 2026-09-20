/**
 * @file shaders/shading.glsl
 * @brief Shared lighting: the scene light/IBL/shadow bindings and the BRDF stack both PBR passes use.
 * @ingroup shaders
 *
 * Included by pbr.frag (the forward path) and deferred.frag (the deferred path's lighting stage).
 * The deferred path gets its surface from the G-buffer instead of from a vertex/fragment interface,
 * but from that point on the two paths must agree EXACTLY: same sun, same shadow test, same punctual
 * lights, same split-sum IBL, same selectable BRDF/diffuse presets, same cel-shading bands. Keeping
 * that in one place is not just tidy - it is what makes the forward path a usable A/B reference for
 * the deferred one, and it removes the whole class of "the deferred image looks slightly different"
 * bugs.
 *
 * What lives here: the scene-set bindings a shading stage needs (camera UBO, the three IBL maps, the
 * light UBO, the shadow map), the light/IBL/shadow/BRDF functions, and shade_surface() - the whole
 * per-fragment lighting evaluation.
 *
 * What stays out: the surface itself (shaders/surface.glsl: material table, textures, alpha test,
 * normal mapping) and the sky (shaders/sky.glsl) - the deferred lighting stage calls the sky function
 * for pixels the G-buffer left empty.
 *
 * A shader including this file must NOT declare the bindings it declares (0, 2, 3, 4, 7, 8) and must
 * use the runtime's shared scene pipeline layout for set 0.
 */

#ifndef VULKAN_RENDER_SHADING_GLSL
#define VULKAN_RENDER_SHADING_GLSL

// Camera UBO (scene set binding 0): view/projection and the world-space eye position (the shading
// path needs the eye to build the view vector and, for the sky, the view ray). The last two matrices
// feed the motion vectors: `proj` is the CURRENT projection INCLUDING the TAA jitter (geometry has to
// be sampled at the jittered offsets), while `view_proj_unjittered` / `prev_view_proj` are the
// jitter-free pair - a jitter that leaked into a motion vector would be read as camera motion and
// would reproject the history to the wrong place every frame.
// THE SLOT GRID, which is what every declaration below addresses (see docs/descriptor_heap_migration.md): the
// constants, the two specialization constants that carry the frame and image indices, and the shared heap arrays.
#include "heap_slots.glsl"

// HEAP-NATIVE (see docs/descriptor_heap_migration.md): the array IS the heap and the slot carries the FRAME, so
// every read is `camera[heap_camera_slot].field` (the slot constants are declared with the rest of the grid
// below). The block itself is unchanged - it is a CPU/GPU contract.
layout(descriptor_heap, descriptor_stride = heap_slot_stride) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    mat4 view_proj_unjittered;
    mat4 prev_view_proj;
} camera[];

// Split-sum IBL: the prefiltered GGX environment (roughness mip chain), the
// irradiance map for the diffuse ambient, and the BRDF integration LUT.
// IMAGES AND SAMPLERS ARE SEPARATE IN A HEAP, so these are textures now and the sampler is the shared one (the
// host picks which at the fetch sites below).
layout(descriptor_heap, descriptor_stride = heap_slot_stride) uniform textureCube env_texture[];
layout(descriptor_heap, descriptor_stride = heap_slot_stride) uniform textureCube irradiance_texture[];
layout(descriptor_heap, descriptor_stride = heap_slot_stride) uniform texture2D brdf_lut_texture[];

// The SLOTS these declarations resolve to live in heap_slots.glsl with every other stage's, because a stage that
// does not include this file needs them too (see that file's note on which index each kind of array takes).
// ... and the specular half of the split sum, which lives in a file of its own so that the two sides of a
// subtraction compute it from the same expressions (see that file's header).
#include "ibl_specular.glsl"

// Light UBO (scene set binding 7): the orthographic light view-proj (world -> shadow map) and the
// light direction, followed by the active punctual lights. The direction is filled by the CPU
// (make_directional_light_ubo) and matches the sky sun, so the direct light, the visible sun disc
// and the shadows all agree. Layout must match vulkan::light_ubo in primitive.cppm (std140):
// mat4 | vec4 | 4 floats | uint + 3 pad floats | PunctualLight[4] - the CPU mirrors the "uint + pad"
// slot with one glm::vec4, so the array starts at byte 112 and the block is 368 bytes. (A vec3 pad
// would force 16-byte alignment to 128 and shift every light by 16.)
const int MAX_PUNCTUAL_LIGHTS = 128; // vulkan::max_punctual_lights
const int CLUSTER_TILE_SIZE = 64;    // vulkan::cluster_tile_size (pixels per cluster tile)
const int CLUSTER_LIGHT_CAPACITY = 32; // vulkan::cluster_light_capacity (lights stored per cluster)
const int MAX_SHADOW_CASCADES = 4; // vulkan::max_shadow_cascades

struct PunctualLight {
    vec4 position; // xyz: world position (w unused)
    vec4 color;    // xyz: linear color * intensity (w unused)
    vec4 spot_dir; // xyz: spot axis, normalized for spot lights (w unused)
    vec4 params;   // x = range (0 = infinite), y = 0 point / 1 spot, z = cos(outer cone), w = cos(inner cone, spot only)
};

layout(descriptor_heap, descriptor_stride = heap_slot_stride) uniform LightUBO {
    // One orthographic world -> light-clip matrix per cascade: entry 0 covers the near range, the
    // rest the ranges given by cascade_splits. With cascade_count == 1 only entry 0 is fitted and
    // used, which is exactly the single-shadow-map behavior.
    mat4 light_view_proj[MAX_SHADOW_CASCADES];
    vec4 light_dir;           // xyz: normalized light direction, w: 1 / shadow map size (uv texel)
    vec4 cascade_splits;      // view-space FAR distance of each cascade
    vec4 cascade_texel_world; // world size of one shadow-map texel, per cascade (normal-offset bias)
    float shadow_enabled; // 1.0 = sample the shadow map, 0.0 = fully lit (runtime::set_shadow_enabled)
    // selectable BRDF theory models (gui combos -> runtime::set_brdf_model / set_diffuse_model,
    // CPU-side, riding the std140 padding of this block):
    //   brdf_model:    0 = GGX + joint Smith (default), 1 = GGX + height-correlated Smith,
    //                  2 = Beckmann + Smith, 3 = Blinn-Phong + Smith
    //   diffuse_model: 0 = Lambert (default), 1 = Oren-Nayar
    float brdf_model;
    float diffuse_model;
    float cascade_blend; // fraction of a cascade's range blended into the next one (0.1 = last 10%)
    float cascade_count; // active cascades (1 = the single-map path)
    float rt_shadows; // 1.0 = the sun's shadow is the ray-traced visibility image (binding 14),
                      // 0.0 = the cascaded shadow maps. Rides the std140 padding that keeps
                      // light_count on its 16-byte boundary; see the CPU's light_ubo.
    float sun_intensity;
    float furnace_level;
    uint light_count;
    float exposure; // y lane of the CPU's light_count vec4: linear exposure scale (pre-tonemap)
    float toon_steps;   // cel-shading quantization steps (LightUBO.light_count.z; 0 = PBR)
    float toon_softness; // band edge width in normalized [0,1] space (LightUBO.light_count.w)
    PunctualLight punctual_lights[MAX_PUNCTUAL_LIGHTS];
    // Clustered light culling (M5), appended after the light array so its offset is unchanged:
    //   cluster_grid  x = active tile columns, y = active tile rows, z = depth slices,
    //                 w = 1.0 = read the per-cluster light lists, 0.0 = loop every active light
    //   cluster_depth x = near view depth, y = far view depth the slices span (z/w = screen size,
    //                 used by the cluster pass only)
    vec4 cluster_grid;
    vec4 cluster_depth;
    // ---- ZZZ-style NPR (the Diffuse Warp and the matcap rim of XIYAG's ZZZ shader, see
    //      shaders/shading.glsl's diffuse_warp below and docs/zzz_shading.md). APPENDED after the
    //      cluster lanes, so every offset above keeps its value and a stage that declares the block
    //      without these two still matches the buffer.
    //   npr_shadow xyz = the colour the shadowed END of the toon ramp lerps towards, as a MULTIPLIER
    //                    of the material's base colour: the reference's five ShadowColor values are
    //                    this one tint walked across the five bands, and (1,1,1) means the warp is
    //                    OFF - which is the compiled default, so a stock frame is unchanged.
    //                    w unused.
    //   npr_rim    x = rim strength (0 = off), y = rim exponent (its falloff in view space),
    //              z = the reference's `MData.z` specular mask (see reference_specular_colors),
    //              w = the per-texel band gain: how much of the surface's own albedo luminance is added to
    //                  the frame's toon_shadow_band (0 = the frame-wide constant, the compiled default).
    vec4 npr_shadow;
    vec4 npr_rim;
    vec4 npr_face;
    // npr_face_forward xyz = the FACE BLOCK's own plane in world space, i.e. the direction the face is
    // facing (the converter averages its vertex normals). The face's shading normal is blended towards it
    // so that a face that is nearly flat in the art STOPS being shaded by its nose, lips and cheeks - and
    // so that the blend does not follow the camera (which made the face right at one angle only).
    vec4 npr_face_forward;
    // ambient_gain_tint xyz = a tint on the environment light, w = its gain ([render] ambient_tint /
    // ambient_gain). Declared LAST to match the CPU struct member for member: this block is read by OFFSET.
    vec4 ambient_gain_tint;
} light[];

// Per-cluster light lists (scene set bindings 11/12), written by shaders/light_cluster.comp: one
// entry per cluster in cluster_counts (how many lights landed in it) and a fixed-capacity row per
// cluster in cluster_indices holding the indices into light[heap_light_slot].punctual_lights. Storage buffers rather
// than more UBO lanes because the grid is thousands of entries - and small enough (16 lights per
// cluster) that no per-cluster linked list / prefix sum is needed.
layout(descriptor_heap, descriptor_stride = heap_slot_stride) readonly buffer ClusterCounts {
    uint counts[];
} cluster_counts[];
layout(descriptor_heap, descriptor_stride = heap_slot_stride) readonly buffer ClusterIndices {
    uint indices[];
} cluster_indices[];

// Shadow map (scene set binding 8): a 2D ARRAY of cascades, sampled with a depth-compare sampler
// (sampler2DArrayShadow) whose LINEAR filtering performs HARDWARE percentage-closer filtering - the
// hardware compares the reference depth against the 2x2 texel neighborhood of the addressed layer
// and returns the lit fraction (no manual 3x3 loop needed). An ARRAY texture rather than an array of
// samplers because the layer is chosen per FRAGMENT: dynamic indexing of a sampler array would need
// dynamically uniform indices, while a texture-array layer is just a coordinate.
// ... and it is a heap TEXTURE (the depth-compare sampler comes from the sampler heap at heap_sampler_shadow),
// per swapchain image, which is what heap_image_index selects.
layout(descriptor_heap, descriptor_stride = heap_slot_stride) uniform texture2DArray shadow_texture[];

const float PI = 3.14159265359;

/**
 * @brief quantize x into @p steps bands with a soft edge of +- @p softness
 * @param x value in [0,1] (a diffuse falloff factor, a shadow factor, ...)
 * @param steps band count; below 1.5 the value is returned unchanged (plain PBR), so the same code
 *              path serves both styles
 * @param softness band edge width in normalized space; smaller = harder cel edges
 * @return the quantized value in [0,1]
 */
float toon_band(float x, float steps, float softness) {
    if (steps < 1.5) {
        return x;
    }
    float scaled = clamp(x, 0.0, 1.0) * steps;
    float base = floor(scaled);
    float frac = scaled - base;
    float edge = smoothstep(0.5 - softness, 0.5 + softness, frac);
    return (base + edge) / steps;
}

/**
 * @brief the shadow-end MULTIPLIER of a material's ramp, DERIVED from its own albedo
 * @param base_color the material's albedo (factor x texture)
 * @return a per-channel multiplier: the colour the shadowed side is multiplied by
 *
 * The reference authors these as MULTIPLIERS - its `ShadowColor1..5` default to pure white, and white means
 * "no shadow at all" - which is why this is a ratio to the albedo rather than an absolute colour.
 *
 * The rule is the one the reference's own numbers describe, measured off an in-game capture (sRGB, same
 * material, lit vs shadowed side): its mint hair goes from luma 241 / saturation 0.08 to luma 180 /
 * saturation 0.25, i.e. 0.75x the value and THREE times the saturation, while its skin - already twice as
 * saturated - gains only 20%. That is a VIBRANCE curve rather than a scale, which is why it is written as
 * one: the less saturated the colour, the more of the boost it gets. A plain multiply darkens without
 * re-saturating, and that is exactly the "grey plastic" a cel ramp reads as on a model like this one.
 */
vec3 derived_shadow_multiplier(vec3 base_color) {
    const float luma = dot(base_color, vec3(0.2126, 0.7152, 0.0722));
    const float peak = max(max(base_color.r, base_color.g), base_color.b);
    const float trough = min(min(base_color.r, base_color.g), base_color.b);
    const float saturation = peak > 1e-4 ? (peak - trough) / peak : 0.0;
    // vibrance 2.0: a fully grey colour is pushed to 3x its (zero) saturation, and a saturated one barely
    // moves - the shape the two reference measurements above have. It was 3.0 until the ported pipeline was
    // measured end to end: at 3.0 the deepest band put this asset's hair shadow at saturation 0.44 against
    // the reference's 0.34, with every other number already matching, so the boost came down rather than the
    // band or the value scale moving.
    const float vibrance = 2.0;
    const float value_scale = 0.75;
    const vec3 saturated = mix(vec3(luma), base_color, 1.0 + vibrance * (1.0 - saturation));
    const vec3 shadowed = clamp(saturated * value_scale, vec3(0.0), vec3(1.0));
    // ... expressed as a MULTIPLIER of the albedo, which is the space the reference's cascade works in
    return clamp(shadowed / max(base_color, vec3(0.02)), vec3(0.0), vec3(2.0));
}

/**
 * @brief the reference's `Shadow Colors` group: a five-colour cascade, gamma 2.2 at the end
 * @param base_color the material's albedo (factor x texture)
 * @param factor the band factor the cascade is walked with (the reference's `MData.x`, its light-map
 *        channel; a PMX carries no light map, so this is the frame's `toon_shadow_band`)
 * @param tint an extra multiplier on the deepest colour; (1,1,1) = the derivation alone
 * @return the shadow-end multiplier of the ramp
 *
 * PORTED FROM THE REFERENCE, structure for structure: five colours blended by `factor * (0.2, 0.4, 0.6,
 * 0.8)` in that order, then `pow(c, 2.2)`. The reference's five colours are AUTHORED per material
 * (`ShadowColor1..5`, default white = no shadow); a PMX carries none, so they are derived here as a ramp
 * from the derived shadow multiplier to WHITE - the lit end of the reference's own ramp.
 *
 * The cascade is a SMOOTH blend in the reference (its Math nodes MULTIPLY the factor), which is why its
 * shadows have no hard borders - the hard stepping is on the SPECULAR side only.
 */
vec3 reference_shadow_colors(vec3 base_color, float factor, vec3 tint) {
    const vec3 sc1 = derived_shadow_multiplier(base_color) * tint;
    const vec3 sc5 = vec3(1.0);
    const vec3 sc2 = mix(sc1, sc5, 0.25);
    const vec3 sc3 = mix(sc1, sc5, 0.50);
    const vec3 sc4 = mix(sc1, sc5, 0.75);
    vec3 c = mix(sc1, sc2, factor * 0.2);
    c = mix(c, sc3, factor * 0.4);
    c = mix(c, sc4, factor * 0.6);
    c = mix(c, sc5, factor * 0.8);
    // the reference's final Gamma node, 2.2 - its authored colours are sRGB values being used as multipliers
    return pow(max(c, vec3(0.0)), vec3(2.2));
}

/**
 * @brief the reference's `Light Factor` and its diffuse composition, ported whole
 * @param base_color the material's albedo (factor x texture)
 * @param ndotl the UNQUANTIZED falloff
 * @param band the band factor (see reference_shadow_colors)
 * @param shadow_tint an extra multiplier on the deepest colour
 * @return the multiplier the albedo is shaded by - NOT a replacement colour
 *
 * Two things in the reference are easy to get wrong from its name alone, and both are load-bearing:
 *   - its shadow colours are MULTIPLIERS (white = no shadow), so the shaded albedo is
 *     `albedo * mix(SC, white, light)` rather than a lerp between two absolute colours;
 *   - its `Light Factor` compresses NdotL into a NARROW band - `smoothstep(0, 0.25, NdotL)`, i.e. every
 *     surface past a quarter-lit is fully lit - which is what gives the terminator its cel edge. It is also
 *     the whole falloff: there is no separate NdotL multiply on top of it.
 * The reference multiplies this by `smoothstep(0, 0.5, 1 - vertex_colour)`; a PMX has no vertex colours, so
 * that term is 1 here and the port notes it rather than inventing a substitute.
 */
vec3 reference_shadow_multiplier(vec3 base_color, float ndotl, float band, vec3 shadow_tint) {
    const float light = smoothstep(0.0, 0.25, ndotl);
    return mix(reference_shadow_colors(base_color, band, shadow_tint), vec3(1.0), light);
}

/**
 * @brief the reference's `Specular Colors` group: the SAME five-colour chain, driven by HARD thresholds
 * @param base_color the material's albedo (factor x texture)
 * @param factor the band factor (the reference's `MData.x`, see reference_shadow_colors)
 * @return the colour the highlight term adds
 *
 * PORTED: the reference's Math nodes here are `LESS_THAN 0.8 / 0.6 / 0.4 / 0.2` on the band factor rather
 * than the shadow side's MULTIPLY - so the highlight STEPS between five colours instead of blending. That
 * asymmetry is the reference's, not a simplification: its shadows are smooth and its highlights are hard,
 * which is the opposite of what "cel shading" suggests and the reason its highlights read as shapes.
 * The reference's five specular colours are authored and default to white; the ramp here runs from a dimmer
 * to a brighter tint of the material's own hue.
 */
vec3 reference_specular_colors(vec3 base_color, float factor) {
    const vec3 sp1 = mix(base_color, vec3(1.0), 0.35);
    const vec3 sp5 = mix(base_color, vec3(1.0), 0.90);
    const vec3 sp2 = mix(sp1, sp5, 0.25);
    const vec3 sp3 = mix(sp1, sp5, 0.50);
    const vec3 sp4 = mix(sp1, sp5, 0.75);
    vec3 c = mix(sp1, sp2, factor < 0.8 ? 1.0 : 0.0);
    c = mix(c, sp3, factor < 0.6 ? 1.0 : 0.0);
    c = mix(c, sp4, factor < 0.4 ? 1.0 : 0.0);
    c = mix(c, sp5, factor < 0.2 ? 1.0 : 0.0);
    return pow(max(c, vec3(0.0)), vec3(2.2));
}

/**
 * @brief the reference's `Matcap` group, operator for operator
 * @param base_color the material's albedo (factor x texture)
 * @param light_factor the reference's `Light Factor`
 * @param matcap_sample the `Eff_MatCap` lookup, which is the matcap or the model's sphere map
 * @return the shaded base colour the shadow multiplier is then applied to
 *
 * PORTED WHOLE, including the part that is easy to mistake for a texture trick: the last node is a
 * NON-UNIFORM mix whose factor is a PER-CHANNEL BOOLEAN (`base > 0.5`), so colours below half are scaled
 * down while colours above it get the `1 - 2(1-base)(1-mix)` term on top. That switch - not a colour ramp -
 * is where the reference's extra saturation in the darks comes from, and it runs on EVERY material, with or
 * without a matcap image.
 *
 * `matcap_sample` is 0 at this port's call site: the .blend ships `Eff_MatCap` as an EMPTY image, and this
 * model's own sphere map is already applied to the albedo in MMD's own mode (see gather_surface), so
 * feeding it here as well would count it twice. Zero is the reference's own no-matcap case.
 */
vec3 reference_matcap_combine(vec3 base_color, float light_factor, vec3 matcap_sample) {
    const vec3 mix3 = (matcap_sample - vec3(0.5)) * 0.5 + matcap_sample; // Mix.003: ADD, factor 1
    const vec3 mix1 = vec3(0.5) + mix3 * (light_factor * 0.5);           // Mix.001: ADD, factor = light * 0.5
    const vec3 v12 = vec3(1.0) - ((vec3(1.0) - mix1) * ((vec3(1.0) - base_color) * 2.0)); // .006 .009 .011 .012
    const vec3 v10 = (base_color * 2.0) * mix1;                          // Vector Math.008, .010
    const vec3 factor = vec3(base_color.r > 0.5 ? 1.0 : 0.0, base_color.g > 0.5 ? 1.0 : 0.0, base_color.b > 0.5 ? 1.0 : 0.0);
    return v10 + v12 * factor;                                           // Mix.002, NON_UNIFORM factor
}

/**
 * @brief shadow factor of ONE cascade for a world-space point: normal-offset bias + 3x3 PCF
 * @param world_pos receiver position in world space
 * @param normal receiver world-space normal
 * @param cascade cascade index (the shadow map array layer)
 * @return 1.0 = fully lit, 0.0 = fully shadowed (also passed through toon_band for cel shading)
 *
 * - the sample point is pushed along the world normal by a couple of light-space texels, which
 *   removes most quantization acne on flat receivers WITHOUT a large depth bias - and a large depth
 *   bias is exactly what erases the shadow of a thin caster (a sword, a railing). The normal offset
 *   lets the depth bias below drop to a fraction of the old 0.0015. The offset is per cascade
 *   because the texel size is: cascade 0's texels are a fraction of cascade 3's, and a fixed world
 *   offset would over-bias the near range (detaching contact shadows) and under-bias the far one.
 * - the 3x3 grid of hardware 2x2 comparison taps (4x4 texel footprint) smooths the edge; a single
 *   tap flickered badly on thin geometry. The grid is unrotated on purpose: a rotated grid needs TAA
 *   to hide its per-pixel noise (the deferred path has TAA, the forward path does not yet).
 * - outside the light frustum the fragment is reported lit: each cascade's map covers its own fitted
 *   box only, and runtime::update_shadow_frustum() keeps those boxes on the part of the scene the
 *   camera can see.
 */
float calc_shadow_cascade(vec3 world_pos, vec3 normal, int cascade) {
    float texel_uv = light[heap_light_slot].light_dir.w;                    // 1 / shadow map size
    float texel_world = light[heap_light_slot].cascade_texel_world[cascade]; // world size of one texel of this cascade

    // normal offset: shift the world position before projecting it into light space
    vec3 offset_pos = world_pos + normal * (texel_world * 2.0);
    vec4 light_clip = light[heap_light_slot].light_view_proj[cascade] * vec4(offset_pos, 1.0);
    vec3 ndc = light_clip.xyz / light_clip.w; // ortho projection: w == 1
    vec2 uv = ndc.xy * 0.5 + 0.5;
    float current_depth = ndc.z; // [0,1] (RH_ZO ortho)

    // Outside the light frustum: fully lit (the shadow map covers the scene bounds only)
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || current_depth < 0.0 || current_depth > 1.0) {
        return 1.0;
    }

    // The sampler's compareOp is LESS_OR_EQUAL, so lit = (ref - bias) <= stored depth.
    float bias = 0.0004;
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 tap = uv + vec2(float(x), float(y)) * texel_uv;
            lit += texture(sampler2DArrayShadow(shadow_texture[heap_shadow_slot], heap_samplers[heap_sampler_shadow]), vec4(tap, float(cascade), current_depth - bias));
        }
    }
    float shadow = lit / 9.0;
    // cel shading hardens the shadow edge into the same bands as the diffuse falloff
    return toon_band(shadow, light[heap_light_slot].toon_steps, light[heap_light_slot].toon_softness);
}

/**
 * @brief shadow factor for a world-space point, selecting (and blending between) the cascades
 * @param world_pos receiver position in world space
 * @param normal receiver world-space normal
 * @return 1.0 = fully lit, 0.0 = fully shadowed
 *
 * The cascade is picked by the fragment's VIEW-SPACE DEPTH against the per-cascade split distances
 * the CPU fitted: cascade i covers [splits[i-1], splits[i]], so the near geometry is shadowed by the
 * small, dense cascade 0 and the far range by the coarse last one - the whole point of cascades is
 * that a single map cannot be dense enough for both.
 *
 * Inside `cascade_blend` of a boundary the two neighbouring cascades are both sampled and mixed:
 * a hard switch would show as a line where the resolution (and the offset) step is.
 */
float calc_shadow(vec3 world_pos, vec3 normal) {
    if (light[heap_light_slot].cascade_count < 1.5) {
        return calc_shadow_cascade(world_pos, normal, 0); // single map: no selection to do
    }
    const float view_depth = -(camera[heap_camera_slot].view * vec4(world_pos, 1.0)).z; // positive distance along the view
    int cascade = int(light[heap_light_slot].cascade_count + 0.5) - 1;                 // past the last split: the farthest
    for (int i = 0; i < MAX_SHADOW_CASCADES; ++i) {
        if (i >= int(light[heap_light_slot].cascade_count + 0.5)) {
            break;
        }
        if (view_depth <= light[heap_light_slot].cascade_splits[i]) {
            cascade = i;
            break;
        }
    }
    float shadow = calc_shadow_cascade(world_pos, normal, cascade);

    // blend into the next cascade across the boundary band
    const int next = cascade + 1;
    if (next < int(light[heap_light_slot].cascade_count + 0.5)) {
        const float boundary = light[heap_light_slot].cascade_splits[cascade];
        const float band = max(boundary * light[heap_light_slot].cascade_blend, 1e-4);
        if (view_depth > boundary - band) {
            const float t = clamp((view_depth - (boundary - band)) / band, 0.0, 1.0);
            shadow = mix(shadow, calc_shadow_cascade(world_pos, normal, next), t);
        }
    }
    return shadow;
}

/**
 * @brief GGX / Trowbridge-Reitz normal distribution (matches UE's D_GGX)
 * @param n world normal, @p h half vector, @p roughness perceptual roughness
 * @return the NDF value, finite even at a perfectly smooth specular hotspot (the denominator is
 *         clamped: roughness 0 with ndoth == 1 would otherwise be 0/0 = NaN and blacken the fragment)
 */
float distribution_ggx(vec3 n, vec3 h, float roughness) {
    float a2 = roughness * roughness;
    a2 = a2 * a2; // perceptual roughness -> alpha^2 (UE passes Pow4(Roughness))
    float ndoth = max(dot(n, h), 0.0);
    float denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
    // Guard the denominator (same NaN class as the Vis guards below): at the exact specular
    // hotspot center of a perfectly smooth surface (roughness 0 AND ndoth == 1) denom is 0 and
    // a2 / (PI * 0) is 0/0 = NaN, turning the fragment black. Clamping keeps D finite (it is
    // 0 there anyway - an infinitely sharp lobe needs no finite value).
    denom = max(denom, 1e-6);
    return a2 / (PI * denom * denom);
}

/// @brief Beckmann NDF preset (brdf_model 2); same alpha^2 mapping as distribution_ggx
float distribution_beckmann(vec3 n, vec3 h, float roughness) {
    float a2 = roughness * roughness;
    a2 = max(a2 * a2, 1e-6); // roughness 0 would make 0/0 (or inf) below at ndoth == 1
    float ndoth = max(dot(n, h), 1e-4); // guard: tan blows up at grazing, exp() dies first
    float cos2 = ndoth * ndoth;
    float tan2 = (1.0 - cos2) / cos2;
    return exp(-tan2 / a2) / (PI * a2 * cos2 * cos2);
}

/// @brief Blinn-Phong NDF preset (brdf_model 3), exponent capped at 4096
float distribution_blinn_phong(vec3 n, vec3 h, float roughness) {
    float a2 = roughness * roughness;
    a2 = max(a2 * a2, 1e-6);
    const float exponent = min(2.0 / a2 - 2.0, 4096.0); // classic n = 2 / alpha^2 - 2
    const float ndoth = max(dot(n, h), 0.0);
    return (exponent + 2.0) * pow(ndoth, exponent) / (2.0 * PI);
}

/**
 * @brief Geometric shadowing-masking as a visibility term: Vis = G / (4 NoV NoL)
 * @param n world normal, @p v view direction, @p l light direction, @p roughness
 * @return Heitz's joint Smith approximation for GGX (UE's Vis_SmithJointApprox)
 *
 * One term shadows AND masks in the half-vector sense, so the BRDF is specular = D * Vis * F with no
 * separate 4 NoV NoL denominator (UE's SpecularGGX structure). The denominator is clamped: at an
 * exact grazing silhouette both ndotv and ndotl are 0, and inf * 0 = NaN would turn the whole
 * fragment black (visible as black flashes on thin skinned limbs).
 */
float geometry_vis_smith_joint_approx(vec3 n, vec3 v, vec3 l, float roughness) {
    float a2 = roughness * roughness;
    a2 = a2 * a2; // perceptual roughness -> alpha^2 (UE passes Pow4(Roughness))
    float a = sqrt(a2);
    float ndotv = max(dot(n, v), 0.0);
    float ndotl = max(dot(n, l), 0.0);
    float vis_v = ndotl * (ndotv * (1.0 - a) + a);
    float vis_l = ndotv * (ndotl * (1.0 - a) + a);
    // Guard the denominator: at an exact grazing silhouette both ndotv and ndotl are 0, the sum
    // below is 0 and 0.5/0 would be +inf. The direct term multiplies by radiance (ndotl * ...),
    // so inf * 0 = NaN would turn the whole fragment black (visible as black flashes on thin
    // skinned limbs like RecursiveSkeletons at certain poses). Clamping the sum keeps Vis large
    // but finite - at ndotl == 0 the product is exactly 0 either way.
    return 0.5 / max(vis_v + vis_l, 1e-5);
}

/// @brief Height-correlated Smith visibility (Heitz 2014), the exact form the approximation above
///        simplifies - selected as brdf_model 1; same alpha^2 and the same grazing guard
float geometry_vis_smith_height_correlated(vec3 n, vec3 v, vec3 l, float roughness) {
    float a2 = roughness * roughness;
    a2 = a2 * a2;
    float ndotv = max(dot(n, v), 0.0);
    float ndotl = max(dot(n, l), 0.0);
    const float sqrt_v = sqrt(ndotv * ndotv * (1.0 - a2) + a2);
    const float sqrt_l = sqrt(ndotl * ndotl * (1.0 - a2) + a2);
    // same grazing guard as geometry_vis_smith_joint_approx
    return 0.5 / max(ndotl * sqrt_v + ndotv * sqrt_l, 1e-5);
}

/// @brief Oren-Nayar diffuse (roughness-dependent), selected as diffuse_model 1: the classic A/B
///        approximation of the paper's integral; roughness 0 reduces to Lambert (A = 1, B = 0)
float oren_nayar_diffuse(vec3 n, vec3 v, vec3 l, float roughness, float ndotv, float ndotl) {
    const float alpha2 = roughness * roughness;
    const float A = 1.0 - 0.5 * alpha2 / (alpha2 + 0.33);
    const float B = 0.45 * alpha2 / (alpha2 + 0.09);
    const float sin_v = sqrt(max(1.0 - ndotv * ndotv, 0.0));
    const float sin_l = sqrt(max(1.0 - ndotl * ndotl, 0.0));
    // cos(phi_i - phi_o): angle between the view/light projections onto the tangent plane
    const vec3 vp = v - n * ndotv;
    const vec3 lp = l - n * ndotl;
    const float vp_len = length(vp);
    const float lp_len = length(lp);
    float cos_diff = 0.0;
    if (vp_len > 1e-6 && lp_len > 1e-6) {
        cos_diff = clamp(dot(vp, lp) / (vp_len * lp_len), 0.0, 1.0);
    }
    // sin(alpha) * tan(beta), alpha/beta = the larger/smaller of the two incident angles
    const float sin_max = max(sin_v, sin_l);
    const float tan_min = min(sin_v / max(ndotv, 1e-4), sin_l / max(ndotl, 1e-4));
    return (A + B * cos_diff * sin_max * tan_min) / PI;
}

/// @brief Schlick's Fresnel approximation: @p f0 + (1 - f0) * (1 - cos(theta))^5
vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

/**
 * @brief Cook-Torrance direct light for ONE light, in radiance units
 * @param n world normal, @p v view direction, @p base_color albedo
 * @param metallic / roughness material factors, @p f0 the Fresnel reflectance at normal incidence
 * @param light_dir surface-to-light direction, @p light_radiance radiance * attenuation
 * @return outgoing radiance (already multiplied by the diffuse ndotl factor)
 *
 * @p light_radiance carries the light's intensity/attenuation (and, for the sun, its shadow factor);
 * ndotl is folded in here, so the directional sun and every punctual light take the exact same code
 * path. The BRDF theory selections (LightUBO.brdf_model / diffuse_model) and the cel-shading bands
 * are applied inside.
 *
 * @note these presets drive the DIRECT lights only. The IBL ambient in shade_surface() always uses
 *       the fixed GGX model (prefiltered GGX environment + GGX BRDF LUT with Fdez-Aguera
 *       multiscatter compensation, Lambert diffuse irradiance) - the GUI preset switch is an honest
 *       direct-light A/B, not a whole-scene model comparison.
 */
//
// `ndotl_wrap` 1 = half-lambert for the diffuse falloff (see the FACE below), 0 = the ordinary ramp.
vec3 evaluate_direct_light(vec3 n, vec3 v, vec3 base_color, float metallic, float roughness, vec3 f0, vec3 light_dir, vec3 light_radiance, vec3 matcap_sample, float ndotl_wrap) {
    vec3 l = normalize(light_dir);
    // Half vector: normalize(v + l) is NaN when the light sits exactly behind the fragment
    // along the view ray (v + l == 0, e.g. a point light placed at the camera). Fall back to
    // the normal - a degenerate lobe with no real reflection, but a finite one.
    const vec3 sum = v + l;
    vec3 h = length(sum) > 1e-6 ? sum / length(sum) : n;

    // ---- Specular NDF / visibility by the selected BRDF preset. Preset 0 = GGX + joint
    //      Smith, byte-for-byte the historic default; each other preset differs by exactly one
    //      piece so the gui is a live A/B compare.
    float ndf;
    float vis;
    const int brdf_model = int(light[heap_light_slot].brdf_model + 0.5);
    if (brdf_model == 1) {
        ndf = distribution_ggx(n, h, roughness);
        vis = geometry_vis_smith_height_correlated(n, v, l, roughness);
    } else if (brdf_model == 2) {
        ndf = distribution_beckmann(n, h, roughness);
        vis = geometry_vis_smith_joint_approx(n, v, l, roughness);
    } else if (brdf_model == 3) {
        ndf = distribution_blinn_phong(n, h, roughness);
        vis = geometry_vis_smith_joint_approx(n, v, l, roughness);
    } else {
        ndf = distribution_ggx(n, h, roughness);
        vis = geometry_vis_smith_joint_approx(n, v, l, roughness);
    }
    vec3 f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    // UE structure: specular = D * Vis * F (Vis already folds in G / (4 NoV NoL))
    vec3 specular = ndf * vis * f;

    vec3 kd = (1.0 - f) * (1.0 - metallic);
    float ndotv = max(dot(n, v), 0.0);
    // The UNQUANTIZED falloff is kept for the rim below: the rim is a silhouette term, and a band edge
    // cutting it would be visible as a step in it.
    const float raw_ndotl = max(dot(n, l), 0.0);
    // ---- A FACE IS NOT LIT LIKE A SPHERE, and the reference does not light it like one either: its face
    // path takes the lit/shadow split from a soft `Face Factor` group and its shadow is a TINT, not an
    // absence of light. With the ordinary ramp a sun that happens to sit to the side of the face turns the
    // whole face grey - measured at 155 luma against 209 with the camera-aligned normal - which is the other
    // half of what a user kept reporting. Half-lambert wraps the falloff so the face keeps its light and
    // only its SHADOW SIDE darkens, smoothly.
    const float wrap = clamp(ndotl_wrap, 0.0, 1.0);
    float ndotl = mix(raw_ndotl, raw_ndotl * 0.5 + 0.5, wrap);

    // ---- cel/toon shading (no-op when LightUBO.toon_steps < 1): quantize the diffuse falloff
    //      into bands and turn the specular lobe into a single hard highlight block. The
    //      visibility terms keep their unquantized ndotl, so only the shading response bands -
    //      the silhouette stays smooth.
    if (light[heap_light_slot].toon_steps > 0.5) {
        ndotl = toon_band(ndotl, light[heap_light_slot].toon_steps, light[heap_light_slot].toon_softness);
        float ndoth = max(dot(n, h), 0.0);
        float highlight_threshold = 0.5 + 0.5 * (1.0 - roughness); // smooth surfaces -> tighter highlight
        specular *= smoothstep(highlight_threshold - light[heap_light_slot].toon_softness, highlight_threshold + light[heap_light_slot].toon_softness, ndoth);
    }

    // ---- THE REFERENCE'S DIFFUSE PATH (XIYAG's `Main Shader`), taken over from the cel ramp while
    //      `toon_steps > 0`. What the port keeps from the reference: the shading is a MULTIPLIER on the
    //      albedo whose falloff is the reference's own narrow Light Factor (smoothstep(0, 0.25, NdotL)), the
    //      shadow end of that multiplier walks a five-colour cascade (reference_shadow_colors), and the
    //      specular is attenuated by the same factor. What it substitutes, and why, is on those functions.
    //
    //      The measured reason the falloff is NOT a separate NdotL multiply on top: this engine's sun
    //      carries a radiance of 7.5 (see shade_surface), so the first version - which dropped the falloff
    //      and let the tinted colour be the whole result - put the lit band at albedo/pi * 7.5 = 2.4x the
    //      albedo and the tonemapper resolved it to white (face region mean 176 against PBR's 154 and pure
    //      albedo's 182).
    const vec3 shadow_tint = light[heap_light_slot].npr_shadow.rgb;
    const float shadow_band = light[heap_light_slot].npr_shadow.w;
    const bool warp = light[heap_light_slot].toon_steps > 0.5;

    // ---- Diffuse by the selected model (LightUBO.diffuse_model): Lambert (default) or the
    //      roughness-dependent Oren-Nayar approximation (0 -> Lambert).
    vec3 radiance;
    if (warp) {
        const float ndotl_falloff = smoothstep(0.0, 0.25, raw_ndotl);
        // The falloff is the reference's own narrow Light Factor. NOTE: an earlier version swapped it for a
        // single head-frame scalar for face materials (its `- Face` mechanism, with the VIEW standing in for
        // the head frame). It measured as a NO-OP on this asset - this sun puts every face normal past the
        // reference's own 0.25 threshold - and it is moot now that painted materials discard this whole
        // result: the flatness comes from the albedo override at the end of shade_surface, which is a
        // stronger version of the same idea. The measurement stands in docs/zzz_shading.md.
        const float light_factor = ndotl_falloff;
        // THE BAND FACTOR, per texel where the reference's light map would supply it. A PMX has no light map,
        // so the band is derived from the surface's OWN albedo luminance - a darker material takes a deeper
        // shadow - which is the direction a painted light map goes, and the frame's toon_shadow_band_gain
        // scales how much of it is used. At gain 0 the band is the frame-wide constant, i.e. exactly the
        // behaviour before this existed.
        //
        // Why it has to be per texel at all is measured rather than assumed: sweeping the frame-wide band on
        // this asset puts the HAIR's best arm at 0.15 (shadow luma 182.7 / saturation 0.244 against the
        // reference's 180.3 / 0.246, a near-exact match) while the FACE only approaches its reference
        // (196.7 / 0.157) above 0.3 - no single value serves both. The luminance split reproduces that
        // direction: the hair's albedo is mid-dark (linear luma 0.42), the skin's is pale (0.53).
        const float albedo_luma = dot(base_color, vec3(0.2126, 0.7152, 0.0722));
        const float band = clamp(shadow_band + light[heap_light_slot].npr_rim.w * albedo_luma, 0.0, 1.0);
        // NOTE: this path's result is DISCARDED for a painted material (see the albedo override at the end of
        // shade_surface), so the face's own band offset and cast-shadow retention that used to be mixed in
        // here are gone: they tuned a lighting result nothing reads any more. Their measurements stand in
        // docs/zzz_shading.md, and the band itself still serves every material that IS lit.
        // the reference's own composition, in its own order: the base colour goes through the per-channel
        // Matcap combine, THAT is multiplied by the shadow multiplier, and the stepped highlight is added on
        // top of it, masked by the reference's `MData.z` (a per-material value the game reads from its ILM
        // texture; a PMX has none, so it is the frame's [render] toon_specular here) and by its own
        // smoothstep(0.75, 1, NdotH) band and the Light Factor.
        // the reference's Matcap combine runs where the material HAS a matcap sample - its own `Eff_MatCap`
        // image, which this port feeds the model's MMD sphere map instead. With no sample the reference's
        // maths degenerates into a per-channel contrast curve driven by the light factor alone, and measured
        // on this asset's hair that LIGHTENS the shadow side (the shadow/lit quartile ratio moves from 0.78
        // to 0.83, away from the reference's own 0.75). That degeneration is an artifact of the reference's
        // template shipping its matcap image EMPTY rather than an intended look, so a material without a
        // sample passes through unchanged.
        const vec3 combined = matcap_sample == vec3(0.0) ? base_color : reference_matcap_combine(base_color, light_factor, matcap_sample);
        const vec3 shadow_mult = reference_shadow_multiplier(base_color, raw_ndotl, band, shadow_tint);
        const float spec_mask = light[heap_light_slot].npr_rim.z;
        // the half-vector dot this branch needs is not the one the BRDF helper computes internally, so it is
        // taken here; v + l degenerates to zero when the view and the light are exactly opposed
        const vec3 half_vector = v + l;
        const float half_length = length(half_vector);
        const float ndoth_band = half_length > 1e-6 ? smoothstep(0.75, 1.0, max(dot(n, half_vector / half_length), 0.0)) : 0.0;
        const vec3 highlight = reference_specular_colors(base_color, band) * (spec_mask * ndoth_band * light_factor);
        const vec3 shaded = kd * (combined * shadow_mult + highlight) / PI;
        radiance = shaded * light_radiance;
    } else {
        vec3 diffuse;
        if (int(light[heap_light_slot].diffuse_model + 0.5) == 1) {
            diffuse = kd * base_color * oren_nayar_diffuse(n, v, l, roughness, ndotv, ndotl);
        } else {
            diffuse = kd * base_color / PI;
        }
        radiance = (diffuse + specular) * (light_radiance * ndotl);
    }

    // ---- the rim (no-op at strength 0, the compiled default). The reference takes it from the matcap
    //      half of its sphere - "CombineMESphere" is the model's own MMD sphere texture combined with an
    //      authored matcap - so it is a VIEW-space term rather than a light one. This is that term with
    //      the texture left out (the tinted sphere sample is the next slice; see docs/zzz_shading.md);
    //      it is added rather than multiplied because it is emitted light, scaled by the unquantized
    //      falloff so the rim does not wrap onto the unlit side, and carried by the sun's radiance so it
    //      cannot glow through a cast shadow.
    const float rim_strength = light[heap_light_slot].npr_rim.x;
    if (rim_strength > 0.0) {
        const float rim = pow(1.0 - ndotv, light[heap_light_slot].npr_rim.y);
        radiance += light_radiance * (rim * rim_strength * raw_ndotl);
    }
    return radiance;
}

/// @brief IBL: split-sum approximation (ported from glTF-Sample-Renderer's ibl.glsl); the ambient
///        always uses this fixed GGX model, independent of the BRDF presets above
/// Diffuse ambient: irradiance map lookup by the world normal
vec3 get_diffuse_light(vec3 n) {
    return texture(samplerCube(irradiance_texture[heap_irradiance_slot], heap_samplers[heap_sampler_texture]), n).rgb;
}

// The specular half - get_specular_sample / get_ibl_ggx_fresnel / get_ibl_radiance_ggx - is
// shaders/ibl_specular.glsl's, included above. It moved there so a subtraction would use the same
// expressions rather than a third copy; see that file's header for what was
// verified. The three below are the call sites' names for it.

/**
 * @brief depth slice a view-space depth falls into (exponential slicing between the cluster range)
 * @param view_depth positive distance along the view direction
 * @param slices active slice count
 * @return slice index in [0, slices - 1]
 *
 * MUST stay identical to the same function in shaders/light_cluster.comp: the compute pass assigns
 * lights by unprojecting exactly these slice boundaries, so a different rounding here would put a
 * fragment in a cluster the lights were never assigned to (a light popping out at a slice edge).
 */
int cluster_slice_of(float view_depth, int slices) {
    const float near = max(light[heap_light_slot].cluster_depth.x, 1e-4);
    const float far = max(light[heap_light_slot].cluster_depth.y, near * 1.0001);
    const float t = clamp(log(max(view_depth, near) / near) / log(far / near), 0.0, 1.0);
    return clamp(int(t * float(slices)), 0, slices - 1);
}

/**
 * @brief cluster index of a pixel, or -1 when clustering is off (the brute-force path)
 * @param pixel the shaded pixel, in PIXELS from the top-left (y down - the same convention the compute
 *        shader's dispatch uses); a fragment passes `ivec2(gl_FragCoord.xy)`
 * @param world_pos the shaded point's world position (its view depth picks the slice)
 * @note THE PIXEL IS A PARAMETER rather than read from gl_FragCoord, and that is what makes this file
 *       usable from a COMPUTE shader: `shaders/megalights_trace.comp` includes it for the same BRDF and
 *       the same cluster lists, and gl_FragCoord does not exist there. The fragment path's wrapper below
 *       is the only place the builtin appears.
 */
int cluster_index_at(ivec2 pixel, vec3 world_pos) {
    if (light[heap_light_slot].cluster_grid.w < 0.5) {
        return -1;
    }
    const int tiles_x = int(light[heap_light_slot].cluster_grid.x);
    const int tiles_y = int(light[heap_light_slot].cluster_grid.y);
    const int slices = int(light[heap_light_slot].cluster_grid.z);
    if (tiles_x <= 0 || tiles_y <= 0 || slices <= 0) {
        return -1;
    }
    const ivec2 tile = clamp(pixel / int(CLUSTER_TILE_SIZE), ivec2(0), ivec2(tiles_x - 1, tiles_y - 1));
    const float view_depth = -(camera[heap_camera_slot].view * vec4(world_pos, 1.0)).z;
    return (cluster_slice_of(view_depth, slices) * tiles_y + tile.y) * tiles_x + tile.x;
}

/**
 * @brief how many punctual lights the loop must visit for this fragment
 * @param cluster the fragment's cluster index, or -1 for the brute-force path
 */
int cluster_light_count_for(int cluster) {
    if (cluster < 0) {
        return int(light[heap_light_slot].light_count);
    }
    return int(min(cluster_counts[heap_cluster_count_slot].counts[cluster], uint(CLUSTER_LIGHT_CAPACITY)));
}

/**
 * @brief the i-th light index of a cluster (i < cluster_light_count_for(cluster))
 * @param cluster the fragment's cluster index, or -1: then the light index IS i
 */
int cluster_light_index(int cluster, int i) {
    if (cluster < 0) {
        return i;
    }
    return int(cluster_indices[heap_cluster_index_slot].indices[cluster * int(CLUSTER_LIGHT_CAPACITY) + i]);
}

/**
 * @brief the radiance one punctual light delivers at a surface point, and where it comes from
 * @param pl the light (the light UBO's entry)
 * @param world_pos the receiving point
 * @param out_to_light the unit direction from the point TOWARDS the light
 * @param out_distance the distance to the light, in world units
 * @return linear radiance arriving at the point, BEFORE the BRDF
 *
 * THE ONE DEFINITION OF A PUNCTUAL LIGHT'S ATTENUATION, extracted so that the stochastic lighting pass
 * (`shaders/megalights_trace.comp`) and this shader's own loop cannot disagree about it: the trace pass
 * has to evaluate a light's contribution to build its sampling PDF, and a second copy of inverse-square
 * plus the range fade plus the spot cone is a second chance to pick different thresholds.
 *
 * The falloff is `1 / (1 + d^2)` rather than a physical `1 / d^2`: it is well behaved at zero distance,
 * which is what an artist-facing intensity wants. `range` (0 = infinite) adds a smooth cutoff - the fade
 * is SQUARED so the value and its slope both reach zero at the range boundary.
 */
vec3 punctual_light_radiance(const PunctualLight pl, vec3 world_pos, out vec3 out_to_light, out float out_distance) {
    const vec3 to_light = pl.position.xyz - world_pos;
    out_distance = length(to_light);
    out_to_light = out_distance > 1e-6 ? to_light / out_distance : vec3(0.0, 1.0, 0.0);
    vec3 radiance = pl.color.xyz / (1.0 + out_distance * out_distance);
    const float range = pl.params.x;
    if (range > 0.0) {
        // smooth range cutoff (no hard pop at the boundary)
        const float d = out_distance / range;
        const float fade = clamp(1.0 - d * d, 0.0, 1.0);
        radiance *= fade * fade;
    }
    if (pl.params.y > 0.5) { // spot light: cone around spot_dir
        const float outer = pl.params.z;
        // inner cone: cos of the inner half-angle (glTF KHR innerConeAngle, set by the CPU);
        // w == 0 means the light did not specify one -> legacy soft-inner mix(outer, 1, 0.6)
        const float inner = pl.params.w > 0.0 ? pl.params.w : mix(outer, 1.0, 0.6);
        const float cone = smoothstep(outer, inner, dot(-out_to_light, normalize(pl.spot_dir.xyz)));
        radiance *= cone;
    }
    return radiance;
}

/**
 * @brief everything the shading stage needs to know about one surface point
 * @note the forward path fills this from its interpolated fragment inputs, the deferred path from
 *       the G-buffer texels - which is the whole point: the lighting below cannot tell them apart
 */
struct shade_input {
    ivec2 pixel;    // the pixel being shaded, in PIXELS (y down); the cluster lookup's tile coordinate
    vec3 world_pos; // world-space position (view vector + shadow lookup)
    vec3 normal;    // world-space shading normal (normal-mapped, double-sided flipped)
    vec3 albedo;    // base color, linear
    vec3 sphere_sample; // the material's MMD sphere/matcap lookup, or 0 when it has none: the reference's
                    // Matcap combine needs the sample and the light factor together, so the lookup travels
                    // with the shading input rather than only being folded into the albedo
    float painted_mask; // 1 when this material is PAINTED: drawn from its albedo rather than from the
                    // lighting stack (the converter's mmd_unlit - the face block and the head's props)
    float face_mask; // 1 for the model's FACE block, which the LIGHTING needs too: a lit face keeps less of
                    // the cast shadow than the body does (the reference's face path reads a light map)
    vec3 emissive;  // emissive radiance, linear (added after the lighting)
    float metallic; // 0 = dielectric, 1 = metal
    float roughness; // perceptual roughness
    float ao;       // ambient occlusion: scales the IBL ambient only, never the direct light
    /**
     * 1 = the PUNCTUAL lights are somebody else's business this frame, so this stage must not add them.
     *
     * The somebody is `shaders/megalights_trace.comp` (docs/megalights.md): it samples a few of the pixel's
     * lights, traces one visibility ray per sample and produces the shadowed estimate that the deferred
     * lighting stage then adds back - so this loop, which knows nothing about occlusion, would double every
     * punctual light in the frame.
     *
     * IT IS A RECORDED FACT RATHER THAN THE KNOB, and the difference matters: the renderer sets it from
     * whether the stochastic pass actually recorded this frame, so a frame whose pass was gated off keeps
     * this loop instead of losing its punctual lights entirely.
     */
    float punctual_replaced;
    // < 0 = no override: the shadow comes from calc_shadow() as it always did. >= 0 = use this factor
    // instead, which is how the deferred path hands in the RAY-TRACED visibility (it is a screen-space
    // lookup, so it cannot be recomputed from world_pos inside the shared lighting code). The forward
    // path and every other caller leave it negative, so their behaviour is unchanged by construction.
    float shadow_override;
};

/**
 * @brief evaluate the full lighting of one surface point
 * @param s the surface properties (see shade_input)
 * @return linear HDR outgoing radiance - no exposure, no tonemapping, no display encoding: the post
 *         pass owns all display-referred work, so a deferred frame and a forward frame of the same
 *         scene go through exactly the same display pipeline
 *
 * The directional sun through evaluate_direct_light() (attenuated by calc_shadow()), the active
 * punctual lights through the same BRDF path, the split-sum IBL ambient (diffuse irradiance +
 * prefiltered specular, always the fixed GGX model), then emissive. The shadow factor attenuates the
 * directional sun only; the IBL ambient stays unshadowed, which is the usual approximation and keeps
 * interiors from going pitch black.
 */
vec3 shade_surface(shade_input s) {
    const vec3 v = normalize(camera[heap_camera_slot].camera_pos - s.world_pos);
    const vec3 f0 = mix(vec3(0.04), s.albedo, s.metallic);

    vec3 direct = vec3(0.0);
    // ---- THE FACE'S FLATTENED SHADING NORMAL (the reference's `- Face` group exists for this): a face is
    // nearly flat and its shading should follow the HEAD, not the nose, the lips and the cheeks - those
    // normals are what put a gradient across a face that has none in the art. It is blended towards the
    // direction the face is facing (at the eye, in a portrait), which leaves the face LIT while the
    // per-feature shading goes away.
    //
    // IT IS USED FOR THE LIGHTING ONLY. s.normal stays the GEOMETRIC one for everything that is not a
    // shading input: the shadow lookup above, the screen-space occlusion that runs before this stage, and
    // the G-buffer the other passes read. A normal bent towards the eye used for the shadow comparison
    // self-shadows (the offset no longer matches the surface) and one used for SSAO occludes the face
    // against itself - both measured as a face going grey on a turned head.
    //
    // AND THE TARGET IS THE CAMERA'S VIEW AXIS, NOT THE PER-FRAGMENT DIRECTION TO THE EYE. The first version
    // used normalize(v), and that breaks at exactly the angles a user reported: looked at from below, the
    // eye vector lies almost IN the face's plane, so a normal blended 85% towards it swings AWAY from the
    // light, n.l falls off the cel threshold and the face reads as shadowed - past about 30 degrees of
    // elevation. One direction for the whole face keeps it a flat plane that always faces the viewer.
    const float face_normal_flatten = 0.85;
    // THE TARGET IS THE MODEL'S OWN FACE PLANE (npr_face_forward, averaged over the face block's vertex
    // normals), NOT THE CAMERA AND NOT A GUESSED WORLD AXIS. Both of those were tried and both were wrong in
    // ways a user could see: the camera axis made the shading follow the viewer, so the face came out right
    // at one angle and wrong at the rest; a world axis measured wrong and darkened the whole face. The model
    // knows its own direction, and this is it - the same thing the reference takes from its `headFwd`.
    const vec3 face_forward = length(light[heap_light_slot].npr_face_forward.xyz) > 0.5
                                  ? normalize(light[heap_light_slot].npr_face_forward.xyz)
                                  : s.normal;
    const vec3 shading_normal = mix(s.normal, face_forward, s.face_mask * face_normal_flatten);

    // directional sun: shadow factor attenuates only this light; IBL ambient stays unshadowed
    {
        // A ray-traced override wins over both: it IS the shadow, already resolved per pixel.
        float shadow = 1.0;
        if (s.shadow_override >= 0.0) {
            shadow = s.shadow_override;
        } else if (light[heap_light_slot].shadow_enabled > 0.5) {
            shadow = calc_shadow(s.world_pos, s.normal);
        }
        // ---- THE FACE TAKES NO SHADOW MAP AT ALL. Two reasons, and the second is the one a user found:
        // (1) the reference's face path reads a light map, not the shadow map - toon rendering puts the
        // face's shadow shape under the artist's control and the body's under NdotL (GDC's 3D Toon
        // Rendering talk puts it as "shadow shapes are determined by NdotL except for the face");
        // (2) the shadow lookup offsets its sample along the NORMAL (`world_pos + normal * 2 texels`), so a
        // face that is nearly PARALLEL to the light - exactly what a face looks like from below or in front
        // of a low sun - samples the map at a grazing angle, and the acne that produces was being KEPT at
        // 35% by the earlier mix. That is the grey speckle a user reported and diagnosed as "the shadow
        // computed wrongly, and the cause is the normal".
        const float sun_shadow = mix(shadow, 1.0, s.face_mask);
        direct += evaluate_direct_light(shading_normal, v, s.albedo, s.metallic, s.roughness, f0, light[heap_light_slot].light_dir.xyz, vec3(7.5 * light[heap_light_slot].sun_intensity) * sun_shadow, s.sphere_sample, s.face_mask);
    }
    // punctual lights (point/spot, no shadow casting in this version): inverse-square falloff
    // (well-behaved at zero distance) with an optional smooth range cutoff; spots add a soft
    // cone mask between the inner and outer half-angles.
    //
    // Clustered culling (M5): light_cluster.comp sorted the lights into screen-space tiles x
    // exponential depth slices, so this loop only touches the lights that can reach THIS pixel -
    // which is what makes a light count two orders of magnitude past the old four affordable. With
    // clustering off (cluster_grid.w == 0) the loop walks every active light instead, the
    // brute-force reference the clustered path is verified against.
    const int frag_cluster = cluster_index_at(s.pixel, s.world_pos);
    const int punctual_count = s.punctual_replaced != 0.0 ? 0 : cluster_light_count_for(frag_cluster);
    for (int i = 0; i < MAX_PUNCTUAL_LIGHTS; ++i) {
        if (i >= punctual_count) {
            break;
        }
        const int light_index = cluster_light_index(frag_cluster, i);
        const PunctualLight pl = light[heap_light_slot].punctual_lights[light_index];
        // The attenuation is `punctual_light_radiance`'s (see its note): the stochastic lighting pass
        // evaluates the same function to build its sampling PDF, so the two cannot drift.
        vec3 dir;
        float dist;
        const vec3 radiance = punctual_light_radiance(pl, s.world_pos, dir, dist);
        if (radiance != vec3(0.0)) {
            direct += evaluate_direct_light(shading_normal, v, s.albedo, s.metallic, s.roughness, f0, dir, radiance, s.sphere_sample, s.face_mask);
        }
    }

    // ---- IBL (split-sum): diffuse irradiance + prefiltered specular ----
    // THE ENVIRONMENT USES THE SHADING NORMAL, i.e. the flattened one on a face. An earlier version kept
    // the geometric normal here, on the grounds that the flattened one pointed at the camera and would
    // sample the sky behind the viewer. That reasoning died with the camera-axis target: the flattened
    // normal is the MODEL'S OWN FACE PLANE now, so the hemisphere it samples IS the one the face is facing.
    // Leaving the geometric normal here was measured as the face's biggest single loss: a face seen from
    // below has geometric normals pointing down-forward, so its ambient came off the GROUND - 155 luma
    // against 209 for the same face with the camera-aligned normal. The shadow lookup keeps the geometric
    // one, which is where the acne lived.
    vec3 ibl_diffuse = get_diffuse_light(shading_normal);
    vec3 ibl_specular = ibl_specular_radiance(shading_normal, v, s.roughness);
    vec3 fresnel_ibl = ibl_specular_fresnel(shading_normal, v, s.roughness, f0, 1.0);

    // Metals have no diffuse term: diffuse ambient is scaled by (1 - metallic),
    // metal color comes entirely from specular environment (matches the official mix(dielectric, metal, metallic))
    vec3 ambient = ibl_diffuse * s.albedo * s.ao * (1.0 - s.metallic);
    // ... and dropped entirely on a frame that does not add the diffuse ambient at all (see
    // shade_input). A BRANCH rather than a fifth factor, and it is not an optimisation: multiplying the
    // finished term by 1.0 is algebraically the identity but NOT bit-identical - folding a fifth operand
    // into the expression changes how the compiler contracts the chain, and it moved one 8-bit texel of
    // 1036800 on one measured frame (blue, one step). Zeroing the finished value leaves every frame that
    // does not use the switch bit for bit what it was, and that is what makes "the frames that do not use
    // it are unchanged" a check rather than a claim.
    vec3 specular_ibl = ibl_specular * fresnel_ibl * s.ao;

    // ---- THE ENVIRONMENT'S OWN GAIN AND TINT ([render] ambient_gain / ambient_tint). The ambient is the
    // sky: bright, and blue enough that this asset's darks came out 59/61/73 where the reference's are
    // 48/40/39, with a dark albedo of 58 rendering at 120.
    //
    // TWO PLAIN MULTIPLIES, NOT A BRANCH, AND THAT IS A MEASURED CHOICE: the first version wrapped them in
    // `if (any(notEqual(scale, vec3(1.0))))`, which never runs with neutral knobs yet still cost four of the
    // nine gate scenarios their bit-identity - control flow in the middle of this function changes how the
    // compiler contracts the cluster loop and the ambient expression around it. Multiplying by exactly 1.0
    // is exact in IEEE-754, and the gate is back to 0 changed with the multiplies unconditional.
    const vec3 ambient_scale = light[heap_light_slot].ambient_gain_tint.rgb * light[heap_light_slot].ambient_gain_tint.w;
    ambient *= ambient_scale;
    specular_ibl *= ambient_scale;

    vec3 color = ambient + direct + specular_ibl + s.emissive;

    // ---- THE NON-PBR PATH: A SURFACE DRAWN FROM ITS ALBEDO, NOT FROM THE LIGHT (the reference's own
    // arrangement for a face: its `- Face` shader
    // reads a face light map and a `TData` mask, never the sun - which is why an anime face keeps its
    // painted eyes, blush and nose mark instead of having them lit and washed out. Face materials (the
    // converter's `mmd_face`) therefore take the albedo with a gain, and none of the stack above.
    //
    // `unlit_gain` (the light UBO's npr_face.x) compensates for the fact that this path no longer receives
    // the sun's radiance: measured against the in-game capture, the reference's lit face sits at luma
    // 236 where this asset's raw albedo sits at 217, so a gain above 1 is what matches it. It is a
    // CONFIG value ([render] unlit_gain, a gui slider) because the frame's sun scale deliberately
    // cannot reach this path - the face is painted, and it needs its own brightness knob.
    const float unlit_gain = light[heap_light_slot].npr_face.x;
    vec3 painted_color = s.albedo * unlit_gain;

    // The nose mark is NOT here: it needs the surface uv, which this stage does not have in the deferred
    // path, so gather_surface draws it on the albedo instead (see surface.glsl).

    // ---- THE LIT FACE'S OWN BRIGHTNESS (light UBO's npr_face.z, [render] face_gain). The face is lit now,
    // so the sun scale reaches it and the painted path's unlit_gain does not apply; what a face wants is a
    // multiplier of its own, applied to the WHOLE lit result (ambient included) so it dims evenly rather than
    // going flat or dark on one side. 1.0 is the neutral value.
    color *= mix(1.0, light[heap_light_slot].npr_face.z, s.face_mask);

    color = mix(color, painted_color, s.painted_mask);

    // The scene target is HDR: this stage writes linear radiance. Exposure, ACES tonemapping and
    // gamma happen once in the post-process pass (post.frag), which also gives the bloom chain a
    // linear image to work on.
    return max(color, vec3(0.0));
}

#endif // VULKAN_RENDER_SHADING_GLSL
