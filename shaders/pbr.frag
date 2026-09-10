#version 450
#extension GL_EXT_nonuniform_qualifier : enable

/**
 * @file shaders/pbr.frag
 * @brief Forward PBR fragment stage: metallic-roughness direct lighting, IBL, shadows, cel shading.
 * @ingroup shaders
 *
 * Structure of main():
 * -# material lookup (texture indices + factors + flags) and the glTF alphaMode tests;
 * -# tangent-space normal mapping (with the double-sided normal flip) and occlusion;
 * -# the directional sun through evaluate_direct_light() - the same BRDF path every punctual light
 *    takes - attenuated by calc_shadow();
 * -# the split-sum IBL ambient (irradiance + prefiltered GGX environment + BRDF LUT), always the
 *    fixed GGX model so the GUI presets are an honest direct-light A/B;
 * -# emissive, then linear HDR output. Exposure, tonemapping and display encoding all belong to
 *    post.frag, so this stage never applies gamma.
 *
 * The BRDF theory model (NDF + visibility) and the diffuse model are switchable per frame through
 * LightUBO.brdf_model / diffuse_model - see the presets in evaluate_direct_light(). Cel/toon shading
 * (LightUBO.toon_steps / toon_softness) bands the shading response without touching geometry.
 */

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
} camera;

// Single flat scene set: shared camera UBO (binding 0), runtime texture array (binding 1,
// descriptor indexing: partially bound + update-after-bind), shared IBL (bindings 2-4) and
// the GPU material table (binding 5: per-material texture indices + factors, see material_record).
layout(set = 0, binding = 1) uniform sampler2D textures[];
layout(set = 0, binding = 2) uniform samplerCube env_sampler;        // prefiltered environment (roughness mip chain)
layout(set = 0, binding = 3) uniform samplerCube irradiance_sampler; // irradiance map (diffuse IBL)
layout(set = 0, binding = 4) uniform sampler2D brdf_lut_sampler;     // BRDF integration LUT

// One entry of the material table; layout matches material_record in vulkan/model.cppm (std430, 80 bytes)
struct Material {
    uvec4 tex_indices; // albedo, metallic-roughness, normal, occlusion (indices into textures[])
    uint emissive_index;
    float alpha_cutoff;       // alphaMode MASK threshold
    float occlusion_strength; // mix(1, sampled AO, strength)
    uint _pad;
    vec4 base_color_factor;
    vec4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    uint flags; // bit0: normal map, bit1: occlusion map, bit2: emissive map, bit3: double-sided, bit4: alphaMode MASK, bit5: alphaMode BLEND
};
layout(set = 0, binding = 5) readonly buffer Materials { Material materials[]; };

// Push constant block: must mirror pbr.vert and material_push_constants in the runtime
// (seven uint fields first, then the aligned mat4) so member offsets agree across stages and
// with the CPU writes. This fragment stage only reads material_index; the remaining fields
// exist to keep the block layout identical.
layout(push_constant) uniform PushConstants {
    uint material_index; // index into the material table (material data lives on the GPU)
    uint flags;          // bit0: instanced draw -> model from instances[...]; bit3: double-sided
    uint skin_base;      // start of this primitive's joint block in skins.matrices (0 = identity)
    uint morph_base;     // float index of this primitive's morph block in morph_data.morphs (0 = none)
    uint morph_targets;  // number of morph targets (0 = not morphable)
    uint morph_vertices; // vertex count of this primitive (morph block stride)
    uint instance_base;  // mat4 start of this instanced primitive's transforms (unused here)
    mat4 model;          // per-model world transform (kept out of the shared camera UBO; unused here)
} push;

// Light UBO (scene set binding 7): the orthographic light view-proj (world -> shadow map) and
// the light direction, followed by the active punctual lights. The direction is filled by the
// CPU (make_directional_light_ubo) and matches the sky sun, so the direct light, the visible
// sun disc and the shadows all agree. Layout must match vulkan::light_ubo in primitive.cppm
// (std140): mat4 | vec4 | 4 floats | uint + 3 pad floats | PunctualLight[4] - the CPU mirrors
// the "uint + pad" slot with one glm::vec4, so the array starts at byte 112 and the block is
// 368 bytes. (A vec3 pad would force 16-byte alignment to 128 and shift every light by 16.)
const int MAX_PUNCTUAL_LIGHTS = 4; // vulkan::max_punctual_lights

struct PunctualLight {
    vec4 position; // xyz: world position (w unused)
    vec4 color;    // xyz: linear color * intensity (w unused)
    vec4 spot_dir; // xyz: spot axis, normalized for spot lights (w unused)
    vec4 params;   // x = range (0 = infinite), y = 0 point / 1 spot, z = cos(outer cone), w = cos(inner cone, spot only)
};

layout(set = 0, binding = 7) uniform LightUBO {
    mat4 light_view_proj;
    vec4 light_dir; // xyz: normalized light direction, w: 1 / shadow map size (uv texel)
    float shadow_enabled; // 1.0 = sample shadow map, 0.0 = fully lit (runtime::set_shadow_enabled)
    // selectable BRDF theory models (gui combos -> runtime::set_brdf_model / set_diffuse_model,
    // CPU-side, riding the std140 padding of this block):
    //   brdf_model:    0 = GGX + joint Smith (default), 1 = GGX + height-correlated Smith,
    //                  2 = Beckmann + Smith, 3 = Blinn-Phong + Smith
    //   diffuse_model: 0 = Lambert (default), 1 = Oren-Nayar
    float brdf_model;
    float diffuse_model;
    float shadow_texel_world; // world size of one shadow-map texel (normal-offset bias)
    uint light_count;
    float exposure; // y lane of the CPU's light_count vec4: linear exposure scale (pre-tonemap)
    float toon_steps;   // cel-shading quantization steps (LightUBO.light_count.z; 0 = PBR)
    float toon_softness; // band edge width in normalized [0,1] space (LightUBO.light_count.w)
    PunctualLight punctual_lights[MAX_PUNCTUAL_LIGHTS];
} light;

// Shadow map (scene set binding 8): depth-compare sampler (sampler2DShadow) with LINEAR
// filtering - one texture() call performs HARDWARE percentage-closer filtering: the hardware
// compares the reference depth against the 2x2 texel neighborhood and returns the lit
// fraction (no manual 3x3 loop needed).
layout(set = 0, binding = 8) uniform sampler2DShadow shadow_map;

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
 * @brief shadow factor for a world-space point: normal-offset bias + 3x3 PCF
 * @param world_pos receiver position in world space
 * @param normal receiver world-space normal
 * @return 1.0 = fully lit, 0.0 = fully shadowed (also passed through toon_band for cel shading)
 *
 * - the sample point is pushed along the world normal by a couple of light-space texels, which
 *   removes most quantization acne on flat receivers WITHOUT a large depth bias - and a large depth
 *   bias is exactly what erases the shadow of a thin caster (a sword, a railing). The normal offset
 *   lets the depth bias below drop to a fraction of the old 0.0015.
 * - the 3x3 grid of hardware 2x2 comparison taps (4x4 texel footprint) smooths the edge; a single
 *   tap flickered badly on thin geometry. The grid is unrotated on purpose: a rotated grid needs TAA
 *   to hide its per-pixel noise, and this renderer has no TAA yet.
 * - outside the light frustum the fragment is reported lit: the map covers the fitted box only, and
 *   runtime::update_shadow_frustum() keeps that box on the part of the scene the camera can see.
 */
float calc_shadow(vec3 world_pos, vec3 normal) {
    float texel_uv = light.light_dir.w;            // 1 / shadow map size
    float texel_world = light.shadow_texel_world;  // world size of one texel

    // normal offset: shift the world position before projecting it into light space
    vec3 offset_pos = world_pos + normal * (texel_world * 2.0);
    vec4 light_clip = light.light_view_proj * vec4(offset_pos, 1.0);
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
            lit += texture(shadow_map, vec3(tap, current_depth - bias));
        }
    }
    float shadow = lit / 9.0;
    // cel shading hardens the shadow edge into the same bands as the diffuse falloff
    return toon_band(shadow, light.toon_steps, light.toon_softness);
}
const float PI = 3.14159265359;

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
 * @note these presets drive the DIRECT lights only. The IBL ambient in main() always uses the fixed
 *       GGX model (prefiltered GGX environment + GGX BRDF LUT with Fdez-Aguera multiscatter
 *       compensation, Lambert diffuse irradiance) - the GUI preset switch is an honest
 *       direct-light A/B, not a whole-scene model comparison.
 */
vec3 evaluate_direct_light(vec3 n, vec3 v, vec3 base_color, float metallic, float roughness, vec3 f0, vec3 light_dir, vec3 light_radiance) {
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
    const int brdf_model = int(light.brdf_model + 0.5);
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
    float ndotl = max(dot(n, l), 0.0);

    // ---- cel/toon shading (no-op when LightUBO.toon_steps < 1): quantize the diffuse falloff
    //      into bands and turn the specular lobe into a single hard highlight block. The
    //      visibility terms keep their unquantized ndotl, so only the shading response bands -
    //      the silhouette stays smooth.
    if (light.toon_steps > 0.5) {
        ndotl = toon_band(ndotl, light.toon_steps, light.toon_softness);
        float ndoth = max(dot(n, h), 0.0);
        float highlight_threshold = 0.5 + 0.5 * (1.0 - roughness); // smooth surfaces -> tighter highlight
        specular *= smoothstep(highlight_threshold - light.toon_softness, highlight_threshold + light.toon_softness, ndoth);
    }

    // ---- Diffuse by the selected model (LightUBO.diffuse_model): Lambert (default) or the
    //      roughness-dependent Oren-Nayar approximation (0 -> Lambert).
    vec3 diffuse;
    if (int(light.diffuse_model + 0.5) == 1) {
        diffuse = kd * base_color * oren_nayar_diffuse(n, v, l, roughness, ndotv, ndotl);
    } else {
        diffuse = kd * base_color / PI;
    }
    return (diffuse + specular) * (light_radiance * ndotl);
}

/// @brief IBL: split-sum approximation (ported from glTF-Sample-Renderer's ibl.glsl); the ambient
///        always uses this fixed GGX model, independent of the BRDF presets above
/// Diffuse ambient: irradiance map lookup by the world normal
vec3 get_diffuse_light(vec3 n) {
    return texture(irradiance_sampler, n).rgb;
}

/// Specular ambient: prefiltered environment sampled at the roughness-derived mip level
vec3 get_specular_sample(vec3 reflection, float lod) {
    return textureLod(env_sampler, reflection, lod).rgb;
}

/// @brief Single-scatter plus multi-scatter-compensated Fresnel weights from the BRDF LUT
///        (Fdez-Aguera); @p specular_weight is the material's monochrome specular amount
vec3 get_ibl_ggx_fresnel(vec3 n, vec3 v, float roughness, vec3 f0, float specular_weight) {
    float ndotv = clamp(dot(n, v), 0.0, 1.0);
    vec2 brdf_sample_point = clamp(vec2(ndotv, roughness), vec2(0.0), vec2(1.0));
    vec2 f_ab = texture(brdf_lut_sampler, brdf_sample_point).rg;
    vec3 fr = max(vec3(1.0 - roughness), f0) - f0;
    vec3 k_s = f0 + fr * pow(1.0 - ndotv, 5.0);
    vec3 fssess = specular_weight * (k_s * f_ab.x + f_ab.y);

    float ems = 1.0 - (f_ab.x + f_ab.y);
    vec3 f_avg = specular_weight * (f0 + (1.0 - f0) / 21.0);
    vec3 fmsems = ems * fssess * f_avg / (1.0 - f_avg * ems);

    return fssess + fmsems;
}

/// @brief Prefiltered GGX environment radiance for a reflection ray, at the mip that matches
///        @p roughness; the level count is queried from the sampler so it always matches whatever
///        env_mip_count the CPU baked (no hardcoded constant)
vec3 get_ibl_radiance_ggx(vec3 n, vec3 v, float roughness) {
    // roughness -> lod across the prefiltered chain; the level count is queried from the
    // sampler so it always matches whatever env_mip_count the CPU baked (no hardcoded constant)
    const float lod = roughness * float(max(textureQueryLevels(env_sampler) - 1, 0));
    vec3 reflection = normalize(reflect(-v, n));
    return get_specular_sample(reflection, lod);
}

/// @brief ACES filmic tonemapping (kept here for the unlit/debug paths; the display-referred work
///        for the frame happens in post.frag)
vec3 aces_tone_mapping(vec3 color) {
    return clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
}

/**
 * @brief shade one fragment: material, normal mapping, sun + shadow, punctual lights, IBL, emissive
 *
 * Output is linear HDR radiance - no exposure, no gamma. The per-frame model selections come from
 * the LightUBO lanes (brdf_model, diffuse_model, toon_steps/toon_softness, exposure is applied in
 * post.frag). The shadow factor attenuates the directional sun only; the IBL ambient stays
 * unshadowed, which is the usual approximation and keeps interiors from going pitch black.
 */
void main() {
    // ---- Material: one GPU-side record (texture indices + factors + flags) ----
    Material mat = materials[push.material_index];

    // ---- Material parameters: factor * texture (indices come from the material record) ----
    vec4 base_color = mat.base_color_factor * texture(textures[mat.tex_indices.x], v_uv);
    // alphaMode MASK (record flag bit4): discard fragments below the cutoff (base_color.a is
    // factor.a * albedo.a) - glTF alphaCutoff semantics
    if ((mat.flags & 16u) != 0u && base_color.a < mat.alpha_cutoff) {
        discard;
    }
    float metallic = mat.metallic_factor * texture(textures[mat.tex_indices.y], v_uv).b;
    float roughness = mat.roughness_factor * texture(textures[mat.tex_indices.y], v_uv).g;
    // occlusion: sampled AO modulated by occlusion_strength; without an occlusion map the slot
    // is the white fallback (ao = 1) and the strength has no effect
    float ao = mix(1.0, texture(textures[mat.tex_indices.w], v_uv).r, mat.occlusion_strength);
    vec3 emissive = mat.emissive_factor.rgb * texture(textures[mat.emissive_index], v_uv).rgb;

    // ---- Normal: optional tangent-space normal map, else interpolated normal ----
    // The TBN frame is derived from screen-space derivatives of the world position and the UVs,
    // so mirrored UV layouts (glTF TANGENT.w = -1) are handled implicitly - no per-vertex
    // tangent sign is needed and the vertex TANGENT attribute is not consumed here.
    vec3 n;
    if ((mat.flags & 1u) != 0u) {
        const vec3 dp1 = dFdx(v_world_pos);
        const vec3 dp2 = dFdy(v_world_pos);
        const vec2 duv1 = dFdx(v_uv);
        const vec2 duv2 = dFdy(v_uv);
        vec3 normal = normalize(v_normal);
        const float denom = duv1.x * duv2.y - duv2.x * duv1.y;
        if (abs(denom) < 1e-8) {
            n = normal; // degenerate UV derivatives: fall back to the interpolated normal
        } else {
            const vec3 sdir = (duv2.y * dp1 - duv1.y * dp2) / denom; // world tangent direction
            const vec3 tdir = (duv1.x * dp2 - duv2.x * dp1) / denom; // world bitangent direction
            vec3 tbn_normal = texture(textures[mat.tex_indices.z], v_uv).rgb * 2.0 - 1.0;
            tbn_normal.xy *= mat.normal_scale;
            tbn_normal = normalize(tbn_normal);
            n = normalize(mat3(normalize(sdir), normalize(tdir), normal) * tbn_normal);
        }
    } else {
        n = normalize(v_normal);
    }
    // double-sided material (record flag bit3): mirror the normal on back faces, as the glTF
    // spec requires, so the inner side of a shell is lit by its inward-facing normal
    if ((mat.flags & 8u) != 0u && !gl_FrontFacing) {
        n = -n;
    }

    // ---- Direct light: the directional sun + the active punctual lights, all through the
    //      shared evaluate_direct_light() (same BRDF theory path, so they agree visually).
    vec3 v = normalize(camera.camera_pos - v_world_pos);
    vec3 f0 = mix(vec3(0.04), base_color.rgb, metallic);

    vec3 direct = vec3(0.0);
    // directional sun: shadow factor attenuates only this light; IBL ambient stays unshadowed
    {
        float shadow = (light.shadow_enabled > 0.5) ? calc_shadow(v_world_pos, n) : 1.0;
        direct += evaluate_direct_light(n, v, base_color.rgb, metallic, roughness, f0, light.light_dir.xyz, vec3(7.5) * shadow);
    }
    // punctual lights (point/spot, no shadow casting in this version): inverse-square falloff
    // (well-behaved at zero distance) with an optional smooth range cutoff; spots add a soft
    // cone mask between the inner and outer half-angles
    for (int i = 0; i < MAX_PUNCTUAL_LIGHTS; ++i) {
        if (i >= int(light.light_count)) {
            break;
        }
        const PunctualLight pl = light.punctual_lights[i];
        vec3 to_light = pl.position.xyz - v_world_pos;
        const float dist = length(to_light);
        const vec3 dir = dist > 1e-6 ? to_light / dist : vec3(0.0, 1.0, 0.0);
        vec3 radiance = pl.color.xyz / (1.0 + dist * dist);
        const float range = pl.params.x;
        if (range > 0.0) {
            // smooth range cutoff (no hard pop at the boundary)
            const float d = dist / range;
            const float fade = clamp(1.0 - d * d, 0.0, 1.0);
            radiance *= fade * fade;
        }
        if (pl.params.y > 0.5) { // spot light: cone around spot_dir
            const float outer = pl.params.z;
            // inner cone: cos of the inner half-angle (glTF KHR innerConeAngle, set by the CPU);
            // w == 0 means the light did not specify one -> legacy soft-inner mix(outer, 1, 0.6)
            const float inner = pl.params.w > 0.0 ? pl.params.w : mix(outer, 1.0, 0.6);
            const float cone = smoothstep(outer, inner, dot(-dir, normalize(pl.spot_dir.xyz)));
            radiance *= cone;
        }
        if (radiance != vec3(0.0)) {
            direct += evaluate_direct_light(n, v, base_color.rgb, metallic, roughness, f0, dir, radiance);
        }
    }

    // ---- IBL (split-sum): diffuse irradiance + prefiltered specular ----
    vec3 ibl_diffuse = get_diffuse_light(n);
    vec3 ibl_specular = get_ibl_radiance_ggx(n, v, roughness);
    vec3 fresnel_ibl = get_ibl_ggx_fresnel(n, v, roughness, f0, 1.0);

    // Metals have no diffuse term: diffuse ambient is scaled by (1 - metallic),
    // metal color comes entirely from specular environment (matches the official mix(dielectric, metal, metallic))
    vec3 ambient = ibl_diffuse * base_color.rgb * ao * (1.0 - metallic);
    vec3 specular_ibl = ibl_specular * fresnel_ibl * ao;

    vec3 color = ambient + direct + specular_ibl + emissive;

    // The scene target is HDR: this forward pass writes linear radiance. Exposure, ACES
    // tonemapping and gamma now happen once in the post-process pass (post.frag), which also
    // gives the bloom chain a linear image to work on.
    color = max(color, vec3(0.0));

    // glTF alpha semantics: only alphaMode BLEND materials carry real coverage in the output
    // alpha. OPAQUE and MASK outputs must write alpha = 1 (their base_color.a / albedo alpha
    // is ignored by the spec), otherwise the always-on blending below would make e.g. an
    // albedo texture with an alpha channel unexpectedly translucent. BLEND keeps base_color.a.
    float out_alpha = ((mat.flags & 32u) != 0u) ? base_color.a : 1.0;
    out_color = vec4(color, out_alpha);
}
