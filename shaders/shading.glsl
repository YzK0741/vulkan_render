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
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    mat4 view_proj_unjittered;
    mat4 prev_view_proj;
} camera;

// Split-sum IBL (bindings 2-4): the prefiltered GGX environment (roughness mip chain), the
// irradiance map for the diffuse ambient, and the BRDF integration LUT.
layout(set = 0, binding = 2) uniform samplerCube env_sampler;        // prefiltered environment (roughness mip chain)
layout(set = 0, binding = 3) uniform samplerCube irradiance_sampler; // irradiance map (diffuse IBL)
layout(set = 0, binding = 4) uniform sampler2D brdf_lut_sampler;     // BRDF integration LUT

// Light UBO (scene set binding 7): the orthographic light view-proj (world -> shadow map) and the
// light direction, followed by the active punctual lights. The direction is filled by the CPU
// (make_directional_light_ubo) and matches the sky sun, so the direct light, the visible sun disc
// and the shadows all agree. Layout must match vulkan::light_ubo in primitive.cppm (std140):
// mat4 | vec4 | 4 floats | uint + 3 pad floats | PunctualLight[4] - the CPU mirrors the "uint + pad"
// slot with one glm::vec4, so the array starts at byte 112 and the block is 368 bytes. (A vec3 pad
// would force 16-byte alignment to 128 and shift every light by 16.)
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

/**
 * @brief everything the shading stage needs to know about one surface point
 * @note the forward path fills this from its interpolated fragment inputs, the deferred path from
 *       the G-buffer texels - which is the whole point: the lighting below cannot tell them apart
 */
struct shade_input {
    vec3 world_pos; // world-space position (view vector + shadow lookup)
    vec3 normal;    // world-space shading normal (normal-mapped, double-sided flipped)
    vec3 albedo;    // base color, linear
    vec3 emissive;  // emissive radiance, linear (added after the lighting)
    float metallic; // 0 = dielectric, 1 = metal
    float roughness; // perceptual roughness
    float ao;       // ambient occlusion: scales the IBL ambient only, never the direct light
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
    const vec3 v = normalize(camera.camera_pos - s.world_pos);
    const vec3 f0 = mix(vec3(0.04), s.albedo, s.metallic);

    vec3 direct = vec3(0.0);
    // directional sun: shadow factor attenuates only this light; IBL ambient stays unshadowed
    {
        float shadow = (light.shadow_enabled > 0.5) ? calc_shadow(s.world_pos, s.normal) : 1.0;
        direct += evaluate_direct_light(s.normal, v, s.albedo, s.metallic, s.roughness, f0, light.light_dir.xyz, vec3(7.5) * shadow);
    }
    // punctual lights (point/spot, no shadow casting in this version): inverse-square falloff
    // (well-behaved at zero distance) with an optional smooth range cutoff; spots add a soft
    // cone mask between the inner and outer half-angles
    for (int i = 0; i < MAX_PUNCTUAL_LIGHTS; ++i) {
        if (i >= int(light.light_count)) {
            break;
        }
        const PunctualLight pl = light.punctual_lights[i];
        vec3 to_light = pl.position.xyz - s.world_pos;
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
            direct += evaluate_direct_light(s.normal, v, s.albedo, s.metallic, s.roughness, f0, dir, radiance);
        }
    }

    // ---- IBL (split-sum): diffuse irradiance + prefiltered specular ----
    vec3 ibl_diffuse = get_diffuse_light(s.normal);
    vec3 ibl_specular = get_ibl_radiance_ggx(s.normal, v, s.roughness);
    vec3 fresnel_ibl = get_ibl_ggx_fresnel(s.normal, v, s.roughness, f0, 1.0);

    // Metals have no diffuse term: diffuse ambient is scaled by (1 - metallic),
    // metal color comes entirely from specular environment (matches the official mix(dielectric, metal, metallic))
    vec3 ambient = ibl_diffuse * s.albedo * s.ao * (1.0 - s.metallic);
    vec3 specular_ibl = ibl_specular * fresnel_ibl * s.ao;

    vec3 color = ambient + direct + specular_ibl + s.emissive;

    // The scene target is HDR: this stage writes linear radiance. Exposure, ACES tonemapping and
    // gamma happen once in the post-process pass (post.frag), which also gives the bloom chain a
    // linear image to work on.
    return max(color, vec3(0.0));
}

#endif // VULKAN_RENDER_SHADING_GLSL
