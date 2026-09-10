/**
 * @file shaders/surface.glsl
 * @brief Shared material-surface gathering: the material table lookup, the glTF alpha tests, the
 *        tangent-space normal mapping and the texture-derived factors.
 * @ingroup shaders
 *
 * Included by both fragment stages that shade a scene surface - pbr.frag (the forward path) and
 * gbuffer.frag (the deferred path's G-buffer write). Both need exactly the same answers to "what
 * is this surface made of", and it is the part of the shading chain where a divergence would be
 * least visible and most confusing (a deferred frame whose albedo disagreed with the forward
 * frame's). Keeping the fetch in one place also keeps the glTF semantics (MASK discard, the
 * double-sided normal flip, occlusion_strength) in one place.
 *
 * What stays OUT of here: the lighting itself (the forward path evaluates it in pbr.frag; the
 * deferred path evaluates it in the lighting pass from what the G-buffer stored), the camera /
 * light / shadow bindings, and anything that depends on the pass.
 *
 * The descriptor bindings declared here (1 = the texture array, 5 = the material table) and the
 * push constant block are the shared scene set convention - see docs/shaders.md. A shader that
 * includes this file must NOT declare them again, and a host that creates a pipeline from such a
 * shader must use the runtime's shared scene pipeline layout (it does: every scene pipeline shares
 * vulkan::core::scene_pipeline_layout).
 */

#ifndef VULKAN_RENDER_SURFACE_GLSL
#define VULKAN_RENDER_SURFACE_GLSL

// The runtime's texture array: every material's image lives in one bindless array, indexed by the
// material record (descriptor indexing: partially bound + non-uniform index).
layout(set = 0, binding = 1) uniform sampler2D textures[];

// One entry of the material table; layout matches material_record in vulkan/primitive.cppm
// (std430, 80 bytes). Field order and the flag bits are a CPU/GPU contract - see register_material.
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
    uint flags; // bit0: normal map, bit1: occlusion map, bit2: emissive map, bit3: double-sided,
                // bit4: alphaMode MASK, bit5: alphaMode BLEND
};
layout(set = 0, binding = 5) readonly buffer Materials { Material materials[]; };

// Push constant block: must mirror the vertex stages and material_push_constants in the runtime
// (seven uint fields first, then the aligned mat4) so member offsets agree across stages and with
// the CPU writes. A fragment stage typically reads material_index and flags only; the other fields
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

/**
 * @brief everything the lighting code needs to know about one surface point
 * @note the names mirror the glTF metallic-roughness material model so the fields map 1:1 onto the
 *       spec (albedo = baseColor, ao = occlusion, ...)
 */
struct surface_sample {
    vec3 albedo;    // base color: base_color_factor * albedo texture (linear, NOT premultiplied)
    float alpha;    // coverage: base_color_factor.a * albedo.a (only meaningful for MASK/BLEND)
    vec3 normal;    // shading normal in world space: normal-mapped, double-sided flipped
    vec3 emissive;  // emissive_factor * emissive texture (linear HDR-ish radiance)
    float roughness; // roughness_factor * metallic-roughness texture .g
    float metallic;  // metallic_factor * metallic-roughness texture .b
    float ao;        // mix(1, occlusion texture .r, occlusion_strength)
    uint flags;      // the material record's flag bits (see Material)
};

/**
 * @brief gather the surface properties of one fragment from the material table and the texture array
 * @param world_pos interpolated world position (needed for the screen-space TBN derivatives)
 * @param geo_normal interpolated geometric normal (used when the material has no normal map)
 * @param uv interpolated texture coordinates
 * @return the surface sample of this fragment (never for a discarded fragment: see below)
 * @note CALLS @c discard FOR alphaMode MASK MATERIALS whose alpha is below the record's
 *       alphaCutoff. That is the glTF rule and it belongs here rather than at the call sites: every
 *       pass that shades a surface (forward, G-buffer) must cut the same fragments out, or the
 *       shadow/deferred/forward views of one object disagree. The caller sees a "clean" surface.
 * @note the TBN frame comes from screen-space derivatives of the world position and the UVs, so
 *       mirrored UV layouts (glTF TANGENT.w = -1) are handled implicitly and the vertex TANGENT
 *       attribute is never consumed; a degenerate UV derivative falls back to the fine normal.
 */
surface_sample gather_surface(vec3 world_pos, vec3 geo_normal, vec2 uv) {
    Material mat = materials[push.material_index];

    surface_sample s;
    const vec4 base_color = mat.base_color_factor * texture(textures[mat.tex_indices.x], uv);
    // alphaMode MASK (record flag bit4): discard fragments below the cutoff (base_color.a is
    // factor.a * albedo.a) - glTF alphaCutoff semantics
    if ((mat.flags & 16u) != 0u && base_color.a < mat.alpha_cutoff) {
        discard;
    }

    s.albedo = base_color.rgb;
    s.alpha = base_color.a;
    s.metallic = mat.metallic_factor * texture(textures[mat.tex_indices.y], uv).b;
    s.roughness = mat.roughness_factor * texture(textures[mat.tex_indices.y], uv).g;
    // occlusion: sampled AO modulated by occlusion_strength; without an occlusion map the slot is
    // the white fallback (ao = 1) and the strength has no effect
    s.ao = mix(1.0, texture(textures[mat.tex_indices.w], uv).r, mat.occlusion_strength);
    s.emissive = mat.emissive_factor.rgb * texture(textures[mat.emissive_index], uv).rgb;
    s.flags = mat.flags;

    // ---- normal: optional tangent-space normal map, else the interpolated normal ----
    if ((mat.flags & 1u) != 0u) {
        const vec3 dp1 = dFdx(world_pos);
        const vec3 dp2 = dFdy(world_pos);
        const vec2 duv1 = dFdx(uv);
        const vec2 duv2 = dFdy(uv);
        const vec3 normal = normalize(geo_normal);
        const float denom = duv1.x * duv2.y - duv2.x * duv1.y;
        if (abs(denom) < 1e-8) {
            s.normal = normal; // degenerate UV derivatives: fall back to the interpolated normal
        } else {
            const vec3 sdir = (duv2.y * dp1 - duv1.y * dp2) / denom; // world tangent direction
            const vec3 tdir = (duv1.x * dp2 - duv2.x * dp1) / denom; // world bitangent direction
            vec3 tbn_normal = texture(textures[mat.tex_indices.z], uv).rgb * 2.0 - 1.0;
            tbn_normal.xy *= mat.normal_scale;
            tbn_normal = normalize(tbn_normal);
            s.normal = normalize(mat3(normalize(sdir), normalize(tdir), normal) * tbn_normal);
        }
    } else {
        s.normal = normalize(geo_normal);
    }
    // double-sided material (record flag bit3): mirror the normal on back faces, as the glTF spec
    // requires, so the inner side of a shell is lit by its inward-facing normal
    if ((mat.flags & 8u) != 0u && !gl_FrontFacing) {
        s.normal = -s.normal;
    }
    return s;
}

#endif // VULKAN_RENDER_SURFACE_GLSL
