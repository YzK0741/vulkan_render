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
// material record. HEAP-NATIVE (see docs/descriptor_heap_migration.md): the array IS the heap - one 64 B slot per
// texture, starting at heap_slots_textures - and the sampler is SEPARATE, because a combined image sampler cannot
// be declared this way at all: every fetch constructs one, `sampler2D(heap_textures[i], heap_samplers[s])`.
#include "heap_slots.glsl"

// One entry of the material table; layout matches material_record in vulkan/primitive.cppm
// (std430, 96 bytes). Field order and the flag bits are a CPU/GPU contract - see register_material.
struct Material {
    uvec4 tex_indices; // albedo, metallic-roughness, normal, occlusion (indices into textures[])
    uint emissive_index;
    float alpha_cutoff;       // alphaMode MASK threshold
    float occlusion_strength; // mix(1, sampled AO, strength)
    uint sphere_index; // MMD sphere map: the texture it was combined from (0 = none); flags bits 8-9 hold the mode
    vec4 base_color_factor;
    vec4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    uint flags; // bit0: normal map, bit1: occlusion map, bit2: emissive map, bit3: double-sided,
                // bit4: alphaMode MASK, bit5: alphaMode BLEND, bit6: painted, bit7: face,
                // bits 8-9: MMD sphere mode, bit10: MMD edge authored
    // MMD's own outline inputs, from the glTF material's `extras` (see docs/zzz_shading.md): xyz is the
    // line colour the model authored, w its thickness multiplier, and flags bit6 says whether the model
    // authored an edge AT ALL - required, because black is a legitimate edge colour and 0 a legitimate
    // size, so the values alone cannot say "absent". APPENDED, so every field above keeps its offset; and
    // declared in EVERY copy of this record, because a storage buffer's array stride is the struct's own
    // size - a copy that omits it indexes the table at the wrong pitch.
    vec4 npr_edge;
};
// The ARRAY name carries the HEAP slot and the block member carries the record index: two index spaces, which is
// why a lookup is `heap_material_tables[heap_slots_materials].materials[push.material_index]`.
layout(descriptor_heap, descriptor_stride = heap_slot_stride) readonly buffer Materials { Material materials[]; } heap_material_tables[];

/**
 * @brief one fetch from the bindless array, with the sampler the descriptor-set path used to bind for it
 * @param texture_index the texture's own index (its slot is heap_slots_textures + this)
 * @param uv the coordinate
 * @note THE TWO HALVES OF A FETCH ARE SEPARATE IN A HEAP: the image comes from the resource heap at its slot, the
 *       sampler from the SAMPLER heap at heap_sampler_texture (linear, repeat, mips - exactly what the set path
 *       bound), and a combined image sampler cannot be declared at all, so every fetch says both.
 */
vec4 heap_sample(uint texture_index, vec2 uv) {
    return texture(sampler2D(heap_textures[heap_slots_textures + texture_index], heap_samplers[heap_sampler_texture]), uv);
}

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
    // THE HEAP INDICES (see heap_slots.glsl), shared by every stage that includes this file: which frame slot and
    // which swapchain image it runs for. LAST on purpose, so every field above keeps its offset, and delivered
    // through vkCmdPushDataEXT because a heap pipeline has no layout to hold push constants.
    uint frame_slot;
    uint image_index;
} push;

// ... and the two names the shared slot macros use (heap_slots.glsl says why these are macros and not constants).
#define heap_frame_slot (push.frame_slot)
#define heap_image_index (push.image_index)

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
    float face_mask; // 1 for the model's FACE block: it shades from its own flattened normal below, and
                     // the lighting gives it a lighter cast shadow (see shade_surface)
    vec3 sphere_sample; // the material's MMD sphere/matcap lookup, or 0 when it has none: the reference's
                        // Matcap combine needs the SAMPLE and the light factor together, so the lookup is
                        // kept here rather than only folded into the albedo (see reference_matcap_combine)
    float painted_mask; // 1 when this material is PAINTED (the converter's mmd_unlit): drawn from its albedo,
                        // not from the lighting stack. The face block is, and so are the head's props
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
surface_sample gather_surface(vec3 world_pos, vec3 geo_normal, vec2 uv, vec3 view_normal, float nose_strength) {
    Material mat = heap_material_tables[heap_slots_materials].materials[push.material_index];

    surface_sample s;
    const vec4 base_color = mat.base_color_factor * heap_sample(mat.tex_indices.x, uv);
    // alphaMode MASK (record flag bit4): discard fragments below the cutoff (base_color.a is
    // factor.a * albedo.a) - glTF alphaCutoff semantics
    if ((mat.flags & 16u) != 0u && base_color.a < mat.alpha_cutoff) {
        discard;
    }

    s.albedo = base_color.rgb;
    s.alpha = base_color.a;
    s.metallic = mat.metallic_factor * heap_sample(mat.tex_indices.y, uv).b;
    s.roughness = mat.roughness_factor * heap_sample(mat.tex_indices.y, uv).g;
    // occlusion: sampled AO modulated by occlusion_strength; without an occlusion map the slot is
    // the white fallback (ao = 1) and the strength has no effect
    s.ao = mix(1.0, heap_sample(mat.tex_indices.w, uv).r, mat.occlusion_strength);
    s.emissive = mat.emissive_factor.rgb * heap_sample(mat.emissive_index, uv).rgb;
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
            vec3 tbn_normal = heap_sample(mat.tex_indices.z, uv).rgb * 2.0 - 1.0;
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
    // NOTE: the face's flattened SHADING normal is NOT applied here, and that is a fix rather than an
    // omission. This pass writes the G-buffer, whose normal is read by the SHADOW lookup, by SSAO and by
    // the ray-traced passes - so a normal bent towards the eye makes all of them wrong: measured on a
    // turned head, the face went grey because the screen-space occlusion was computed against a normal that
    // pointed at the camera. The flattening happens in shade_surface instead, where a normal is a shading
    // input and nothing else.

    // ---- MMD's SPHERE map (see docs/zzz_shading.md), which is where a model like this gets its saturation:
    //      the diffuse texture is authored pale and the sphere map is combined on top of it. The lookup is
    //      matcap-shaped - the VIEW-space normal's xy, remapped to [0,1] - and the mode is the record's
    //      flags bits 7-8: 1 multiply, 2 add. Mode 3 (sub-texture, which samples the model's extra UV sets
    //      instead) is NOT implemented; no material in the asset this was built for uses it, and pretending
    //      otherwise would sample a texture with the wrong coordinates.
    const uint sphere_mode = (mat.flags >> 8u) & 3u;
    s.painted_mask = (mat.flags & 64u) != 0u ? 1.0 : 0.0; // record bit6
    s.face_mask = (mat.flags & 128u) != 0u ? 1.0 : 0.0;    // record bit7
    s.sphere_sample = vec3(0.0);
    if (sphere_mode == 1u || sphere_mode == 2u) {
        const vec3 sphere_normal = normalize(view_normal);
        // MMD's sphere textures are stored with the opposite vertical convention to glTF's UV origin, which
        // a matcap lookup makes look like a flipped gradient rather than an obvious error.
        const vec2 sphere_uv = vec2(sphere_normal.x * 0.5 + 0.5, 1.0 - (sphere_normal.y * 0.5 + 0.5));
        const vec3 sphere = heap_sample(mat.sphere_index, sphere_uv).rgb;
        s.sphere_sample = sphere;
        s.albedo = sphere_mode == 1u ? s.albedo * sphere : s.albedo + sphere;
    }
    // ---- THE NOSE MARK, redrawn at a FIXED size in UV space. The texture paints one - a ~10x20 texel dot
    // at uv (0.5000, 0.5073) in this model's 2048 atlas - but this render puts the whole face into about a
    // hundred pixels, so a texel-true mark is sub-pixel and vanishes. The artist's mark is what makes a
    // face read as anime, so it is redrawn here as an ellipse at the painted mark's own position and shape:
    // a model-specific constant, which is exactly what the reference's per-model face light map carries.
    //
    // IT LIVES IN THIS PASS, NOT IN THE LIGHTING, and that is not a style choice: the mark needs the
    // SURFACE uv, and the deferred path's lighting stage only has a screen uv - the G-buffer carries no
    // per-surface coordinates. Its first version sat in shade_surface and drew its ellipse at the middle of
    // the FRAME, on fragments nothing had marked as face: measured, a strength sweep from 1.0 to 0.0 moved
    // 392 pixels, which is the run-to-run jitter. Here both the forward and the deferred path get it.
    if ((mat.flags & 128u) != 0u && nose_strength > 0.0) { // record bit7: the FACE block alone

        const vec2 nose_uv = vec2(0.5000, 0.5073);
        // SIZED IN PIXELS, NOT IN UV, and that is the correction a measurement forced: the painted mark is
        // 10x20 texels of a 2048 atlas, i.e. 0.5% of the face, and this render puts the face into about a
        // hundred pixels - so a uv-space ellipse of the painted size covers 0.38 of ONE pixel and moved
        // nothing (a whole-face probe proved the path live, a strength sweep of the real size proved it
        // invisible). fwidth(uv) is how many uv units one pixel spans, so the mark keeps a constant SIZE ON
        // SCREEN at any zoom, which is what line art does.
        const vec2 uv_per_pixel = max(fwidth(uv), vec2(1e-6));
        const vec2 nose_radius = uv_per_pixel * 2.0; // about a 4-pixel-wide mark
        const float nose_d = length((uv - nose_uv) / nose_radius);
        s.albedo *= mix(1.0, mix(1.0, 0.12, smoothstep(1.0, 0.35, nose_d)), nose_strength);
    }
    return s;
}

#endif // VULKAN_RENDER_SURFACE_GLSL
