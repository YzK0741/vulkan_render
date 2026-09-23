// shaders/shadow.slang - the SLANG source of the depth-only shadow fragment stage.
//
// PRODUCES `shadow.frag.spv`, the same name the GLSL version produced, so the host side is untouched.
// Ported from shaders/shadow.frag (glsl.old/ once the migration finishes).
//
// The stage has no color attachment: rasterization depth is all it writes, which is why the Slang entry
// point returns void and declares only its one varying. What it DOES do is discard below the material's
// alpha cutoff, so a masked caster (foliage, curtains, grates) casts a cut-out shadow.
//
// THIS LEAF SHARES NO BODY and is therefore the shape a stage takes when the shim is the only thing it
// needs: the material table and the albedo fetch go through the shared accessors (`material_at` and
// `heap_sample`), which is exactly the pair that the deferred and forward paths call too, so the shaded
// alpha edge and the shadow edge keep agreeing by construction rather than by a copied expression.
//
// The push-constant block must stay byte-identical to shadow.vert's: one pipeline shares a single
// push-constant range across both stages.

#include "heap_access.slang"

/// One entry of the material table; layout matches material_record in vulkan/model.cppm (std430)
struct Material
{
    uint4 tex_indices;        // albedo, metallic-roughness, normal, occlusion (indices into the texture heap)
    uint emissive_index;
    float alpha_cutoff;       // alphaMode MASK threshold
    float occlusion_strength; // mix(1, sampled AO, strength)
    uint _pad;
    float4 base_color_factor;
    float4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    uint flags; // bit0: normal map, bit1: occlusion map, bit2: emissive map, bit3: double-sided, bit4: alphaMode MASK, bit5: alphaMode BLEND
};

struct ShadowPush
{
    uint material_index; // index into the material table (the mask test's only input)
    uint flags;          // unused here: declared to keep the block layout identical to shadow.vert
    uint skin_base;
    uint morph_base;
    uint morph_targets;
    uint morph_vertices;
    uint instance_base;
    VR_MAT4 model;
};
[[vk::push_constant]] ConstantBuffer<ShadowPush> push;

/// @brief the alphaMode MASK test: discard below the material's alpha cutoff
/// @note cost is one uniform branch per draw plus one texture fetch for the masked materials only: the
///       material record's flag bit4 is a push-constant-decided value, so opaque casters never sample the
///       albedo texture at all.
[shader("fragment")]
void main([[vk::location(0)]] float2 v_uv : TEXCOORD0)
{
    const Material mat = material_at(heap_slots_materials, push.material_index);
    if ((mat.flags & 16u) == 0u)
    {
        return; // OPAQUE / BLEND: nothing to test, so the albedo texture is never sampled
    }
    // same expression as pbr.frag: base_color.a is factor.a * albedo.a
    if (mat.base_color_factor.a * heap_sample(mat.tex_indices.x, v_uv).a < mat.alpha_cutoff)
    {
        discard;
    }
}
