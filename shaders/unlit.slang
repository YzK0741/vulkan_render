// shaders/unlit.slang - the SLANG source of the unlit fragment stage.
//
// PRODUCES `unlit.frag.spv`, THE SAME NAME THE GLSL VERSION PRODUCED, so the host side (which loads
// <name>.spv by file name) needs no change at all. That is the acceptance for the whole migration: the
// capture gate must stay byte-identical, and the gate's `unlit` scenario covers this stage.
//
// Ported from shaders/unlit.frag (now in glsl.old/ once the migration finishes). The GLSL's model is
// "the array IS the heap, indexed by an absolute slot"; Slang's DescriptorHandle is a handle VALUE, so
// every fetch here builds one from a literal slot - see docs/slang_migration.md, which records why a
// handle ARRAY (the obvious port) is wrong: Slang would wrap it in a synthesized descriptor set, and a
// heap-native pipeline has no descriptor sets at all.
//
// The grid's numbers come from the SHARED constants file the GLSL side includes too, so there is one
// source of truth for the slots (tests/test_render_resources.cpp compares it against core::heap_slots).

#include "heap_slot_constants.glsl"

// One entry of the material table; layout matches material_record (std430, 80 bytes) - must stay identical
// to the GLSL Material and to the host's struct.
struct Material
{
    uint4 tex_indices; // albedo, metallic-roughness, normal, occlusion (indices into the texture heap)
    uint emissive_index;
    float alpha_cutoff;       // alphaMode MASK threshold
    float occlusion_strength; // mix(1, sampled AO, strength)
    uint pad_;
    float4 base_color_factor;
    float4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    uint flags; // bit0: normal map, bit1: occlusion map, bit2: emissive map, bit3: double-sided, bit4: alphaMode MASK, bit5: alphaMode BLEND
};

// Push constant block: identical layout to pbr.vert / pbr.frag (only material_index is read here).
// -fvk-use-gl-layout gives std430, which is what the GLSL block and the CPU struct use.
struct PushConstants
{
    uint material_index;
    uint flags;
    uint skin_base;
    uint morph_base;
    uint morph_targets;
    uint morph_vertices;
    uint instance_base;
    float4x4 model;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> push;

// A handle for an absolute slot: `.x` is the index into the resource heap, or - for a SamplerState - into
// the sampler heap (Slang uses the same component for both; only a CombinedTextureSampler uses `.y`).
DescriptorHandle<T> heap_at<T>(uint slot) where T : IOpaqueDescriptor
{
    return (DescriptorHandle<T>)uint2(slot, 0u);
}

// The GLSL's `heap_texel(tex, sampler_slot, uv)`: one texel through the sampler heap entry named. Written
// as a function here rather than a macro because Slang has no such restriction - the GLSL's macro exists
// because a `texture2D` PARAMETER does not survive a function boundary in SPIR-V (see heap_slots.glsl).
float4 heap_texel(DescriptorHandle<Texture2D<float4>> texture, uint sampler_slot, float2 uv)
{
    return texture.Sample(heap_at<SamplerState>(sampler_slot), uv);
}

[shader("fragment")]
float4 main([[vk::location(2)]] float2 v_uv : TEXCOORD0) : SV_Target
{
    Material mat = heap_at<StructuredBuffer<Material>>(heap_slots_materials)[push.material_index];
    float4 base_color = mat.base_color_factor * heap_texel(heap_at<Texture2D<float4>>(heap_slots_textures + mat.tex_indices.x), heap_sampler_texture, v_uv);
    if ((mat.flags & 16u) != 0u && base_color.a < mat.alpha_cutoff)
    {
        discard;
    }
    float out_alpha = ((mat.flags & 32u) != 0u) ? base_color.a : 1.0;
    return float4(base_color.rgb, out_alpha);
}
