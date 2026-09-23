// shaders/heap_probe.slang - the SLANG source of the heap-native PROBE pair (a fullscreen triangle and the
// fragment stage that reads the material table out of the heap).
//
// PRODUCES `heap_probe.vert.spv` AND `heap_probe.frag.spv`, the same names the GLSL versions produced, so the
// host side is untouched. Ported from shaders/heap_probe.vert and shaders/heap_probe.frag.
//
// THIS PAIR IS THE ONE GROUP OF SHADERS THAT VERIFIES ITSELF, and that is why it is worth porting even though
// no capture-gate scenario renders it: the runtime dispatches it during initialization, reads a pixel back, and
// logs the result against a value it knows in advance (element 0 of the material table is the white default
// material this renderer reserves). The verification of this port is therefore a LOG COMPARISON, not a frame:
// the two `descriptor heap: the heap-native GRAPHICS probe rendered grid slot ... read back rgba ...` lines
// must read exactly as they did with the GLSL builds - including the deliberately WRONG slot, which is what
// makes the probe able to fail at all.
//
// There is deliberately NO vertex buffer: what the probe tests is the FRAGMENT stage's access to the heap, and
// a vertex buffer would add a second thing that could be wrong.

#include "heap_access.slang"

// The material record, under the name the shim's accessor expects: the GLSL file calls its copy ProbeMaterial
// (it declares its own block), while `material_at` in heap_access.slang is defined over `Material`. The LAYOUT
// is the contract and it is identical in both - byte-identical to material_record in vulkan/primitive.cppm.
struct Material
{
    uint4 tex_indices;
    uint emissive_index;
    float alpha_cutoff;
    float occlusion_strength;
    uint _pad;
    float4 base_color_factor;
    float4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    uint flags;
};

// The slot is a PARAMETER rather than a constant, and that is the point: the host pushes it, so the probe can
// be run a second time with a deliberately WRONG slot - a probe that could only say "fine" would pass every
// check. With no pipeline layout (what the heap flag requires) this block arrives through vkCmdPushDataEXT.
struct HeapProbePush
{
    uint material_slot; // ABSOLUTE grid slot of the material table, e.g. heap_slots_materials
};
[[vk::push_constant]] ConstantBuffer<HeapProbePush> pc;

struct heap_probe_vertex_output
{
    [[vk::location(0)]] float2 uv : UV_OUT;
    float4 position : SV_Position;
};

/// @brief the fullscreen triangle, synthesized from the vertex index alone
[shader("vertex")]
heap_probe_vertex_output main(uint vertex_index : SV_VertexID)
{
    const float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    const float2 p = positions[vertex_index];

    heap_probe_vertex_output out;
    out.uv = (p + float2(1.0, 1.0)) * 0.5;
    out.position = float4(p, 0.0, 1.0);
    return out;
}

/// @brief read element 0 of the material table at @p pc.material_slot and write its base colour
/// @note the uv is unused on purpose: the value is a constant of the heap's contents, not of the fragment's
///       position, so the whole 4x4 target must come back the same colour - which is what makes a single read
///       pixel a valid probe.
[shader("fragment")]
float4 frag_main([[vk::location(0)]] float2 uv : UV_IN) : SV_Target
{
    return float4(material_at(pc.material_slot, 0).base_color_factor.rgb, 1.0);
}
