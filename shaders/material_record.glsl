// shaders/material_record.glsl - THE MATERIAL RECORD, declared where BOTH a scene stage and a fullscreen
// stage can reach it.
//
// WHY IT IS A FILE OF ITS OWN, and this is the constraint that forced it: a fullscreen stage cannot include
// `surface.glsl`. That file declares the shared scene push block as well as this struct, and the block is
// `mesh_stage_block_size` = 144 bytes - over the 128-byte `max_push_bytes` the heap-native pipeline path
// allows - because it carries the mesh geometry lanes. So every fullscreen stage in this renderer
// (`deferred`, `taa`, `gbuffer_debug`, `post`) declares a push block of its own, and any of them that needs
// the MATERIAL TABLE had no way to name the type it reads. This file is that way.
//
// IT IS ONLY A LAYOUT: the struct, nothing else - no heap block, no push block, no accessor. The table is
// reached through `material_at(slot, index)`, which `heap_access.slang` defines, so a stage that wants the
// table includes the shim and this file and is done.
//
// THE SPELLINGS ARE GLSL'S (`uvec4`, `vec4`) ON PURPOSE: under `-allow-glsl`, Slang accepts them and keeps
// their GLSL meaning, which `surface.glsl` already relies on - it is compiled by the Slang leaves today and
// uses these same types. One spelling, one layout, both consumers.
//
// THE LAYOUT MUST MATCH `vulkan::material_record` (std430, 80 bytes, `static_assert`ed on the host side).
// A field added here without the host - or the host without here - moves every material's texture indices,
// and the failure is a wrong texture rather than a build error.

struct Material {
    uvec4 tex_indices; // albedo, metallic-roughness, normal, occlusion (indices into textures[])
    uint emissive_index;
    float alpha_cutoff;       // alphaMode MASK threshold
    float occlusion_strength; // mix(1, sampled AO, strength)
    uint toon_family;         // the toon material family (see gltf_loader's toon_family_of); 0 == none.
                              // It completes the std430 group of four uints that starts at emissive_index, so
                              // naming it changes no offset - see material_record's note in
                              // vulkan/primitive/primitive.cppm.
    vec4 base_color_factor;
    vec4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    uint flags; // bit0: normal map, bit1: occlusion map, bit2: emissive map, bit3: double-sided,
                // bit4: alphaMode MASK, bit5: alphaMode BLEND
};
