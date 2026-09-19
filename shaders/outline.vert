#version 450

/**
 * @file shaders/outline.vert
 * @brief The inverted hull's vertex stage: the surface pushed out along its normal (see docs/zzz_shading.md).
 * @ingroup shaders
 *
 * THE INVERTED HULL is how the reference draws its outline: a second copy of the model expanded along its
 * own normals and drawn with the FRONT faces culled, so the only part that survives the depth test is the
 * sliver the real surface does not already cover - a silhouette. The expansion is done in WORLD space, by
 * `camera[].outline.w` world units, which is what keeps the line the same width on a model whose own
 * transform scales it.
 *
 * The transform chain below is pbr.vert's - morph, then skin, then the world matrix - and it is a COPY
 * rather than a shared include, deliberately: pbr.vert is the forward path's stage, and the capture gate
 * compares SPIR-V-derived frames byte for byte, so a refactor there would have to prove it changed nothing.
 * The cost is that a change to the chain has to be made in both files; the benefit is that adding an
 * outline cannot move a single gate reference. The one piece that is NOT copied is the previous-frame
 * world matrix: the hull writes a zero motion vector (see outline.frag), because an outline is a
 * silhouette and TAA's history has nothing better to reproject it from.
 *
 * The vertex layout, the push block and the heap slots are the shared ones - a hull draws the same
 * geometry through the same bindings, which is why this stage declares exactly what pbr.vert declares
 * minus the attributes it does not read.
 */

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
// DECLARED AND UNUSED, and it is load-bearing: the pipeline's vertex input layout is DERIVED from the
// locations a stage declares, with the stride accumulated over them in order (see vulkan::make_pipeline),
// so a stage that skips location 2 gets a 48-byte stride against a 64-byte vertex - every vertex after the
// first is read misaligned and the hull rasterizes as garbage triangles spanning the screen. pbr.vert says
// the same thing from its side ("shadow.vert must declare exactly the same inputs").
layout(location = 2) in vec2 in_uv;
layout(location = 4) in uvec4 in_joints; // skin joint indices (JOINTS_0); 0 when unskinned
layout(location = 5) in vec4 in_weights;  // skin weights (WEIGHTS_0); (1,0,0,0) when unskinned

layout(location = 0) out vec3 v_normal; // world normal, for the G-buffer's normal target (see outline.frag)

#extension GL_EXT_descriptor_heap : require
#extension GL_EXT_nonuniform_qualifier : enable
#include "heap_slots.glsl"

// The camera block, declared IN FULL so the outline lane lands on its real offset: std140 gives the
// `vec3 camera_pos` a 16-byte slot, and the two unjittered matrices sit between it and the outline lane -
// declaring only the members this stage reads would put `outline` where `view_proj_unjittered` lives, and
// the hull would be expanded by a matrix row (measured: the camera ends up inside the model).
//   outline: xyz = the line's colour, w = its width in WORLD units (0 = the outline is off, which is also
//   why the leaves are never asked to draw a hull when it is 0).
layout(descriptor_heap, descriptor_stride = heap_slot_stride) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    float camera_padding;
    mat4 view_proj_unjittered;
    mat4 prev_view_proj;
    vec4 outline;
} camera[];

layout(descriptor_heap, descriptor_stride = heap_slot_stride) readonly buffer InstanceTransforms {
    mat4 transforms[];
} instances[];

layout(descriptor_heap, descriptor_stride = heap_slot_stride) readonly buffer SkinMatrices {
    mat4 matrices[];
} skins[];

// Morph data: same block pbr.vert reads, same layout (see vulkan/primitive/primitive.cppm).
layout(descriptor_heap, descriptor_stride = heap_slot_stride) readonly buffer MorphData {
    float morphs[];
} morph_data[];

// The material table, for THIS draw's record (see surface.glsl for the canonical field list): the model's
// own outline thickness, which is the thing MMD authors per material and the reason this stage reads the
// table at all. Declared in full even though one lane is used: a storage buffer's array stride is the
// struct's own size, so a copy that stops early indexes the table at the wrong pitch.
struct Material {
    uvec4 tex_indices;
    uint emissive_index;
    float alpha_cutoff;
    float occlusion_strength;
    uint _pad;
    vec4 base_color_factor;
    vec4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    uint flags; // bit6: the model authored MMD edge data (hand-appended in the converter, see docs)
    vec4 npr_edge;
};
layout(descriptor_heap, descriptor_stride = heap_slot_stride) readonly buffer Materials { Material materials[]; } heap_material_tables[];

// The material push block. Mirrors material_push_constants (vulkan/primitive/primitive.cppm) - the hull
// reads the model matrix and the morph/skin/skinning indices, and ignores the rest.
// `motion_base` is DECLARED AND UNUSED for the same reason `in_uv` above is: it sits in the 4 bytes
// std430 leaves between instance_base and model, so a stage that omits it and still declares `model`
// gets the right offset only by that padding - and a stage that omits it and declares the fields AFTER
// model reads them four bytes early. That is exactly how this shader first came out invisible: `model`
// resolved to the wrong bytes, the hull was transformed somewhere off screen, and every draw in the
// frame was valid.
layout(push_constant) uniform PushConstants {
    uint material_index;
    uint flags;          // bit0: instanced draw -> model from instances[...]
    uint skin_base;
    uint morph_base;
    uint morph_targets;
    uint morph_vertices;
    uint instance_base;
    uint motion_base;
    mat4 model;
    uint frame_slot;
    uint image_index;
} push;

// ... and the two names the shared slot macros use (heap_slots.glsl says why these are macros rather than
// constants): every per-frame heap array's slot is its base plus the frame the push block carries.
#define heap_frame_slot (push.frame_slot)
#define heap_image_index (push.image_index)

void main() {
    // ---- morph (local-space deltas), exactly pbr.vert's
    vec4 local_pos = vec4(in_position, 1.0);
    vec3 morph_normal = in_normal;
    if (push.morph_targets > 0u) {
        const uint vert = gl_VertexIndex;
        const uint weight_base = push.morph_base + push.morph_targets * push.morph_vertices * 6u;
        vec3 pos_delta = vec3(0.0);
        vec3 nrm_delta = vec3(0.0);
        for (uint t = 0u; t < push.morph_targets; ++t) {
            const float w = morph_data[heap_morph_slot].morphs[weight_base + t];
            const uint base = push.morph_base + (vert * push.morph_targets + t) * 6u;
            const vec3 dpos = vec3(morph_data[heap_morph_slot].morphs[base], morph_data[heap_morph_slot].morphs[base + 1u], morph_data[heap_morph_slot].morphs[base + 2u]);
            const vec3 dnrm = vec3(morph_data[heap_morph_slot].morphs[base + 3u], morph_data[heap_morph_slot].morphs[base + 4u], morph_data[heap_morph_slot].morphs[base + 5u]);
            pos_delta += w * dpos;
            nrm_delta += w * dnrm;
        }
        local_pos = vec4(in_position + pos_delta, 1.0);
        morph_normal = in_normal + nrm_delta;
    }

    // ---- skin, exactly pbr.vert's (the identity block at skin_base 0 leaves a static mesh unchanged)
    vec3 skinned_normal = morph_normal;
    const float wsum = in_weights.x + in_weights.y + in_weights.z + in_weights.w;
    if (wsum > 0.0) {
        vec4 pos = vec4(0.0);
        vec3 nrm = vec3(0.0);
        pos += in_weights.x * (skins[heap_skin_slot].matrices[push.skin_base + in_joints.x] * local_pos);
        pos += in_weights.y * (skins[heap_skin_slot].matrices[push.skin_base + in_joints.y] * local_pos);
        pos += in_weights.z * (skins[heap_skin_slot].matrices[push.skin_base + in_joints.z] * local_pos);
        pos += in_weights.w * (skins[heap_skin_slot].matrices[push.skin_base + in_joints.w] * local_pos);
        nrm += in_weights.x * mat3(skins[heap_skin_slot].matrices[push.skin_base + in_joints.x]) * morph_normal;
        nrm += in_weights.y * mat3(skins[heap_skin_slot].matrices[push.skin_base + in_joints.y]) * morph_normal;
        nrm += in_weights.z * mat3(skins[heap_skin_slot].matrices[push.skin_base + in_joints.z]) * morph_normal;
        nrm += in_weights.w * mat3(skins[heap_skin_slot].matrices[push.skin_base + in_joints.w]) * morph_normal;
        local_pos = pos / wsum;
        skinned_normal = nrm / wsum;
    }

    // ---- world, then THE EXPANSION along the world-space normal
    const mat4 world = (push.flags & 1u) != 0u ? instances[heap_instance_slot].transforms[push.instance_base + gl_InstanceIndex] : push.model;
    vec4 world_pos = world * local_pos;
    // THE WIDTH IS THE MATERIAL'S OWN when the model authored one: MMD stores a per-material edge size
    // (エッジ倍率) that its own renderer scales the extrusion by - the hair and the face are 0.5 here, the
    // body 1.0 - and the frame's [render] outline_width is the overall multiplier on top of it. A model
    // without the data (any glTF this project's converter did not write) keeps the frame's width flat.
    const Material material = heap_material_tables[heap_slots_materials].materials[push.material_index];
    const float width = (material.flags & 64u) != 0u ? camera[heap_camera_slot].outline.w * material.npr_edge.w : camera[heap_camera_slot].outline.w;
    // A degenerate normal (a vertex no triangle uses, or a mesh with no normals) has nothing to push
    // along, and normalizing it would be a NaN that rasterizes the whole hull away - so the guard is on
    // the length rather than on a zero comparison.
    const vec3 world_normal = mat3(world) * skinned_normal;
    const float normal_length = length(world_normal);
    if (normal_length > 1e-8) {
        world_pos.xyz += (world_normal / normal_length) * width;
    }

    gl_Position = camera[heap_camera_slot].proj * camera[heap_camera_slot].view * world_pos;
    // The hull's world normal goes to the fragment stage only so the G-buffer's normal target holds
    // something meaningful: with metallic 0 and roughness 1 the shading barely depends on it.
    v_normal = world_normal;
}
