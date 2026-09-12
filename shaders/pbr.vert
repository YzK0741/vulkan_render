#version 450

/**
 * @file shaders/pbr.vert
 * @brief Forward vertex stage: morph blend, GPU skinning, instancing and the world transform.
 * @ingroup shaders
 *
 * One vertex layout serves every draw strategy (locations 0,1,2,4,5, interleaved 64-byte stride;
 * shadow.vert must declare exactly the same inputs). Per-draw data arrives through the material push
 * constants; per-frame scene data through the shared scene set:
 * - binding 6 @c InstanceTransforms - the world matrix of an instanced draw (push flag bit0)
 * - binding 9 @c SkinMatrices - per-joint matrices, identity block at skin_base 0
 * - binding 10 @c MorphData - per-vertex position/normal deltas plus per-target weights
 *
 * Order matters and must match shadow.vert: morph first (local-space deltas), then skinning (the
 * four joint matrices selected by the vertex's joints, weighted), then the world transform. The
 * outputs feed pbr.frag's world-space lighting: a world position, a world normal and the UV.
 */

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 4) in uvec4 in_joints; // skin joint indices (JOINTS_0); 0 when unskinned
layout(location = 5) in vec4 in_weights;  // skin weights (WEIGHTS_0); (1,0,0,0) when unskinned

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
} camera;

// Per-instance world transforms for instanced draws (one mat4 per instance, written by the
// runtime via set_instanced_draw); only read when the push flag bit0 is set
layout(set = 0, binding = 6) readonly buffer InstanceTransforms {
    mat4 transforms[];
} instances;

// Previous-frame world matrices (scene set binding 13), one per MOTION SLOT: the runtime walks the
// scene tree once per frame and writes, for every leaf, the world matrix that leaf had one frame
// ago. This draw's own entry is indexed by push.motion_base (+ gl_InstanceIndex for an instanced
// draw, whose slots run parallel to the instance block), and the resulting previous world position
// is what the fragment stage turns into TAA's motion vector - which is how a MOVING object gets one
// at all. Without it the vector could only describe camera motion, and a character walking through
// a static scene smeared.
layout(set = 0, binding = 13) readonly buffer PreviousTransforms {
    mat4 matrices[];
} previous_transforms;

// Per-joint skin matrices (scene set binding 9, written per frame by set_skin_matrices):
// indices 0-3 are the identity block (the fallback for unskinned draws, skin_base = 0), the
// per-skin joint blocks follow. Each vertex blends the four matrices selected by its joints.
layout(set = 0, binding = 9) readonly buffer SkinMatrices {
    mat4 matrices[];
} skins;

// Morph data (scene set binding 10): per-morphable-primitive block laid out by the caller as
//   [ per vertex v: per target t: posDelta(xyz) normalDelta(xyz) ]  [ weights per target ]
// referenced through push.morph_base (float index) / morph_targets / morph_vertices.
layout(set = 0, binding = 10) readonly buffer MorphData {
    float morphs[];
} morph_data;

layout(push_constant) uniform PushConstants {
    uint material_index; // unused here (vertex stage), declared to keep the block layout identical to pbr.frag
    uint flags;          // bit0: instanced draw -> model comes from instances[instance_base + gl_InstanceIndex]
    uint skin_base;      // start of this primitive's joint block in skins.matrices (0 = identity)
    uint morph_base;     // float index of this primitive's morph block in morph_data.morphs (0 = none)
    uint morph_targets;  // number of morph targets (0 = not morphable)
    uint morph_vertices; // vertex count of this primitive (morph block stride)
    uint instance_base;  // mat4 start of this instanced primitive's transforms (binding 6)
    uint motion_base;    // start of this draw's previous-frame world matrices (binding 13); this
                         // field fits the 4 bytes std430 leaves between instance_base and model
    mat4 model;          // per-model world transform (kept out of the shared camera UBO)
} push;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec3 v_prev_world_pos; // where this vertex was last frame (see binding 13)

/**
 * @brief morph blend -> skin -> world transform; outputs the world position, normal and UV
 *
 * Each stage is skipped by a uniform branch when the primitive does not need it
 * (push.morph_targets == 0, weight sum == 0, flag bit0 clear), so one pipeline serves static,
 * skinned, morphed and instanced geometry without variants.
 */
void main() {
    // morph blend first (local deltas + active target weights; skipped when not morphable)
    vec4 local_pos = vec4(in_position, 1.0);
    vec3 morph_normal = in_normal;
    if (push.morph_targets > 0u) {
        const uint vert = gl_VertexIndex;
        const uint weight_base = push.morph_base + push.morph_targets * push.morph_vertices * 6u;
        vec3 pos_delta = vec3(0.0);
        vec3 nrm_delta = vec3(0.0);
        for (uint t = 0u; t < push.morph_targets; ++t) {
            const float w = morph_data.morphs[weight_base + t];
            const uint base = push.morph_base + (vert * push.morph_targets + t) * 6u;
            const vec3 dpos = vec3(morph_data.morphs[base], morph_data.morphs[base + 1u], morph_data.morphs[base + 2u]);
            const vec3 dnrm = vec3(morph_data.morphs[base + 3u], morph_data.morphs[base + 4u], morph_data.morphs[base + 5u]);
            pos_delta += w * dpos;
            nrm_delta += w * dnrm;
        }
        local_pos = vec4(in_position + pos_delta, 1.0);
        morph_normal = in_normal + nrm_delta;
    }

    // skinning after morph: blend the four joint transforms with the vertex weights (the
    // identity block at skin_base 0 leaves unskinned vertices unchanged). Normals use the
    // weighted rotation part of the joint matrices.
    vec3 skinned_normal = morph_normal;
    const float wsum = in_weights.x + in_weights.y + in_weights.z + in_weights.w;
    if (wsum > 0.0) {
        vec4 pos = vec4(0.0);
        vec3 nrm = vec3(0.0);
        pos += in_weights.x * (skins.matrices[push.skin_base + in_joints.x] * local_pos);
        pos += in_weights.y * (skins.matrices[push.skin_base + in_joints.y] * local_pos);
        pos += in_weights.z * (skins.matrices[push.skin_base + in_joints.z] * local_pos);
        pos += in_weights.w * (skins.matrices[push.skin_base + in_joints.w] * local_pos);
        nrm += in_weights.x * mat3(skins.matrices[push.skin_base + in_joints.x]) * morph_normal;
        nrm += in_weights.y * mat3(skins.matrices[push.skin_base + in_joints.y]) * morph_normal;
        nrm += in_weights.z * mat3(skins.matrices[push.skin_base + in_joints.z]) * morph_normal;
        nrm += in_weights.w * mat3(skins.matrices[push.skin_base + in_joints.w]) * morph_normal;
        local_pos = pos / wsum;
        skinned_normal = nrm / wsum;
    }

    mat4 world = (push.flags & 1u) != 0u ? instances.transforms[push.instance_base + gl_InstanceIndex] : push.model;
    vec4 world_pos = world * local_pos;
    v_world_pos = world_pos.xyz;
    v_normal = normalize(mat3(world) * skinned_normal);
    v_uv = in_uv;

    // The SAME local position through the previous frame's world matrix: exact object motion for a
    // rigid transform (the only kind of movement a model matrix can describe), and the standard
    // approximation for a deforming one - a skinned or morphed vertex moved within its own object
    // space too, and nothing here knows that yet.
    const uint motion_index = push.motion_base + ((push.flags & 1u) != 0u ? gl_InstanceIndex : 0u);
    v_prev_world_pos = (previous_transforms.matrices[motion_index] * local_pos).xyz;

    gl_Position = camera.proj * camera.view * world_pos;
}
