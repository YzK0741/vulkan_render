#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_tangent;
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

// Per-joint skin matrices (scene set binding 9, written per frame by set_skin_matrices):
// indices 0-3 are the identity block (the fallback for unskinned draws, skin_base = 0), the
// per-skin joint blocks follow. Each vertex blends the four matrices selected by its joints.
layout(set = 0, binding = 9) readonly buffer SkinMatrices {
    mat4 matrices[];
} skins;

layout(push_constant) uniform PushConstants {
    uint material_index; // unused here (vertex stage), declared to keep the block layout identical to pbr.frag
    uint flags;          // bit0: instanced draw -> model comes from instances[gl_InstanceIndex]
    uint skin_base;      // start of this primitive's joint block in skins.matrices (0 = identity)
    uint _pad;
    mat4 model;          // per-model world transform (kept out of the shared camera UBO)
} push;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec3 v_tangent;

void main() {
    // skinning: blend the four joint transforms with the vertex weights (the identity block at
    // skin_base 0 leaves unskinned vertices unchanged). Normals/tangents use the rotation part.
    vec4 local_pos = vec4(in_position, 1.0);
    vec3 skinned_normal = in_normal;
    vec3 skinned_tangent = in_tangent;
    const float wsum = in_weights.x + in_weights.y + in_weights.z + in_weights.w;
    if (wsum > 0.0) {
        vec4 pos = vec4(0.0);
        vec3 nrm = vec3(0.0);
        vec3 tan = vec3(0.0);
        pos += in_weights.x * (skins.matrices[push.skin_base + in_joints.x] * local_pos);
        pos += in_weights.y * (skins.matrices[push.skin_base + in_joints.y] * local_pos);
        pos += in_weights.z * (skins.matrices[push.skin_base + in_joints.z] * local_pos);
        pos += in_weights.w * (skins.matrices[push.skin_base + in_joints.w] * local_pos);
        nrm += in_weights.x * mat3(skins.matrices[push.skin_base + in_joints.x]) * in_normal;
        nrm += in_weights.y * mat3(skins.matrices[push.skin_base + in_joints.y]) * in_normal;
        nrm += in_weights.z * mat3(skins.matrices[push.skin_base + in_joints.z]) * in_normal;
        nrm += in_weights.w * mat3(skins.matrices[push.skin_base + in_joints.w]) * in_normal;
        tan += in_weights.x * mat3(skins.matrices[push.skin_base + in_joints.x]) * in_tangent;
        tan += in_weights.y * mat3(skins.matrices[push.skin_base + in_joints.y]) * in_tangent;
        tan += in_weights.z * mat3(skins.matrices[push.skin_base + in_joints.z]) * in_tangent;
        tan += in_weights.w * mat3(skins.matrices[push.skin_base + in_joints.w]) * in_tangent;
        local_pos = pos / wsum;
        skinned_normal = nrm / wsum;
        skinned_tangent = tan / wsum;
    }

    mat4 world = (push.flags & 1u) != 0u ? instances.transforms[gl_InstanceIndex] : push.model;
    vec4 world_pos = world * local_pos;
    v_world_pos = world_pos.xyz;
    v_normal = normalize(mat3(world) * skinned_normal);
    v_uv = in_uv;
    v_tangent = normalize(mat3(world) * skinned_tangent);
    gl_Position = camera.proj * camera.view * world_pos;
}
