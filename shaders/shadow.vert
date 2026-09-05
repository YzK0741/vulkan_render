#version 450

// Depth-only vertex shader for the directional shadow pass: transforms the model into the
// light's orthographic clip space, so rasterization writes the depth seen from the light.
//
// The vertex input layout MUST stay identical to pbr.vert (positions 0-5, interleaved 76-byte
// stride): the shadow pass draws the very same vertex/index buffers, and the pipeline derives
// its vertex input stride from the shader's inputs. All six inputs are declared and referenced
// (in a never-taken branch) purely so a driver/compiler cannot prune them and shrink the stride.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_tangent;
layout(location = 4) in uvec4 in_joints; // skin joint indices (JOINTS_0); 0 when unskinned
layout(location = 5) in vec4 in_weights;  // skin weights (WEIGHTS_0); (1,0,0,0) when unskinned

// Per-instance world transforms (same storage as pbr.vert, scene set binding 6)
layout(set = 0, binding = 6) readonly buffer InstanceTransforms {
    mat4 transforms[];
} instances;

// Per-joint skin matrices (same storage as pbr.vert, scene set binding 9)
layout(set = 0, binding = 9) readonly buffer SkinMatrices {
    mat4 matrices[];
} skins;

// Light UBO (scene set binding 7): the orthographic light view-proj maps world -> shadow map.
layout(set = 0, binding = 7) uniform LightUBO {
    mat4 light_view_proj;
    vec4 light_dir;
} light;

layout(push_constant) uniform PushConstants {
    uint material_index; // unused here (vertex stage), declared to keep the block layout identical to pbr.vert
    uint flags;          // bit0: instanced draw -> model comes from instances[gl_InstanceIndex]
    uint skin_base;      // start of this primitive's joint block in skins.matrices (0 = identity)
    uint _pad;
    mat4 model;
} push;

void main() {
    // skinning identical to pbr.vert: weighted joint transforms (identity block at skin_base 0)
    vec4 local_pos = vec4(in_position, 1.0);
    const float wsum = in_weights.x + in_weights.y + in_weights.z + in_weights.w;
    if (wsum > 0.0) {
        vec4 pos = vec4(0.0);
        pos += in_weights.x * (skins.matrices[push.skin_base + in_joints.x] * local_pos);
        pos += in_weights.y * (skins.matrices[push.skin_base + in_joints.y] * local_pos);
        pos += in_weights.z * (skins.matrices[push.skin_base + in_joints.z] * local_pos);
        pos += in_weights.w * (skins.matrices[push.skin_base + in_joints.w] * local_pos);
        local_pos = pos / wsum;
    }

    mat4 world = (push.flags & 1u) != 0u ? instances.transforms[gl_InstanceIndex] : push.model;
    vec4 world_pos = world * local_pos;
    gl_Position = light.light_view_proj * world_pos;

    // Keep every input alive so the vertex input layout (and thus the bound buffer stride)
    // matches pbr.vert exactly (76 bytes, locations 0-5). This branch can never run.
    if (isnan(in_position.x) && isinf(in_normal.x) && in_uv.x > 1e30 && isinf(in_tangent.x)) {
        gl_Position = vec4(0.0);
    }
}
