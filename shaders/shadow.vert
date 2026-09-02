#version 450

// Depth-only vertex shader for the directional shadow pass: transforms the model into the
// light's orthographic clip space, so rasterization writes the depth seen from the light.
//
// The vertex input layout MUST stay identical to pbr.vert (positions 0-3, interleaved 44-byte
// stride): the shadow pass draws the very same vertex/index buffers, and the pipeline derives
// its vertex input stride from the shader's inputs. All four inputs are declared and referenced
// (in a never-taken branch) purely so a driver/compiler cannot prune them and shrink the stride.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_tangent;

// Per-instance world transforms (same storage as pbr.vert, scene set binding 6)
layout(set = 0, binding = 6) readonly buffer InstanceTransforms {
    mat4 transforms[];
} instances;

// Light UBO (scene set binding 7): the orthographic light view-proj maps world -> shadow map.
layout(set = 0, binding = 7) uniform LightUBO {
    mat4 light_view_proj;
    vec4 light_dir;
} light;

layout(push_constant) uniform PushConstants {
    uint material_index; // unused here (vertex stage), declared to keep the block layout identical to pbr.vert
    uint flags;          // bit0: instanced draw -> model comes from instances[gl_InstanceIndex]
    mat4 model;
} push;

void main() {
    mat4 world = (push.flags & 1u) != 0u ? instances.transforms[gl_InstanceIndex] : push.model;
    vec4 world_pos = world * vec4(in_position, 1.0);
    gl_Position = light.light_view_proj * world_pos;

    // Keep every input alive so the vertex input layout (and thus the bound buffer stride)
    // matches pbr.vert exactly. This branch can never run (in_position.x is never NaN/Inf).
    if (isnan(in_position.x) && isinf(in_normal.x) && in_uv.x > 1e30 && isinf(in_tangent.x)) {
        gl_Position = vec4(0.0);
    }
}
