#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_tangent;

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

layout(push_constant) uniform PushConstants {
    uint material_index; // unused here (vertex stage), declared to keep the block layout identical to pbr.frag
    uint flags;          // bit0: instanced draw -> model comes from instances[gl_InstanceIndex]
    mat4 model;          // per-model world transform (kept out of the shared camera UBO)
} push;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec3 v_tangent;

void main() {
    mat4 world = (push.flags & 1u) != 0u ? instances.transforms[gl_InstanceIndex] : push.model;
    vec4 world_pos = world * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = normalize(mat3(world) * in_normal);
    v_uv = in_uv;
    v_tangent = normalize(mat3(world) * in_tangent);
    gl_Position = camera.proj * camera.view * world_pos;
}
