#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_tangent;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
} camera;

layout(push_constant) uniform PushConstants {
    vec4 base_color_factor;
    vec4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    uint flags; // bit0: has normal map, bit1: has occlusion map, bit2: has emissive map
} push;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec3 v_tangent;

void main() {
    vec4 world_pos = camera.model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = normalize(mat3(camera.model) * in_normal);
    v_uv = in_uv;
    v_tangent = normalize(mat3(camera.model) * in_tangent);
    gl_Position = camera.proj * camera.view * world_pos;
}
