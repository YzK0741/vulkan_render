#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 mvp;
} scene;

layout(push_constant) uniform PushConstants {
    mat4 model;
} push;

layout(location = 0) out vec3 frag_color;

void main() {
    gl_Position = scene.mvp * push.model * vec4(in_position, 1.0);
    frag_color = in_color;
}
