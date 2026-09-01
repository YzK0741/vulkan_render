#version 450

layout(location = 0) out vec4 out_color;

// The prefiltered environment's base mip is the original environment (roughness 0)
layout(set = 0, binding = 2) uniform samplerCube env_sampler;

layout(location = 0) in vec3 v_dir;

void main() {
    vec3 dir = normalize(v_dir);
    out_color = vec4(texture(env_sampler, dir).rgb, 1.0);
}
