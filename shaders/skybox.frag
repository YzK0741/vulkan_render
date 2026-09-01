#version 450

layout(location = 0) out vec4 out_color;

// The prefiltered environment's base mip is the original environment (roughness 0)
layout(set = 0, binding = 2) uniform samplerCube env_sampler;

layout(location = 0) in vec3 v_dir;

void main() {
    vec3 dir = normalize(v_dir);
    // Force mip 0: implicit LOD over a fullscreen triangle would pick the darkest prefiltered mip
    out_color = vec4(textureLod(env_sampler, dir, 0.0).rgb, 1.0);
}
