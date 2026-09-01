#version 450

layout(location = 0) out vec4 out_color;

// The prefiltered environment's base mip is the original environment (roughness 0)
layout(set = 0, binding = 2) uniform samplerCube env_sampler;

layout(location = 0) in vec3 v_dir;

// Same exposure pipeline as pbr.frag: ACES filmic tonemapping + gamma, so the sky matches the
// lit models instead of appearing as raw dark linear values on the sRGB swapchain
vec3 aces_tone_mapping(vec3 color) {
    return clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec3 dir = normalize(v_dir);
    // Force mip 0: implicit LOD over a fullscreen triangle would pick the darkest prefiltered mip
    vec3 color = textureLod(env_sampler, dir, 0.0).rgb;
    color = aces_tone_mapping(color);
    color = pow(color, vec3(1.0 / 2.2));
    out_color = vec4(color, 1.0);
}
