#version 450

// Post-processing: reads the HDR scene target (linear radiance written by pbr.frag / skybox.frag /
// unlit.frag), applies the exposure scale, tonemaps with ACES and gamma-encodes for the sRGB
// swapchain. The bloom chain will be composited here as well (the push constants already reserve
// its parameters), so all display-referred processing lives in one place.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D hdr_color;

layout(push_constant) uniform PostPush {
    float exposure;        // linear exposure scale (runtime::set_exposure)
    float bloom_intensity; // reserved: bloom blend weight (next step)
    float bloom_threshold; // reserved: bloom bright-pass threshold
    float _pad;
} pc;

// ACES filmic tonemapping (same curve the forward pass used to apply inline)
vec3 aces_tone_mapping(vec3 color) {
    return clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec3 color = texture(hdr_color, v_uv).rgb;
    color *= pc.exposure;
    color = aces_tone_mapping(color);
    color = pow(color, vec3(1.0 / 2.2));
    out_color = vec4(color, 1.0);
}
