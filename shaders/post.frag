#version 450

// Post-processing + multi-level bloom (mode selects the stage):
//   mode 0: bright-pass prefilter   HDR -> bloom level 0 (1/2 resolution)
//   mode 1: downsample              binding 0 (level k) -> the current target (level k+1)
//   mode 2: composite               HDR + weighted bloom levels -> exposure -> ACES -> gamma
// Every mode reads binding 0; only the composite samples bindings 1..4 (the four bloom levels).

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D source_color; // pass input (HDR, or a bloom level)
layout(set = 0, binding = 1) uniform sampler2D bloom_l0;     // composite only
layout(set = 0, binding = 2) uniform sampler2D bloom_l1;
layout(set = 0, binding = 3) uniform sampler2D bloom_l2;
layout(set = 0, binding = 4) uniform sampler2D bloom_l3;

layout(push_constant) uniform PostPush {
    float exposure;        // linear exposure scale (runtime::set_exposure)
    float bloom_intensity; // blend weight of the bloom sum (0 = off)
    float bloom_threshold; // linear value subtracted in the bright pass
    float mode;            // 0 prefilter / 1 downsample / 2 composite
} pc;

// ACES filmic tonemapping
vec3 aces_tone_mapping(vec3 color) {
    return clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
}

// 4-tap box average over one texel of the given sampler's own resolution
vec3 sample_box(sampler2D s, vec2 uv) {
    vec2 texel = 1.0 / vec2(textureSize(s, 0));
    vec3 sum = texture(s, uv + vec2(-0.5, -0.5) * texel).rgb;
    sum += texture(s, uv + vec2(0.5, -0.5) * texel).rgb;
    sum += texture(s, uv + vec2(-0.5, 0.5) * texel).rgb;
    sum += texture(s, uv + vec2(0.5, 0.5) * texel).rgb;
    return sum * 0.25;
}

void main() {
    if (pc.mode < 0.5) {
        // bright pass: soft threshold in linear space, then let the hardware bilinear downsample
        vec3 color = texture(source_color, v_uv).rgb;
        out_color = vec4(max(color - vec3(pc.bloom_threshold), vec3(0.0)), 1.0);
        return;
    }

    if (pc.mode < 1.5) {
        // downsample: 4-tap box into the next level
        out_color = vec4(sample_box(source_color, v_uv), 1.0);
        return;
    }

    // composite: weighted sum of the four levels (each box-filtered at its own resolution)
    vec3 bloom = sample_box(bloom_l0, v_uv) * 0.50;
    bloom += sample_box(bloom_l1, v_uv) * 0.30;
    bloom += sample_box(bloom_l2, v_uv) * 0.20;
    bloom += sample_box(bloom_l3, v_uv) * 0.12;

    vec3 color = texture(source_color, v_uv).rgb;
    color += bloom * pc.bloom_intensity;
    color *= pc.exposure;
    color = aces_tone_mapping(color);
    color = pow(color, vec3(1.0 / 2.2));
    out_color = vec4(color, 1.0);
}