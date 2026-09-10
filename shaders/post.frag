#version 450

// Post-processing, two passes (selected by pc.mode):
//   mode 0: bright pass + horizontal blur  -> HDR scene target into the quarter-res bloom target
//   mode 1: vertical blur + composite      -> HDR (+ bloom * intensity) -> exposure -> ACES ->
//                                             gamma -> swapchain
// The forward passes write linear HDR radiance; all display-referred processing happens here.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D hdr_color;   // scene target (full resolution)
layout(set = 0, binding = 1) uniform sampler2D bloom_color; // bloom target (quarter resolution)

layout(push_constant) uniform PostPush {
    float exposure;        // linear exposure scale (runtime::set_exposure)
    float bloom_intensity; // blend weight of the blurred bright pass (0 = off)
    float bloom_threshold; // linear value subtracted in the bright pass
    float mode;            // 0 = bright/blur pass, 1 = composite pass
} pc;

// ACES filmic tonemapping (the same curve the forward pass used to apply inline)
vec3 aces_tone_mapping(vec3 color) {
    return clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
}

// 9-tap gaussian weights (sigma ~2)
float tap_weight(int i) {
    float fi = float(i);
    return exp(-0.5 * fi * fi / 4.0);
}

void main() {
    if (pc.mode < 0.5) {
        // ---- pass A: bright pass (soft threshold) + horizontal blur ----
        // sampled at quarter resolution, so one output texel spans 4 HDR texels; the offset is
        // doubled again to widen the glow
        vec2 hdr_texel = 1.0 / vec2(textureSize(hdr_color, 0));
        vec3 sum = vec3(0.0);
        float weight_sum = 0.0;
        for (int i = -4; i <= 4; ++i) {
            float w = tap_weight(i);
            vec3 sample_color = texture(hdr_color, v_uv + vec2(float(i) * hdr_texel.x * 8.0, 0.0)).rgb;
            sum += max(sample_color - vec3(pc.bloom_threshold), vec3(0.0)) * w;
            weight_sum += w;
        }
        out_color = vec4(sum / weight_sum, 1.0);
        return;
    }

    // ---- pass B: vertical blur of the bloom target + composite + display transform ----
    vec2 bloom_texel = 1.0 / vec2(textureSize(bloom_color, 0));
    vec3 bloom = vec3(0.0);
    float weight_sum = 0.0;
    for (int i = -4; i <= 4; ++i) {
        float w = tap_weight(i);
        bloom += texture(bloom_color, v_uv + vec2(0.0, float(i) * bloom_texel.y * 2.0)).rgb * w;
        weight_sum += w;
    }
    bloom /= weight_sum;

    vec3 color = texture(hdr_color, v_uv).rgb;
    color += bloom * pc.bloom_intensity;
    color *= pc.exposure;
    color = aces_tone_mapping(color);
    color = pow(color, vec3(1.0 / 2.2));
    out_color = vec4(color, 1.0);
}