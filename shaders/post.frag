#version 450

/**
 * @file shaders/post.frag
 * @brief Post-processing and the multi-level bloom chain; @c pc.mode selects the stage.
 * @ingroup shaders
 *
 * - @b mode 0 - bright-pass prefilter: HDR resolve target -> bloom level 0 (half resolution). One
 *   fetch into a half-resolution target lands exactly between four source texels, so the LINEAR
 *   sampler returns their 2x2 average and the soft threshold is applied to that average.
 * - @b mode 1 - downsample: binding 0 (level k) -> the current target (level k+1), 4-tap box.
 * - @b mode 2 - composite: HDR + weighted bloom levels -> exposure -> ACES -> display.
 *
 * Every mode reads binding 0; only the composite samples bindings 1..4 (the four bloom levels).
 * Display encoding is NOT done here unless the target is a non-sRGB format (pc.encode_gamma): the
 * swapchain is normally an sRGB attachment and the hardware encodes on write, so applying gamma in
 * the shader as well would encode twice. With FXAA enabled the composite instead renders the
 * gamma-encoded LDR image (core::ldr_images) that fxaa.frag reads.
 *
 * Requires the pipelines built by runtime::make_post_pipeline(): one per color format - the
 * composite writes the swapchain, the bloom stages write the R16F levels.
 */

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
    float encode_gamma;    // composite only: 1 = encode to sRGB by hand (non-sRGB swapchain), 0 = the
                           // attachment is an sRGB format and the hardware encodes on write
} pc;

/**
 * @brief ACES filmic tonemapping (Narkowicz fit)
 * @param color linear HDR colour, already multiplied by the exposure scale
 * @return display-referred colour in [0,1]
 */
vec3 aces_tone_mapping(vec3 color) {
    return clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
}

/**
 * @brief linear -> sRGB, for a swapchain that is NOT an sRGB format
 * @param color linear colour
 * @return display-encoded colour
 * @note with an sRGB attachment the hardware encodes on write, and encoding by hand on top of that
 *       applies gamma TWICE (measured: a linear 0.5 came out as 188, which is already sRGB(0.5), so
 *       the extra pow(1/2.2) turned x^0.45 into x^0.207 - a washed-out image). This path therefore
 *       only runs when pc.encode_gamma is set, i.e. for a UNORM swapchain.
 */
vec3 linear_to_srgb(vec3 color) {
    vec3 low = color * 12.92;
    vec3 high = 1.055 * pow(max(color, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, step(vec3(0.0031308), color));
}

/**
 * @brief 4-tap box average over one texel of @p s 's own resolution
 * @param s the sampler to filter (its textureSize() drives the tap spacing, so it works at any
 *          bloom level without a resolution constant)
 * @param uv the sample position
 * @return the averaged RGB
 */
vec3 sample_box(sampler2D s, vec2 uv) {
    vec2 texel = 1.0 / vec2(textureSize(s, 0));
    vec3 sum = texture(s, uv + vec2(-0.5, -0.5) * texel).rgb;
    sum += texture(s, uv + vec2(0.5, -0.5) * texel).rgb;
    sum += texture(s, uv + vec2(-0.5, 0.5) * texel).rgb;
    sum += texture(s, uv + vec2(0.5, 0.5) * texel).rgb;
    return sum * 0.25;
}

/**
 * @brief the three post-process stages, selected by pc.mode
 *
 * mode 0 bright-pass prefilter (soft threshold, bloom level 0), mode 1 downsample, mode 2 composite
 * (weighted bloom sum, exposure, ACES, display encoding). The bloom level weights are the artistic
 * part: 0.50 / 0.30 / 0.20 / 0.12 over the four levels, so the widest level contributes least.
 */
void main() {
    if (pc.mode < 0.5) {
        // bright pass: one fetch into a half-resolution target. Sampling a full-resolution source
        // from a pixel centre of the half-resolution target lands exactly halfway between four source
        // texels, so the LINEAR sampler returns their 2x2 average - no box filter needed here - and
        // the soft threshold is applied to that average.
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
    // Display encoding: normally the sRGB swapchain attachment does it (pc.encode_gamma == 0);
    // this path only exists so a UNORM swapchain still gets correct output.
    if (pc.encode_gamma > 0.5) {
        color = linear_to_srgb(color);
    }
    out_color = vec4(color, 1.0);
}