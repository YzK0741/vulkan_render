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
 * @brief 13-tap downsample (Karis, "Next Generation Post Processing in Call of Duty"), the filter UE
 *        uses for its bloom chain: a 5x5 footprint built from a 3x3 ring plus a 2x2 inner quad, with
 *        the corners weighted lowest
 * @param s the level being read (its textureSize() drives the tap spacing, so no resolution constant)
 * @param uv the sample position
 * @return the filtered RGB (weights sum to 1)
 *
 * A plain box downsample moves a single bright pixel into the next level almost unchanged, and after
 * two or three levels that pixel IS a whole texel of a very coarse image - which the composite then
 * magnifies into a visible block. The 13 taps spread it over its neighbourhood on the way down. This
 * is the point UE's higher-quality downsample path makes too (PostProcessDownsample.usf).
 */
vec3 downsample_13(sampler2D s, vec2 uv) {
    const vec2 t = 1.0 / vec2(textureSize(s, 0));
    const vec3 a = texture(s, uv + t * vec2(-2.0, -2.0)).rgb;
    const vec3 b = texture(s, uv + t * vec2(0.0, -2.0)).rgb;
    const vec3 c = texture(s, uv + t * vec2(2.0, -2.0)).rgb;
    const vec3 d = texture(s, uv + t * vec2(-2.0, 0.0)).rgb;
    const vec3 e = texture(s, uv).rgb;
    const vec3 f = texture(s, uv + t * vec2(2.0, 0.0)).rgb;
    const vec3 g = texture(s, uv + t * vec2(-2.0, 2.0)).rgb;
    const vec3 h = texture(s, uv + t * vec2(0.0, 2.0)).rgb;
    const vec3 i = texture(s, uv + t * vec2(2.0, 2.0)).rgb;
    const vec3 j = texture(s, uv + t * vec2(-1.0, -1.0)).rgb;
    const vec3 k = texture(s, uv + t * vec2(1.0, -1.0)).rgb;
    const vec3 l = texture(s, uv + t * vec2(-1.0, 1.0)).rgb;
    const vec3 m = texture(s, uv + t * vec2(1.0, 1.0)).rgb;
    vec3 sum = e * 0.125;
    sum += (a + c + g + i) * 0.03125;
    sum += (b + d + f + h) * 0.0625;
    sum += (j + k + l + m) * 0.125;
    return sum;
}

/**
 * @brief 3x3 tent filter (1 2 1 / 2 4 2 / 1 2 1) over one texel of @p s 's own resolution: the
 *        footprint an upsample step of the bloom pyramid stands for
 * @param s the level to filter
 * @param uv the sample position
 * @return the filtered RGB (weights sum to 1)
 *
 * Sampling a coarse level with a single bilinear fetch (or a half-texel box) makes one of ITS texels
 * a block on screen - at 1/16 resolution that is 16x16 pixels of flat light with square edges. The
 * tent is what UE's bloom upsample uses (PostProcessBloom.usf), and it is the reason a coarse level
 * reads as a wide glow instead of a tile.
 */
vec3 sample_tent(sampler2D s, vec2 uv) {
    const vec2 t = 1.0 / vec2(textureSize(s, 0));
    vec3 sum = texture(s, uv).rgb * 4.0;
    sum += texture(s, uv + vec2(-t.x, 0.0)).rgb * 2.0;
    sum += texture(s, uv + vec2(t.x, 0.0)).rgb * 2.0;
    sum += texture(s, uv + vec2(0.0, -t.y)).rgb * 2.0;
    sum += texture(s, uv + vec2(0.0, t.y)).rgb * 2.0;
    sum += texture(s, uv + vec2(-t.x, -t.y)).rgb;
    sum += texture(s, uv + vec2(t.x, -t.y)).rgb;
    sum += texture(s, uv + vec2(-t.x, t.y)).rgb;
    sum += texture(s, uv + vec2(t.x, t.y)).rgb;
    return sum / 16.0;
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
        // texels, so the LINEAR sampler returns their 2x2 average.
        //
        // The threshold is a RAMP over luminance, not a per-channel subtract-and-clamp. Subtracting
        // and clamping makes the bloom switch on and off along the iso-luminance contour of whatever
        // is in the frame, so a whole region pops in at once with geometrically straight edges as the
        // camera moves - the "bright spot suddenly becomes a big square" artifact. UE does exactly
        // this ramp (PostProcessBloom.usf, BloomSetupCommon):
        //     BloomAmount = saturate((Luminance - BloomThreshold) * 0.5);  out = BloomAmount * Color
        // which fades in over a 2.0-wide luminance window above the threshold and keeps the colour.
        vec3 color = texture(source_color, v_uv).rgb;
        const float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));

        // Firefly ceiling. The ramp above bounds the FRACTION a pixel may contribute, not its
        // magnitude: with a specular highlight three orders of magnitude above the sky, one blown
        // pixel still entered the chain at full value and the whole pyramid carried it - which is
        // what turns a tiny highlight into a large round patch that appears and vanishes as the
        // camera moves. UE clamps the bloom input for exactly this reason (Bloom/BloomClampKernel)
        // and its own ramp assumes an exposure-scaled input, which ours is not.
        //
        // The curve leaves everything at or below the ceiling untouched and folds the rest towards
        // 2x the ceiling (an excess of 97.5 becomes 4.94 at a ceiling of 2.5) - smoothly, so the
        // clamp introduces no plateau and no edge. It is a luminance scale, so the hue survives.
        const float firefly_ceiling = 2.5;
        const float base = min(luminance, firefly_ceiling);
        const float excess = max(luminance - firefly_ceiling, 0.0);
        const float compressed = base + excess / (1.0 + excess / firefly_ceiling);
        color *= compressed / max(luminance, 1e-5);

        const float amount = clamp((compressed - pc.bloom_threshold) * 0.5, 0.0, 1.0);
        out_color = vec4(color * amount, 1.0);
        return;
    }

    if (pc.mode < 1.5) {
        // downsample: 13-tap filter into the next level (see downsample_13)
        out_color = vec4(downsample_13(source_color, v_uv), 1.0);
        return;
    }

    // composite: weighted sum of the four levels, each read through the tent filter that the
    // upsample step of a bloom pyramid stands for (see sample_tent)
    vec3 bloom = sample_tent(bloom_l0, v_uv) * 0.50;
    bloom += sample_tent(bloom_l1, v_uv) * 0.30;
    bloom += sample_tent(bloom_l2, v_uv) * 0.20;
    bloom += sample_tent(bloom_l3, v_uv) * 0.12;

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