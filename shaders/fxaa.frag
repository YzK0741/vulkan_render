#version 450

// FXAA (Fast Approximate Anti-Aliasing), Lottes' FXAA 3.11 "quality" variant, on the LDR image the
// composite produced.
//
// Position in the chain: forward -> HDR -> bloom -> composite -> [this] -> swapchain. It has to sit
// AFTER tonemapping (the luma thresholds below are tuned for display-referred, gamma-encoded data)
// and it needs its own input, because a pass cannot read the image it renders into: that is what
// core::ldr_images are for. The composite writes them *gamma-encoded* (post_push_constants::
// encode_gamma - they are R16F, so nothing decodes them again) and this shader therefore works
// directly in the perceptual space FXAA was designed for. Its own output is LINEAR, because the
// swapchain is an sRGB attachment whose write path encodes to display values in hardware.
//
// What it does: luma of a 3x3 neighbourhood -> if the local contrast is under a relative threshold
// the pixel is left alone (that is what keeps flat/gradual shading and text from being blurred) ->
// otherwise the edge direction is estimated from the diagonal luma gradients and the pixel is
// blended along it, with a two-step (near/far) search that stops as soon as the blended luma leaves
// the neighbourhood's range. Cost is ~9 texture fetches, no depth/normal/motion input.
//
// Because it is a screen-space blur it cannot fix sub-pixel shimmer in motion (that needs TAA), and
// it softens fine detail; the debug overlay is therefore drawn AFTER this pass, not before.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

// binding 5 = the gamma-encoded LDR image (see the post set layout; binding 0 is the HDR target the
// composite reads, which this pass has no use for).
layout(set = 0, binding = 5) uniform sampler2D display_color;

// Same block as post.frag / post.vert (the pipeline layout shares one push-constant range); only the
// FXAA lanes are read here, the rest exist so the offsets stay identical.
layout(push_constant) uniform PostPush {
    float exposure;
    float bloom_intensity;
    float bloom_threshold;
    float mode;
    float encode_gamma;
    float fxaa_subpixel;      // 0 = pure directional blend, up to 1 = also blend away single-pixel aliasing
    float fxaa_edge_threshold; // relative luma contrast below which a pixel counts as flat (0.166 = FXAA default)
} pc;

const float FXAA_EDGE_THRESHOLD_MIN = 0.0833; // absolute floor for the contrast test (dark areas)
const float FXAA_DIR_STEP_CAP = 8.0;          // max length of the edge-direction step, in texels
const float FXAA_SUBPIXEL_CAP = 1.0;

float luma_of(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

// sRGB -> linear, for the final write: the swapchain attachment encodes linear -> sRGB in hardware,
// so handing it an already-encoded value would double-encode (see the gamma note in post.frag).
vec3 srgb_to_linear(vec3 color) {
    // NOTE: no local `const` - glslc rejects it ("unexpected CONST")
    vec3 low = color / 12.92;
    vec3 high = pow(max((color + 0.055) / 1.055, vec3(0.0)), vec3(2.4));
    return mix(low, high, step(vec3(0.04045), color));
}

void main() {
    vec2 texel = 1.0 / vec2(textureSize(display_color, 0));

    vec3 rgb_m = texture(display_color, v_uv).rgb;
    vec3 rgb_nw = texture(display_color, v_uv + vec2(-1.0, -1.0) * texel).rgb;
    vec3 rgb_ne = texture(display_color, v_uv + vec2(1.0, -1.0) * texel).rgb;
    vec3 rgb_sw = texture(display_color, v_uv + vec2(-1.0, 1.0) * texel).rgb;
    vec3 rgb_se = texture(display_color, v_uv + vec2(1.0, 1.0) * texel).rgb;

    float luma_m = luma_of(rgb_m);
    float luma_nw = luma_of(rgb_nw);
    float luma_ne = luma_of(rgb_ne);
    float luma_sw = luma_of(rgb_sw);
    float luma_se = luma_of(rgb_se);

    float luma_min = min(luma_m, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    float luma_max = max(luma_m, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));

    // flat area: keep the pixel untouched (this early-out is what leaves text and smooth gradients
    // alone, and it is why most pixels cost only the five taps above)
    if ((luma_max - luma_min) < max(FXAA_EDGE_THRESHOLD_MIN, luma_max * pc.fxaa_edge_threshold)) {
        out_color = vec4(srgb_to_linear(rgb_m), 1.0);
        return;
    }

    // edge direction: perpendicular to the local gradient, from the diagonal luma differences
    vec2 dir = vec2(-((luma_nw + luma_ne) - (luma_sw + luma_se)), ((luma_nw + luma_sw) - (luma_ne + luma_se)));

    // shorten the direction where the gradients nearly cancel (|dir| small), then normalise so the
    // step is measured in texels and clamp its length
    float dir_reduce = max((luma_nw + luma_ne + luma_sw + luma_se) * 0.25 * 0.03125, 1.0 / 128.0);
    float rcp_dir_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + dir_reduce);
    dir = clamp(dir * rcp_dir_min, vec2(-FXAA_DIR_STEP_CAP), vec2(FXAA_DIR_STEP_CAP)) * texel;

    // two-step search along the edge: a near pair and a far pair; pick the far pair unless its luma
    // already left the neighbourhood's range (that would mean the blend ran into another feature)
    vec3 rgb_a = 0.5 * (texture(display_color, v_uv + dir * (1.0 / 3.0 - 0.5)).rgb +
                        texture(display_color, v_uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 rgb_b = rgb_a * 0.5 + 0.25 * (texture(display_color, v_uv + dir * -0.5).rgb +
                                       texture(display_color, v_uv + dir * 0.5).rgb);
    float luma_b = luma_of(rgb_b);
    vec3 rgb_result = (luma_b < luma_min || luma_b > luma_max) ? rgb_a : rgb_b;

    // sub-pixel term: on top of the directional blend, mix toward the centre pixel where the pixel
    // sits noticeably above the local average (removes the single-pixel "sparkle" FXAA otherwise
    // leaves on near-horizontal/vertical edges). 0 disables it.
    if (pc.fxaa_subpixel > 0.0) {
        float luma_lowpass = (luma_nw + luma_ne + luma_sw + luma_se + luma_m * 4.0) * 0.125;
        float range = luma_max - luma_min;
        float subpixel = clamp(abs(luma_lowpass - luma_m) / max(range, 1e-5), 0.0, 1.0);
        subpixel = smoothstep(0.0, 1.0, subpixel);
        subpixel = subpixel * subpixel * clamp(pc.fxaa_subpixel, 0.0, FXAA_SUBPIXEL_CAP);
        rgb_result = mix(rgb_result, rgb_m, subpixel);
    }

    // Hand the result back in the encoding the TARGET expects: an sRGB swapchain attachment encodes
    // linear -> sRGB in hardware, so it must receive linear values (encode_gamma == 0); a UNORM
    // swapchain does no encoding, so the display-encoded result is stored as-is (encode_gamma == 1).
    // Ignoring this branch made the UNORM case too dark by a whole gamma.
    if (pc.encode_gamma > 0.5) {
        out_color = vec4(rgb_result, 1.0);
    } else {
        out_color = vec4(srgb_to_linear(rgb_result), 1.0);
    }
}
