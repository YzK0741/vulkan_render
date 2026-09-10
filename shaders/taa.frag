#version 450

/**
 * @file shaders/taa.frag
 * @brief Temporal anti-aliasing: blend this frame with a reprojected, neighborhood-clamped history.
 * @ingroup shaders
 *
 * The deferred path renders at 1x, so it has no multisampling to hide edges with. TAA replaces it:
 * every frame the projection is offset by a different sub-pixel amount (a Halton(2,3) sequence, see
 * runtime::taa_jitter), so consecutive frames sample the image at different positions, and this pass
 * accumulates them - the resolution the eye sees is the accumulated one, not the per-frame one. It
 * also removes the sub-pixel shimmer that no post-process edge filter (FXAA) can touch, because that
 * shimmer IS the missing samples.
 *
 * Inputs (its own descriptor set):
 * - binding 0: the current frame's scene color (HDR, jittered - the image to accumulate)
 * - binding 1: the history image (the previous RESOLVED frame, one per swapchain image)
 * - binding 2: the motion vectors (G-buffer RG16F: current_uv - previous_uv)
 * - binding 3: the G-buffer depth (D32_SFLOAT, sampled)
 *
 * Output: the resolved HDR image, which the runtime writes into the HDR target the post chain reads -
 * so bloom/composite/FXAA are untouched by TAA existing, and a copy of the result becomes the next
 * frame's history.
 *
 * The three things that make temporal accumulation work rather than smear:
 * -# REPROJECTION: the history is sampled at `uv - velocity`, so camera motion (and, once the G-buffer
 *    carries it, object motion) reads from the pixel that actually shows the same surface;
 * -# REJECTION: the history sample is clamped into an axis-aligned box around the current frame's 3x3
 *    neighborhood. A history value outside the range of what this pixel currently looks like is
 *    stale (an occlusion change, a disocclusion, a lighting change) and is pulled back to the closest
 *    valid value instead of being blended in as a ghost;
 * -# DEPTH GUARD: a history sample whose reprojected depth disagrees with this frame's depth belongs
 *    to a different surface (the classic disocclusion at a silhouette) and is dropped entirely. The
 *    velocity reprojection alone cannot know that - the previous frame's surface there was simply
 *    something else.
 *
 * The blend weight is fixed but motion-aware: a fast-moving pixel trusts the current frame more
 * (a long history for a surface that is barely on screen is mostly stale data), while a static pixel
 * converges to a smooth result.
 */

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D current_color; // this frame's scene color (HDR, jittered)
layout(set = 0, binding = 1) uniform sampler2D history_color; // the previous resolved frame
layout(set = 0, binding = 2) uniform sampler2D velocity;      // motion vectors in UV space
layout(set = 0, binding = 3) uniform sampler2D depth;         // G-buffer depth (linearized for the guard)

layout(push_constant) uniform TaaPush {
    // 1 = trust the history (the normal case), 0 = use the current frame only (the first frame for
    // this image, or a swapchain recreation: the history image holds nothing meaningful then)
    float history_valid;
    float blend_static; // history weight for a static pixel (0.9 = 10% of the current frame each frame)
    float blend_min;    // history weight floor under motion (never go fully current-frame: that is a
                        // non-temporal image with extra cost, which is what FXAA already is)
    // The texel size stays two SCALARS: a vec2 aligns to 8 bytes in the push constant block, which
    // would insert padding the C++ struct does not have (validation caught it as a block range of 36
    // bytes against the layout's 32 - and the fields after it would have been read from the wrong
    // offsets, which is worse than the size mismatch).
    float texel_size_x; // 1 / target width  (the 3x3 neighborhood + the depth guard's radius)
    float texel_size_y; // 1 / target height
    float depth_scale;  // the projection's depth-linearization terms, for the disocclusion guard
    float depth_offset;
    float _pad;
} pc;

/// @brief the target's texel size as a vector
vec2 texel_size() {
    return vec2(pc.texel_size_x, pc.texel_size_y);
}

/**
 * @brief view-space distance of a depth-buffer sample (same inversion as gbuffer_debug.frag)
 */
float view_depth(float raw_depth) {
    return pc.depth_offset / (raw_depth + pc.depth_scale);
}

/**
 * @brief resolve one pixel: clamp the reprojected history into the current neighborhood and blend
 * @param uv the pixel's texture coordinate
 * @return the resolved linear HDR color
 */
vec3 resolve(vec2 uv) {
    const vec3 center = texture(current_color, uv).rgb;
    const vec2 motion = texture(velocity, uv).rg;

    // ---- the current frame's local range (a 3x3 cross of the resampled color) ----
    vec3 neighborhood_min = center;
    vec3 neighborhood_max = center;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            const vec3 sample_color = texture(current_color, uv + vec2(float(x), float(y)) * texel_size()).rgb;
            neighborhood_min = min(neighborhood_min, sample_color);
            neighborhood_max = max(neighborhood_max, sample_color);
        }
    }

    // ---- reproject the history ----
    const vec2 history_uv = uv - motion;
    // outside the frame there is no history: whatever the clamp below does with the border texel
    // would be a guess, so fall back to the current frame entirely
    if (history_uv.x < 0.0 || history_uv.x > 1.0 || history_uv.y < 0.0 || history_uv.y > 1.0) {
        return center;
    }
    vec3 history = texture(history_color, history_uv).rgb;

    // ---- disocclusion guard: is the reprojected texel the same surface? ----
    // The motion vector alone cannot tell: at a silhouette the previous frame's pixel at that
    // location showed the background, and blending it in is the classic TAA ghost trail. Comparing
    // the two view depths (with a relative tolerance that scales with distance) rejects those.
    const float current_depth = view_depth(texture(depth, uv).r);
    const float history_depth = view_depth(texture(depth, history_uv).r);
    const float depth_tolerance = max(0.02 * current_depth, 0.1); // metres-ish, scale-free enough
    if (abs(current_depth - history_depth) > depth_tolerance) {
        return center;
    }

    // ---- rejection: pull a stale history value into what this pixel looks like now ----
    history = clamp(history, neighborhood_min, neighborhood_max);

    // ---- blend, trusting the history less the faster the pixel moves ----
    const float speed = length(motion / texel_size()); // motion in pixels
    const float history_weight = mix(pc.blend_static, pc.blend_min, clamp(speed, 0.0, 1.0)) * pc.history_valid;
    return mix(center, history, history_weight);
}

/**
 * @brief resolve the pixel and write the accumulated HDR color
 */
void main() {
    out_color = vec4(resolve(v_uv), 1.0);
}
