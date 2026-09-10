#version 450

/**
 * @file shaders/skybox.frag
 * @brief Analytic sky background, evaluated per pixel from the view ray.
 * @ingroup shaders
 *
 * The sky is written as linear HDR radiance like the lit geometry; the post-process pass
 * (post.frag) applies exposure + ACES + display encoding once for the whole frame, so sky and models
 * stay consistent without the skybox needing any descriptor set or push constant beyond the shared
 * scene set.
 */

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec3 v_dir;

/**
 * @brief analytic sky radiance for a world-space direction
 * @param dir normalized world-space view direction
 * @return linear HDR sky radiance (ground / horizon / sky bands plus a soft sun disc)
 *
 * Matches vulkan.math::environment_color (which bakes the IBL cubemap), so the visible sky and the
 * environment reflections agree exactly. Computed rather than sampled from the env cubemap - like
 * UE's SkyAtmosphere - which eliminates every cubemap face/texel artifact (the "inside a cube"
 * look): no face seams, no banding, no texel steps. The three bands are blended with smoothstep
 * (C1-smooth) rather than piecewise linear, and the sun disc sits along the same fixed sun direction
 * that pbr.frag and the shadow pass use.
 */
vec3 sky_color(vec3 dir) {
    // elevation t: 0 = nadir, 1 = zenith
    float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 ground = vec3(0.05, 0.05, 0.07) * 0.75;
    vec3 horizon = vec3(0.17, 0.20, 0.27) * 0.75;
    vec3 sky = vec3(0.28, 0.45, 0.75) * 0.75;
    // C1-smooth blending between the three bands (smoothstep, not piecewise linear)
    float g = smoothstep(0.28, 0.50, t); // ground -> horizon
    float s = smoothstep(0.50, 0.92, t); // horizon -> sky
    vec3 env = ground + (horizon - ground) * g;
    env += (sky - env) * s;
    // soft-edged sun disc along sun_dir
    vec3 sun_dir = normalize(vec3(0.3, 1.0, 0.5));
    env += vec3(1.0, 0.95, 0.85) * smoothstep(0.98, 1.0, dot(dir, sun_dir)) * 1.5;
    return env;
}

/**
 * @brief evaluate the sky for the interpolated view ray and write linear HDR radiance
 *
 * No exposure, tonemapping or encode happens here: the post pass owns all display-referred work, so
 * the sky and the lit geometry cannot drift apart.
 */
void main() {
    vec3 color = sky_color(normalize(v_dir));
    out_color = vec4(color, 1.0);
}
