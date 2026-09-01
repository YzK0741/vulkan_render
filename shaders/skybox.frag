#version 450

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec3 v_dir;

// Analytic sky, matching vulkan.math::environment_color (which bakes the IBL cubemap), so the
// visible sky and environment reflections agree exactly. Computed per-pixel from the view ray —
// like UE's SkyAtmosphere — instead of sampling the env cubemap, which eliminates every cubemap
// face/texel artifact (the "inside a cube" look): no face seams, no banding, no texel steps.
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

// Same exposure pipeline as pbr.frag: ACES filmic tonemapping + gamma, so the sky matches the
// lit models instead of appearing as raw dark linear values on the sRGB swapchain
vec3 aces_tone_mapping(vec3 color) {
    return clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec3 color = sky_color(normalize(v_dir));
    color = aces_tone_mapping(color);
    color = pow(color, vec3(1.0 / 2.2));
    out_color = vec4(color, 1.0);
}
