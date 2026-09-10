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

// The sky is written as linear HDR radiance like the lit geometry; the post-process pass
// (post.frag) applies exposure + ACES + gamma once for the whole frame, so sky and models stay
// consistent without the skybox needing any descriptor set or push constant.
void main() {
    vec3 color = sky_color(normalize(v_dir));
    out_color = vec4(color, 1.0);
}
