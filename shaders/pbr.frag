#version 450

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec3 v_tangent;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
} camera;

layout(set = 1, binding = 0) uniform sampler2D base_color_texture;
layout(set = 1, binding = 1) uniform sampler2D metallic_roughness_texture;
layout(set = 1, binding = 2) uniform sampler2D normal_texture;
layout(set = 1, binding = 3) uniform sampler2D occlusion_texture;
layout(set = 1, binding = 4) uniform sampler2D emissive_texture;

layout(push_constant) uniform PushConstants {
    vec4 base_color_factor;
    vec4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    uint flags; // bit0: has normal map, bit1: has occlusion map, bit2: has emissive map
} push;

const float PI = 3.14159265359;

// 法线分布函数：GGX / Trowbridge-Reitz
float distribution_ggx(vec3 n, vec3 h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float ndoth = max(dot(n, h), 0.0);
    float denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// 几何遮蔽：Schlick-GGX（直接光照变体）
float geometry_schlick_ggx(float ndotv, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return ndotv / (ndotv * (1.0 - k) + k);
}

float geometry_smith(vec3 n, vec3 v, vec3 l, float roughness) {
    float ndotv = max(dot(n, v), 0.0);
    float ndotl = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(ndotv, roughness) * geometry_schlick_ggx(ndotl, roughness);
}

// 菲涅尔：Schlick 近似
vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// 菲涅尔：粗糙度变体（IBL 环境反射用，粗糙度越高反射越弱）
vec3 fresnel_schlick_roughness(float cos_theta, vec3 f0, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// 程序化环境光：按方向采样渐变天空/地面 + 太阳亮斑，低成本模拟 IBL
vec3 environment_color(vec3 dir) {
    float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0); // 0 朝下，1 朝上
    const vec3 ground = vec3(0.03, 0.03, 0.05) * 0.75;
    const vec3 horizon = vec3(0.16, 0.19, 0.26) * 0.75;
    const vec3 sky = vec3(0.28, 0.45, 0.75) * 0.75;
    vec3 env = t < 0.5 ? mix(ground, horizon, t * 2.0) : mix(horizon, sky, (t - 0.5) * 2.0);
    // 太阳亮斑：方向与主光对齐时最亮，金属表面的"光泽"主要来自这里
    const vec3 sun_dir = normalize(vec3(0.3, 1.0, 0.5));
    float sun = pow(max(dot(dir, sun_dir), 0.0), 64.0);
    env += vec3(1.0, 0.95, 0.85) * sun * 1.35;
    return env;
}

// ACES 电影级色调映射
vec3 aces_tone_mapping(vec3 color) {
    return clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    // ---- 材质参数：因子 × 贴图 ----
    vec4 base_color = push.base_color_factor * texture(base_color_texture, v_uv);
    float metallic = push.metallic_factor * texture(metallic_roughness_texture, v_uv).b;
    float roughness = push.roughness_factor * texture(metallic_roughness_texture, v_uv).g;
    float ao = texture(occlusion_texture, v_uv).r;
    vec3 emissive = push.emissive_factor.rgb * texture(emissive_texture, v_uv).rgb;

    // ---- 法线：可选切线空间法线贴图，否则使用插值法线 ----
    vec3 n;
    if ((push.flags & 1u) != 0u) {
        vec3 tangent = normalize(v_tangent);
        vec3 normal = normalize(v_normal);
        vec3 bitangent = normalize(cross(normal, tangent));
        vec3 tbn_normal = texture(normal_texture, v_uv).rgb * 2.0 - 1.0;
        tbn_normal.xy *= push.normal_scale;
        tbn_normal = normalize(tbn_normal);
        n = normalize(mat3(tangent, bitangent, normal) * tbn_normal);
    } else {
        n = normalize(v_normal);
    }

    // ---- Cook-Torrance BRDF（单一方向光） ----
    vec3 v = normalize(camera.camera_pos - v_world_pos);
    vec3 l = normalize(vec3(0.3, 1.0, 0.5));
    vec3 h = normalize(v + l);

    vec3 f0 = mix(vec3(0.04), base_color.rgb, metallic);

    float ndf = distribution_ggx(n, h, roughness);
    float g = geometry_smith(n, v, l, roughness);
    vec3 f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    vec3 numerator = ndf * g * f;
    float denominator = 4.0 * max(dot(n, v), 0.0) * max(dot(n, l), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;

    vec3 kd = (1.0 - f) * (1.0 - metallic);
    float ndotl = max(dot(n, l), 0.0);
    vec3 radiance = vec3(7.5) * ndotl;

    vec3 diffuse = kd * base_color.rgb / PI;

    // ---- 模拟 IBL 环境光：增强金属/光滑表面的光泽 ----
    // 漫反射按法线方向采样环境；镜面反射按粗糙度模糊后的反射方向采样，
    // 再用菲涅尔（粗糙度变体）控制反射强度：金属反射最强、粗糙表面最弱
    vec3 r = reflect(-v, n);
    r = normalize(mix(r, n, roughness * 0.7));
    vec3 ibl_diffuse = environment_color(n);
    vec3 ibl_specular = environment_color(r) * ao;
    vec3 fresnel_ibl = fresnel_schlick_roughness(max(dot(n, v), 0.0), f0, roughness);

    vec3 ambient = ibl_diffuse * base_color.rgb * ao;
    vec3 specular_ibl = ibl_specular * fresnel_ibl;

    vec3 color = ambient + (diffuse + specular) * radiance + specular_ibl + emissive;

    // ---- 色调映射 + Gamma 校正 ----
    color = aces_tone_mapping(color);
    color = pow(color, vec3(1.0 / 2.2));

    out_color = vec4(color, base_color.a);
}
