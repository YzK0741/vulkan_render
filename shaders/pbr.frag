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
layout(set = 1, binding = 5) uniform samplerCube env_sampler;        // 预过滤环境（粗糙度 mip 链）
layout(set = 1, binding = 6) uniform samplerCube irradiance_sampler; // 辐照度图（漫反射 IBL）
layout(set = 1, binding = 7) uniform sampler2D brdf_lut_sampler;     // BRDF 积分 LUT

layout(push_constant) uniform PushConstants {
    vec4 base_color_factor;
    vec4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    uint flags; // bit0: has normal map, bit1: has occlusion map, bit2: has emissive map
} push;

const float PI = 3.14159265359;
const float ENV_MIP_COUNT = 5.0; // 与 CPU 端生成的预过滤环境 mip 数一致

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

// ---- IBL：split-sum 近似（移植自 glTF-Sample-Renderer 的 ibl.glsl） ----

// 漫反射环境：辐照度图
vec3 get_diffuse_light(vec3 n) {
    return texture(irradiance_sampler, n).rgb;
}

// 镜面环境：按 lod 采样预过滤环境
vec3 get_specular_sample(vec3 reflection, float lod) {
    return textureLod(env_sampler, reflection, lod).rgb;
}

// 单次散射 + 多次散射补偿的 Fresnel 权重（BRDF LUT），来自 Fdez-Aguera
vec3 get_ibl_ggx_fresnel(vec3 n, vec3 v, float roughness, vec3 f0, float specular_weight) {
    float ndotv = clamp(dot(n, v), 0.0, 1.0);
    vec2 brdf_sample_point = clamp(vec2(ndotv, roughness), vec2(0.0), vec2(1.0));
    vec2 f_ab = texture(brdf_lut_sampler, brdf_sample_point).rg;
    vec3 fr = max(vec3(1.0 - roughness), f0) - f0;
    vec3 k_s = f0 + fr * pow(1.0 - ndotv, 5.0);
    vec3 fssess = specular_weight * (k_s * f_ab.x + f_ab.y);

    float ems = 1.0 - (f_ab.x + f_ab.y);
    vec3 f_avg = specular_weight * (f0 + (1.0 - f0) / 21.0);
    vec3 fmsems = ems * fssess * f_avg / (1.0 - f_avg * ems);

    return fssess + fmsems;
}

vec3 get_ibl_radiance_ggx(vec3 n, vec3 v, float roughness) {
    float lod = roughness * (ENV_MIP_COUNT - 1.0);
    vec3 reflection = normalize(reflect(-v, n));
    return get_specular_sample(reflection, lod);
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

    // ---- IBL（split-sum）：漫反射辐照度 + 预过滤镜面反射 ----
    vec3 ibl_diffuse = get_diffuse_light(n);
    vec3 ibl_specular = get_ibl_radiance_ggx(n, v, roughness);
    vec3 fresnel_ibl = get_ibl_ggx_fresnel(n, v, roughness, f0, 1.0);

    // 金属没有漫反射项：漫反射环境光按 (1 - metallic) 衰减，
    // 金属的颜色完全来自镜面反射环境（与官方 mix(dielectric, metal, metallic) 行为一致）
    vec3 ambient = ibl_diffuse * base_color.rgb * ao * (1.0 - metallic);
    vec3 specular_ibl = ibl_specular * fresnel_ibl * ao;

    vec3 color = ambient + (diffuse + specular) * radiance + specular_ibl + emissive;

    // ---- 色调映射 + Gamma 校正 ----
    color = aces_tone_mapping(color);
    color = pow(color, vec3(1.0 / 2.2));

    out_color = vec4(color, base_color.a);
}
