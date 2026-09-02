#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec3 v_tangent;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
} camera;

// Single flat scene set: shared camera UBO (binding 0), runtime texture array (binding 1,
// descriptor indexing: partially bound + update-after-bind), shared IBL (bindings 2-4) and
// the GPU material table (binding 5: per-material texture indices + factors, see material_record).
layout(set = 0, binding = 1) uniform sampler2D textures[];
layout(set = 0, binding = 2) uniform samplerCube env_sampler;        // prefiltered environment (roughness mip chain)
layout(set = 0, binding = 3) uniform samplerCube irradiance_sampler; // irradiance map (diffuse IBL)
layout(set = 0, binding = 4) uniform sampler2D brdf_lut_sampler;     // BRDF integration LUT

// One entry of the material table; layout matches material_record in vulkan/model.cppm (std430, 80 bytes)
struct Material {
    uvec4 tex_indices; // albedo, metallic-roughness, normal, occlusion (indices into textures[])
    uint emissive_index;
    uint _pad[3];
    vec4 base_color_factor;
    vec4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    uint flags; // bit0: has normal map, bit1: has occlusion map, bit2: has emissive map
};
layout(set = 0, binding = 5) readonly buffer Materials { Material materials[]; };

layout(push_constant) uniform PushConstants {
    uint material_index; // index into the material table (material data lives on the GPU)
    mat4 model; // per-model world transform (kept out of the shared camera UBO)
} push;

const float PI = 3.14159265359;
const float ENV_MIP_COUNT = 5.0; // must match the prefiltered-env mip count generated on the CPU

// Normal distribution function: GGX / Trowbridge-Reitz
float distribution_ggx(vec3 n, vec3 h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float ndoth = max(dot(n, h), 0.0);
    float denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Geometric shadowing: Schlick-GGX (direct-lighting variant)
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

// Fresnel: Schlick approximation
vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// ---- IBL: split-sum approximation (ported from glTF-Sample-Renderer's ibl.glsl) ----

// Diffuse ambient: irradiance map
vec3 get_diffuse_light(vec3 n) {
    return texture(irradiance_sampler, n).rgb;
}

// Specular ambient: sample the prefiltered env by lod
vec3 get_specular_sample(vec3 reflection, float lod) {
    return textureLod(env_sampler, reflection, lod).rgb;
}

// Single-scatter + multi-scatter-compensated Fresnel weights (BRDF LUT), from Fdez-Aguera
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

// ACES filmic tonemapping
vec3 aces_tone_mapping(vec3 color) {
    return clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    // ---- Material: one GPU-side record (texture indices + factors + flags) ----
    Material mat = materials[push.material_index];

    // ---- Material parameters: factor * texture (indices come from the material record) ----
    vec4 base_color = mat.base_color_factor * texture(textures[mat.tex_indices.x], v_uv);
    float metallic = mat.metallic_factor * texture(textures[mat.tex_indices.y], v_uv).b;
    float roughness = mat.roughness_factor * texture(textures[mat.tex_indices.y], v_uv).g;
    float ao = texture(textures[mat.tex_indices.w], v_uv).r;
    vec3 emissive = mat.emissive_factor.rgb * texture(textures[mat.emissive_index], v_uv).rgb;

    // ---- Normal: optional tangent-space normal map, else interpolated normal ----
    vec3 n;
    if ((mat.flags & 1u) != 0u) {
        vec3 tangent = normalize(v_tangent);
        vec3 normal = normalize(v_normal);
        vec3 bitangent = normalize(cross(normal, tangent));
        vec3 tbn_normal = texture(textures[mat.tex_indices.z], v_uv).rgb * 2.0 - 1.0;
        tbn_normal.xy *= mat.normal_scale;
        tbn_normal = normalize(tbn_normal);
        n = normalize(mat3(tangent, bitangent, normal) * tbn_normal);
    } else {
        n = normalize(v_normal);
    }
    // double-sided material (record flag bit3): mirror the normal on back faces, as the glTF
    // spec requires, so the inner side of a shell is lit by its inward-facing normal
    if ((mat.flags & 8u) != 0u && !gl_FrontFacing) {
        n = -n;
    }

    // ---- Cook-Torrance BRDF (single directional light) ----
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

    // ---- IBL (split-sum): diffuse irradiance + prefiltered specular ----
    vec3 ibl_diffuse = get_diffuse_light(n);
    vec3 ibl_specular = get_ibl_radiance_ggx(n, v, roughness);
    vec3 fresnel_ibl = get_ibl_ggx_fresnel(n, v, roughness, f0, 1.0);

    // Metals have no diffuse term: diffuse ambient is scaled by (1 - metallic),
    // metal color comes entirely from specular environment (matches the official mix(dielectric, metal, metallic))
    vec3 ambient = ibl_diffuse * base_color.rgb * ao * (1.0 - metallic);
    vec3 specular_ibl = ibl_specular * fresnel_ibl * ao;

    vec3 color = ambient + (diffuse + specular) * radiance + specular_ibl + emissive;

    // ---- Tonemapping + gamma correction ----
    color = aces_tone_mapping(color);
    color = pow(color, vec3(1.0 / 2.2));

    out_color = vec4(color, base_color.a);
}
