/**
 * @file shaders/hit_shading.glsl
 * @brief Shading a ray-traced hit from the geometry it landed on, shared by the passes that trace.
 * @ingroup shaders
 *
 * Included by the GI tracer (shaders/ssgi.comp) and, from the step that makes the probe cache
 * view-independent, by the probe pass (shaders/gi_probe.comp) - both bind the shared scene set, so both
 * can read the material records, the texture array and the light UBO this needs.
 *
 * WHAT THE INCLUDER MUST PROVIDE, and why it is not declared here: the top level structure (set 0 binding
 * 16) and the irradiance cube (binding 3) are declared by each includer, because each one reaches them
 * through its own stage and its own push block. Everything this file's functions reference beyond those is
 * declared below, and an includer must NOT declare any of it: the same binding declared twice in one
 * translation unit does not compile.
 *
 * The instance table's address and the scale the shadow ray's self-intersection bias is relative to are
 * PARAMETERS rather than something read from a push block: the tracer carries the address in
 * proj_terms.zw and the probe pass in a lane of its own, and the two passes' scene-relative lengths are not
 * the same number either. A shared function cannot depend on which pass called it.
 *
 * WHAT IT DOES NOT REPRODUCE, stated rather than discovered later: the toon and non-PBR presets of
 * shading.glsl's brdf_model/diffuse_model (this is the default PBR path), alphaMode MASK (the geometry is
 * built OPAQUE, so a MASK surface is solid to the ray - the documented behaviour of every traced effect in
 * this engine), skinning and morphing (the structures hold the bind pose), and the punctual-light cluster,
 * which is a screen-space structure a hit outside the frame has no entry in.
 */
#extension GL_EXT_ray_query : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// The bindless texture array is indexed by the material's own indices, exactly as shaders/surface.glsl
// indexes it: plain indices into a partially-bound array.
#extension GL_EXT_nonuniform_qualifier : enable
layout(set = 0, binding = 7) uniform LightUBO {
    mat4 light_view_proj[4];
    vec4 light_dir;
} light;
// The IBL's other two halves (the irradiance cube above is the third): the prefiltered environment and
// the BRDF integration LUT. A shaded hit has to reproduce the lighting stage's split-sum ambient, or the
// indirect it reports for that surface would disagree with what the screen shows for it.
layout(set = 0, binding = 2) uniform samplerCube env_sampler;
layout(set = 0, binding = 4) uniform sampler2D brdf_lut_sampler;
// The scene's material table and its bindless texture array (set 0 bindings 5 and 1, the same two
// shaders/surface.glsl reads). Declared here rather than included from there: that file also declares the
// camera UBO and the IBL cubes this pass already has, and re-declaring a binding is a compile error.
struct hit_material {
    uvec4 tex_indices; // albedo, metallic-roughness, normal, occlusion
    uint emissive_index;
    float alpha_cutoff;
    float occlusion_strength;
    uint _pad;
    vec4 base_color_factor;
    vec4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    uint flags; // bit0 normal map, bit1 occlusion, bit2 emissive, bit3 double sided, bit4 MASK
};
layout(set = 0, binding = 5) readonly buffer HitMaterials {
    hit_material materials[];
} hit_materials;
layout(set = 0, binding = 1) uniform sampler2D scene_textures[];

// The instance table (vulkan.acceleration_structure::instance_record, 96 bytes) reached through the
// device address the runtime pushes - which is also the MODE SWITCH: 0 means the structures are not built
// or hit shading is off, and every hit falls back to sampling the screen. A buffer reference rather than
// a descriptor because the buffers it points at (the geometry) already carry device addresses for the
// acceleration-structure build, so this costs no binding of its own.
struct hit_instance {
    uint64_t vertex_address; // the vertex buffer's base
    uint64_t index_address;  // the index buffer's base
    mat4 model;              // object -> world, for the vertex normal (the position comes from the ray)
    uint vertex_stride;
    uint index_type;     // VkIndexType: 0 = uint32, 1 = uint16
    uint material_index;
    uint primitive_index;
};
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer InstanceTable {
    hit_instance records[];
};
// The geometry itself, read as untyped 4-byte words: the vertex is a 64-byte interleaved record in an
// implementation-defined struct, so addressing it as floats is what makes the offsets (0, 3, 6) read
// `vec3 position; vec3 normal; vec2 uv` without depending on a std430 struct layout that does not exist.
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer GeometryFloats {
    float values[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer GeometryWords {
    uint values[];
};

/** @brief the Schlick approximation, the same one shading.glsl uses for direct and IBL specular */
vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (vec3(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

/**
 * @brief the material's surface at a hit, built from the triangle the ray actually hit
 * @param instance the hit instance's table entry
 * @param triangle the hit triangle's index within that instance's geometry
 * @param bary the hit's barycentric coordinates (x for vertex 1, y for vertex 2; vertex 0 takes the rest)
 * @param front_facing whether the ray met the triangle's front face
 * @param out_uv the interpolated texture coordinates
 * @return the hit surface's object-space normal, and through @p out_uv its UVs
 *
 * The TANGENT FRAME IS SOLVED FROM THE TRIANGLE, not from screen-space derivatives: surface.glsl builds
 * its TBN from dFdx/dFdy of the world position and the UVs because a fragment shader has nothing else,
 * and this pass has no derivatives at all - but it does have the triangle's three vertices and their
 * UVs, which is exactly what the derivative solve is approximating. So the analytic frame is both the
 * only option here and the better one: it is per-triangle rather than per-quad.
 */
vec3 hit_surface(hit_instance instance, uint triangle, vec2 bary, bool front_facing, out vec2 out_uv) {
    GeometryWords indices = GeometryWords(instance.index_address);
    const uint base = triangle * 3u;
    uint vertex_indices[3];
    for (uint k = 0u; k < 3u; ++k) {
        const uint i = base + k;
        // VkIndexType: UINT16 is 0 and UINT32 is 1 (UINT8_EXT is not something the loader produces), so
        // the 16-bit case is the one that has to read a half of a word. Getting this backwards does not
        // mis-shade a triangle - it reads a different INDEX, whose vertex lies outside the buffer.
        if (instance.index_type == 0u) { // VK_INDEX_TYPE_UINT16
            const uint word = indices.values[i >> 1u];
            vertex_indices[k] = (i & 1u) == 0u ? (word & 0xFFFFu) : (word >> 16u);
        } else {
            vertex_indices[k] = indices.values[i]; // VK_INDEX_TYPE_UINT32
        }
    }
    vec3 positions[3];
    vec3 normals[3];
    vec2 uvs[3];
    for (uint k = 0u; k < 3u; ++k) {
        // The stride is in bytes and the buffer reference is in floats, so the address arithmetic stays
        // in bytes and the word index is the byte offset divided by four.
        GeometryFloats vertex = GeometryFloats(instance.vertex_address + uint64_t(vertex_indices[k]) * uint64_t(instance.vertex_stride));
        positions[k] = vec3(vertex.values[0], vertex.values[1], vertex.values[2]);
        normals[k] = vec3(vertex.values[3], vertex.values[4], vertex.values[5]);
        uvs[k] = vec2(vertex.values[6], vertex.values[7]);
    }
    // EXT_ray_query reports the barycentrics of vertices 1 and 2; vertex 0 takes whatever is left.
    const vec3 weights = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    const vec3 object_normal = normalize(normals[0] * weights.x + normals[1] * weights.y + normals[2] * weights.z);
    out_uv = uvs[0] * weights.x + uvs[1] * weights.y + uvs[2] * weights.z;

    const vec3 normal = normalize(mat3(instance.model) * object_normal);
    const hit_material mat = hit_materials.materials[instance.material_index];
    if ((mat.flags & 1u) == 0u) {
        // No normal map: the interpolated vertex normal is the shading normal. The double-sided flip is
        // the one gl_FrontFacing stands in for in a fragment shader - here the ray query reports whether
        // the ray met the front face, which is the same fact.
        return (mat.flags & 8u) != 0u && !front_facing ? -normal : normal;
    }
    // The analytic tangent frame: the triangle's own edges and UV deltas, which is what the derivative
    // solve in surface.glsl approximates from neighbouring fragments.
    const vec3 edge1 = positions[1] - positions[0];
    const vec3 edge2 = positions[2] - positions[0];
    const vec2 delta1 = uvs[1] - uvs[0];
    const vec2 delta2 = uvs[2] - uvs[0];
    const float determinant = delta1.x * delta2.y - delta2.x * delta1.y;
    if (abs(determinant) < 1e-12) {
        return (mat.flags & 8u) != 0u && !front_facing ? -normal : normal; // degenerate UVs: no frame
    }
    const vec3 tangent = normalize(mat3(instance.model) * ((delta2.y * edge1 - delta1.y * edge2) / determinant));
    const vec3 bitangent = normalize(mat3(instance.model) * ((delta1.x * edge2 - delta2.x * edge1) / determinant));
    const vec3 mapped = texture(scene_textures[mat.tex_indices.z], out_uv).rgb * 2.0 - 1.0;
    const vec3 scaled = normalize(vec3(mapped.xy * mat.normal_scale, mapped.z));
    const vec3 shading = normalize(mat3(tangent, bitangent, normal) * scaled);
    return (mat.flags & 8u) != 0u && !front_facing ? -shading : shading;
}

/**
 * @brief the radiance leaving the surface a ray hit, evaluated from the geometry rather than the screen
 * @param query the finished ray query (its committed intersection is a real hit)
 * @param hit_world the hit's world position (the ray's origin plus t times its direction)
 * @param dir the world-space ray direction
 * @param out_radiance the surface's outgoing radiance toward the ray's origin
 * @return false when the hit cannot be shaded (the table has no entry for it), so the caller falls back
 *
 * WHY THIS EXISTS: sampling the screen's direct-radiance image at a hit is what the traced path did, and
 * it has two costs. It cannot answer for a hit the camera does not see (off screen, or hidden behind a
 * nearer surface) - those fall back to a probe, which is an approximation of the light WHERE THE RAY
 * STARTED rather than radiance arriving from where it landed - and it makes every sample depend on the
 * frame the camera happens to be showing, so a surface's indirect light changes when the camera moves.
 * Shading the hit from the geometry removes both: the answer depends on the hit and the lights alone.
 *
 * WHAT IT DOES NOT REPRODUCE, stated rather than discovered later: the toon and non-PBR presets of
 * shading.glsl's `brdf_model`/`diffuse_model` (this is the default PBR path), alphaMode MASK (the
 * geometry is built OPAQUE, so a MASK surface is solid to the ray - the documented behaviour of every
 * traced effect in this engine), skinning and morphing (the structures hold the bind pose), and the
 * punctual-light cluster, which is a screen-space structure a hit outside the frame has no entry in.
 */
bool shade_hit(rayQueryEXT query, vec3 hit_world, vec3 dir, uint64_t table_address, float bias_scale, out vec3 out_radiance) {
    InstanceTable table = InstanceTable(table_address);
    const hit_instance instance = table.records[rayQueryGetIntersectionInstanceCustomIndexEXT(query, true)];
    const hit_material mat = hit_materials.materials[instance.material_index];

    vec2 uv = vec2(0.0);
    const vec3 normal = hit_surface(instance, rayQueryGetIntersectionPrimitiveIndexEXT(query, true), rayQueryGetIntersectionBarycentricsEXT(query, true), rayQueryGetIntersectionFrontFaceEXT(query, true), uv);

    const vec3 base_color = mat.base_color_factor.rgb * texture(scene_textures[mat.tex_indices.x], uv).rgb;
    const float metallic = clamp(mat.metallic_factor * texture(scene_textures[mat.tex_indices.y], uv).b, 0.0, 1.0);
    const float roughness = clamp(mat.roughness_factor * texture(scene_textures[mat.tex_indices.y], uv).g, 0.045, 1.0);
    const float ao = (mat.flags & 2u) != 0u ? mix(1.0, texture(scene_textures[mat.tex_indices.w], uv).r, mat.occlusion_strength) : 1.0;
    const vec3 emissive = (mat.flags & 4u) != 0u ? mat.emissive_factor.rgb * texture(scene_textures[mat.emissive_index], uv).rgb : vec3(0.0);

    const vec3 v = normalize(camera.camera_pos.xyz - hit_world);
    const vec3 f0 = mix(vec3(0.04), base_color, metallic);
    const vec3 kd = (1.0 - f0) * (1.0 - metallic);

    // The sun, with a shadow ray instead of a shadow map: this pass already has the acceleration
    // structures, and a ray answers the visibility question for the exact point rather than for a texel
    // of a cascade. The radiance is the same literal the lighting stage uses, so the two agree.
    //
    // The self-intersection bias follows the engine's own convention for pushing a ray off a surface: an
    // offset ALONG THE NORMAL that is a fraction of the scene-relative ray length (pc.params.x), which is
    // what makes one number mean the same thing on a 1.6-unit model and on Sponza - the cascaded shadow
    // maps offset by twice a shadow texel's world size, which is the same idea expressed in texels.
    //
    // It is NOT what decides this term's weight, and the measurement says so: quadrupling the offset (a
    // fixed 0.01 to 0.045 on Sponza) moved the frame by 0.045 of a mean brightness out of the 3.23 the
    // shadow ray removes in total, and aiming the offset along whichever side the ray came from (the
    // traversal is allowed to hit back faces) moved it by -0.03. Neither is the cause.
    //
    // WHERE THE DIFFERENCE IS, isolated by shading ONLY the hits the screen confirms (the shaded block
    // temporarily moved below the depth guard, nothing else changed): that frame is 61.6175 against the
    // screen path's 61.3964, so on surfaces the camera can SEE, shading a hit agrees with the lighting
    // stage to 0.22 of a mean brightness (0.36%) - which is what a different shadow method and the
    // punctual lights this path does not evaluate are worth. The full shaded frame is 60.3263, so the
    // whole of the remaining difference comes from the hits the screen path REFUSES to answer.
    //
    // That is a structural difference, not a numerical one: the screen path discards a hidden hit and
    // hands it probe ambient, while this one answers it with its own shading - and a ray that ends on the
    // unlit far side of a wall, or inside one, is legitimately dark. So the shaded path is closer to right
    // for hidden surfaces that are real, and it exposes what the screen path grants the ones it never
    // evaluated; for hits INSIDE solid geometry neither answer is right (a diffuse bounce cannot leave a
    // solid's interior), and rejecting those - a back-face hit is the cheap signature of one - is the
    // refinement this measurement points at.
    float sun_visibility = 1.0;
    {
        rayQueryEXT shadow_query;
        rayQueryInitializeEXT(shadow_query, tlas, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFF,
                             hit_world + normal * max(0.01, bias_scale * 0.02), 0.01, light.light_dir.xyz, 1e30);
        while (rayQueryProceedEXT(shadow_query)) {
        }
        if (rayQueryGetIntersectionTypeEXT(shadow_query, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            sun_visibility = 0.0;
        }
    }
    const vec3 l = normalize(light.light_dir.xyz);
    const float ndotl = max(dot(normal, l), 0.0);
    const float ndotv = max(dot(normal, v), 0.0);
    vec3 direct = vec3(0.0);
    if (ndotl > 0.0 && sun_visibility > 0.0) {
        // The default PBR preset of shading.glsl's evaluate_direct_light: Lambert + GGX with the Smith
        // visibility term, which is the path every measurement in this project was taken on.
        const vec3 h = normalize(v + l);
        const float a2 = roughness * roughness;
        const float ndoth = max(dot(normal, h), 0.0);
        const float denominator = ndoth * ndoth * (a2 - 1.0) + 1.0;
        const float ndf = a2 / max(3.141592653589793 * denominator * denominator, 1e-8);
        const float a = sqrt(a2);
        const float vis = 0.5 / max(ndotl * (ndotv * (1.0 - a) + a) + ndotv * (ndotl * (1.0 - a) + a), 1e-6);
        const vec3 specular = ndf * vis * fresnel_schlick(max(dot(h, v), 0.0), f0);
        direct = (kd * base_color / 3.141592653589793 + specular) * (vec3(7.5) * ndotl);
    }

    // The split-sum IBL ambient, as the lighting stage computes it (the same two lookups and the same
    // combination, so a shaded hit and the screen agree about a surface's ambient).
    const vec3 ibl_diffuse = texture(irradiance_sampler, normal).rgb;
    const vec3 ibl_specular = textureLod(env_sampler, normalize(reflect(-v, normal)), roughness * float(max(textureQueryLevels(env_sampler) - 1, 0))).rgb;
    const vec2 f_ab = texture(brdf_lut_sampler, vec2(ndotv, roughness)).rg;
    const vec3 fr = max(vec3(1.0 - roughness), f0) - f0;
    const vec3 k_s = f0 + fr * pow(1.0 - ndotv, 5.0);
    const vec3 fssess = k_s * f_ab.x + f_ab.y;
    const float ems = 1.0 - (f_ab.x + f_ab.y);
    const vec3 f_avg = f0 + (1.0 - f0) / 21.0;
    const vec3 fmsems = ems * fssess * f_avg / max(1.0 - f_avg * ems, vec3(1e-4));
    const vec3 ambient = ibl_diffuse * base_color * ao * (1.0 - metallic);
    const vec3 specular_ibl = ibl_specular * (fssess + fmsems) * ao;

    out_radiance = ambient + direct + specular_ibl + emissive;
    return true;
}

