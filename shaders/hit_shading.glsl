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
 * through its own stage and its own push block. The CAMERA is not needed at all: the direction towards the
 * viewer is a parameter, so an includer that asks on behalf of a ray rather than a pixel does not declare
 * it, and the probe pass does not. Everything this file's functions reference beyond those is
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
    vec4 cascade_splits;
    vec4 cascade_texel_world;
    float shadow_enabled;
    float brdf_model;
    float diffuse_model;
    float cascade_blend;
    float cascade_count;
    float rt_shadows;
    float sun_intensity;
    float furnace_level;
} light;
// The IBL's other two halves (the irradiance cube above is the third): the prefiltered environment and
// the BRDF integration LUT. A shaded hit has to reproduce the lighting stage's split-sum ambient, or the
// indirect it reports for that surface would disagree with what the screen shows for it.
layout(set = 0, binding = 2) uniform samplerCube env_sampler;
layout(set = 0, binding = 4) uniform sampler2D brdf_lut_sampler;
// ... and the shared specular half of the split sum (see shaders/ibl_specular.glsl): this file used to
// carry its own copy of the Fresnel weights inline, and the GI chain's new subtraction is the reason the
// copy became a shared definition instead of a third one.
#include "ibl_specular.glsl"
// The world-space cache's basis and its reconstruction - the ONE copy of the SH-2 basis (see that file's
// header: two copies would be two chances to disagree). Included here because a hit can now take its DIFFUSE
// AMBIENT from the cache instead of from the sky cube, which is what gives the shaded path a second bounce:
// the ambient a hit surface receives is answered by a world-space structure that knows where the hit IS.
#include "probe_sh.glsl"
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
    const uint base = triangle * 3u;
    uint vertex_indices[3];
    if (instance.index_address == 0u) {
        // NO INDEX BUFFER: this geometry is the mask bake's EXPANDED copy - three unique vertices per
        // triangle, with the ones a material's alphaMode MASK cut out written at a single position so no ray
        // can hit them (see shaders/mask_bake.comp) - so the triangle's vertices are simply consecutive.
        vertex_indices = uint[3](base, base + 1u, base + 2u);
    } else {
        GeometryWords indices = GeometryWords(instance.index_address);
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
 * @param cache_sh0 / @p cache_sh1 / @p cache_sh2 / @p cache_sh3 the world-space cache's four SH-2
 *        coefficient images, AS THE CALLER BINDS THEM - parameters rather than bindings, because the tracer
 *        reads them through the G-buffer set and the probe pass through its own ping-pong (see
 *        shaders/probe_sh.glsl's probe_sh_sample)
 * @param cache_grid the cache's mapping as (grid minimum corner.xyz, cell size.w), the same value the
 *        tracer already pushes for its own lookups
 * @param cache_gain how much of the cache's answer this hit's diffuse ambient may take. 0.0 uses the sky
 *        cube alone, which is what makes the callers that pass it bit-identical to before this parameter
 *        existed; the tracer passes [render] ssgi_bounce here, the knob that until now had no meaning on
 *        this path at all (see the ambient block below and shaders/ssgi.comp's push comment)
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
bool shade_hit(rayQueryEXT query, vec3 hit_world, vec3 dir, vec3 to_viewer, uint64_t table_address, float bias_scale,
               sampler3D cache_sh0, sampler3D cache_sh1, sampler3D cache_sh2, sampler3D cache_sh3, vec4 cache_grid, float cache_gain,
               out vec3 out_radiance) {
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

    // The direction towards whoever is ASKING for this radiance, passed in rather than read from the
    // camera. The tracer asks on behalf of a screen pixel, so it passes the direction to the eye; a probe
    // cell asks on behalf of its own ray, so it passes the direction back along that ray. Reading the
    // camera here instead made a probe ray's hit reflect towards the CAMERA - which is a view dependence
    // inside a cache whose whole purpose is to not have one, and it was invisible until the probe pass
    // stopped being a screen projection.
    const vec3 v = normalize(to_viewer);
    const vec3 f0 = mix(vec3(0.04), base_color, metallic);
    const vec3 kd = (1.0 - f0) * (1.0 - metallic);

    // The sun, with a shadow ray instead of a shadow map: this pass already has the acceleration
    // structures, and a ray answers the visibility question for the exact point rather than for a texel
    // of a cascade. The radiance is the same literal the lighting stage uses, so the two agree.
    //
    // The self-intersection bias follows the engine's own convention for pushing a ray off a surface: an
    // offset ALONG THE NORMAL that is a fraction of a length the CALLER supplies (`bias_scale`), floored at
    // 0.01. What that length is differs per caller and has to, because the thing being avoided differs:
    // the probe pass (shaders/gi_probe.comp) passes two of its own cell diagonals, since a probe's ray
    // starts in a cell whose size is the only scale it knows; the two traced lobes pass the WORLD-SPACE
    // bias they already computed for their own ray origin (see shaders/ssgi.comp's push comment), which is
    // a fraction of the scene radius rather than a fraction of the ray length.
    //
    // IT USED TO BE THE RAY LENGTH for both traced lobes, and that is a defect of the same class the ray
    // origin bias had: it made the shadow ray's start scale with a knob that means REACH, so the same
    // config meant a different distance on every scene - 0.045 world units above the surface inside Sponza
    // at the traced default, 0.093 at the glossy pass's own. Measured when both callers were changed: on
    // the material sweep, where hit shading is on, +0.0348 of mean green with 2.01% of pixels differing and
    // a worst pixel of 15 - and NOTHING at all in `sponza_gi`, because that scenario shades no hits (the
    // parameter is only read on the shaded path). The SIGN is not what the obvious story predicts: a larger
    // offset should let a shadow ray clear a nearby occluder and brighten the frame, and the SMALLER one
    // brightened it instead. Recorded as measured rather than explained, the way the offset's own earlier
    // measurement was (see the paragraph below). The probe pass's expression is untouched, so the
    // world-space cache's cells are bit-identical across this change.
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
        direct = (kd * base_color / 3.141592653589793 + specular) * (vec3(7.5 * light.sun_intensity) * ndotl);
    }

    // The split-sum IBL ambient, as the lighting stage computes it (the same two lookups and the same
    // combination, so a shaded hit and the screen agree about a surface's ambient). Both halves are
    // shaders/ibl_specular.glsl's, which is also what a subtraction elsewhere in the chain computes.
    vec3 ibl_diffuse = texture(irradiance_sampler, normal).rgb;
    // ... AND THE SECOND BOUNCE, when the caller asks for one. The sky cube above knows the light arriving
    // from a DIRECTION and nothing about where the surface is, so every interior wall in the scene is lit as
    // though it stood outdoors - that is the whole reason the world-space cache exists. The cache's answer
    // is mixed in by the cell's own TRUST, exactly as the tracer's probe_radiance does, so a cell with no
    // evidence degrades to the sky and this term can never make a hit darker than the ambient it replaces:
    // it is a REPLACEMENT of the diffuse ambient, not an addition to it (adding it would count the sky
    // twice, which is the defect the traced chain's subtraction exists to avoid on the screen side).
    //
    // WHERE THE GAIN COMES FROM: [render] ssgi_bounce, which the tracer already carries in its push and
    // which the shaded path did not read at all - the screen-sampled path is the only one that fed the
    // previous frame's accumulation back, so on the default path the knob was dead. It is alive here with
    // the meaning its own documentation always claimed: how much of the previous frame's accumulated
    // indirect a hit re-emits, where "accumulated indirect" is the world-space cache's SH-2 radiance rather
    // than a screen image (which a hit the camera cannot see has no entry in).
    if (cache_gain > 0.0) {
        // The same mapping the tracer does for its own lookups (see shaders/ssgi.comp's probe_radiance):
        // world -> cell -> normalized texture coordinate, all four coefficient images sharing the extent.
        const vec3 cell = (hit_world - cache_grid.xyz) / max(cache_grid.w, 1e-6);
        const vec3 uvw = (cell + 0.5) / vec3(textureSize(cache_sh0, 0));
        const vec4 cached = probe_sh_sample(cache_sh0, cache_sh1, cache_sh2, cache_sh3, uvw, normal);
        ibl_diffuse = mix(ibl_diffuse, cached.rgb, clamp(cache_gain * cached.a, 0.0, 1.0));
    }
    const vec3 ibl_specular = ibl_specular_radiance(normal, v, roughness);
    const vec3 f_ibl = ibl_specular_fresnel(normal, v, roughness, f0, 1.0);
    const vec3 ambient = ibl_diffuse * base_color * ao * (1.0 - metallic);
    const vec3 specular_ibl = ibl_specular * f_ibl * ao;

    out_radiance = ambient + direct + specular_ibl + emissive;
    return true;
}

