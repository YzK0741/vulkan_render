#version 450

/**
 * @file shaders/shadow.vert
 * @brief Depth-only vertex shader for the directional shadow pass.
 * @ingroup shaders
 *
 * Transforms the model into the light's orthographic clip space, so rasterization writes the depth
 * seen from the light. The light matrices come from the LightUBO (scene set binding 7) that
 * runtime::update_shadow_frustum() refits to the camera every time the camera or the scene moves.
 *
 * The vertex input layout MUST stay identical to pbr.vert (locations 0,1,2,4,5, interleaved 64-byte
 * stride): the shadow pass draws the very same vertex/index buffers, and the pipeline derives its
 * vertex input stride from the shader's inputs. Normals/uv have no role in the depth pass but are
 * declared and kept alive by a never-taken branch below so a driver/compiler cannot prune them and
 * shrink the stride below pbr.vert's.
 *
 * Skinning and morphing are applied exactly like pbr.vert (both passes must agree on where the
 * geometry is), and the pass rasterizes two-sided: runtime::record_shadow_content() sets
 * render_environment::two_sided so a single-sided caster can never be dropped for facing away from
 * the light.
 */
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 4) in uvec4 in_joints; // skin joint indices (JOINTS_0); 0 when unskinned
layout(location = 5) in vec4 in_weights;  // skin weights (WEIGHTS_0); (1,0,0,0) when unskinned

// Per-instance world transforms (same storage as pbr.vert, scene set binding 6)
layout(set = 0, binding = 6) readonly buffer InstanceTransforms {
    mat4 transforms[];
} instances;

// Per-joint skin matrices (same storage as pbr.vert, scene set binding 9)
layout(set = 0, binding = 9) readonly buffer SkinMatrices {
    mat4 matrices[];
} skins;

// Morph data (same storage as pbr.vert, scene set binding 10)
layout(set = 0, binding = 10) readonly buffer MorphData {
    float morphs[];
} morph_data;

// Light UBO (scene set binding 7): the orthographic light view-proj maps world -> shadow map.
layout(set = 0, binding = 7) uniform LightUBO {
    mat4 light_view_proj;
    vec4 light_dir;
} light;

layout(push_constant) uniform PushConstants {
    uint material_index; // unused here (vertex stage), declared to keep the block layout identical to pbr.frag
    uint flags;          // bit0: instanced draw -> model comes from instances[instance_base + gl_InstanceIndex]
    uint skin_base;      // start of this primitive's joint block in skins.matrices (0 = identity)
    uint morph_base;     // float index of this primitive's morph block in morph_data.morphs (0 = none)
    uint morph_targets;  // number of morph targets (0 = not morphable)
    uint morph_vertices; // vertex count of this primitive (morph block stride)
    uint instance_base;  // mat4 start of this instanced primitive's transforms (binding 6)
    mat4 model;
} push;

// Albedo UV, consumed by shadow.frag's alphaMode MASK test (the pass has no other use for it)
layout(location = 0) out vec2 v_uv;

/**
 * @brief morph, skin, transform into light clip space, and pass the albedo UV through
 *
 * The morph/skin order and math are identical to pbr.vert; the only difference is the final
 * projection (light.light_view_proj instead of camera.proj * camera.view) and the v_uv passthrough
 * that shadow.frag's mask test needs.
 */
void main() {
    // morph blend first (same layout as pbr.vert)
    vec4 local_pos = vec4(in_position, 1.0);
    if (push.morph_targets > 0u) {
        const uint vert = gl_VertexIndex;
        const uint weight_base = push.morph_base + push.morph_targets * push.morph_vertices * 6u;
        vec3 pos_delta = vec3(0.0);
        for (uint t = 0u; t < push.morph_targets; ++t) {
            const float w = morph_data.morphs[weight_base + t];
            const uint base = push.morph_base + (vert * push.morph_targets + t) * 6u;
            pos_delta += w * vec3(morph_data.morphs[base], morph_data.morphs[base + 1u], morph_data.morphs[base + 2u]);
        }
        local_pos = vec4(in_position + pos_delta, 1.0);
    }

    // skinning (identical to pbr.vert: weighted joint transforms, identity block at skin_base 0)
    const float wsum = in_weights.x + in_weights.y + in_weights.z + in_weights.w;
    if (wsum > 0.0) {
        vec4 pos = vec4(0.0);
        pos += in_weights.x * (skins.matrices[push.skin_base + in_joints.x] * local_pos);
        pos += in_weights.y * (skins.matrices[push.skin_base + in_joints.y] * local_pos);
        pos += in_weights.z * (skins.matrices[push.skin_base + in_joints.z] * local_pos);
        pos += in_weights.w * (skins.matrices[push.skin_base + in_joints.w] * local_pos);
        local_pos = pos / wsum;
    }

    mat4 world = (push.flags & 1u) != 0u ? instances.transforms[push.instance_base + gl_InstanceIndex] : push.model;
    vec4 world_pos = world * local_pos;
    gl_Position = light.light_view_proj * world_pos;
    v_uv = in_uv;

    // Keep the input without a role in the depth pass (location 1, the normal) alive so the vertex
    // input layout - and thus the bound buffer stride - stays identical to pbr.vert's (64 bytes,
    // locations 0,1,2,4,5). This branch can never run.
    if (isnan(in_position.x) && isinf(in_normal.x)) {
        gl_Position = vec4(0.0);
    }
}
