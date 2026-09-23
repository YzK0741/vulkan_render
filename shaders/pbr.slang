// shaders/pbr.slang - the SLANG source of the FORWARD PBR shader PAIR (fragment + vertex).
//
// PRODUCES `pbr.frag.spv` AND `pbr.vert.spv`, the same names the GLSL versions produced, so the host side is
// untouched. Ported from shaders/pbr.frag and shaders/pbr.vert (glsl.old/ once the migration finishes): the
// bodies are the GLSL bodies - gather the surface, fill a shade_input, call shade_surface; and morph blend ->
// skin -> world transform - and what changes is only what a different language requires. In THESE stages
// that is exactly three things:
//
//   - the entry point's varyings and its output carry `[[vk::location(n)]]` / `SV_Target` instead of
//     `layout(location = n)`;
//   - `gl_FragCoord.xy` becomes the `SV_Position` parameter's `.xy`, which is the same value (pixel
//     centres, y down) - and it is why shading.glsl takes the pixel as a PARAMETER rather than reading
//     gl_FragCoord itself: the same shade path runs in compute stages on the deferred side, where that
//     builtin does not exist;
//   - the heap is reached through the shared accessors, which is the shim's whole job.
//
// THIS IS THE STAGE THAT PROVES THE WHOLE SHADING PATH COMPILES UNDER SLANG. gbuffer.slang (the previous
// port) exercises gather_surface and the material table; this one calls shade_surface, i.e. the sun and the
// cascaded shadow maps, the punctual and clustered light loops, the split-sum IBL with env_cube /
// irradiance / BRDF LUT, and the selectable BRDF and cel-shading branches - every fetch helper in
// heap_access.slang, in the order a lit pixel reaches them.
//
// THE INCLUDE ORDER IS PART OF THE CONTRACT (see docs/slang_migration.md section 8): the shim first, then
// surface.glsl, which declares the push-constant block and the heap_frame_slot / heap_image_index aliases
// the slot macros in heap_slot_constants.glsl expand against.

#include "heap_access.slang"
#include "surface.glsl"
#include "shading.glsl"

/// @brief shade one fragment: gather the surface, light it, apply the glTF output alpha
/// @note glTF alpha semantics: only alphaMode BLEND materials carry real coverage in the output alpha.
///       OPAQUE and MASK outputs must write alpha = 1 (their base color alpha is ignored by the spec),
///       otherwise the always-on blending below would make e.g. an albedo texture with an alpha channel
///       unexpectedly translucent. BLEND keeps base_color.a.
[shader("fragment")]
float4 main(
    [[vk::location(0)]] float3 v_world_pos : TEXCOORD0,
    [[vk::location(1)]] float3 v_normal : TEXCOORD1,
    [[vk::location(2)]] float2 v_uv : TEXCOORD2,
    float4 sv_position : SV_Position) : SV_Target
{
    const surface_sample s = gather_surface(v_world_pos, v_normal, v_uv);

    shade_input si;
    si.pixel = int2(sv_position.xy); // gl_FragCoord.xy: the cluster grid's tile coordinate (see shade_input)
    si.world_pos = v_world_pos;
    si.normal = s.normal;
    si.albedo = s.albedo;
    si.emissive = s.emissive;
    si.metallic = s.metallic;
    si.roughness = s.roughness;
    si.ao = s.ao;
    // The forward path adds the diffuse ambient itself; the deferred path leaves it to the resolve.

    const float out_alpha = ((s.flags & 32u) != 0u) ? s.alpha : 1.0;
    return float4(shade_surface(si), out_alpha);
}

// =============================================================================================================
// THE VERTEX STAGE, in the same file because it is the same CPU/GPU contract: the push block carries the
// per-draw state (it is declared by surface.glsl above, which is why `motion_base` lives in it) and the heap
// arrays carry the per-frame one. THIS IS THE STAGE THE G-BUFFER PASS LOADS FOR EVERY OPAQUE DRAW, so unlike
// `pbr.frag` - which only the transparent pass's forward pipeline records - this entry point is on the gate's
// path in all ten scenarios.
//
// The body is the GLSL body: morph blend (local deltas, current and previous weights), then skinning (the
// four joint matrices selected by the vertex's joints, weighted, through BOTH the current and the previous
// frame's blocks), then the world transform, with the previous world position projected by the previous
// frame's world matrix. The two "previous" reads are what give a DEFORMING vertex a motion vector instead of
// only its node's rigid motion (see docs/deformation_motion_vectors.md).
// =============================================================================================================

struct pbr_vertex_output
{
    [[vk::location(0)]] float3 world_pos : WORLD_POS;
    [[vk::location(1)]] float3 normal : WORLD_NORMAL;
    [[vk::location(2)]] float2 uv : UV_OUT;
    [[vk::location(3)]] float3 prev_world_pos : PREV_WORLD_POS;
    float4 position : SV_Position;
};

/// @brief morph blend -> skin -> world transform; outputs the world position, normal and UV
/// @note each stage is skipped by a uniform branch when the primitive does not need it
///       (push.morph_targets == 0, weight sum == 0, flag bit0 clear), so one pipeline serves static,
///       skinned, morphed and instanced geometry without variants.
[shader("vertex")]
pbr_vertex_output vertex_main(
    [[vk::location(0)]] float3 in_position : POSITION0,
    [[vk::location(1)]] float3 in_normal : NORMAL0,
    [[vk::location(2)]] float2 in_uv : UV_IN,
    [[vk::location(4)]] uint4 in_joints : JOINTS0,
    [[vk::location(5)]] float4 in_weights : WEIGHTS0,
    uint vertex_index : SV_VertexID,
    uint instance_index : SV_InstanceID)
{
    // morph blend first (local deltas + active target weights; skipped when not morphable)
    float4 local_pos = float4(in_position, 1.0);
    // ... and where this vertex was in object space BEFORE the morph blend, which is the same position
    // whenever the primitive is not morphable (the copy is why the branch below can be a uniform one).
    float4 prev_local_pos = local_pos;
    float3 morph_normal = in_normal;
    if (push.morph_targets > 0u)
    {
        const uint vert = vertex_index;
        // TWO weight regions per morph block: the current weights, then the weights one frame ago (see the
        // layout note on MorphData in pbr.vert and the weight write in animation::controller::update).
        const uint weight_base = push.morph_base + push.morph_targets * push.morph_vertices * 6u;
        float3 pos_delta = float3(0.0, 0.0, 0.0);
        float3 prev_pos_delta = float3(0.0, 0.0, 0.0);
        float3 nrm_delta = float3(0.0, 0.0, 0.0);
        for (uint t = 0u; t < push.morph_targets; ++t)
        {
            const float w = morph_value_at(heap_morph_slot, weight_base + t);
            const float w_prev = morph_value_at(heap_morph_slot, weight_base + push.morph_targets + t);
            const uint base = push.morph_base + (vert * push.morph_targets + t) * 6u;
            const float3 dpos = float3(morph_value_at(heap_morph_slot, base), morph_value_at(heap_morph_slot, base + 1u), morph_value_at(heap_morph_slot, base + 2u));
            const float3 dnrm = float3(morph_value_at(heap_morph_slot, base + 3u), morph_value_at(heap_morph_slot, base + 4u), morph_value_at(heap_morph_slot, base + 5u));
            pos_delta += w * dpos;
            prev_pos_delta += w_prev * dpos;
            nrm_delta += w * dnrm;
        }
        local_pos = float4(in_position + pos_delta, 1.0);
        // the SAME blend with the previous weights: the morph half of the deformation term, and the deltas
        // are read once because they are static - only the weights moved between the two frames
        prev_local_pos = float4(in_position + prev_pos_delta, 1.0);
        morph_normal = in_normal + nrm_delta;
    }

    // skinning after morph: blend the four joint transforms with the vertex weights (the identity block at
    // skin_base 0 leaves unskinned vertices unchanged). Normals use the weighted rotation part.
    float3 skinned_normal = morph_normal;
    const float wsum = in_weights.x + in_weights.y + in_weights.z + in_weights.w;
    if (wsum > 0.0)
    {
        float4 pos = float4(0.0, 0.0, 0.0, 0.0);
        float4 prev_pos = float4(0.0, 0.0, 0.0, 0.0);
        float3 nrm = float3(0.0, 0.0, 0.0);
        pos += in_weights.x * (skin_matrix_at(heap_skin_slot, push.skin_base + in_joints.x) * local_pos);
        pos += in_weights.y * (skin_matrix_at(heap_skin_slot, push.skin_base + in_joints.y) * local_pos);
        pos += in_weights.z * (skin_matrix_at(heap_skin_slot, push.skin_base + in_joints.z) * local_pos);
        pos += in_weights.w * (skin_matrix_at(heap_skin_slot, push.skin_base + in_joints.w) * local_pos);
        // ... and the SAME blend through the matrices the previous frame drew with, applied to the position
        // the previous frame morphed to: the two halves compose as skin_previous(morph_previous(v)), which is
        // the deformation term for a mesh that both morphs and skins.
        prev_pos += in_weights.x * (skin_matrix_previous_at(heap_skin_previous_slot, push.skin_base + in_joints.x) * prev_local_pos);
        prev_pos += in_weights.y * (skin_matrix_previous_at(heap_skin_previous_slot, push.skin_base + in_joints.y) * prev_local_pos);
        prev_pos += in_weights.z * (skin_matrix_previous_at(heap_skin_previous_slot, push.skin_base + in_joints.z) * prev_local_pos);
        prev_pos += in_weights.w * (skin_matrix_previous_at(heap_skin_previous_slot, push.skin_base + in_joints.w) * prev_local_pos);
        nrm += in_weights.x * mat3(skin_matrix_at(heap_skin_slot, push.skin_base + in_joints.x)) * morph_normal;
        nrm += in_weights.y * mat3(skin_matrix_at(heap_skin_slot, push.skin_base + in_joints.y)) * morph_normal;
        nrm += in_weights.z * mat3(skin_matrix_at(heap_skin_slot, push.skin_base + in_joints.z)) * morph_normal;
        nrm += in_weights.w * mat3(skin_matrix_at(heap_skin_slot, push.skin_base + in_joints.w)) * morph_normal;
        local_pos = pos / wsum;
        prev_local_pos = prev_pos / wsum;
        skinned_normal = nrm / wsum;
    }

    // VR_MAT4 for the local as well, so the ternary is between two values of the SAME type: assigning a
    // row-major-declared matrix into a differently-declared local is a conversion, and that conversion is a
    // transpose (the same trap as the push block's `model` above).
    const VR_MAT4 world = (push.flags & 1u) != 0u ? instance_transform_at(heap_instance_slot, push.instance_base + instance_index) : push.model;
    const float4 world_pos = world * local_pos;

    pbr_vertex_output o;
    o.world_pos = world_pos.xyz;
    o.normal = normalize(mat3(world) * skinned_normal);
    o.uv = in_uv;

    // The previous LOCAL position through the previous frame's world matrix. The node's rigid motion comes
    // from that matrix, and a vertex's own movement inside its object space comes from prev_local_pos above -
    // the previous MORPH weights and the previous SKIN matrices, composed in that order. Which is why this is
    // no longer "the current local position through last frame's model": that version reported a motion
    // vector of exactly zero for a mesh whose node never moved, no matter how it deformed.
    // NOT COVERED HERE, and not by anything in this stage: alpha-blended geometry, which is composited
    // outside the G-buffer and so writes no motion vector at all.
    const uint motion_index = push.motion_base + (((push.flags & 1u) != 0u) ? instance_index : 0u);
    o.prev_world_pos = (previous_transform_at(heap_previous_slot, motion_index) * prev_local_pos).xyz;

    o.position = camera_at(heap_camera_slot).proj * camera_at(heap_camera_slot).view * world_pos;
    return o;
}
