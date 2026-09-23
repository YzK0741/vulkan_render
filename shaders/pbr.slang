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

/// @brief morph blend -> skin -> world transform; the body BOTH geometry entries share
/// @param v the vertex record, from the input assembler (the vertex entry) or fetched by address (the mesh entry,
///        docs/mesh_shaders.md step 2) - the two paths differ ONLY in how the bytes arrive
/// @param vertex_index SV_VertexID in the vertex entry, the fetched index in the mesh entry (the morph block is
///        indexed by it)
/// @param instance_index SV_InstanceID in the vertex entry, the mesh workgroup's Y in the mesh entry
/// @note each stage is skipped by a uniform branch when the primitive does not need it
///       (push.morph_targets == 0, weight sum == 0, flag bit0 clear), so one pipeline serves static,
///       skinned, morphed and instanced geometry without variants.
pbr_vertex_output pbr_shade_vertex(const MeshVertex v, const uint vertex_index, const uint instance_index)
{
    // morph blend first (local deltas + active target weights; skipped when not morphable)
    float4 local_pos = float4(v.position, 1.0);
    // ... and where this vertex was in object space BEFORE the morph blend, which is the same position
    // whenever the primitive is not morphable (the copy is why the branch below can be a uniform one).
    float4 prev_local_pos = local_pos;
    float3 morph_normal = v.normal;
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
        local_pos = float4(v.position + pos_delta, 1.0);
        // the SAME blend with the previous weights: the morph half of the deformation term, and the deltas
        // are read once because they are static - only the weights moved between the two frames
        prev_local_pos = float4(v.position + prev_pos_delta, 1.0);
        morph_normal = v.normal + nrm_delta;
    }

    // skinning after morph: blend the four joint transforms with the vertex weights (the identity block at
    // skin_base 0 leaves unskinned vertices unchanged). Normals use the weighted rotation part.
    float3 skinned_normal = morph_normal;
    const float wsum = v.weights.x + v.weights.y + v.weights.z + v.weights.w;
    if (wsum > 0.0)
    {
        float4 pos = float4(0.0, 0.0, 0.0, 0.0);
        float4 prev_pos = float4(0.0, 0.0, 0.0, 0.0);
        float3 nrm = float3(0.0, 0.0, 0.0);
        pos += v.weights.x * (skin_matrix_at(heap_skin_slot, push.skin_base + v.joints.x) * local_pos);
        pos += v.weights.y * (skin_matrix_at(heap_skin_slot, push.skin_base + v.joints.y) * local_pos);
        pos += v.weights.z * (skin_matrix_at(heap_skin_slot, push.skin_base + v.joints.z) * local_pos);
        pos += v.weights.w * (skin_matrix_at(heap_skin_slot, push.skin_base + v.joints.w) * local_pos);
        // ... and the SAME blend through the matrices the previous frame drew with, applied to the position
        // the previous frame morphed to: the two halves compose as skin_previous(morph_previous(v)), which is
        // the deformation term for a mesh that both morphs and skins.
        prev_pos += v.weights.x * (skin_matrix_previous_at(heap_skin_previous_slot, push.skin_base + v.joints.x) * prev_local_pos);
        prev_pos += v.weights.y * (skin_matrix_previous_at(heap_skin_previous_slot, push.skin_base + v.joints.y) * prev_local_pos);
        prev_pos += v.weights.z * (skin_matrix_previous_at(heap_skin_previous_slot, push.skin_base + v.joints.z) * prev_local_pos);
        prev_pos += v.weights.w * (skin_matrix_previous_at(heap_skin_previous_slot, push.skin_base + v.joints.w) * prev_local_pos);
        nrm += v.weights.x * mat3(skin_matrix_at(heap_skin_slot, push.skin_base + v.joints.x)) * morph_normal;
        nrm += v.weights.y * mat3(skin_matrix_at(heap_skin_slot, push.skin_base + v.joints.y)) * morph_normal;
        nrm += v.weights.z * mat3(skin_matrix_at(heap_skin_slot, push.skin_base + v.joints.z)) * morph_normal;
        nrm += v.weights.w * mat3(skin_matrix_at(heap_skin_slot, push.skin_base + v.joints.w)) * morph_normal;
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
    o.uv = v.uv;

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

/// @brief the model-to-clip transform the input assembler feeds: five vertex attributes, SV_VertexID and
///        SV_InstanceID, handed to the body above as one record
/// @note THE FIVE ATTRIBUTES ARE THE PIPELINE'S VERTEX INPUT LAYOUT, which the host derives from this entry's own
///       declarations: dropping one would silently shrink the bound buffer's stride, so they must stay written to
///       the record the body reads (locations 0,1,2,4,5 - 64 bytes, the engine's interleaved vertex).
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
    MeshVertex v;
    v.position = in_position;
    v.normal = in_normal;
    v.uv = in_uv;
    v.joints = in_joints;
    v.weights = in_weights;
    return pbr_shade_vertex(v, vertex_index, instance_index);
}

// =============================================================================================================
// THE MESH STAGE (entry `mesh_main`), which replaces the input assembler AND the vertex entry above:
// docs/mesh_shaders.md step 2. It fetches the very vertices the input assembler would have bound - through the
// geometry lanes the host pushes into this stage's block - projects them with THE SAME body, and emits exactly the
// triangles the draw's index window describes, so the G-buffer (and, through the same file, the forward and unlit
// pipelines) must come out identical, and the capture gate is what decides that they do.
//
// IT IS THE SHADOW PASS'S STAGE ONE FAMILY LARGER: the same loop, the same 85-triangle workgroup budget (see
// shaders/mesh_geometry.slang), the same `SV_GroupID.y` as the instance index - and the difference is what the
// shared body does with the vertex, which here is the whole shading-relevant transform: morph, skin, the
// previous-frame position that becomes TAA's motion vector, and the camera projection.
// =============================================================================================================
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh_main(out vertices pbr_vertex_output verts[mesh_vertices_per_workgroup], out indices uint3 tris[mesh_triangles_per_workgroup], uint3 group_id : SV_GroupID)
{
    const MeshGeometryLanes geometry = push.geometry;
    const uint triangle_count = geometry.index_count / mesh_indices_per_triangle;
    // this workgroup's first triangle, and how many of the draw's triangles are left for it
    const uint first_triangle = group_id.x * mesh_triangles_per_workgroup;
    const uint remaining = first_triangle < triangle_count ? triangle_count - first_triangle : 0u;
    const uint emit = min(remaining, mesh_triangles_per_workgroup);

    // ONE SetMeshOutputCounts for the whole workgroup, before any output is written.
    SetMeshOutputCounts(emit * mesh_indices_per_triangle, emit);

    for (uint t = 0u; t < emit; ++t)
    {
        for (uint corner = 0u; corner < mesh_indices_per_triangle; ++corner)
        {
            const uint raw_index = mesh_index_at(geometry, geometry.first_index + (first_triangle + t) * mesh_indices_per_triangle + corner);
            const uint vertex_index = uint(int(raw_index) + geometry.base_vertex);
            const MeshVertex v = mesh_vertex_at(geometry, vertex_index);
            verts[t * mesh_indices_per_triangle + corner] = pbr_shade_vertex(v, vertex_index, group_id.y);
        }
        tris[t] = uint3(t * mesh_indices_per_triangle + 0u, t * mesh_indices_per_triangle + 1u, t * mesh_indices_per_triangle + 2u);
    }
}

// =============================================================================================================
// THE MESHLET ENTRY (entry `meshlet_main`): ONE workgroup per meshlet, its window read out of the table, and the
// same per-meshlet frustum culling the shadow pass's entry does - against the CAMERA's clip volume here, since
// this is the pass the camera sees (docs/mesh_shaders.md step 3).
//
// The projection body is `pbr_shade_vertex`, shared with the vertex and mesh entries, so a kept meshlet lands
// where the other two forms would have put it - which is what lets the capture gate accept this path.
// =============================================================================================================
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshlet_main(out vertices pbr_vertex_output verts[mesh_vertices_per_workgroup], out indices uint3 tris[mesh_triangles_per_workgroup], uint3 group_id : SV_GroupID)
{
    const MeshGeometryLanes geometry = push.geometry;
    const MeshletRecord meshlet = meshlet_at(geometry.first_index + group_id.x);

    // PER-MESHLET CULLING against the camera's clip volume: object -> world (the push's model matrix, or the
    // instance table for an instanced draw) -> clip, with the radius scaled by an upper bound on both matrices'
    // axis scales. A rejected meshlet emits NOTHING, which is the whole effect a task stage would have had on the
    // frame (its advantage is skipping the launch, not the pixels).
    const VR_MAT4 world = (push.flags & 1u) != 0u ? instance_transform_at(heap_instance_slot, push.instance_base + group_id.y) : push.model;
    const VR_MAT4 view_proj = camera_at(heap_camera_slot).proj * camera_at(heap_camera_slot).view;
    const float4 clip = view_proj * (world * float4(meshlet.center, 1.0));
    const float radius = meshlet.radius * matrix_max_axis_scale(world) * matrix_max_axis_scale(view_proj);
    if (!clip_sphere_visible(clip, radius))
    {
        SetMeshOutputCounts(0u, 0u);
        return;
    }

    const uint triangle_count = min(meshlet.index_count, mesh_triangles_per_workgroup * mesh_indices_per_triangle) / mesh_indices_per_triangle;
    SetMeshOutputCounts(triangle_count * mesh_indices_per_triangle, triangle_count);

    for (uint t = 0u; t < triangle_count; ++t)
    {
        for (uint corner = 0u; corner < mesh_indices_per_triangle; ++corner)
        {
            const uint raw_index = mesh_index_at(geometry, meshlet.first_index + t * mesh_indices_per_triangle + corner);
            const uint vertex_index = uint(int(raw_index) + meshlet.base_vertex);
            const MeshVertex v = mesh_vertex_at(geometry, vertex_index);
            verts[t * mesh_indices_per_triangle + corner] = pbr_shade_vertex(v, vertex_index, group_id.y);
        }
        tris[t] = uint3(t * mesh_indices_per_triangle + 0u, t * mesh_indices_per_triangle + 1u, t * mesh_indices_per_triangle + 2u);
    }
}