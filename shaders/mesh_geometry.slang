// shaders/mesh_geometry.slang - THE GEOMETRY FETCH A MESH STAGE USES INSTEAD OF THE INPUT ASSEMBLER.
//
// WHY THIS FILE EXISTS (docs/mesh_shaders.md step 1): a mesh stage REPLACES the vertex stage and the input
// assembler together. There is no VkVertexInputBindingDescription to read a vertex through and no
// SV_VertexID to index one with, so the mesh stage has to name the buffers itself - which is the whole
// reason a mesh stage can cull before it emits anything, and also the whole cost of porting a vertex
// stage to one.
//
// IT IS SLANG, NOT GLSL, and that is deliberate: the fetch is a device ADDRESS cast to a pointer
// (`(uint*)address`), which GLSL spells GL_EXT_buffer_reference and Slang spells as a C-style cast (see
// shaders/compute_skin.slang, which does exactly this against the same 64-byte interleaved vertex). The
// GLSL sources are archived in shaders/glsl.old/ and no longer built, so there is no second consumer to
// keep a GLSL spelling alive for.
//
// THE LAYOUT IS NOT NEGOTIATED HERE: `mesh_vertex_at` reads the engine's ONE interleaved vertex record,
// byte for byte the one the vertex pipelines declare as locations 0,1,2,4,5 and the one
// shaders/compute_skin.slang skins. Position (float3, byte 0), normal (float3, byte 12), uv (float2, byte
// 24), joints (uint4, byte 32), weights (float4, byte 48), stride 64 - sorted by location, so the input
// assembler's derived offsets and these are the same numbers. A mesh whose layout differed would need
// this file changed with it; the caller refuses such a mesh instead (see the host's mesh draw path).
//
// IT DOES NOT DECIDE WHERE A TRIANGLE GOES: the stage's own transform body is shared with the vertex
// entry of the same file, so the two paths emit the same clip positions by construction rather than by a
// copied expression - which is what makes the capture gate a valid acceptance for a mesh port.

#ifndef MESH_GEOMETRY_SLANG
#define MESH_GEOMETRY_SLANG

/// @brief the 64-bit device address held in two 32-bit push lanes (little-endian, low half first)
/// @note the same shape shaders/compute_skin.slang pushes its two buffer addresses in, so a reader who
///       has seen one pass has seen both
uint64_t mesh_lane_address(uint2 halves)
{
    return (uint64_t(halves.y) << 32u) | uint64_t(halves.x);
}

/// @brief where a draw's geometry is, and which window of it this draw covers
///
/// @note A MESH STAGE HAS NO DRAW COMMAND TO READ THESE FROM. `vkCmdDrawMeshTasksEXT` carries group
///       counts and nothing else: no vertex buffer binding, no index buffer, no first index, no
///       instance count. Everything the input assembler used to consume therefore has to arrive as
///       DATA, and this struct is that data - pushed by the primitive's own draw() right before its
///       mesh dispatch, at the offset its stage block reserves for it (see the host's
///       `mesh_geometry_offset_*`).
///
/// @note THE ADDRESSES ARE FOUR 32-BIT HALVES RATHER THAN TWO `uint2`s, and that is not style: a `uint2`
///       aligns to 8 bytes, which leaves a 4-byte hole between the heap index lanes and this struct - and
///       with a descriptor-heap pipeline EVERY byte of the declared block must have been written by
///       `vkCmdPushDataEXT` before the draw, so a hole is a validation error
///       (VUID-vkCmdDrawMeshTasksEXT-None-11376, measured: "uses push-constant statically at range
///       [0, 144), but vkCmdPushDataEXT was never called for range [108, 112)"). Four-byte fields leave
///       no hole at all.
struct MeshGeometryLanes
{
    uint vertex_address_low; // device address of the vertex buffer's base (byte 0 of vertex 0)
    uint vertex_address_high;
    uint index_address_low;  // device address of the index buffer's base (index 0)
    uint index_address_high;
    uint first_index;        // the draw's first index inside the index buffer
    uint index_count;        // indices this draw covers (a multiple of 3: triangles)
    int base_vertex;         // added to every fetched index (static-draw chunks; 0 otherwise)
    uint index_width;        // bytes per index: 2 or 4
};

/// @brief one record of the engine's interleaved vertex, as a mesh stage needs it
struct MeshVertex
{
    float3 position;
    float3 normal;
    float2 uv;
    uint4 joints;
    float4 weights;
};

/// @brief the engine's interleaved vertex stride, in bytes
/// @note NOT a push lane: a mesh whose vertex layout differed could not be drawn by the raster pipelines
///       at all (their input state is derived from the vertex stage's own declarations), so the stride is
///       a property of the RENDERER rather than of the draw. shaders/compute_skin.slang makes the same
///       assumption and the caller refuses a source whose stride is not this.
static const uint mesh_vertex_stride = 64u;

/// @brief read vertex @p index (base-vertex included by the caller) off the draw's vertex buffer
/// @param geometry the draw's geometry lanes
/// @param index the ABSOLUTE vertex index (the caller adds `base_vertex` to the fetched index)
/// @note untyped words, bit-cast to float, exactly as compute_skin reads them: the bit pattern is the
///       payload, so the type is a view rather than a conversion. THE WORD OFFSETS ARE THE LAYOUT'S, not
///       the stride's: 0 = position, 3 = normal, 6 = uv, 8 = joints, 12 = weights.
MeshVertex mesh_vertex_at(const MeshGeometryLanes geometry, const uint index)
{
    MeshVertex v;
    const uint64_t base = mesh_lane_address(uint2(geometry.vertex_address_low, geometry.vertex_address_high));
    uint* words = (uint*)(base + uint64_t(index) * uint64_t(mesh_vertex_stride));
    v.position = float3(asfloat(words[0]), asfloat(words[1]), asfloat(words[2]));
    v.normal = float3(asfloat(words[3]), asfloat(words[4]), asfloat(words[5]));
    v.uv = float2(asfloat(words[6]), asfloat(words[7]));
    v.joints = uint4(words[8], words[9], words[10], words[11]);
    v.weights = float4(asfloat(words[12]), asfloat(words[13]), asfloat(words[14]), asfloat(words[15]));
    return v;
}

/// @brief read index @p i (absolute, i.e. `first_index` already added) off the draw's index buffer
/// @note THE 16-BIT CASE IS READ AS TWO HALVES OF A 32-BIT WORD rather than as a `uint16_t` load: a
///       16-bit buffer load would pull the `StorageBuffer16BitAccess` capability into every mesh stage
///       for a fetch that a shift and a mask express exactly, and Vulkan's index buffers are
///       little-endian, so the odd index is the word's high half.
uint mesh_index_at(const MeshGeometryLanes geometry, const uint i)
{
    const uint64_t base = mesh_lane_address(uint2(geometry.index_address_low, geometry.index_address_high));
    if (geometry.index_width == 2u)
    {
        uint* words = (uint*)(base + uint64_t(i >> 1u) * 4ull);
        const uint word = words[0];
        return (i & 1u) != 0u ? (word >> 16u) : (word & 0xFFFFu);
    }
    uint* words = (uint*)(base + uint64_t(i) * 4ull);
    return words[0];
}

/// @brief how many triangles one mesh workgroup emits, and the vertices that costs
///
/// @note BOTH NUMBERS ARE LIMITS OF THE DEVICE, measured on this one and reported at startup (see
///       docs/mesh_shaders.md section 2): `maxMeshOutputVertices` and `maxMeshOutputPrimitives` are 256
///       each, and a triangle whose three vertices are written out separately - which is what a stage
///       without index sharing does - costs three of the former per one of the latter. 85 triangles =
///       255 vertices, i.e. the largest count that fits BOTH: 86 triangles would want 258 vertices.
///
/// @note THE INDEX SHARING A MESHLET WOULD DO IS NOT HERE ON PURPOSE. Emitting each triangle's three
///       vertices independently is the DIRECT port of what the input assembler produced (it is the same
///       triangle with the same three vertices in the same order), which is what lets the capture gate
///       decide the port - the sharing, and the meshlet structure it needs, is step 3.
static const uint mesh_triangles_per_workgroup = 85u;
static const uint mesh_vertices_per_workgroup = 255u;
static const uint mesh_indices_per_triangle = 3u;

#endif // MESH_GEOMETRY_SLANG
