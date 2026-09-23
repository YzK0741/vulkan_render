// shaders/heap_access.slang - THE SLANG HALF OF THE BINDING MODEL.
//
// A shared body (surface.glsl, shading.glsl, sky.glsl, ibl_specular.glsl) is included by both languages and
// touches the heap ONLY through the names defined here or in heap_slots.glsl's GLSL branch. This file is
// that GLSL branch's counterpart, and the two are the only place the languages differ.
//
// WHY IT IS NEEDED AT ALL, measured: Slang's GLSL mode parses every function, type, macro and constant in
// the shared code, and rejects exactly one construct - `layout(descriptor_heap, ...)` (E31217,
// "unrecognized GLSL layout qualifier"). So the DECLARATIONS cannot cross, and this side reaches the same
// descriptors through Slang's `DescriptorHandle<T>` instead.
//
// HOW THE TWO HEAPS MAP, and this is the part that makes the port small: a `DescriptorHandle<T>` is a
// uint2 whose `.x` is an index into the RESOURCE heap and, for a CombinedTextureSampler, whose `.y` is an
// index into the SAMPLER heap - Slang then emits the OpSampledImage at the point of use. That is exactly
// this renderer's model (one flat 64-byte resource grid, one 32-byte sampler grid, the two halves bound
// separately), so every fetch helper the GLSL files already have keeps its name and its signature and only
// its BODY differs.
//
// THE ACCESSOR NAMES MUST MATCH heap_slots.glsl's GLSL definitions, because both sides have to satisfy the
// same shared bodies. A drift is not silent: the capture gate compares every stage's frame against the
// same references, and the two sides render the same scenes.

// THE GRID'S NUMBERS come from the same file the GLSL side includes, so the two cannot drift (and a stage
// that also includes heap_slots.glsl does not define them twice - that file has its own guard).
#include "heap_slot_constants.glsl"

// A MATRIX MEMBER'S TYPE on this side: `row_major` is what emits SPIR-V ColMajor, which is what the GLSL
// `mat4` members and the host's column-major glm::mat4 are (the keyword reads backwards - `column_major`
// emits RowMajor). heap_slots.glsl defines the GLSL half.
#define VR_MAT4 row_major float4x4

// A handle for an absolute slot: `.x` is the resource heap index, or - for a SamplerState - the sampler
// heap index (Slang uses the same component for both; only a combined sampler uses `.y`).
DescriptorHandle<T> heap_at<T>(uint slot) where T : IOpaqueDescriptor
{
    return (DescriptorHandle<T>)uint2(slot, 0u);
}

// A combined image sampler from the two heap slots, which is what a GLSL `sampler2D(tex, samp)` is here.
DescriptorHandle<Sampler2D> combined_at(uint texture_slot, uint sampler_slot)
{
    return (DescriptorHandle<Sampler2D>)uint2(texture_slot, sampler_slot);
}

// ---- THE ACCESSORS (the GLSL definitions live in heap_slots.glsl) ----
//
// The struct types in these macros (`Material`, `CameraUBO`, `LightUBO`) are declared by the shared bodies
// themselves, which is fine because a macro is text: it is expanded at the USE site, by which time the
// bodies have declared them. That is also why the two cluster accessors name `StructuredBuffer<uint>`
// rather than the GLSL blocks - those blocks' `uint counts[];` / `uint indices[];` members are just uint
// arrays, and Slang has no block-member indirection over a structured buffer.
#define material_at(slot, index) heap_at<StructuredBuffer<Material>>(slot)[index]
#define camera_at(slot) heap_at<ConstantBuffer<CameraUBO>>(slot)
#define light_at(slot) heap_at<ConstantBuffer<LightUBO>>(slot)
#define cluster_count_at(slot, cluster) heap_at<StructuredBuffer<uint>>(slot)[cluster]
#define cluster_indices_at(slot, i) heap_at<StructuredBuffer<uint>>(slot)[i]

// ---- THE VERTEX-SIDE MATRIX ARRAYS, and why they are structured buffers here ----
//
// Each GLSL declaration in the vertex leaves is `readonly buffer X { mat4 arr[]; } x[];`, i.e. a heap slot
// holding a runtime array of `mat4`. The Slang counterpart of "a runtime array of mat4" is a structured
// buffer whose ELEMENT is the matrix, and the element type is `VR_MAT4` (`row_major float4x4`) for the same
// measured reason a block member is: that spelling is what emits SPIR-V `ColMajor` with `MatrixStride 16`,
// which is the layout the host's column-major glm::mat4 arrays have. The shared `= 0u`-style slot argument
// is the caller's, so these macros only name the SHAPE.
//
// THEY ARE DEFINED HERE RATHER THAN IN A SHARED BODY because the families are read by the vertex LEAVES: a
// leaf may touch the heap directly, and the GLSL side of these five spells the same reads inline in its own
// file (there is no shared body to keep in step). They exist so that pbr.slang and the shadow.vert port
// that follows it read the same way in both languages.
#define instance_transform_at(slot, index) heap_at<StructuredBuffer<VR_MAT4>>(slot)[index]
#define previous_transform_at(slot, index) heap_at<StructuredBuffer<VR_MAT4>>(slot)[index]
#define skin_matrix_at(slot, index) heap_at<StructuredBuffer<VR_MAT4>>(slot)[index]
#define skin_matrix_previous_at(slot, index) heap_at<StructuredBuffer<VR_MAT4>>(slot)[index]
#define morph_value_at(slot, index) heap_at<StructuredBuffer<float>>(slot)[index]

// ---- THE FETCH HELPERS the GLSL side defines as macros or functions ----

/// one texel from a heap image, through the sampler at @p sampler_slot (the GLSL side is the
/// `heap_texel` macro: a `texture2D` parameter does not survive a function boundary in GLSL, while Slang
/// has no such restriction - so this side can be, and is, a function).
float4 heap_texel(DescriptorHandle<Texture2D<float4>> texture, uint sampler_slot, float2 uv)
{
    return texture.Sample(heap_at<SamplerState>(sampler_slot), uv);
}

/// the bindless-array fetch the shared bodies call by name (the GLSL definition is in surface.glsl): the
/// material's own image through the shared texture sampler. The GLSL side builds `sampler2D(tex, samp)` at
/// the point of use; here the two slots travel in ONE combined handle instead.
float4 heap_sample(uint texture_index, float2 uv)
{
    return combined_at(heap_slots_textures + texture_index, heap_sampler_texture).Sample(uv);
}

// ---- THE IBL FETCHES (the GLSL definitions live in ibl_specular.glsl) ----

/// the prefiltered chain's mip count, QUERIED from the texture rather than hardcoded
int env_cube_levels()
{
    uint width, height, levels;
    heap_at<TextureCube<float4>>(heap_slots_env_cube).GetDimensions(0u, width, height, levels);
    return int(levels);
}

/// one sample of the prefiltered environment at an explicit mip (GLSL's `textureLod`)
float3 env_cube_sample_lod(float3 dir, float lod)
{
    return heap_at<TextureCube<float4>>(heap_slots_env_cube).SampleLevel(heap_at<SamplerState>(heap_sampler_texture), dir, lod).rgb;
}

/// the BRDF integration LUT's `.rg` at a (ndotv, roughness) point
float2 brdf_lut_sample(float2 uv)
{
    return heap_at<Texture2D<float4>>(heap_slots_brdf_lut).Sample(heap_at<SamplerState>(heap_sampler_texture), uv).rg;
}

// ---- THE SHADOW AND IRRADIANCE FETCHES (the GLSL definitions live in shading.glsl) ----

/// one comparison fetch from the cascade array (GLSL's `sampler2DArrayShadow` + `texture(...)`): Slang
/// names the comparison sampler as a type instead, and `SampleCmpLevelZero` is the comparison fetch.
/// @param slot the shadow map's heap slot, passed in by the caller: a shim is compiled before the push
///        block that `heap_shadow_slot` needs, so the macro cannot be expanded here
float shadow_sample(uint slot, float2 tap, int cascade, float depth)
{
    return heap_at<Texture2DArray<float>>(slot).SampleCmpLevelZero(heap_at<SamplerComparisonState>(heap_sampler_shadow), float3(tap, float(cascade)), depth);
}

/// the diffuse ambient: the irradiance map at the world normal
float3 irradiance_sample(float3 n)
{
    return heap_at<TextureCube<float4>>(heap_slots_irradiance_cube).Sample(heap_at<SamplerState>(heap_sampler_texture), n).rgb;
}
