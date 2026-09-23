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
#define material_at(slot, index) heap_at<StructuredBuffer<Material>>(slot)[index]

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
