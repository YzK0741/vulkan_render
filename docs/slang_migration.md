# Migrating this renderer's shaders to Slang: the binding contract, and what is proven

**STATUS: reconnaissance and the binding contract are settled; the migration itself has not started.**
This document exists because the migration spans many sessions: the recipe below is what a later pass
needs to pick the work up, and the two dead ends are recorded so nobody walks into them again.

## 1. Why this is possible at all, and why it looked impossible for an afternoon

This renderer has ONE binding model - a descriptor heap - and it deleted every other one: no set layout,
no pool, no descriptor set, no pipeline layout (`vkCreateDescriptorSetLayout`, `vkAllocateDescriptorSets`,
`vkCreateDescriptorPool`, `vkCreatePipelineLayout` and `vkCmdBindDescriptorSets` appear zero times).
Every stage declares its resources as heap arrays with a 64-byte stride and addresses them by an
ABSOLUTE slot constant, e.g. `layout(descriptor_heap, descriptor_stride = heap_slot_stride) uniform
texture2D heap_textures[];` indexed as `heap_textures[heap_slots_textures + mat.tex_indices.x]`.

Slang's bindless model is NOT that shape: `DescriptorHandle<T>` is a *handle value* (a `uint2`, `.x` =
resource heap index, `.y` = sampler heap index) that has to be **stored somewhere** - a buffer, a
parameter block, or a synthesized global-parameter block. A first attempt that declared handle ARRAYS
produced exactly what this renderer cannot accept:

    OpDecorate %globalParams DescriptorSet 0
    OpDecorate %globalParams Binding 0
    OpMemberName %GlobalParams_std140 0 "heap_material_tables"
    ...                                     <-- Slang wrapped the handles into a synthesized block

so the honest preliminary reading was "Slang cannot express this project". **That reading was wrong**,
and the correction is the whole value of this document: a handle can be built from a literal slot by a
cast, and the compiler then emits a literal-indexed heap access with no descriptor sets at all.

## 2. The recipe (all four lines matter)

```slang
// 1. a handle from an absolute slot: .x = resource heap slot, .y = sampler heap slot
DescriptorHandle<T> slot_handle<T>(uint resource_slot, uint sampler_slot) where T : IOpaqueDescriptor
{
    return (DescriptorHandle<T>)uint2(resource_slot, sampler_slot);
}

// 2. use it inline; a handle dereferences at the member access (`.Sample`, `.Load`, `.camera_pos`),
//    NOT in a plain initializer - `CameraUBO c = slot_handle<...>(...);` is a type error.
acc += slot_handle<Texture2D<float4>>(16384u, 0u).Sample(slot_handle<SamplerState>(0u, 2048u), uv);
acc += slot_handle<ConstantBuffer<CameraUBO>>(16384u + 514u, 0u).camera_pos;
```

```bash
# 3. the capability IS the switch: without it Slang emits the descriptor-set model instead
slangc shader.slang -target spirv -profile spirv_1_6 \
  -capability spvDescriptorHeapEXT \
  -spirv-resource-heap-stride 64 -spirv-sampler-heap-stride 32 \
  -fvk-use-gl-layout \
  -entry fragMain -stage fragment -o shader.frag.spv
```

- `-capability spvDescriptorHeapEXT` is what turns on `UntypedPointersKHR` + `DescriptorHeapEXT`.
  Measured: with it, `OpCapability DescriptorHeapEXT`; without it, `OpCapability RuntimeDescriptorArray`
  and `DescriptorSet`/`Binding` decorations - the model this renderer removed.
- The two stride options exist because Slang's default heap stride is `OpConstantSizeOfEXT` of the
  resource type; this renderer's grid is a flat 64-byte slot grid, and its sampler heap is 32.
  Measured: `ArrayStride 64` and `ArrayStride 32`, i.e. the same numbers as `heap_slot_stride` and the
  device's sampler descriptor size.
- `-fvk-use-gl-layout` selects std430, which is the layout the CPU structs already use. Measured: the
  `Material` record's member offsets came out `0,16,20,24,28,32,48,64,68,72,76` - identical to what
  glslc emits for the same struct.

## 3. What is proven, family by family

One Slang file that reaches EVERY family through literal-slot handles compiles both as a fragment and
as a compute stage with **ZERO `DescriptorSet`/`Binding` decorations** and both `BuiltIn` heaps:

| family | Slang type | used by |
| --- | --- | --- |
| sampled 2D | `DescriptorHandle<Texture2D<float4>>` | `heap_textures`, every G-buffer/post image |
| sampled cube | `DescriptorHandle<TextureCube<float4>>` | `env_texture`, `irradiance_texture` |
| sampled 2D array | `DescriptorHandle<Texture2DArray<float4>>` | `shadow_texture` |
| shadow compare | `...SampleCmpLevelZero(SamplerComparisonState handle, ...)` | the shadow PCF |
| storage buffer (read) | `DescriptorHandle<StructuredBuffer<T>>` | `Materials`, `SkinMatrices`, `MorphData`, ... |
| storage buffer (write) | `DescriptorHandle<RWStructuredBuffer<T>>` | the cluster passes |
| uniform buffer IN THE HEAP | `DescriptorHandle<ConstantBuffer<T>>` | `CameraUBO`, `LightUBO` |
| storage image | `DescriptorHandle<RWTexture2D<float>>` | `rt_visibility`, `ml_trace`/`ml_resolved` |
| the TLAS IN THE HEAP | `DescriptorHandle<RaytracingAccelerationStructure>` + `RayQuery` | `rt_shadow`, MegaLights |
| sampler | `DescriptorHandle<SamplerState>` | the sampler heap |

## 4. Open items, stated before any work starts

1. **Storage-image formats - MEASURED, NOT YET DECIDED.** The project declares `r16f` / `rgba16f` on its
   heap storage images. Two forms were compiled and compared:

   | form | emitted image type | descriptor sets |
   | --- | --- | --- |
   | handle (`slot_handle<RWTexture2D<float>>(slot).Load(...)`) | `OpTypeImage %float 2D 2 0 0 2 Unknown` | **0** |
   | global `[[vk::image_format("r16f")]] RWTexture2D<float> vis;` | `... 2 R16f` | **1** |

   So the format and the set-less model are, in this Slang build, mutually exclusive: the attribute
   cannot ride a handle (a type alias carrying it did not compile either), and a *global* resource is a
   shader parameter, which is what brings the descriptor set back.
   **The engine does not currently enable `shaderStorageImageReadWithoutFormat` /
   `shaderStorageImageWriteWithoutFormat`** (grep: zero hits in `core.constructor.cppm` /
   `core.declarations.cppm`), and Slang's format-less output requests exactly those capabilities.
   THE DECISION TO MAKE, before porting `rt_visibility`, `megalights_trace` and `megalights_temporal`:
   (a) enable those two device features and keep the set-less model (a small host change, and the
   `*WithoutFormat` path is what a typed `Unknown` storage image needs), or
   (b) accept one descriptor set for those three shaders (undoing the property the whole migration is
   measured against), or
   (c) find a third construct (a `ParameterBlock`-carried handle with a format, or a custom
   `getDescriptorFromHandle`). (a) is the one that preserves the model; it must be verified on the
   device and with validation before the port, not after.
2. **One source of truth for the slot grid - DONE.** `shaders/heap_slot_constants.glsl` now holds the grid's
   constants and slot macros, `heap_slots.glsl` includes it (the GLSL heap arrays and the `heap_texel` helper
   stay there, being GLSL syntax), and `tests/test_render_resources.cpp` parses the NEW file against
   `core::heap_slots`. Verified as a pure refactor: build clean, ctest 8/8 (the grid comparison still passes,
   so every number survived the move), capture gate 10/10 with 0 changed.
3. **The build - DONE, in two phases, and CI is the last phase's problem.** `CMakeLists.txt` has a
   `VR_SLANG_SOURCES` list whose entries are `<source>:<entry>:<stage>:<output .spv name>`; the output name is
   the SAME `.spv` the GLSL version produced, so the host is untouched. A GLSL source whose `.spv` a Slang
   source now produces is dropped from the glslc list (both would write one file).
   PHASE ONE (now): slangc is used when found, and a machine without it builds that shader's GLSL source
   instead - so the build stays green, including CI. PHASE TWO (the last step of the migration): the GLSL
   sources move to `glsl.old/`, the fallback disappears and slangc becomes REQUIRED.
   **CI will need a source for slangc in phase two**: the workflow installs MSYS2's shaderc for glslc because
   its runner has no SDK, and MSYS2 ships no slang package (measured: `pacman -Ss slang` returns nothing).
   So phase two adds either a pinned Slang release binary or a Vulkan SDK install to the workflow.
4. **Unverified Slang maturity.** Slang's own feature-matureness table lists "Capabilities" as Experimental
   with "Implementation not done", and the SPIR-V chapter of its manual does not document descriptor heaps at
   all. Everything in section 6 was found by experiment, not from documentation - which is why it is written
   down with the exact flags.

## 5. The migration's acceptance, which does not change

Every shader ported keeps the SAME `.spv` name, so the host side is untouched, and is accepted only on:
the Release capture gate **10 scenarios x 2, 0 changed, 0 flaky** (byte-identical frames - the strongest
form available and free here), every run validation-clean, ctest 8/8, and the heap-native pipelines
unchanged (`layout = VK_NULL_HANDLE`, the heap bit set, the push block layout identical). Only when
every stage passes do the GLSL sources move to `glsl.old/`, with CMake, `Doxyfile`'s EXTENSION_MAPPING,
`docs/shaders.md`, the README and `shaders/compile_shaders.ps1`/`.sh` updated in the same commit.

## 6. The port recipe, and what has landed

THE EXACT `slangc` INVOCATION a port needs (this is what `CMakeLists.txt` runs, one entry per shader):

```
slangc <src>.slang -I shaders -allow-glsl \
  -target spirv -profile spirv_1_6 \
  -capability spvDescriptorHeapEXT \
  -spirv-resource-heap-stride 64 -spirv-sampler-heap-stride 32 \
  -fvk-use-gl-layout -matrix-layout-column-major \
  -entry <entry> -stage <stage> -o <the same .spv name>.spv
```

Each flag was forced by a measurement, and three of them are not guessable:

- **`-allow-glsl` is required for the SHARED CONSTANTS.** They are `const uint` at global scope (valid GLSL);
  Slang refuses a non-static global const (E31224) and `constexpr` is "not a supported Slang feature".
  `static const` is impossible because glslc rejects `static` outright ("'static' : Reserved word"), so
  `-allow-glsl` is the only form that lets one file serve both languages. `#include` works for a `.glsl`
  file (`__include` does not: it searches module paths, not `-I`).
- **A MATRIX MEMBER MUST SAY `row_major` TO GET `ColMajor`**, which is what glslang emits for a GLSL `mat4`
  and therefore what matches the host's column-major `glm::mat4`. MEASURED, and the keyword reads backwards:
  `column_major float4x4` emits `RowMajor`, `row_major float4x4` emits `ColMajor`. Slang's
  `-matrix-layout-column-major` flag did NOT change this in this build (all four flag combinations emitted
  the same decoration), and Slang's default is row-major, so the per-member keyword is the only reliable
  control. Getting this wrong transposes every matrix read from a UBO or a push block - which would show up
  as a wildly different frame rather than a subtle one.
- **Storage images need no format.** A handled `RWTexture2D<float>` emits `OpTypeImage ... Unknown` plus
  `StorageImageReadWithoutFormat` / `StorageImageWriteWithoutFormat`; the project already enables both (they
  are Vulkan 1.0 core features, the capability query fills the whole `VkPhysicalDeviceFeatures2` and
  `device_capabilities::device_pnext()` hands it to `vkCreateDevice`), and the device supports both
  (`vulkaninfo`: true, true). So the values still follow the format of the view the host creates.
- The push block keeps its layout: with `-fvk-use-gl-layout` the members come out at
  `0,4,8,12,16,20,24,32` - the same 96 bytes the CPU struct and the GLSL block use.

WHAT HAS LANDED (each entry: verified by the gate, not by inspection):

| stage | source | result |
| --- | --- | --- |
| `unlit.frag` | `shaders/unlit.slang` | **byte-identical frame**, 10/10 gate, ctest 8/8, zero validation findings. 4036 B of SPIR-V against glslc's 4848 B; zero descriptor sets; heaps and strides identical to the GLSL build's; the material slot folded to `16896`, the texture slot to `16384 + index`, the sampler to `2048` |

## 7. The architecture the migration settled on: ONE copy of the shading code, two thin shims

THE FINDING THAT DECIDED IT, and it was not the expected one. A Slang shader can `#include` a file written
in GLSL and call its functions. Measured: a file with `const float`, `vec3 tint(vec3, vec2) { return
mix(...); }` and `float luma(vec3)` was included by a Slang entry point that called it with `float3`
arguments and compiled clean. Then the real test: a Slang shader including `heap_slots.glsl`,
`surface.glsl` AND `shading.glsl` as they are produced **exactly eleven errors, every one of them
`E31217: unrecognized GLSL layout qualifier`** - the `layout(descriptor_heap, ...)` declarations and
nothing else. Every function body, type, macro and constant in the shared code is accepted as-is.

SO THE PLAN IS NOT "one copy per language". It is:

     heap_slot_constants.glsl   the numbers, shared                                  (DONE)
     <body>.glsl                surface / shading / sky / ibl_specular: ONE copy, touching the heap
                                only through named accessors
     the guards                 each GLSL-only declaration in those bodies wrapped in
                                #ifndef VR_SLANG ... #endif (NO separate declarations file is needed:
                                see section 8, which is what was actually built and verified)
     heap_access.slang          the Slang shim: DescriptorHandle accessors + the fetch helpers

and then each leaf stage is a `.slang` file including the shared body plus the Slang shim. This is what
keeps the shading math in ONE place: the only per-language code is a shim whose whole job is to say how a
slot becomes a resource, and a drift between the two shims cannot change a frame silently - the capture
gate compares every stage against the same references.

THE SHIM'S TWO NON-OBVIOUS RULES, both measured:

- **A GLSL named buffer block hides a member that Slang indexes directly.** GLSL writes
  `heap_material_tables[slot].materials[i]`; Slang's `StructuredBuffer<Material>` is indexed as
  `buffer[i]` and has no `.materials`. So the accessor takes BOTH parameters:
  `#define material_at(slot, index) heap_at<StructuredBuffer<Material>>(slot)[index]`.
- **`heap_texel` stays a macro in GLSL and becomes a function in Slang.** The GLSL side cannot make it a
  function at all (glslang: a `texture2D` parameter does not survive a function boundary - see
  `heap_slots.glsl`), while Slang has no such restriction; the shared body calls it either way.

PROVEN END TO END as a compile: a shared GLSL-syntax body using only `camera_at(slot)`,
`texture_at(slot)`, `material_at(slot, index)` and `heap_texel(...)`, plus the Slang shim, emits ZERO
descriptor sets, both `BuiltIn` heaps, and four `OpUntypedAccessChainKHR` heap accesses.

A TEMPTING SHORTCUT, TRIED AND RETIRED, recorded so it is not retried: making the shim reproduce the
GLSL access syntax EXACTLY - so the shared bodies need no edit at all - by shadowing the resource names
with `static` view objects that carry a `__subscript`:
`struct CameraView { __subscript(uint slot) -> ConstantBuffer<CameraUBO> { get { return
heap_at<ConstantBuffer<CameraUBO>>(slot); } } }; static CameraView camera;`. It fails, and the failures
are Slang language limits rather than mistakes in the idea: a `DescriptorHandle` cannot be a struct field
the way the view wants it (E30019 type mismatch where the field is declared, and the language's own
legalization HOISTS opaque members out of structs into globals - so a view struct cannot hold a handle and
hand it back), and the inline `property ... { get { ... } }` form used to give the material table its
`.materials` member is rejected (E20001 unexpected token). Four errors, all inside the shim, and the
workaround would be one bespoke view type per resource family. The named-accessor route is smaller, is
explicit, and is already proven.

WHAT REMAINS, as four steps with a gate on each: **the mechanism is settled and VERIFIED on both sides -
see section 8 - so the rest is mechanical.**

## 8. The mechanism, verified on a real shared body (surface.glsl)

THE DECLARATIONS ARE HIDDEN FROM SLANG BY A PREPROCESSOR GUARD, not moved to another file. Each
GLSL-only part of a shared file is wrapped in `#ifndef VR_SLANG ... #else <why> #endif`, and the Slang
build passes `-DVR_SLANG`. GLSL never defines that macro, so the guards remove nothing on that side -
which is why the GLSL build stays byte-identical rather than merely equivalent, and the gate proves it.

`shaders/heap_access.slang` is the other half: the same NAMES the shared bodies use, defined over
`DescriptorHandle<>` values built from literal slots. The two halves now in place:

| name | GLSL (heap_slots.glsl / surface.glsl) | Slang (heap_access.slang) |
| --- | --- | --- |
| `material_at(slot, index)` | `#define ... heap_material_tables[slot].materials[index]` | `#define ... heap_at<StructuredBuffer<Material>>(slot)[index]` |
| `heap_sample(texture_index, uv)` | `texture(sampler2D(heap_textures[..], heap_samplers[..]), uv)` | `combined_at(texture_slot, sampler_slot).Sample(uv)` |
| `heap_texel(tex, sampler_slot, uv)` | a MACRO (a `texture2D` parameter dies across a function boundary in glslang) | a FUNCTION (Slang has no such limit) |

THE COMBINED-SAMPLER MAPPING is what keeps this small: `DescriptorHandle<Sampler2D>` is a `uint2` whose
`.x` indexes the resource heap and whose `.y` indexes the SAMPLER heap, and Slang emits the
`OpSampledImage` at the point of use - which is exactly this renderer's two-grid model, so a fetch helper
keeps its name and signature and only its body differs.

MEASURED RESULT of the slice:

- The GLSL side, after the guards and the accessor rewrite: build clean, ctest 8/8, capture gate **10/10
  with 0 changed** - the rewrite is neutral, as designed.
- The Slang side: a fragment stage that includes the UNMODIFIED `surface.glsl` (plus `heap_access.slang`
  and `-DVR_SLANG`) compiles to **zero descriptor sets**, both `BuiltIn` heaps, three
  `OpUntypedAccessChainKHR` accesses. Nothing in `surface.glsl`'s functions had to change - only the two
  sites that touched the heap directly.

TWO THINGS THE SLICE TAUGHT, both worth not rediscovering: the push-constant block is declared ONCE, in
`surface.glsl` (with the two heap-index lanes and `mat4 model`), and Slang's GLSL mode accepts
`layout(push_constant) uniform` - so a Slang leaf that includes a shared body declares no push block at
all; and the twelve remaining heap accesses in `shading.glsl` are mostly `light[heap_light_slot].field`
(25 occurrences of one string) or combined-sampler fetches, i.e. one accessor plus a handful of fetch
helpers, not sixty bespoke edits.

THE REMAINING STEPS, unchanged in shape: the same guards + accessors + fetch helpers for `shading.glsl`
(camera/light/env/irradiance/lut/shadow/cluster) and `ibl_specular.glsl`; then each leaf stage is a
`.slang` file with its entry point, its varyings, and the two includes.

PROGRESS ON `shading.glsl` (the bulk of the mechanical work is done and gate-green): the 25
`light[heap_light_slot]` and 4 `camera[heap_camera_slot]` sites now call `light_at(...)` / `camera_at(...)`,
with the GLSL definitions in `heap_slots.glsl` expanding to exactly the expressions they replaced - ctest
8/8 and the capture gate 10/10 with 0 changed.

AN ACCESSOR NAME IS A MACRO NAME, AND THE SHARED FILES ALREADY HAVE FUNCTIONS - which cost one failed
build and is worth knowing before inventing the next one: `cluster_index_at` was the obvious name for the
cluster-index accessor and `shading.glsl` ALREADY HAS a function called `cluster_index_at(ivec2, vec3)`,
so the macro rewrote that function's own declaration and glslc reported
`shading.glsl:476: syntax error, unexpected IDENTIFIER, expecting LEFT_PAREN`. The accessor is
`cluster_indices_at` instead. Before adding an accessor, grep the shared files for the name. What is left in that file is FOUR code sites
(`sampler2DArrayShadow(shadow_texture[...])` at 205, `samplerCube(irradiance_texture[...])` at 441, and the
two block-member reads `cluster_counts[...].counts[...]` / `cluster_indices[...].indices[...]` at 499 and
510) plus the declarations to guard. Note that the UBO struct member lists cannot be shared verbatim: a
GLSL `mat4` member must be `row_major float4x4` on the Slang side (see section 6 - the keyword reads
backwards), so those members need a per-language macro for the type, and the guards wrap the
`layout(...) uniform X {` opening and the `} name[];` closing while the member list stays shared.

1. **`scene_structs.glsl`** - extract the language-neutral struct definitions (`Material` from
   `surface.glsl`; `CameraUBO`, `LightUBO` and the cluster structs from `shading.glsl`) into one file that
   both languages parse. Nothing about behaviour changes: the two bodies include it, so the GLSL side
   compiles to the same SPIR-V and the gate stays byte-identical.
2. **`heap_declarations.glsl`** (GLSL shim) - every `layout(descriptor_heap, ...)` declaration that is in
   `heap_slots.glsl`, `surface.glsl` and `shading.glsl` today, PLUS the accessor macros
   (`#define camera_at(slot) camera[slot]`, `#define texture_at(slot) heap_textures[slot]`,
   `#define material_at(slot, index) heap_material_tables[slot].materials[index]`, one per family, and the
   existing `heap_texel`). `heap_slots.glsl` includes it, so every GLSL stage keeps working.
3. **The body rewrite** - the shared bodies stop indexing the heap directly and call the accessors. This is
   PROVABLY NEUTRAL on the GLSL side: each accessor macro expands to exactly the expression it replaces,
   so the gate must not move by one pixel, and a moved frame means a macro is wrong.
4. **`heap_access.slang`** (Slang shim) - the same accessor names, defined with `DescriptorHandle`s and a
   `heap_texel` function. After this the shared bodies are includable by both languages and every
   remaining stage is a mechanical `.slang` file: the Slang shim, the shared bodies, and its own entry
   point, varying and push block.
