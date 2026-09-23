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
| `fxaa.frag` | `shaders/fxaa.slang` | gate 10/10; a post stage: push block only, no heap reads |
| `post.vert` | `shaders/post.slang` | gate 10/10; the fullscreen triangle, no heap reads |
| `gbuffer_debug.frag` | `shaders/gbuffer_debug.slang` | gate 10/10; the channel view, push layout verified member by member (0,4,8,12,16,20) |
| `taa.frag` | `shaders/taa.slang` | gate 10/10; push block only |
| `gbuffer.frag` | `shaders/gbuffer.slang` | **the first stage with shared-body + heap-UBO reads**, and the one that found the two traps in section 9: the transposed `mul` and the storage class. Wired, gate-green, and the motion channel is a real field again (598 distinct values over the swept fixture against 1 when it was broken) |
| `pbr.frag` | `shaders/pbr.slang` | **the whole shading path under Slang**: `shade_surface` - sun + cascaded shadows, punctual and clustered loops, split-sum IBL, the selectable BRDF/diffuse models and the cel bands - plus every fetch helper in the shim. 55520 B of SPIR-V, 0 descriptor sets, both BuiltIn heaps, 76 heap accesses, and the calls land as the right instructions (3 `OpImageSampleDref` for the PCF shadow fetch, 7 cube samples for env/irradiance). Verified on a real frame rather than only compiled: this is the demo's FORWARD DEFAULT pipeline, which the transparent pass records ("the forward default" in `pass/transparent.cppm`), so the `transparent_blend` scenario draws with it - and that scenario is byte-identical |
| `shadow.frag` | `shaders/shadow.slang` | the depth-only stage with the alphaMode MASK test: shares NO body, so it is the shape a leaf takes when the shim alone is enough (`material_at` + `heap_sample`), and its entry returns void because it has no output. 3496 B, 0 descriptor sets, 3 heap accesses. It also carries a measured codegen difference: **Slang lowers `discard` to `OpDemoteToHelperInvocation`** where glslang emits `OpKill`/`OpTerminateInvocation`, and the two behave identically here - the `sponza` scenario's masked casters (curtains, foliage) are byte-identical, and no validation finding asks for the demote capability, i.e. the device's 1.3 features are enabled |
| `pbr.vert` | `shaders/pbr.slang` (entry `vertex_main`) | **the G-buffer pass's VERTEX stage**, so it is the best-verified port: every opaque scenario records it, and the one that matters most is `deformation` - this is where the morph weights and the skin matrices are read TWICE, as they are now and as they were one frame ago. 15972 B, 0 descriptor sets, ONE BuiltIn heap (a vertex stage samples nothing), 24 heap accesses, and `OpVectorTimesMatrix` count **0**: every one of the 16 multiplies is the column-vector product glslc emits, with `proj * view` as a real `OpMatrixTimesMatrix`. It needed two shared-file fixes of the same kind: the push block's `model` had to be `VR_MAT4` (Slang's default majorness is ROW-major, so a plain `mat4` member is read transposed - harmless while only fragment stages included that block, since none of them read `model`), and `motion_base` had to be declared in it rather than riding the padding before `model` |
| `shadow.vert` | `shaders/shadow.slang` (entry `vertex_main`) | the shadow pass's VERTEX stage, on the path of every shadow-casting scenario, reusing the vertex-side accessors `pbr.vert` introduced. 8836 B, 0 descriptor sets, 10 heap accesses, `OpVectorTimesMatrix` 0. It is where TRAP 3 was found (the first attempt failed 8 scenarios with an empty shadow map), and it is also the stage that needed the whole push block: `frame_slot`/`image_index`/`spare_lane` at 96/100/104 and `cascade` at **108** - reading the cascade from 96 is a bug the GLSL file already records, because `frame_slot` overwrites it |
| `post.frag` | `shaders/post.slang` (entry `frag_main`) | all THREE post modes (bright pass, 13-tap downsample, composite), so every frame of every scenario records it - the widest verification a single port can get. 24896 B, 0 descriptor sets, 107 heap accesses, 51 `OpSampledImage` (the two filter kernels), 5 image-size queries for the tap spacing, push block at 0..36. It also carries the one place where the Slang side is SIMPLER than the GLSL: the GLSL filters take a heap SLOT and index the array inside, because a `texture2D` parameter does not survive a function boundary in glslang, while Slang passes the texture handle itself into `heap_texel` |
| `light_cluster.comp` | `shaders/light_cluster.slang` | **the first COMPUTE stage**, dispatched once per frame by default (`clustered_lights = true`), so every scenario records it. 11232 B, 0 descriptor sets, 14 heap accesses, `LocalSize 64 1 1`, `OpAtomicIAdd` for the per-cluster counter from `atomicAdd`, and the two writable buffers are the same heap slots the GLSL declares `writeonly buffer`, read as `RWStructuredBuffer<uint>`. It also settles the builtin that looked riskiest: `inverse()` is NOT expanded inline by either compiler - glslang and slangc both emit `OpExtInst MatrixInverse` from GLSL.std.450 - so the cluster boxes agree bit for bit and the `inverse` needs no workaround |
| `heap_probe.vert` + `.frag` | `shaders/heap_probe.slang` (entries `main`, `frag_main`) | the graphics half of the heap-native probe: a fullscreen triangle and the fragment stage that reads element 0 of the material table at the pushed slot. **No gate scenario renders it, and it does not need one** - the runtime dispatches it during initialization, reads a pixel back and logs it against the value it knows, so this port is verified by comparing LOG LINES, which is the first use of that route: both `descriptor heap: the heap-native GRAPHICS probe rendered grid slot ...` lines come out byte-identical to the GLSL build's, including the deliberately WRONG slot (16897 -> `0,0,0,255`), and that wrong slot is what makes the probe able to fail rather than always say "fine" |
| `rt_shadow.rchit` | `shaders/rt_shadow.slang` (entry `chit_main`) | the ray-traced shadow's closest-hit stage, whose whole job is the payload contract: the raygen deliberately does NOT initialise the payload, so this `1.0` and the miss shader's `0.0` are the two halves of it. 400 B, 0 descriptor sets, `OpEntryPoint ClosestHitKHR ... "main"`, `OpCapability RayTracingKHR`, and the store lands as `OpStore %payload %float_1` exactly as the GLSL's `rayPayloadInEXT` write did. **Shape-verified only**, and the one thing the A/B route must check first: Slang declares the payload as a 4-byte STRUCT where the GLSL declares a bare `float`, which is the same payload SIZE (all stages of one RT pipeline must agree on the size, not the type) - a difference worth watching when the raygen lands, since the raygen is the stage that declares it in GLSL |
| `rt_shadow.rmiss` | `shaders/rt_shadow.slang` (entry `miss_main`) | the other half of that contract: the ray escaped, so the light is not blocked. 400 B, `OpEntryPoint MissKHR ... "main"`. The check that matters here is CODEGEN, not structure: the GLSL file records that a payload store the compiler can prove redundant gets DROPPED (a raygen storing 0.0 plus a miss that does not write made every escaped ray come back occluded - 61.22 mean over the DamagedHelmet model against 76.48 for the raster shadow map, 100% of differing pixels darker), so the acceptance for this port is that `OpStore %12 %float_0` SURVIVES - and it does, in a module with no raygen-side initialisation to make it redundant |

The five non-`gbuffer` stages above were already byte-identical to their GLSL builds; `gbuffer` is the one that
needed a fix outside the shader (the camera's descriptor type), so it moved the GLSL side too - see section 9.

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

WHAT REMAINS is now a per-stage list with the order to do it in - see section 10. **The mechanism is settled
and VERIFIED on both sides** (section 8), the first real leaf is wired and green (section 9), so the rest is
mechanical, with one warning learned the hard way: the two traps in section 9 were both invisible in the
SPIR-V's SHAPE and only showed up in the picture.

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

ALL THREE SHARED BODIES ARE NOW PORTED AND VERIFIED ON BOTH SIDES, which is what makes the leaf stages
mechanical. `surface.glsl`, `shading.glsl` and `ibl_specular.glsl` are unchanged in their functions; what
they gained is named accessors/fetches and `#ifndef VR_SLANG` guards around the GLSL-only declarations
(including the UBO/blocks, WRAPPED rather than moved so the member lists stay in one place), and a
`VR_MAT4` macro - `mat4` in GLSL, `row_major float4x4` in Slang - because a matrix MEMBER cannot be
shared verbatim. Verified: the GLSL side stayed byte-identical at every step (ctest 8/8, gate 10/10 with 0
changed), and a Slang stage that includes `heap_access.slang` + `surface.glsl` + `shading.glsl` compiles
with ZERO descriptor sets and both BuiltIn heaps.

THREE ORDERING/CONSTRAINT FACTS a leaf must respect, all learned the hard way:

1. **The include order is: `heap_access.slang`, then `surface.glsl`, then the other bodies.** The push
   constant block and the `heap_frame_slot` / `heap_image_index` aliases are declared by `surface.glsl`,
   and the slot macros in `heap_slot_constants.glsl` expand against them - so a body compiled BEFORE that
   include reports `heap_frame_slot` as an undefined identifier.
2. **A fetch helper may not expand a frame-slot-dependent macro**, for the same reason: `shadow_sample`
   takes the slot as a PARAMETER and the caller passes `heap_shadow_slot`, because the shim is compiled
   before the push block exists. Helpers over the frame-INVARIANT slots (`env_cube_sample_lod`,
   `irradiance_sample`, `brdf_lut_sample`, `heap_sample`) take no slot at all.
3. **The two cluster accessors name `StructuredBuffer<uint>`**, not the GLSL blocks: `ClusterCounts`'
   `uint counts[];` member is just a uint array, and Slang has no block-member indirection over a
   structured buffer, so `cluster_count_at(slot, i)` is `heap_at<StructuredBuffer<uint>>(slot)[i]`.

## 9. The first leaf port: two traps, both measured, both fixed

`shaders/gbuffer.slang` was written, wired, and came back **fully black** (`mean.py`: R/G/B all 0.0000 over
the fixture's motion channel; the 4.1 MB PNG size is NOT evidence of content, this encoder does not
compress), with the gate reporting `deferred_taa_fxaa` and `deformation` CHANGED - exactly the two
scenarios that consume motion vectors. Two independent causes, both in the port, neither in the shared
bodies.

**TRAP 1: `mul(M, v)` ON A `row_major float4x4` IS THE TRANSPOSED PRODUCT.** Measured with a three-entry
probe (`build-release-clang64/dvm/probe/mulorder.slang`), all three entries loading the same matrix member
through a heap `ConstantBuffer` handle with the production flags:

| source spelling | emitted instruction | |
| --- | --- | --- |
| `mul(M, v)` | `OpVectorTimesMatrix(v, M)` | = Mᵀ·v, WRONG |
| `mul(v, M)` | `OpMatrixTimesVector(M, v)` | correct |
| `M * v` (the GLSL spelling, legal under `-allow-glsl`) | `OpMatrixTimesVector(M, v)` | correct, and the exact instruction glslc emits |

The rest of the black frame follows arithmetically: a perspective matrix's 4th column is `(0,0,0,0)`, so
Mᵀ·v has `w = 0` for every vertex, `(x / 0) * 0.5 + 0.5` is inf/NaN, and `clamp` turns that into 0 - a black
motion channel with everything else in the shader correct. The fix is to keep the GLSL spelling: the shared
bodies always used `M * v`, and the port is what "improved" it.

THE TRAP INSIDE THE OLDER EVIDENCE, recorded so it is not trusted again: the note that "`mul(M, v)` and
`M * v` produced the same frame hash" was measured on a STATIC camera, where the correct velocity is zero
and so is the transposed one. A hash comparison is evidence only when the quantity it covers is non-zero in
the scenario used.

**TRAP 2: THE STORAGE CLASS A HEAP READ GOES THROUGH MUST MATCH THE DESCRIPTOR'S TYPE, AND A MISMATCH IS
SILENT.** Reproduced all four combinations on this machine (host descriptor type x the storage class of the
pointer the shader fetches it through):

| heap descriptor written as | shader's pointer | result |
| --- | --- | --- |
| `STORAGE_BUFFER` | `StorageBuffer` (Slang's `DescriptorHandle<ConstantBuffer<T>>`) | works - the material table, i.e. the albedo channel |
| `UNIFORM_BUFFER` | `Uniform` (what glslc emits for a GLSL `uniform` block) | works - the GLSL camera, the reference |
| `UNIFORM_BUFFER` | `StorageBuffer` (Slang) | **zeros**: no validation finding, a zero camera matrix, NaN velocity, black motion |
| `STORAGE_BUFFER` | `Uniform` (the GLSL block left as `uniform`) | **zeros**: no validation finding, and the geometry disappears entirely (albedo black) |

Zero validation findings in both failing cases - only the picture. Since **Slang's
`DescriptorHandle<ConstantBuffer<T>>` never emits a `Uniform` pointer** (verified both in the probe and in
the built `gbuffer.frag.spv`), the only workable direction is to make the heap descriptor a STORAGE one,
and THREE things then have to agree:

1. `runtime.constructor.cppm` writes the camera slot as `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`;
2. the camera buffer carries `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` - without it the validation layer rejects
   the write (`vkWriteResourceDescriptorsEXT(): ... has no buffer(s) associated that are valid`) while the
   render still comes out right, which is exactly the kind of finding this project treats as a failure;
3. the GLSL side declares the block `buffer` instead of `uniform` (`shading.glsl`, `pbr.vert`,
   `light_cluster.comp`, `rt_shadow.rgen`).

IT IS LAYOUT-NEUTRAL, which is why it can be done at all: std430 and std140 give `CameraUBO` the same
offsets (0, 64, 128, 144, 208) and the same 272 bytes, because the `vec3 camera_pos` sits where the next
`mat4` has to be 16-aligned anyway; the three partial declarations read a prefix at the same offsets.
Verified by capture: the albedo channel comes back to the identical two-value plateau it had before.

WHAT THIS MEANS FOR EVERY STAGE STILL TO PORT: every heap buffer a Slang stage reads has to be a storage
descriptor, and both of the scene's buffers are now converted. The camera first, and then the LIGHT, whose
offsets were checked member by member BEFORE the change rather than after (that is the habit this section
exists to teach): every member is a vec4/mat4 or a scalar in a packed run, and its only array holds a
16-aligned 64-byte struct, so std430 lands `punctual_lights` on 352, `cluster_grid` on 8544 and
`cluster_depth` on 8560 exactly where std140 had them - 8576 bytes either way, confirmed against the
compiled modules. No scene buffer remains that a Slang stage would have to read through a `Uniform` pointer.

**RULED OUT, so a next attempt does not re-check it:**

- the push block, the earlier prime suspect: member NAMES and OFFSETS are identical in both builds
  (`0,4,8,12,16,20`, with `frame_slot`/`image_index` at 96/100), so the block's `std140` name is cosmetic
  and the heap indices the slot macros expand against are read correctly;
- the slot arithmetic: both builds compute `16898 + frame_slot` and index the same 64-byte grid;
- the camera UBO's layout: `OpMemberDecorate` for `CameraUBO` is `ColMajor` + `MatrixStride 16` at
  0/64/128/144/208 in BOTH modules. Layout was never the problem - the storage class was.

**THE TWO INSTRUMENTS THAT MADE IT VISIBLE**, worth reusing because the screen is not a value readout:

- the debug view runs through exposure -> ACES -> display encode, so an 8-bit plateau is NOT
  `255 * clamp(...)`. What identifies a state is STRUCTURE: the geometry is exactly 22.11% of the frame and
  the background is forced black, so each channel reads as `(0,0,0)` plus one plateau colour;
- a KNOWN-VALUE probe. Writing `float2(0.001, 0.0)` into the velocity output put the plateau at
  `(224, 206, 0)` - the y lane sitting exactly at the "did not move" bias - which proved the write path and
  `motion_gain` (width/4) were both fine. Writing `camera_at(heap_camera_slot).camera_pos.xz * 0.001`,
  whose value is the KNOWN `(0,0,4)` at offset 128, came back as the ZERO-velocity plateau `(206,206,0)`
  instead: what a zeroed camera UBO looks like.

**TRAP 3: A MATRIX *ARRAY* INSIDE A SLANG BLOCK LOSES ITS MAJORNESS.** Found by the `shadow.vert` port, and
it is why the light's cascade matrices now go through an accessor on BOTH sides. MEASURED, in the built
modules, for the `mat4 light_view_proj[4]` member:

| module | decoration on that member |
| --- | --- |
| glslc's `deferred.frag.spv` and `light_cluster.comp.spv` | `OpDecorate %_arr_mat4v4float_uint_4 ArrayStride 64` + `OpMemberDecorate %LightUBO 0 ColMajor` + `MatrixStride 16` |
| slangc's `pbr.frag.spv` and the block-form `shadow.vert.spv` | `ArrayStride 64` on the array, wrapped in an extra inner struct, and **NO `ColMajor`/`MatrixStride` on the member** |

A SINGLE matrix member - the camera's - IS decorated correctly; it is specifically the ARRAY that loses it.
What it costs: the shadow pass projected every draw out of the light's frustum, the shadow map came out empty,
and `deferred` and `shadow_single` rendered IDENTICAL frames because both were uniformly lit. That signature
- two scenarios with different cascade settings agreeing exactly - is what identified it, and it is worth
recognising again. Reproduced twice: the block form failed the same three scenarios with the same hashes on
both runs, while reading the SAME heap slot as `StructuredBuffer<VR_MAT4>` (element `i` is
`light_view_proj[i]`, because a heap descriptor IS an address range) is byte-identical to the GLSL build -
its decorations are `ArrayStride 64` + `ColMajor` + `MatrixStride 16`, verified in the module. The shared
body now calls `light_matrix_at(heap_light_slot, cascade)` on either side of the shim, which ALSO removes a
LATENT instance of the same defect from `pbr.frag`: its module has carried the undecorated array since the
day it was ported, and `transparent_blend` simply never executed that read. The lesson is the acceptance
rule's own limit: "the gate is green" means the paths the gate RECORDS are right - it is not a statement
that every line was exercised.

### THE FIRST STAGE THE BYTE-IDENTICAL STANDARD REFUSED: deferred.frag, by ONE PIXEL

`shaders/deferred.slang` is a faithful port of the deferred lighting stage - the shared shading code again
plus the screen-space AO it owns - and it is **NOT wired**, because the gate reports 8 of 10 scenarios
changed. The diff is the whole story:

    1 pixel of 1036800 differs, at the SAME coordinate (636,153) in every scenario that shows sky, by ONE
    8-bit step, in blue only: reference (112,124,145) against the port's (112,124,146). No geometry moved, no
    lighting changed, no descriptor is misread - the mean absolute difference over the whole frame is 0.000.

That is two compilers' floating-point codegen for the same source meeting a quantisation boundary: the sky's
`smoothstep` bands are smooth enough that a last-bit difference almost never flips an 8-bit value, and the
sun disc's `smoothstep(0.98, 1.0, ...)` is steep enough to amplify one where it does. Everything that CAN be
checked about the module is right: push block at 0/64/80/84/88/92, zero descriptor sets, 94 heap accesses, 0
transposed multiplies, and the SSAO's TBN built with GLSL's COLUMN-constructing `mat3` (verified against the
gate-green shared surface.glsl, which builds its normal-map TBN the same way).

THE OBVIOUS KNOB WAS TRIED AND IS WORSE: `-fp-mode precise` - "disable optimization that could change the
output of floating-point computations" - made NINE scenarios differ, including `sponza` and `deformation`,
which had been identical for the eleven stages before it. Slang's DEFAULT mode is therefore the one whose
codegen matches glslang's for this codebase, and the flag is deliberately absent from the recipe
(`CMakeLists.txt` records the measurement where the flag would have gone).

WHAT TO TRY NEXT, in this order, if that pixel is worth chasing:

  - compare the `NoContraction` decorations between the glslc-built and the slangc-built module for the same
    source. SPIR-V has no FMA instruction, so a last-bit difference in `inv_view_proj * vec4(...)` or
    `normalize(...)` comes from the DRIVER contracting a multiply-add on one side only, and those
    decorations are what tells it to;
  - if that is the cause, either express the sky's direction so its contraction cannot differ, or precompute
    the inverse view-projection's rows as four push lanes and drop the matrix multiply from the shader;
  - and if neither works, revisit the standard for THIS stage explicitly - a re-baseline with the reason
    written down, which is what the gate's `-Update` exists for - rather than quietly accepting a one-pixel
    difference.

**A NEARBY HAZARD from the same driver area**, worth knowing before the matrix-heavy stages are ported:NVIDIA has an open report of a different descriptor-heap defect on this driver family - a whole-matrix
`OpLoad` through an untyped pointer ignoring `MatrixStride` (it gathers 4-bytes-apart columns), with
per-element loads correct and a per-element + `dot` workaround:
[forums.developer.nvidia.com/t/.../383932](https://forums.developer.nvidia.com/t/vulkan-615-71-09-rtx-4050-descriptor-heap-incorrect-rowmajor-matrixstride-handling-for-whole-matrix-opload-through-an-untyped-uniform-pointer/383932)
(RTX 4050 Laptop, driver 615.71.09, Vulkan 1.4.351, SDK 1.4.357 - the same SDK this project builds with).
It is NOT what this port hit: glslc's and slangc's camera reads have the same shape (`OpBufferPointerEXT`
then a typed `OpAccessChain`) and both are correct once the storage class matches. But if a future stage's
matrix read comes out GARBLED rather than zeroed, that report is the first thing to test against.

PROGRESS ON `shading.glsl` (the bulk of the mechanical work is done and gate-green): the 25
`light[heap_light_slot]` and 4 `camera[heap_camera_slot]` sites now call `light_at(...)` / `camera_at(...)`,
with the GLSL definitions in `heap_slots.glsl` expanding to exactly the expressions they replaced - ctest
8/8 and the capture gate 10/10 with 0 changed.

AN ACCESSOR NAME IS A MACRO NAME, AND THE SHARED FILES ALREADY HAVE FUNCTIONS - which cost one failed
build and is worth knowing before inventing the next one: `cluster_index_at` was the obvious name for the
cluster-index accessor and `shading.glsl` ALREADY HAS a function called `cluster_index_at(ivec2, vec3)`,
so the macro rewrote that function's own declaration and glslc reported
`shading.glsl:476: syntax error, unexpected IDENTIFIER, expecting LEFT_PAREN`. The accessor is
`cluster_indices_at` instead. Before adding an accessor, grep the shared files for the name. The last four
code sites in that file (`sampler2DArrayShadow(shadow_texture[...])`, `samplerCube(irradiance_texture[...])`,
and the two block-member reads `cluster_counts[...].counts[...]` / `cluster_indices[...].indices[...]`) are
done too, and the UBO member lists are shared through the `VR_MAT4` type macro: the guards wrap the
`layout(...) buffer X {` opening and the `} name[];` closing while the member list stays in one place.

## 10. What remains

The mechanism is settled, the shim is proven, and six stages are wired: what is left is the same recipe
applied per stage, easy ones first so that each new hazard is met in isolation. Ordered by what they read:

1. **Push-block-only stages** (no heap reads at all): DONE - `fxaa.frag`, `post.vert`, `taa.frag`, and now
   `post.frag`, which turned out to read heap images after all and is on every scenario's path.
2. **Heap readers that touch no buffer**: `deferred.frag` is the one left that the gate RECORDS, and it is
   the best next port: it is the default path's lighting stage and reads the G-buffer, the light block, the
   shadow map, the cluster lists and the ray-traced visibility image in one shader. Then `heap_probe.vert`,
   `.frag` and `.comp` - the runtime's own heap probe, which nothing in the gate renders.
   (Both PBR stages and both shadow stages are DONE - see the table in section 6. The vertex leaves share the
   shim's five matrix-array accessors plus `light_matrix_at`, which every remaining stage that samples the
   sun's shadow also needs.)

### WHAT THE GATE CANNOT VERIFY, stated before the last stages are ported

The gate records what the ten scenarios render, and three groups of shaders are NOT on any of their paths:
`rt_shadow.*` (no scenario enables `rt_shadows`), `megalights_trace/temporal.comp` (no scenario enables the
megalights path) and `heap_probe.*` (the runtime's own PBR/heap probe, which the demo app does not run in
capture mode). For those, "the gate stays 10/10" only proves nothing regressed; the port has to be verified
by its SHAPE (descriptor sets, heaps, the instructions the fetches land as) plus the family-by-family proof
from section 3, and that limit should be stated in the commit rather than implied away. This is the same
limit that hid TRAP 3 in `pbr.frag` for four commits.

TWO ROUTES EXIST FOR THOSE STAGES, and neither needs the gate's references to be touched:

- **The probe stages verify themselves - but only the GRAPHICS half does, measured.** `heap_probe.vert` and
  `heap_probe.frag` are DONE and were accepted by comparing log lines: the runtime dispatches the graphics
  probe during initialization, reads a 4x4 target back and logs the pixel against the value it knows, so the
  two `descriptor heap: the heap-native GRAPHICS probe rendered grid slot ...` lines (including the
  deliberately WRONG slot) are the acceptance, and they come out byte-identical to the GLSL build's. **The
  COMPUTE half does not offer that**: `run_heap_probe` writes its answer into a buffer the host reads back and
  this build logs NO comparable line for it - grepping `debug.log` for every `descriptor heap:` line finds the
  graphics probe's two and nothing from the compute probe. `heap_probe.comp` also uses
  `GL_EXT_buffer_reference` (a raw device address pushed as two 32-bit halves) rather than a heap handle,
  which is a DIFFERENT mechanism from everything ported so far and has no Slang spelling established yet. So
  it needs either a host-side log line added first (a runtime change, not a shader port) or shape-only
  verification, and it should not be described as self-verifying until one of those exists.
- **An A/B capture covers the RT path.** For `rt_shadow.*`, `compute_skin.comp` and `mask_bake.comp`, the
  verification is the same standard applied by hand: capture a frame with `rt_shadows = true` from the GLSL
  build, capture the same frame from the Slang build, and compare the two PNGs pixel for pixel. That is what
  the gate does for its ten scenarios; doing it manually for one RT config extends the same standard to stages
  the scenario list does not reach, without re-baselining anything.


3. **Compute stages**: `compute_skin.comp`, `mask_bake.comp`, `megalights_trace.comp`,
   `megalights_temporal.comp`. (`light_cluster.comp` is DONE - the first compute stage, and the one that
   proves the compute spellings and `atomicAdd`/`inverse`.) These are the ones that WRITE heap buffers, so
   their storage-buffer declarations are the mirror of the read-side contract - the same class/type agreement
   applies, and a mismatch is equally silent.
4. **DONE, and it was item 1 of the critical path**: the LIGHT buffer has made the same three-part change
   the camera did (section 9) - `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`, a `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`
   on the buffer, and `buffer` instead of `uniform` in the four files that declare the block
   (`shading.glsl`, `shadow.vert`, `light_cluster.comp`, `rt_shadow.rgen`). Its layout was verified equal
   BEFORE the switch, which is the opposite of the order that cost this session a day on the camera: a
   block with an ARRAY is exactly where std140 and std430 can disagree, so it is compared first. This is
   what unblocks every light-reading stage below.
5. **Ray tracing**: `rt_shadow.rgen/.rchit/.rmiss/.rahit`. The rgen already has the camera as a `buffer`
   block; it also declares the TLAS and the visibility storage image, so this is where the acceleration
   structure and the storage-image declarations get their Slang spelling (the shim's `Texture2D`/heap
   handling covers the images already).
6. **Then the migration's own endgame**, none of it started: move the GLSL sources to `glsl.old/`, make
   `slangc` a required tool instead of a two-phase fallback, drop the `glslc` rules from `CMakeLists.txt`,
   and sync `Doxyfile` (`EXTENSION_MAPPING` for `.slang`), `docs/shaders.md`, the README's shader section
   and `scripts/windows/compile_shaders.ps1` / `compile_shaders.sh`. CI cannot get `slangc` from MSYS2
   (`pacman -Ss slang` has no package), so the release job needs the LunarG SDK's `Bin` on PATH or a Slang
   build step.

   `Doxyfile` IS DONE (`.slang` maps to C++ exactly as `.glsl`/`.vert`/`.frag` do, and `*.slang` is in
   FILE_PATTERNS - without that Doxygen does not read the files at all). It was verified by RUNNING doxygen:
   801 HTML files and, as the proof that the files are now read, a warning that names a `.slang` file
   (`shaders/pbr.slang:81`, an initializer-list confusion from Doxygen parsing shader syntax as C++ - the
   same class of warning the GLSL files produce).

   THE COMPILE SCRIPTS ARE A LIVE HAZARD UNTIL THEY ARE CONVERTED, and that is why they are called out here
   rather than left to the last commit: `compile_shaders.ps1` and `.sh` compile a HARDCODED list of `.glsl`
   sources with glslc only. Run today, they would overwrite the `.spv` files of all fourteen ported stages
   with GLSL-derived modules - the runtime would then load something the build never produced, and nothing
   would say so. They are the documented escape hatch for recompiling shaders by hand, so the fix is part of
   the endgame step that moves the `.glsl` sources away (after which the scripts cannot compile them at all
   and must be rewritten around `VR_SLANG_SOURCES` anyway). Until then, the canonical build is CMake's.

