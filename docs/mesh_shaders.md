# Mesh shaders: what the stage buys here, and how the migration runs

**STATUS: step 0 is DONE and measured (the heap-native probe runs through a mesh pipeline and reports the
same pixel as the vertex one). Steps 1-4 are planned, not started.**

This document exists for the same reason `docs/slang_migration.md` does: the work spans sessions, so the
recipe, the acceptance route and the traps belong somewhere durable. Every number below was measured on
this machine, not read out of a spec - where a spec is quoted it is because validation quoted it first.

## 1. Why a mesh stage, in this renderer specifically

Geometry currently reaches the rasterizer through four vertex stages, and three of them are real geometry:

| `.spv` | source | who loads it |
| --- | --- | --- |
| `pbr.vert.spv` | `pbr.slang:vertex_main` | every opaque draw in the G-buffer pass, the forward default, and the transparent pass |
| `shadow.vert.spv` | `shadow.slang:vertex_main` | every cascade of the shadow pass |
| `post.vert.spv` | `post.slang:main` | the post chain's fullscreen triangle (no input) |
| `heap_probe.vert.spv` | `heap_probe.slang:main` | the binding probe (no input) |

A MESH stage replaces the vertex stage **and** the input assembler: the shader fetches its own vertices
from buffers it names itself, and it emits primitives per WORKGROUP, which means a workgroup can decide
not to emit anything after looking at all of its geometry. The vertex path can only reject a vertex, or a
whole draw from the CPU.

What that makes possible, and what it does not:

- **possible**: per-meshlet frustum/occlusion culling on the GPU with no CPU draw list, per-meshlet LOD
  selection, and - with a TASK stage and `vkCmdDrawMeshTasksIndirectEXT` - a culling dispatch the GPU
  feeds itself. This renderer already builds its own TLAS from the SAME vertex/index buffers, so a
  meshlet path would have a second consumer of one structure rather than a parallel one.
- **not possible**: the deferred lighting, SSAO, TAA, bloom, FXAA and cluster passes are fragment/compute
  work and cannot move to a mesh stage at all. A fullscreen triangle does not become cheaper either.
- **therefore not the goal**: step 0 does not claim a win. It proves the stage is reachable from this
  renderer's binding model, which is the question that could have ended the whole idea - see section 3.

## 2. What this device offers

Measured by the capability printout (`print_device_capabilities`), which now reports mesh shaders next to
the descriptor heap because the two are the same kind of thing: independent extensions that change how
work is submitted.

```
 mesh shaders  : available (VK_EXT_mesh_shader, task shaders available)
   output      : 256 vertices / 256 primitives / 128 components / 32768 B per workgroup, 128 invocations max
   workgroup   : mesh 128x128x128, task 128x128x128
```

| fact | value | where it comes from |
| --- | --- | --- |
| extension | `VK_EXT_mesh_shader`, revision 1 | `vkEnumerateDeviceExtensionProperties` |
| `VK_NV_mesh_shader` | also advertised, deliberately unused | the vendor extension predates the ratified one and has a different feature struct, so enabling both invites a mismatch for no gain |
| features | `meshShader`, `taskShader`, `meshShaderQueries` all supported | `VkPhysicalDeviceMeshShaderFeaturesEXT` |
| `maxMeshOutputVertices` | 256 | `VkPhysicalDeviceMeshShaderPropertiesEXT` |
| `maxMeshOutputPrimitives` | 256 | idem |
| `maxMeshOutputComponents` | 128 | idem - a vertex carrying position + uv + normal + tangent + 2 uv sets is 4 of them, so this is the limit that sizes a meshlet's vertex format, not its count |
| `maxMeshOutputMemorySize` | 32768 B | idem |
| `maxMeshWorkGroupInvocations` | 128 | idem (both workgroup-size triples report 128 per dimension) |
| extension dependency | `VK_KHR_spirv_1_4`, `VK_VERSION_1_2` | `vk.xml`; the 1.2 one is satisfied by the 1.3 device this renderer creates, so no dependency NAME is pushed at device creation - unlike the descriptor heap's pair |

The feature struct is enabled at device creation under the same rule the heap follows: the query sets
`mesh_shader_available` only when the extension **and** the `meshShader` feature are both there, and
`core.constructor.cppm` pushes `VK_EXT_mesh_shader` only when that flag is set, so "enabled" and
"available" cannot disagree. `taskShader` is deliberately NOT part of the flag - a device may have mesh
shaders without task shaders - so a task stage must gate on `mesh_shader_features.taskShader`.

## 3. The binding-model question, answered by the probe

The one thing that could have made this unworkable is this renderer's binding model: no descriptor sets,
no pipeline layout, a flat 64-byte slot grid addressed by absolute slot constants, and one
`vkCmdPushDataEXT` per draw. A mesh stage is a NEW EXECUTION MODEL, and nothing about the shim that makes
`heap_textures[heap_slots_textures + ...]` work had ever been tried in one.

So step 0 is: run the SAME fullscreen triangle through a mesh stage, through the same fragment stage and
the same 4x4 readback the probe already uses for its vertex stage, and compare the log lines. The probe
is the only shader group in this renderer that verifies itself - the runtime dispatches it during
initialization, reads a pixel back, and logs the result against a value it knows in advance - which is
exactly what a stage no scenario renders needs.

`shaders/heap_probe.slang` therefore has three entries: `main` (vertex), `mesh_main` (mesh) and
`frag_main` (fragment). The mesh entry carries the same heap fetch as the compute probe, through the
shim, and the host runs the pair back to back:

```
descriptor heap: the heap-native GRAPHICS probe rendered grid slot 16896 into a 4x4 target and read back rgba 255,255,255,255 (...)
descriptor heap: the heap-native GRAPHICS probe rendered grid slot 16897 into a 4x4 target and read back rgba 0,0,0,255 (...)
descriptor heap: the heap-native MESH probe rendered grid slot 16896 into a 4x4 target and read back rgba 255,255,255,255 (...)
```

| probe run | stage | dispatch | slot | read back | meaning |
| --- | --- | --- | --- | --- | --- |
| 1 | `VK_SHADER_STAGE_VERTEX_BIT` | `vkCmdDraw(3, 1, 0, 0)` | material table | `255,255,255,255` | the default material's white base colour: the fetch works |
| 2 | `VK_SHADER_STAGE_VERTEX_BIT` | `vkCmdDraw(3, 1, 0, 0)` | one slot past it | `0,0,0,255` | the WRONG slot proves the index selects the descriptor, not just "something non-zero" |
| 3 | `VK_SHADER_STAGE_MESH_BIT_EXT` | `vkCmdDrawMeshTasksEXT(1, 1, 1)` | material table | `255,255,255,255` | a mesh stage reaches the heap through the same shim, and the same triangle covers the read pixel |

Run 3 equals run 1, and it fails the same way run 2 does (measured: with the slot still one past the
table it read `0,0,0,255`), so the mesh stage's fetch is the same fetch rather than a coincidence.

The probe runs once per frame, so a 2-frame capture logs this triple 103 times - 103 identical triples is
also the flakiness check: a stage that produced a different pixel on some frames would show it here.

### 3.1 The three failures step 0 caught, and what each one cost

1. **An advertised extension is not an enabled feature.**
   `vkCreateShaderModule` refused the module: *"output vertices count exceeds the `maxMeshOutputVertices`
   of 0 by 3"* and *"...`maxMeshOutputPrimitives` of 0 by 1"* (VUID-RuntimeSpirv-MeshEXT-07115/07116).
   The physical device reports 256/256 (section 2), but the CREATED DEVICE reports zero for every mesh
   shader limit until the feature is enabled through the `vkCreateDevice` pNext chain - and a mesh
   pipeline cannot be created at all on such a device. This is the single most important fact in this
   document: the probe's failure mode was a limit of 0, not a missing extension.
2. **The pass-through feature policy is not safe for this struct.**
   `vkCreateDevice` reported VUID-VkPhysicalDeviceMeshShaderFeaturesEXT-primitiveFragmentShadingRateMeshShader-07033:
   that bit is only legal when `VkPhysicalDeviceFragmentShadingRateFeaturesKHR::primitiveFragmentShadingRate`
   is enabled, and this renderer never enables that extension. The query fills the struct with the
   driver's support and the chain enables what it reports, so the bit is now forced off explicitly, the
   way `features_1_1.protectedMemory` already was. `multiviewMeshShader` needs no such treatment: its
   dependency is the 1.1 `multiview` feature, which this device has on.
3. **The probe caught a geometry bug a screenshot would have hidden.**
   The first mesh entry decoded the three positions out of the loop index and got `(0,0),(2,0),(0,2)` - a
   quarter-screen triangle in the wrong corner - instead of the vertex entry's `(-1,-1),(3,-1),(-1,3)`.
   The readback returned the CLEAR colour, which is exactly what a probe is for: nothing about the
   pipeline creation, the SPIR-V or the validation messages said "wrong triangle". The constants are now
   written out literally, because a mesh stage has no `SV_VertexID` to share a decode with.

### 3.2 The chain rebuild this forced

The mesh shader is the SECOND independent extension link in `device_capabilities::query` (after the
descriptor heap, before the ray-tracing chain), and the first version hung it off the heap's link - so on
a device with mesh shaders but WITHOUT the heap, `vkGetPhysicalDeviceFeatures2` would never reach the mesh
struct, `meshShader` would read back as zero, and the extension would be reported unavailable for a
reason that has nothing to do with mesh shaders. The per-link `if (!available) unlink` cuts had the same
shape of bug for every link ahead of another.

They are replaced by one rebuild pass that relinks the extension chain from the core tail using the
availability flags only, in a fixed order, followed by a terminating null. The pre-query chain (built
from extension PRESENCE, because the feature bits are unknown until the query has run) is unchanged in
spirit; the post-query rebuild is what `vkCreateDevice` actually receives, and it is now independent of
link order.

## 4. What step 0 does and does not prove

| proven | not proven |
| --- | --- |
| the shim's `DescriptorHandle` cast, the two `BuiltIn` heaps and `vkCmdPushDataEXT` all reach a mesh stage | that a mesh pipeline is FASTER here (nothing was timed) |
| a mesh pipeline is created with `layout = VK_NULL_HANDLE` like every other pipeline | that a mesh stage can fetch vertices from the scene's vertex/index buffers (the probe has no vertex buffer by design - the triangle is synthesized) |
| the same fragment stage reads the same heap through both pipelines | anything about a TASK stage or indirect dispatch (step 3) |
| the mesh stage's heap fetch is index-sensitive, i.e. it is a real fetch | that any SCENARIO renders through a mesh pipeline - which is why the acceptance here is the log, not the gate |

The capture gate is still the right acceptance for a geometry stage, because a mesh pipeline produces the
same pixels as the vertex one when it emits the same triangles: **step 0's pipeline is byte-identical to
the vertex path by construction** (same fragments, same triangle), which is why its proof is the probe's
self-comparison. Steps 1-2 are the other way round - real geometry, verified by the gate.

## 5. The plan

Acceptance routes are the ones `docs/slang_migration.md` section 10 settled on: **gate** (byte-identical
frame), **log comparison** (a probe that knows its own answer), **manual A/B** (an opt-in config key, for
a pass the gate's scenarios do not reach) and **shape-only** (SPIR-V/limit reasoning when neither is
possible). A step is not done until its route says so, plus `ctest` 8/8, `spirv-val --target-env
vulkan1.3` on every emitted `.spv`, and zero validation findings.

### Step 0 - the stage is reachable (DONE)

`heap_probe.slang:mesh_main` + `run_heap_graphics_probe(slot, mesh_shader)` + the device feature.
Acceptance: **log comparison** (section 3). Gate stays byte-identical: the mesh feature changes no
existing pass.

### Step 1 - `shadow.vert` becomes a mesh stage

The smallest real geometry stage: one vertex stream (position only, the shadow pass's shader reads no
material), a depth-only fragment stage, and a pass whose only variability is the cascade's projection -
which is a uniform, not a per-vertex input. Its vertex-fetch function is also the one `pbr.vert` needs, so
porting it first produces the shared function the second port consumes.

The interesting part is the two mirrored vertex-input declarations (`shadow.slang`'s and `pbr.slang`'s)
collapsing into one fetch that takes a buffer address and an index - a mesh stage cannot use the input
assembler, so the fetch is explicit and the two stages can share it.

Acceptance: **gate** - all ten scenarios record the shadow pass, so a wrong vertex, a wrong cascade or a
wrong winding changes pixels. `shadow_single` is the one-cascade case and the most sensitive.

### Step 2 - `pbr.vert` becomes a mesh stage

`pbr.vert` is loaded by three consumers (G-buffer opaque, forward default, transparent), so the gate
covers it broadly - and the transparent pass is the one whose ORDER is decided on the CPU (back to front
by transformed origin), which a mesh stage must preserve: a mesh stage changes who emits the triangles,
not which draw is issued when.

Acceptance: **gate** (all ten scenarios; `transparent_blend` and `sponza` are the sensitive ones), plus a
per-draw comparison against the archived vertex-stage captures if any scenario moves.

### Step 3 - a TASK stage, meshlets, and indirect dispatch

The step where mesh shaders would actually pay, and the one with the most unknowns:

- **`vkCmdDrawMeshTasksEXT` has no `instanceCount`.** One `VkDrawMeshTasksIndirectCommandEXT` is three
  `groupCount`s and nothing else, so the instanced opaque draw this renderer does today has to become
  either one dispatch per instance, or an instance index computed from the workgroup id. This is a host
  side change to how draws are issued, not a shader change.
- **A meshlet split must respect the limits of section 2**: 256 vertices, 256 primitives, 128 components
  and 32768 B of output memory per workgroup. The components limit is the binding one for this renderer's
  vertex format, not the vertex count.
- **The TLAS is built from the same vertex/index buffers by device address**, so a meshlet path must not
  move or rewrite those buffers - meshlets are an additional index structure, and ray-traced shadows must
  keep seeing the same geometry.
- **A task stage must gate on `taskShader`**, not on `mesh_shader_available`.

Acceptance: **gate** with the meshlet path as the only geometry path for a scenario, plus a **manual A/B**
click for the culling decision (a meshlet that is wrongly culled is a missing object, which the gate sees;
a meshlet that is wrongly KEPT is invisible to it).

### Step 4 - remove the vertex path

Only once steps 1-3 are green: delete the vertex-stage pipelines and their `.spv`, sync
`CMakeLists.txt`'s `VR_SLANG_SOURCES`, `compile_shaders.ps1`/`.sh`, `chores.cpp`, `Doxyfile`, the CI
workflow, `README` and this document.

## 6. Traps already measured (so nobody re-measures them)

- **A mesh stage's outputs are WRITE-ONLY.** `E54005: cannot read values from mesh shader outputs` - so
  the "keep a fetch alive through a branch the fragment stage never takes" idiom does not compile. The
  probe writes the fetched value into an output lane no fragment stage reads
  (`[[vk::location(1)]] float heap_probe_lane`); a multiply by zero would be folded away WITH the fetch.
- **`vkCmdDrawMeshTasksEXT` is not exported by the import library.** Calling it directly is a link error
  (`ld.lld: undefined symbol: vkCmdDrawMeshTasksEXT`); it is fetched with `vkGetDeviceProcAddr` into
  `PFN_vkCmdDrawMeshTasksEXT` and the probe is skipped with a log line when that returns null.
- **Slang names every SPIR-V entry `main`** regardless of the source function's name, which is why three
  entries of one `.slang` file are three `-entry`/`-stage` pairs (`main`/`vertex`, `mesh_main`/`mesh`,
  `frag_main`/`fragment`).
- **`topology` must be `TRIANGLE_LIST`** for a mesh pipeline, and the vertex-input state is IGNORED for
  one - so the pipeline builder keeps passing its existing state and no host change is needed for it.
- **`serialize` the strings, not the code**: the mesh `.spv` is 3660 B against the vertex entry's 920 B for
  the same triangle, which is a cheap way to tell which compiler produced the module you are looking at
  when a probe's result does not move.

## 7. Open questions

- **Meshlet size.** Nothing here has been split into meshlets yet, so the interaction between
  `maxMeshOutputComponents` (128) and this renderer's vertex format is a documented limit rather than a
  measured one.
- **Whether the win is in the culling or the vertex reuse.** A mesh stage that emits one triangle per
  workgroup - which is what step 0 does - is strictly worse than the vertex path; the case for the stage
  is entirely in step 3.
- **Task shader vs compute culling.** A compute pass writing an indirect command buffer would also remove
  the CPU draw list. The task stage's advantage is that it runs per workgroup without a second dispatch;
  nothing has been measured, and the honest position is that step 3 should try the one that fits the
  existing pass structure with the smaller change.
