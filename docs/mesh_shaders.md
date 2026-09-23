# Mesh shaders: what the stage buys here, and how the migration runs

**STATUS: steps 0, 1, 2 and step 3 are DONE, with three measured exceptions.**

- The binding model is proven (the heap-native probe runs through a mesh pipeline and reports the same pixel as the
  vertex one), and EVERY geometry stage this renderer draws leaves with has a mesh form: shadow, G-buffer, forward
  `pbr`/`unlit`. The capture gate is byte-identical to the committed vertex-path references with either path
  active, and forced probes prove which one produced the frames.
- THE MESHLET PATH IS THE ONE IN USE: every primitive's geometry is cut into 85-triangle meshlets with object-space
  bounding spheres and normal cones (`vulkan.meshlet`, 3145 records over 103 primitives on Sponza), the records live
  in a heap table written once at import, and both geometry passes draw them ONE WORKGROUP PER MESHLET - the shadow
  pass and the G-buffer - each meshlet culled against its pass's clip volume before it emits anything. Every mesh
  dispatch goes through `vkCmdDrawMeshTasksIndirectEXT`, which is the seam a compute culling pass writes into.
- What is NOT done, each measured and recorded above: per-meshlet BACKFACE culling (the cone is computed and
  tested, but enabling the test changes six scenarios, so it is reverted - see the negative result in step 3);
  a COMPUTE pass that writes the indirect commands (the seam exists, the pass does not); and a measurement of what
  the culling buys, which the log does not carry.
- Step 4 (removing the vertex path) has not started. It should now be possible - every geometry stage is
  gate-green in its mesh form - but the vertex forms are the only fallback a device without `VK_EXT_mesh_shader`
  has, so removing them is a portability decision rather than a cleanup, and the gate cannot see the difference
  (it runs on a device with the extension).

The measurements behind each of those sentences - the device's limits, the two compiler crashes, the layout bug
that wedged the GPU, the culling's two-way acceptance and the backface negative - are in section 5, step by step.



- Step 3's culling half is BLOCKED BY A COMPILER BUG, not by design: `slangc` v2026.18.2 crashes (0xC0000005) on a
  task stage that takes push data or touches the heap, with a six-line reproducer (section 5, step 3).
- The meshlet path ITSELF is in the tree and gate-green: the shadow pass and the G-buffer both prefer a meshlet
  pipeline, one workgroup per meshlet, each meshlet culled against the pass's own frustum in the entry point. The GPU
  hang that stood in its way first was a LAYOUT BUG - a 28-byte host record against the shader's std430 stride, so
  every record after the first was read four bytes off - and not anything about meshlets.
- The SECOND mechanism step 3 names - cull in a COMPUTE pass and dispatch with `vkCmdDrawMeshTasksIndirectEXT` - has
  its SEAM in the tree: every meshlet dispatch already goes through the indirect entry point, reading its counts out
  of a command table at a slot the primitive owns. What it lacks is the pass that writes those counts after culling.
  (The first, cursor-driven version of that seam made `sponza` flaky and was reverted; the design that replaced it is
  in section 5.)
- Step 4 (removing the vertex path) has not started, and should not until a culling path exists, because the vertex
  forms are the fallback every device without mesh shaders runs.

The measurements behind each of those sentences - the device's limits, the two compiler crashes, the GPU hang and
the exact next experiments - are in section 5, step by step.

> **BLOCKER (one of step 3's two named mechanisms, and it is not ours).** A TASK stage cannot be given host data on
> this toolchain: `slangc` v2026.18.2 dies with `0xC0000005` on a six-line entry point that declares
> `[[vk::push_constant]]` and reads one member from it, and dies again on one that only `#include`s
> `heap_access.slang`. Both crashes reproduce with `-target spirv -profile spirv_1_6` alone, so they are not caused
> by this renderer's flags, and the same file's `mesh_main` / `meshlet_main` entries compile and validate. A task
> stage that cannot receive a push block cannot be told the frame slot, the cascade index or the model matrix -
> which is how every stage in this renderer reaches the heap, because a heap pipeline has no layout. **The whole
> reproducer is these six lines** (no engine code, `slangc t.slang -target spirv -profile spirv_1_6 -entry task_main
> -stage task -o t.spv` dies with 0xC0000005; delete the push block and it compiles):
>
> ```slang
> struct P { uint slot; };
> [[vk::push_constant]] ConstantBuffer<P> pc;
> struct Payload { uint slot; };
> [shader("amplification")]
> [numthreads(1, 1, 1)]
> void task_main(out Payload payload)
> {
>     payload.slot = pc.slot;
>     DispatchMesh(1, 1, 1, payload);
> }
> ```
>
> (It is inlined here rather than left in `build-release-clang64/dvm/t_push.slang`, where the first version of this
> note pointed: that directory is a build artifact and a clean build deletes it.) The objective names a second
> mechanism for the same culling - a compute pass writing `VkDrawMeshTasksIndirectCommandEXT` - and that one does not
> crash, so this is a blocked MECHANISM rather than a blocked feature.

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
possible). A step is not done until its route says so, plus `ctest` 10/10, `spirv-val --target-env
vulkan1.3` on every emitted `.spv`, and zero validation findings.

**"The gate" means the CORE set, and the exact scope matters because the numbers above were not all
taken over the same one.** `check_render.ps1` defines ten scenarios and tags five of them `core`; the
default run is those five x 2 runs = 10 renders. The earlier rounds in this document ran all ten (20
renders) because the harness had no tiers yet. The rule for a step here: iterate on the core round,
and run `-Full` (all ten, 20 renders) once when the change is meant to be the step's final state, so
that the five extra references cannot go stale unwatched - `-Update` only re-baselines what ran.

### Step 0 - the stage is reachable (DONE)

`heap_probe.slang:mesh_main` + `run_heap_graphics_probe(slot, mesh_shader)` + the device feature.
Acceptance: **log comparison** (section 3). Gate stays byte-identical: the mesh feature changes no
existing pass.

### Step 1 - `shadow.vert` becomes a mesh stage (DONE)

The smallest real geometry stage: one vertex stream, a depth-only fragment stage, and a pass whose only variability
is the cascade's projection - a uniform, not a per-vertex input. `shaders/shadow.slang` now carries a third entry,
`mesh_main`, and the pass builds a second pipeline from it (`shadow.mesh.spv`, MESH + the SAME fragment stage).
`shadow_pass::pipeline()` answers with the mesh form whenever it exists, so the frame loop is unchanged: the two
pipelines are one pass drawn two ways, and `record_cascade` now says which one it handed over.

**How the geometry reaches the stage.** `shaders/mesh_geometry.slang` is the shared fetch: a device address cast to
a pointer, the engine's 64-byte interleaved vertex read as untyped words bit-cast to float, and the index buffer
read as 32-bit words with the 16-bit case taken as the half its index falls in (the same two tricks
`compute_skin.slang` and `rt_shadow.slang` already use). The DRAW hands the stage its whole geometry window -
two addresses, first index, index count, base vertex, index width - as push data, because a mesh dispatch
(`vkCmdDrawMeshTasksEXT`) has no vertex binding, no index buffer, no `firstIndex`, no `baseVertex` and no
`instanceCount`:

| what the input assembler had | where a mesh stage gets it now |
| --- | --- |
| `vkCmdBindVertexBuffers` | `MeshGeometryLanes::vertex_address` (device address, pushed per draw) |
| `vkCmdBindIndexBuffer` | `MeshGeometryLanes::index_address` + `index_width` |
| the draw's `firstIndex` / `baseVertex` | the same lanes (a static draw's CHUNK is exactly this window) |
| `instanceCount` | the dispatch's `groupCountY`, read as `SV_GroupID.y` - a mesh workgroup grid has three dimensions and a mesh stage has no `SV_InstanceID` |
| the vertex input layout's stride | a renderer constant (64 B), refused rather than guessed for any other layout |

**The output budget is why one workgroup emits 85 triangles.** `maxMeshOutputVertices` and
`maxMeshOutputPrimitives` are both 256 here, and a stage that writes each triangle's three vertices separately -
which is the direct port of what the input assembler produced - spends three of the former per one of the latter:
85 triangles = 255 vertices fits both, 86 would want 258. The host dispatches `ceil(triangles / 85)` workgroups and
the STAGE decides how many of its 85 are real, so an index window that is not a multiple of 85 needs no second
command.

**The push budget, and the one measured deviation from the engine's 128-byte rule.** The stage block is
`96` (material) + `12` (the three heap index lanes the frame's endpoint appends) + `4` (the cascade) + `32` (the
geometry lanes) = **144 bytes**, and the engine's rule is to stay inside the 128 bytes every implementation
guarantees. Sixteen free bytes cannot hold two device addresses and a draw window, so the honest options were a
144-byte block or a new per-frame geometry TABLE in the heap (4 bytes of lanes, the rest read through a slot).
The table is the better long-term shape - it is what step 3's meshlets need anyway - but it is a new heap resource,
slot region and upload path for one consumer, while the block is measurable NOW. So: the mesh path is enabled only
when the device reports room for it, and the vertex path stays where it does not.

```
mesh shaders: available - a pass may build a MESH pipeline (push block 144 B, within 256 B push constants / 256 B push data)
```

That line is the gate: `core::mesh_shader_available` AND `maxPushConstantsSize >= 144` AND the heap's
`maxPushDataSize >= 144`, checked once in `runtime::create_passes` before any pass is created (a pass that tried
and was refused would already have produced a validation ERROR at `vkCreateShaderModule`). On this device the two
limits are 256/256; the same box's other GPU reports 128, which is exactly why the check exists and why the
fallback is the vertex path rather than a failure. A mesh pipeline also has to be built before `create_passes`
returns, and it is not: the pass builds it at create time only when `pass_context::mesh_shaders` says the device
can run one.

**Acceptance: the gate, both ways.** The ten scenarios were captured with the VERTEX path (the committed
references), and the mesh path reproduces every one of them **byte for byte**:

| run | result |
| --- | --- |
| step 0 (mesh feature enabled, vertex shadow path) | 10/10 passed, 0 changed, 0 flaky |
| step 1 (mesh shadow path, `shadow.mesh.spv` in use) | 10/10 passed, 0 changed, 0 flaky - the SAME 10 hashes |
| forced probe: `SetMeshOutputCounts(0, 0)` in the mesh entry, then reverted | `deferred` CHANGED (61770EA9EBFE0714 vs FA1C1BED4DD611C5), and the reference came back on revert |

The third row is what makes the second one mean something: a mesh stage that emits the same triangles produces the
same picture BY CONSTRUCTION, so a green gate alone cannot distinguish "the mesh path drew this" from "the mesh
path was never used". The startup log says which one it was (`shadow: the casters are DISPATCHED (mesh stage) - 103
casters`), and the forced probe proves the log is telling the truth.

The vertex entry, the fragment entry and the pass's fallback are all still there, and `ctest` (8/8), `spirv-val`
(26 modules) and the validation layer stay clean with the mesh path active.

### Step 2 - `pbr.vert` becomes a mesh stage (DONE: the G-buffer, the forward/unlit/transparent leaves and the shadow pass all have mesh forms)

`shaders/pbr.slang` carries a third entry, `mesh_main`, built into a second G-buffer pipeline
(`pbr.mesh.spv` + the SAME `gbuffer.frag`), and the scene session prefers it the way the shadow pass prefers its
own: `runtime::make_gbuffer_pipeline` builds both pipelines from the code the app hands over, and
`make_scene_environment` picks the mesh one and marks the session `mesh_stage`. The vertex body is now
`pbr_shade_vertex(const MeshVertex, vertex_index, instance_index)` - the second half of the objective's "one shared
fetch instead of two mirrored vertex-input declarations": `shadow.slang` and `pbr.slang` each declare their five
attributes once, hand them to a body they share with their own `mesh_main`, and read the same 64-byte record through
the same `shaders/mesh_geometry.slang` fetch.

This is the widest coverage a geometry port can have: the G-buffer pass draws every opaque leaf of every scenario,
so nine of the ten gate scenarios exercise it (`unlit` shades through the forward pipeline, `transparent_blend`
composites through the transparent one - both still vertex stages, and both stay byte-identical, which also proves
the widened stage block did not disturb the vertex path).

**THE VUID THIS STEP FOUND, which changes what "one block" means.** A heap-native pipeline requires EVERY byte of
each declared push block to have been written by `vkCmdPushDataEXT` before the draw
(VUID-vkCmdDrawMeshTasksEXT-None-11376 - and its `vkCmdDrawIndexed` twin):

```
uses push-constant statically at range [0, 144), but vkCmdPushDataEXT was never called for range [108, 112)
```

Three consequences, each measured:

1. **The lanes must sit at the block's previous member's end.** `MeshGeometryLanes` was declared with two `uint2`
   addresses, which align to 8 bytes and left a 4-byte hole - the exact range validation named. It is now eight
   `uint` fields (four address halves, then the four window words), so nothing pads.
2. **A std140 struct member is 16-byte aligned**, which put the struct at 112 even with the hole gone, so the scene
   block declares the word that ends at 112 explicitly (`geometry_pad`) - the offset is now a member's end rather
   than a consequence of the layout rules.
3. **The push must run to the BLOCK's end, not to the lanes' end**: std140 rounds a block's size up to a multiple
   of 16, and the declared block is 144 while the scene's members end at 140 - validation named that range too
   ("[140, 144)"). `primitive::push_geometry_lanes` therefore pushes a zero-filled payload from
   `mesh_geometry_push_offset` (108 scene / 112 shadow) to `mesh_stage_block_size` (144), with the lanes copied to
   where the shader reads them (112 in both).

**AND EVERY DRAW PUSHES THE LANES, INCLUDING THE VERTEX PATH'S.** One source file is one block layout for every
entry it contains, so `pbr.vert` and `gbuffer.frag` declare the lanes that only `mesh_main` reads - and a
descriptor-heap pipeline requires the declared bytes to be written whatever stage reads them. The vertex path
therefore pays 32 bytes and one extra push per draw that it never looks at; the alternative was two block layouts
for one shader file, which the language does not express. Verified both ways: forcing the gate off (both passes on
their vertex pipelines) produces a log with zero validation findings, and `unlit`/`transparent_blend` run through
the vertex path with the widened block in the gate itself.

**Acceptance: the gate, both ways, again.**

| run | result |
| --- | --- |
| step 2 (mesh shadow AND mesh G-buffer) | 10/10 passed, 0 changed, 0 flaky - the same ten hashes as the committed references |
| forced probe: `SetMeshOutputCounts(0, 0)` in `pbr.slang`'s mesh entry, then reverted | `deferred` CHANGED (8687703DA3BCA7EF vs FA1C1BED4DD611C5) - the model leaves the frame entirely |
| gate forced off (`evaluate_mesh_shaders` returning false), then reverted | "mesh shaders: not used", zero validation findings, i.e. a device without the extension runs the vertex path cleanly |

The middle row is again what makes the first mean something, and this time the two failure modes have their own
hashes: `61770EA9EBFE0714` was the frame with the G-buffer RIGHT and no shadows (a lane-offset bug that fed the
shadow stage a window belonging to no draw), `8687703DA3BCA7EF` is the frame with no geometry at all.

The forward, unlit and transparent pipelines were the second half of this step, and they are DONE: each NAMED
pipeline in the registry can carry a mesh form under the same name (`runtime::make_pipeline(name, vertex, fragment,
mesh_vertex)` fills a second map), and a session that binds by name picks the mesh form when there is one - which is
what makes the transparent pass's leaves and the flat (unlit) render mode dispatches too. The choice is made PER BIND
rather than per session there, because a forward session's leaves name their own pipelines; the G-buffer pass has one
fixed pipeline per session, which is why that half came first. Evidence, one line per name at startup:

```
SUCCESS: pipeline 'pbr' created with a MESH form (its leaves are dispatched)
SUCCESS: pipeline 'unlit' created with a MESH form (its leaves are dispatched)
scene: the leaves of 'pbr' are DISPATCHED (mesh stage)
```

The third line is the transparent scenario (`transparent_blend`), whose blend leaf is the one that draws through the
forward `pbr` pipeline. The flat render mode reaches the same code path through its own default name.

### Step 3 - a TASK stage, meshlets, and indirect dispatch

**STATUS: this step is DONE except for the compute-driven form.** The splitter, the heap table, the meshlet entries
for both the shadow pass and the G-buffer, the per-meshlet frustum culling and the INDIRECT SEAM a compute culling
pass writes into are in the tree and gate-green; per-meshlet backface culling was implemented, measured to change six
scenarios, and reverted (see the negative result below); the task stage the objective names as the pre-culling
mechanism is blocked by a compiler bug (see the blocker note at the top) and turned out not to be needed for the
culling itself.

`vulkan/meshlet/meshlet.cppm` is a pure-CPU module that cuts one draw window into meshlets: runs of at most
`meshlet_max_triangles` (85, the mesh stage's output budget from section 2) triangles in index order, each with an
object-space bounding sphere over the axis-aligned box of the vertices it touches. `runtime::create_primitive`
builds them at upload - where the geometry bytes and the layout are still in hand - appends them to the GPU table,
keeps the primitive's own run (`meshlet_base`/`meshlet_count`, which the geometry lanes carry), and the scene import
logs what it produced:

```
descriptor heap: meshlet table written (address 0x1cc30290, 65536 records of 48 B, offset 1096256)
meshlets: 3145 over 103 primitives (85 triangles each at most, object-space spheres)
```

THE TABLE is `heap_slots_meshlets` (slot 745, at the end of the used region so nothing after it renumbers), one
48-byte record per meshlet, host-visible and written ONCE while the scene imports - which is why it owns ONE slot
rather than the per-frame pair every frame-varying buffer has: there is no frame in flight whose contents could
disagree with it (the TLAS is the counter-example, and the reason that rule exists). The host record's fields land
at the same byte offsets the shader reads them at - first index (0), count (4), base vertex (8), a pad word (12),
centre (16/20/24), radius (28), the normal cone's axis (32/36/40) and its cosine (44) - and `tests`-time
`static_assert`s pin every one of them, because this layout is where the GPU hang below came from.

**THE 48 BYTES ARE THE FIX FOR A WEDGED GPU, so the three blocks are not decoration.** The record started at 28
bytes with `center` at 12, while the shader's copy is a std430 `StructuredBuffer` element where a `float3` has a
16-byte ALIGNMENT - so the shader read `center` at 16, `radius` at 28 and the next record 32 bytes on. Every record
after the first was read four bytes off, the fields landed in the wrong members, the index fetch left the buffer and
the device never signalled another fence. The symptom was not a wrong picture but a hang, and validation said only
that a command buffer was reused before its work finished. The normal cone (added for the backface experiment) then
made it 48, and the lesson is the one the layout asserts encode: **a host struct that a shader reads as a buffer
element is a LAYOUT, not a struct** - compare the declared offsets against `spirv-dis` before blaming the GPU.

IT IS A MODULE OF ITS OWN AND A PURE ONE on purpose: the split is arithmetic over vertex and index bytes, and its
bugs are the invisible kind - a sphere that is too small culls a visible meshlet (a hole in the shadow map, on the
frames where it happens to face the light) and an overlapping window draws triangles twice (invisible in a depth
pass, wrong for anything blended). Nothing in a captured frame says which of those happened, so the properties are
asserted on the CPU instead, by `tests/test_meshlet.cpp`, which CI runs:

| property | check |
| --- | --- |
| coverage | the meshlets are contiguous, in order, and cover every index of the window exactly once |
| conservative bounds | every vertex every meshlet indexes is inside its sphere (a box's corner sphere contains the box) |
| limits and determinism | at most 85 triangles each, and the same input twice is byte-identical output |
| both index widths | 16-bit and 32-bit index buffers are read with the same result |
| malformed input | an empty span, no vertices, a stride shorter than one position, a window past the buffer end: an EMPTY result, never a read past it (an out-of-range index inside a valid window keeps the meshlet - the draw is not silently dropped - and simply does not extend the bounds) |

**THE TASK STAGE IS UNUSABLE ON THIS SLANG, FOR TWO INDEPENDENT REASONS - BOTH COMPILER CRASHES.** Throwaway probes
in `build-release-clang64/dvm/` settled the syntax and then closed the fork. `slangc` v2026.18.2 dies with
0xC0000005 (`FAILED: [code=3221225477]`) on a task stage that either takes host data or touches the heap, and it
does so with SIX LINES of shader:

| probe | task entry contains | result |
| --- | --- | --- |
| `task_probe.slang` | payload + `DispatchMesh`, nothing else | **compiles** - 572 B, `OpEntryPoint TaskEXT`, `OpVariable ... TaskPayloadWorkgroupEXT`, `OpEmitMeshTasksEXT gx gy gz %p`, `spirv-val` clean |
| `t_push.slang` | the same plus `[[vk::push_constant]] ConstantBuffer<P> pc;` and `payload.slot = pc.slot;` | **CRASH** - and it crashes with the production flags AND with `-target spirv -profile spirv_1_6` alone, so it is neither the heap capability nor `-allow-glsl` |
| `tp0.slang` | `#include "heap_access.slang"` and NOTHING else (no heap read, no push use) | **CRASH** |
| `t_heaphard.slang` | no push block at all, but a heap read by LITERAL slot (`heap_at<StructuredBuffer<uint>>(...)[0]`) | **CRASH** |
| `t_cull.slang`, `t_addr.slang`, `t_b1.slang`, `t_b2.slang` | culling maths and device-address reads in various shapes | **CRASH** (all of them) |

So the rule is not "the shim" (which is what the previous round's note guessed) and not "device addresses": a task
stage may have a payload and launch mesh workgroups, and NOTHING ELSE. Two consequences, and together they close the
fork:

- **no push data**: the task stage cannot be told the frame slot, the cascade index or the model matrix - and every
  stage in this renderer gets its heap indices exactly that way, because a heap pipeline has no layout;
- **no heap**: it cannot read the light matrices, the meshlet table, or anything else through the shim.

That leaves the task stage with builtins only, which is not enough to cull anything with. The remaining fork for
per-meshlet culling is therefore the one the objective already names as the alternative: **cull in a COMPUTE pass and
dispatch with `vkCmdDrawMeshTasksIndirectEXT`** - a compute stage reaches the heap normally (this renderer's
skinning and mask-bake passes are the proof), and the indirect command's shape (only `groupCountX/Y/Z`, no
`instanceCount`) is the constraint the objective measured for that path. The minimal reproducer above is also the
thing to report upstream: six lines, no engine code, crashing a released compiler.

The probes are NOT in the tree: each was reverted the moment the compiler refused it, so the shader list, the build
and the gate are exactly as this document's step 3 status describes. (One of them briefly WAS in the tree, which is
how the crash was found: `heap_probe.slang`'s `task_main` failed the build, and reverting it plus re-running `cmake`
- `CMAKE_SUPPRESS_REGENERATION=ON` means the build dir keeps the old rules - put the list back.)

**THE MESHLET CONSUMER WAS BUILT, AND IT WEDGES THE GPU.** One attempt is worth recording even though it was
reverted, because it narrows the next one. The whole chain was wired: a `meshlet_main` entry in `shadow.slang` that
reads its window out of the table (`meshlet_at(push.geometry.first_index + group_id.x)` - the lanes carrying the
primitive's run instead of the draw's window), one workgroup per meshlet (`groups = geometry.meshlet_count`), the
lanes switched by a new `render_environment::meshlets` flag, and a third pipeline (`shadow.meshlet.spv`) the shadow
pass preferred over the other two. It BUILT and CREATED:

```
SUCCESS: shadow MESHLET pipeline created (one workgroup per meshlet, window read from the table)
```

... and then the frame never completed. The validation layer reported exactly the signature of a GPU-side hang, one
line each:

```
[ERROR] vkAcquireNextImageKHR(): Semaphore must not have any pending operations.
[ERROR] vkBeginCommandBuffer(): on active VkCommandBuffer 0x... before it has completed. You must check command buffer fence before this call.
[ERROR] vkQueueSubmit(): pSubcommandBuffers[0] VkCommandBuffer 0x... is already in use and is not marked for simultaneous use.
```

i.e. the PREVIOUS frame's commands were still running when the next frame began - not an interface error, and not a
wrong picture. Clamping the shader's `SetMeshOutputCounts` to the 85/255 budget (`min(meshlet.index_count,
mesh_triangles_per_workgroup * mesh_indices_per_triangle)`, which is defensive against a record read at the wrong
index and is worth keeping in any case) did NOT change the outcome, so the hang is not an over-large output count.

The suspects, in the order the next attempt should test them:
1. **the lanes' meshlet base is not what the shader thinks** - the meshlet entry reads `push.geometry.first_index`
   as the record index, so a session that pushes the DRAW's window instead of the run's base reads an arbitrary
   record (a wrong window and a wrong triangle count, i.e. vertices fetched far outside the buffer). A host-side
   dump of the first records plus the lanes' values for one caster settles it in one run;
2. **a static draw dispatches its whole run once per CHUNK** - `static_draw_primitive::draw` loops over chunks, and
   the meshlet path ignores chunks (the run covers the merged buffer), so Sponza's shadow would emit every meshlet
   once per chunk. Wasteful rather than fatal, but it is a real defect of that wiring;
3. **the table's visibility** - it is written once at import through a mapped buffer and read by a stage that never
   ran before, which should be a non-issue (the write happens before any command buffer is recorded) but has not
   been proven with a barrier.

**SUSPECT 3 IS ELIMINATED, AND SO IS "THE TABLE HOLDS GARBAGE".** `runtime::create_primitive` now checks every
record as it leaves the host - non-zero, a multiple of three, at most `meshlet_max_indices`, entirely inside the
draw's index window, and a finite non-negative radius - and logs once if any primitive fails. On the Sponza scene
(3145 records over 103 primitives) it reports nothing:

```
no malformed records: every meshlet window is inside its draw and under the 255-index budget
```

So the records a mesh stage would read are sound, and the hang is not an over-large window arriving from the TABLE.
What is left is how the SHADER gets to a record (the lanes' base, suspect 1) and how many times it is dispatched
(the per-chunk loop of suspect 2) - both host-side facts, both testable without guessing at the GPU. That check is
worth keeping for the same reason the splitter's own tests are: a record's window is what a MESH stage hands to
`SetMeshOutputCounts`, so a malformed one is not a wrong picture but a dispatch asking for output the device does
not have.

**THE HANG WAS NEITHER SUSPECT: IT WAS THE RECORD LAYOUT, AND THAT IS HOW IT WAS CLOSED.** The 28-byte host record
against the shader's std430 stride (see THE TABLE above) was found by a diagnostic dispatch of one workgroup per DRAW
instead of one per meshlet - which still wedged, ruling out the volume of work in one dispatch - and then by
comparing the host struct's field offsets against what the shader declares, field by field. With the padding in and
asserted, the consumer went in for real: `shadow.slang`'s `meshlet_main` reads its own window out of the table, the
shadow pass prefers `shadow.meshlet.spv`, the lanes carry the primitive's run, and the gate came back 10/10 with 0
changed and 0 flaky (commit `00b614f`, step by step below). (The first attempt was reverted in full first - shader,
lanes, session flag, pipeline, registration, both script lists and the CMake rule, with `cmake` re-run because
`CMAKE_SUPPRESS_REGENERATION=ON` otherwise keeps compiling a reverted `VR_SLANG_SOURCES` - and re-landed once the
layout was fixed.) The G-buffer got the same treatment next (`f67c92a`), with the camera frustum as its plane, and
the NAMED pipelines followed (`pbr` and `unlit`, i.e. the forward and transparent leaves too): `make_pipeline` now
takes a third SPIR-V file and keeps a third map, so a leaf that binds by name gets one workgroup per meshlet when the
device has it, then a mesh stage, then the vertex path. `transparent_blend` is the scenario that proves it - its
BLEND leaves log `scene: the leaves of 'pbr' are DISPATCHED per meshlet (meshlet stage)` and the frame is
byte-identical to the reference captured before that form existed, on 30 indirect dispatches with 0 direct.

**THE INDIRECT SEAM IS IN, AND ITS FIRST DESIGN IS THE REASON IT IS WORTH DESCRIBING.** `vkCmdDrawMeshTasksIndirectEXT`
reads `{groupCountX, groupCountY, groupCountZ}` from a buffer, so the counts can be decided on the GPU - which is
what a compute culling pass needs. Every meshlet dispatch now goes through it. The first attempt handed slots out
from an atomic cursor reset per frame in flight, with a fixed capacity and a fall-back to the direct call on
overflow; all ten scenarios stayed byte-identical, but `sponza` came out FLAKY (two runs, `FEF4F9E4AA0C2E6B` and
`4AFE5C90284A1245`), i.e. a command the GPU read was not the command the host meant - a cursor makes WHEN a slot is
rewritten a property of the frame's dispatch COUNT, and that count is not the same on every run. It was reverted in
full (`ffe52fb`). THE REPLACEMENT MAKES THE SLOT THE PRIMITIVE'S OWN: it is `meshlet_base`, which the table's
append-at-import order already makes stable and unique per primitive, so a dispatch rewrites the same slot with the
same counts, a frame's region belongs to one frame in flight (whose previous read the frame-slot wait has already
finished), there is no cursor and no capacity left to exhaust.

Two measured notes on that seam, both worth keeping:

- **a compare-and-fall-back guard cannot be written against a slot that parallel recordings share.** The first
  version of the replacement compared the slot before writing it and sent a mismatch down the direct call; the
  shadow pass's cascades record in PARALLEL, so one thread read another's half-finished store and the log said
  `slot 0 already holds 4x1x1 and this dispatch wants 4x1x1` (the counts are re-read for the line, which is why
  they printed equal). The write is unconditional instead, and safely so: every writer of a slot writes the same
  bytes, because a primitive's meshlet run is cut once at import and its instance count is set when the draw
  primitive is created. A primitive whose counts DID change per frame would need one command per (frame, draw) -
  which is exactly what the compute pass will write.
- **the route is logged, because a seam that silently falls back is a seam nobody tests.** The first version of the
  ORIGINAL attempt resolved the entry point correctly and never bound the buffer (`vk_buffer::handle()` is the
  allocator's id, not a `VkBuffer`), so every dispatch went down the direct path and looked perfect. There are now
  three lines: which route a dispatch takes (once), how many went each way (once, after the first frame - the
  acceptance is "0 through the direct call"), and any reason it fell back.

Acceptance for the seam, both ways: the gate byte-identical on all ten scenarios (`-Full`, 10/10, 0 changed, 0
flaky, `sponza` twice on one hash) with the log reading `4 meshlet dispatch(es) went through the INDIRECT entry
point, 0 through the direct call`; and a FORCED PROBE - the command table written with zero counts - which turned
`deferred` into `8687703DA3BCA7EF` (the no-geometry hash this document already recorded for the earlier probe), so
the counts the GPU used are demonstrably the ones in the table.

**WHAT THE CULLING BUYS, COUNTED (and this is the measurement the log could not carry before).** A rejected meshlet
and a meshlet nobody dispatched are the same pixels, so the culling needed a witness of its own: the mesh entries now
add to counters on the heap (slot `meshlet_stats`, `core::heap_slots::meshlet_stats` - the heap is a mesh stage's
ONLY route to memory) and the host reads the buffer back once, at shutdown, where `wait_idle` has already made the
numbers final. One atomic per workgroup per counter, eight uints per frame in flight, cumulative for the session -
so the figures below are totals over a scenario's 40 frames and are comparable between runs:

| scenario (40 frames) | meshlet workgroups | emitted | culled | share culled | triangles emitted |
| --- | --- | --- | --- | --- | --- |
| `sponza` (3145 meshlets over 103 primitives) | 118541 | 75741 | **42800** | **36 %** | 6325695 |
| `deferred` (DamagedHelmet) | 8554 | 7844 | 710 | 8 % | 665930 |

WHAT THAT SAYS, and what it does not. It says the frustum test rejects a third of the meshlet workgroups on the
heavy scene - that geometry is never fetched and never rasterized, which is the culling's actual effect. It does NOT
say what that is worth in time: **nothing here was timed**, and the counters are counts of work, not of milliseconds.
It also does not separate the shadow pass from the G-buffer (both add to the same counters), and a DEFORMING draw is
never culled at all by design, so its meshlets are in `emitted` whatever the frustum says.

The same counter answers the question item ② is about: a rejected meshlet still costs a WORKGROUP LAUNCH today, so
`culled` is exactly how many dispatches a compute pass that culled before the dispatch would not have recorded -
42,800 over 40 frames on Sponza, i.e. about 1070 a frame. Whether that is worth a pass of its own is a decision the
counts inform and the timings (which do not exist yet) would settle.

**WHAT REMAINS IN THIS STEP** is the pass that writes those counts: a compute pass that culls each meshlet against
the cascade's (or the camera's) frustum - the meshlet table and the light matrices are both heap reads a compute
stage can make - and writes the surviving counts into the command table slot the primitive already owns. One known
defect has to be settled with it: **a static draw in a meshlet session dispatches its whole run once per CHUNK**
(`static_draw_primitive::draw` loops chunks, and the lanes carry the primitive's run rather than the chunk's), which
is invisible in a depth pass and wrong the moment two chunks of one primitive disagree about a material - so the
culling pass either learns the per-chunk material or the meshlet record grows one. Acceptance stays the same:
conservative culling must leave the gate byte-identical, and a forced probe that culls everything must change the
picture.

The constraints the objective measured still stand and are designed around: `vkCmdDrawMeshTasksEXT` has no
`instanceCount` (the dispatch's Y carries it today, and the task payload will carry it once a task stage launches
the workgroups), `VkDrawMeshTasksIndirectCommandEXT` carries only group counts (so an indirect path needs one
command per instance or a task stage that partitions), and the TLAS is built from the SAME vertex and index buffers
by device address - which is why every meshlet path so far keeps FETCHING those buffers rather than replacing them.

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

- **A shader the build compiles and the escape-hatch scripts do not is invisible until someone uses the
  scripts.** `shaders/compile_shaders.ps1`/`.sh` MIRROR `CMakeLists.txt`'s `VR_SLANG_SOURCES` by hand, and the
  mesh migration left `heap_probe.slang:mesh_main:mesh:heap_probe.mesh.spv` out of both - so a machine without
  CMake would have lost the mesh probe and its log-line proof. `tests/test_shader_sources.cpp` now parses all
  four places a `.spv` is named (the CMake list, both scripts, and `chores.cpp`'s loads) and requires them to
  agree, requires the three `.mesh.spv` names to exist, and requires every `mesh_main` a `.slang` file declares
  to have an entry. It runs in CI (both jobs) and is written to FAIL: deleting a mesh entry, or a script entry,
  fails it by name.
- **AN INCLUDE THE BUILD DOES NOT KNOW ABOUT IS STALE SPIR-V.** `mesh_geometry.slang` (and, since the Slang
  migration, `heap_access.slang`) were included by every geometry leaf and absent from CMakeLists'
  `VR_SHADER_INCLUDES`, so the build tree held binaries compiled before the geometry lanes existed while the
  mesh entries sharing their source had been rebuilt - one pipeline whose two stages declared different push
  blocks, and no validation finding, no wrong picture and no build error anywhere. It surfaced as a byte
  difference against the scripts' output (21 of 27 identical, 6 not) and is now a test check: every `#include`
  found in `shaders/` must be in `VR_SHADER_INCLUDES`.
- **A mesh stage's outputs are WRITE-ONLY.** `E54005: cannot read values from mesh shader outputs` - so
  the "keep a fetch alive through a branch the fragment stage never takes" idiom does not compile. The
  probe writes the fetched value into an output lane no fragment stage reads
  (`[[vk::location(1)]] float heap_probe_lane`); a multiply by zero would be folded away WITH the fetch.
- **The vertex-input state must not be PARSED from a mesh module.** `make_pipeline` derives a raster pipeline's
  vertex input (and therefore its buffer stride) from the first stage's Input variables; a mesh stage declares
  none, so the derived list is empty and the pipeline would silently read no geometry. Its `first_stage`
  parameter is what turns that branch off, and the shader must be passed as the first stage either way.
- **The normal has to stay "used" in the shared depth body.** `shadow.vert`'s vertex input layout is derived from
  its own declarations, so a normal that Slang can prove unused disappears from the module and the stride shrinks
  from 64 to 52 with no error anywhere. The never-taken `isnan && isinf` branch that keeps it alive now lives in
  the SHARED body, so both entries keep it - and both entries must keep reading the same five attributes.
- **Slang rejects west-const in a parameter list and in a pointer declarator.** `MeshVertex const v` /
  `uint const* words` are both `E20001 unexpected token`; Slang's spelling is `const MeshVertex v` and `uint*`.
- **`vkCmdDrawMeshTasksEXT` is not exported by the import library.** Calling it directly is a link error
  (`ld.lld: undefined symbol: vkCmdDrawMeshTasksEXT`); it is fetched with `vkGetDeviceProcAddr` (into
  `core::mesh_dispatch` for the draw path, and ad hoc for the probe) and a null answer means "skip the dispatch".
- **Slang names every SPIR-V entry `main`** regardless of the source function's name, which is why three
  entries of one `.slang` file are three `-entry`/`-stage` pairs (`main`/`vertex`, `mesh_main`/`mesh`,
  `frag_main`/`fragment`).
- **`topology` must be `TRIANGLE_LIST`** for a mesh pipeline - the engine's builders already are - and the
  vertex-input state is IGNORED for one, so a mesh draw binds nothing: `vkCmdBindVertexBuffers` on a session whose
  pass dispatched would be a command with no effect (the draw paths skip it for that reason).
- **A 144-byte push block is legal on THIS device and not on every device.** The check is
  `maxPushConstantsSize` (256 here, 128 on the same box's other GPU) and the heap's `maxPushDataSize` (256 here);
  see step 1 for why the block is that size and why the gate exists.
- **`serialize` the strings, not the code**: the mesh `.spv` is 12996 B against the vertex entry's 8288 B for the
  same depth pass (and 3660 B against 920 B for the probe's one triangle), which is a cheap way to tell which
  compiler produced the module you are looking at when a probe's result does not move.

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
