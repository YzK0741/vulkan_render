# Descriptor heap migration: handover

**The tree builds. The renderer does not run yet. This is not the finished state.**

Branch `descriptor-heap-migration` holds the work. `pass-chain` (7fdeb86) is the last commit whose
gate passes, and it is untouched, which is why this work is on a branch: the migration has **no green
intermediate state**, because its unit of work is the whole frame. A half-migrated frame renders
nothing - measured, not assumed: with every mapping off, all nine gate scenarios returned hash
`DC5F6D66428C26D8` and mean `0.00` against the reference's `88.1`, with validation silent. So a
mid-migration commit *cannot* pass the gate, and the choice was between saving the work off the
gate-checked branch and losing twenty-five rounds of it.

## What is done, and what verifies it

1. **Every descriptor is in the heaps.** A fixed slot grid at a 1 MiB base with a 64 B stride
   (`core::heap_slots`, mirrored in `shaders/heap_slots.glsl`), the sampler grid at 64 KiB with a 32 B
   stride, and a host write site for every slot the shader header names - `tests/test_render_resources.cpp`
   fails if a name appears in one and not the other, and it also pins the values, the scalars and the
   sampler order.
2. **Every shader is heap-native.** `layout(set = ` appears **zero** times in `shaders/`; the shader
   build is green. Nineteen files were converted, including the ray-tracing pair and the three compute
   stages.
3. **Every pipeline is created against `layout = VK_NULL_HANDLE`** - graphics through
   `core::pipeline`, compute and ray tracing through the builders in `pipelines.cppm`.
4. **Every push travels as data.** `vkCmdPushConstants` appears **zero** times outside comments in
   `vulkan/`: all 21 sites now go through `vkCmdPushDataEXT`. The frame's passes use
   `resolved_io::push_block`; the primitive draws use `render_environment::push_block`; the mask bake,
   the skinning dispatch and the shadow cascade use endpoints of their own shape.
5. **The frame binds the heaps once**, in `runtime::begin_recording`, and the per-frame and
   per-swapchain-image indices ride in each stage's own push block, appended by the endpoints
   (`push_stage_block` = 3 lanes, `push_index_block` = 2, `push_raw_block` = 0).
6. **The mechanism was proven before it was applied**: a compute probe and a graphics probe read a
   written slot and returned the written value, and a deliberately wrong slot returned a different
   one. That negative proof is the reason the rest was worth doing at all.

## What remains

### (a) Descriptor binds - DONE

`vkCmdBindDescriptorSets` now appears **zero** times in `vulkan/`. Every bind is gone: the scene pass's
per-segment bind and the transparent pass's per-secondary one (each of which existed because a secondary
inherits no state from its primary - the heap bind has the same property, and the runtime makes it when
it records the buffer), plus the sets in `fxaa`, `gbuffer_debug`, `cluster`, `deferred`, `post` (both
record paths), `taa`, `megalights_trace`, `megalights_temporal`, `rt_shadow` and the shadow content in
`runtime.cpp`. `scene_pass::record_segment` lost its `scene_set` parameter with them, which is why its
call sites and its declaration in `scene.cppm` changed too, and `runtime::record_shadow_content` lost
its only use of `pipeline_layout` (now `[[maybe_unused]]`, because the callback signature it must match
belongs to the shadow pass).

Note that the guards which gate on these binds (`io.pipeline_layout == VK_NULL_HANDLE`,
`env.layout != VK_NULL_HANDLE`) still *pass*, because the runtime still hands out
`scene_pipeline_layout` - so none of this was caught by an early return, and none of it was dead code.

### (b) The post chain's source slot

`shaders/post.frag` takes a third push lane, `post_source_slot`, because which image a post stage reads
is the stage's own business (the HDR target for the prefilter and the composite, the previous bloom
level for the downsampler). `vulkan/pass/post.cpp` must pass that base as `extra_lane`; today it pushes
the default `0`, so the shader would read slot 0 plus the image index rather than its source.

### (c) The objects nothing points at any more

Still created, still written, no longer read by any shader: `core::scene_pipeline_layout` and
`scene_descriptor_set_layout` (`vulkan/core/core.cpp:1536`, `:1412`), `core::create_descriptor_pool`
(`:1353`), `runtime::scene_sets`, `runtime::gbuffer_family`, `runtime::post_family`,
`pipelines::make_post_set_layout` / `make_gbuffer_set_layout`, `bindings::make_set_layout` /
`write_set` / `image_set_family`, the passes' `pipeline_layout_` / `set_layout_` members and their
`pipeline_layout()` / `set_layout()` accessors, and the `shared_set_layout` / `shared_pipeline_layout` /
`descriptor_set` context plumbing. Deleting them is the whole point of the migration's title, and it is
purely subtractive work that does not affect what the frame does.

### (d) The mapping shim

`pipelines::scene_heap_layout`, `scene_heap_stage_mapping`, `set_scene_heap_layout`,
`map_from_heap` in `build_cluster`, `runtime::push_heap_frame_slot` and
`descriptor_heap::make_mapping` are now unreachable - `map_from_heap` is `static constexpr bool = false`
and the mapping pointer it guards is therefore never built. They can go.

### (e) What to check first when it runs

These are the three things that have *not* been measured, in the order they will show up:

1. **Push block sizes.** `render_resource`'s `push_block::size` values were written when a block ended
   where the shader's block ended. The shaders now declare the index lanes as their last fields, so each
   converted stage's declared size grew by 8 bytes (12 for the post chain). If validation reports a push
   whose range exceeds the declared block, or a shader reading a field that was never written, this is
   the first place to look.
2. **Lane count against declaration.** Each endpoint appends exactly what its shader declares (0, 2 or
   3 lanes) *by construction*, but the assignment was made by reading each shader's push block, so a
   stage whose block was extended without a matching endpoint will show up here.
3. **The heap bind's timing.** The bind takes the whole command buffer, and `begin_recording` is where
   it happens; if a frame comes back black, the ordering of that bind against the first dispatch is the
   thing to instrument.

## How to finish and verify

```powershell
cmake --build build-release-clang64            # must be exit 0
ctest --test-dir build-release-clang64         # 8 binaries
pwsh scripts\windows\check_render.ps1          # 9 scenarios, changed must be 0
```

Then re-do the negative proof as a *frame-level* check (push a wrong slot on purpose and confirm the
frame changes), and merge to `pass-chain` only when the gate reports nine unchanged scenarios with
validation silent.

The rules the extension actually enforces - the five VUIDs, the null-layout requirement, the glslang
restrictions that shaped the shader side, and the counted work list this document is the tail of - are
in `docs/descriptor_heap_migration.md`.
