# Descriptor heap migration: handover

**The tree builds, and the renderer now runs end to end** - it initialises, creates every pipeline,
records frames and saves a screenshot (exit 0). **The frame is black**, and the log says exactly why.
This is still not the finished state, but it is no longer a migration that cannot execute at all.

Branch `descriptor-heap-migration` holds the work. `pass-chain` (7fdeb86) is the last commit whose
gate passes, and it is untouched, which is why this work is on a branch. The migration's unit is the
whole frame and it has no green intermediate state - a half-migrated frame renders nothing, and that
measurement (hash `DC5F6D66428C26D8`, mean `0.00` against the reference's `88.1`) is now *explained*
rather than merely observed: see (e) below. So a mid-migration commit cannot pass the gate, and the
choice was between saving the work off the gate-checked branch and losing twenty-five rounds of it.

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

`shaders/post.frag` reads `post_source_texture[pc.post_source_slot]` - and that index is an
**absolute** heap slot, not a base plus `heap_image_index` (contrast the four bloom levels two lines
below it, which are `bloom_lN_texture[heap_slots_bloom_lN + heap_image_index]`). So the *host* has to
resolve the source, per frame, per swapchain image:

| which stage | its source slot |
| --- | --- |
| prefilter (`post_pass`, `level_ == 0`, mode 0) | `post_color + io.frame.image_index` |
| downsample (mode 1, `level_` > 0) | `bloom_l0 + (level_ - 1) * 8 + io.frame.image_index` |
| composite (mode 2) | `post_color + io.frame.image_index` (it takes the bloom levels itself, by constant) |

`post_color` is 639 and the bloom chain is 647/655/663/671 - stride 8, those are the
`core::heap_slots` / `heap_slots.glsl` constants.

**Why this is not a two-line fix.** A pass may not import `vulkan.core`: no pass does today (they import
`vulkan.core.handles` and `vulkan.render_resource`, and the split is deliberate - the grid is the
renderer's, and a pass that knows it could take over an image family). So the pass cannot name 639. The
host must hand the slot over, exactly as it hands over `shared_set_layout`, `shared_pipeline_layout` and
`push_block`: a callback on `pass_context`/`pass_host` that answers "what is the post chain's source
slot for this image at this level", filled in `runtime::make_pass_host` and in the pass context the
runtime builds - the same shape as the three callbacks already there. Until then `post.cpp` pushes the
default lane of `0`, and the shader would read slot `0 + 0` - the bindless texture array's first
entry - rather than its source.

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

### (e) What was measured on the first runs

Two runs of `--config build-release-clang64\render-check\default.toml --capture-frames 1` with
validation on. **The first** found two blockers; both were fixed and the second run passed them:

1. **The heap flag is not optional when the layout is null.** Measured:
   `vkCreateComputePipelines(): pCreateInfos[0].flags (VkPipelineCreateFlags2(0)) does not include
   VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT while layout is VK_NULL_HANDLE`
   (`VUID-VkComputePipelineCreateInfo-None-11367`). The rule is *both or neither*. Four compute
   builders had no `flags2` struct at all, `build_cluster` had one but chained it only on the mapping
   path, and the ray-tracing builder chained none - all now set the bit. **Fixed.**
2. **`spirv-val` rejected the SPIR-V the shared `heap_texel` helper generated**:
   `OpFunctionCall Argument <id>'s type does not match Function <id>'s parameter type`, every failing
   call a call to that helper (`VUID-VkShaderModuleCreateInfo-pCode-08737`). A `texture2D` parameter
   does not survive a function boundary. `heap_texel` is a **macro** now, which expands the fetch at the
   point of use. **Fixed** - and note for the next reader: glslang's preprocessor rejects
   `__VA_ARGS__` outright ("'#define' : bad argument"), so it is a fixed three-parameter macro, which is
   enough because every uv argument keeps its commas inside parentheses.

**The second run** then created every pipeline - pbr, unlit, gbuffer, gbuffer-debug, shadow, deferred,
megalights trace and temporal, TAA, ray-traced shadow (raygen + miss + hit + any-hit), the post chain,
fxaa, the mask bake and the compute skinning - ran the frame loop and saved
`screenshot_20260918_133008.png`. The heap probes still pass (slot 16896 reads `255,255,255,255`,
16897 reads `0,0,0,255`). Three things remain, all precisely identified:

**1. The helper functions - FIXED.** `gbuffer_texel` (deferred.frag) and `downsample_13` / `sample_tent`
(post.frag) took a `texture2D` **parameter**, the same thing that broke `heap_texel`. They take the image's
heap **slot** (a `uint`) now and index the array inside, so the fetch is built where it is written and no
image crosses a function boundary. Indexing one declared array for every caller is sound because every
heap array view is a view of the *same* resource heap - the slot chooses the image, the name does not. After
this change the run reports **no `OpFunctionCall` error at all**: every shader module validates.

**2. The inherited heap bind - FIXED, and it removed every validation error.** A secondary is validated on
its own, so the primary's bind never reached it (`VUID-vkCmdDrawIndexed-None-11308`). The two bind infos
`record_bind` builds are now a separate `descriptor_heap::bind_infos` (one definition, two destinations),
and all three secondaries chain a `VkCommandBufferInheritanceDescriptorHeapInfoEXT` whose `pNext` is the
rendering info they already passed: `scene.cpp`'s segment begin, `transparent.cpp`, and the shadow
cascade in `runtime.cpp`. The passes receive the infos through a `fill_heap_bind` callback on their frame
structs - the heap itself stays the renderer's, exactly as with `make_environment` and `push_block`.

**3. The post chain's source slot - FIXED.** The third push lane now crosses the line it needed to: a post
stage passes WHICH source it reads (0 = the HDR target the chain starts from, N > 0 = bloom level N - 1)
and `runtime::push_stage_block` turns that into the absolute slot the shader indexes, image index
included. Before this every level read the HDR slot, so the whole bloom chain and the composite were
sampling the wrong image.

**After all of that the run is CLEAN: zero `[ERROR]` lines, everything created, frames recorded,
screenshot saved.** And the frame is still a uniform black image whose bytes are identical across
scenarios and across every one of the fixes (`DC5F6D66428C26D8`). That is the state to pick up, and it
is a *data* problem now, not a binding one: the draws are legal, the heaps are bound and inherited, the
probes prove that a heap read and a heap draw both work, and no push is refused.

**What is measured away, all of it by experiment.** Take these as settled - do not re-derive them:

* **The scene pass draws.** A one-shot diagnostic in `record_segment` printed `1 leaf/leaves, heap push
  endpoint SET`: the pass iterates its leaves and the environment carries the endpoint.
* **The pushed frame slot.** Forcing `heap_frame_slot` to a literal `0u` in `pbr.vert` changed nothing.
* **The slot arithmetic.** Both sides agree: a per-image array steps by the RAW image index
  (`gbuffer_albedo + image_index`; `core.cpp` writes image `i` at `heap_slot_base + i`), while different
  arrays are `heap_image_capacity` (8) apart (`bloom_l0 + level * heap_image_capacity`).
* **The push-lane length.** Two lanes instead of three produced six validation errors *and* the same hash.
* **The push block layout.** 8 uints + `alignas(16) glm::mat4` = 96 = `scene_push_constant_size`, and the
  shader's block is the same 8 uints then `mat4`, so the lanes land at 96/100 exactly as declared.
* **The vertex stage, entirely.** With `gl_Position` computed by hand - no camera UBO at all - and the model
  matrix replaced by the identity, the frame is *still* black. Geometry that cannot be mis-projected and
  cannot be mis-placed still produces nothing.
* **The image index, and the capture path.** Painting the post chain flat red changed the captured hash
  (`90BC07F22EE6BBD2` against the black `DC5F6D66428C26D8`), and that red reached the capture *through*
  `fxaa.frag`'s read of `display_texture[heap_slots_display_color + heap_image_index]`. So the index the
  shaders are given is the image being captured, the capture reads the image the passes write, and the post
  chain runs.

**What that leaves.** The G-buffer debug view - which bypasses lighting, TAA, FXAA and the post chain -
is black, so the G-buffer itself is black, and it stays black with a hand-projected identity model. The
scene pass therefore rasterises nothing *visible* into it, and the surviving suspects are the fragment
stage's own reads (the material record `push.material_index` names - note the graphics probe reads record
0 of that same table as white) or the draws not reaching the target at all (they are recorded into
secondaries). The next probe is to read back the material record the primitive names, and the G-buffer
albedo target itself, exactly as `heap_probe.frag` reads its material: the probes are the one instrument
in this migration that has never misled.

**The measurement worth carrying forward:** the black frame's hash is `dc5f6d66428c26d8…` - byte for byte
the hash this migration recorded earlier as "a half-migrated frame renders nothing". It is a *uniform*
image, which is why it survived every fix that changed what the frame does.

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
