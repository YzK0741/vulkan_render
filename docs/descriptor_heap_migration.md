# Descriptor heap: the whole-frame conversion

The target is a renderer with **no descriptor sets, no set layouts, no pools and no
`VkDescriptorSetAndBindingMappingEXT` shim**: every descriptor lives in the resource/sampler heap,
every shader names the heap natively, and every pipeline is created heap-native.

## Why it is one commit and not one pass at a time (measured)

Binding the heap is command-buffer state that takes over **every** stage recorded after it. With the bind
recorded at the start of the frame and every mapping switched off - so the heap held nothing any shader had
asked for - all nine gate scenarios came back as the **same** frame:

```
hash DC5F6D66428C26D8   mean 0.00   (unlit reference: 88.1)   validation: SILENT
```

A stage whose descriptors came from a set reads the heap instead once one is bound, so a half-migrated frame
renders *nothing*. The unit of this migration is the frame.

Two rules cost one gate run each and are not optional:

1. Without `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT` a mapping is **silently ignored** ("the
   VkDescriptorSetLayout will be read instead"). The bit is `0x1000000000`, i.e. past the 32-bit `flags` field,
   so it arrives through `VkPipelineCreateFlags2CreateInfo`.
2. That bit **requires `VkPipelineLayout == VK_NULL_HANDLE`** - the layout is exactly what the heap replaces.
   (This holds for the mapping path; the native path below drops layouts for the same reason.)

## The shader model: heap-native GLSL, proven with this toolchain

`glslc` (shaderc v2026.3, glslang 11.1.0) supports `GL_EXT_descriptor_heap`. Verified compiling, in one
fragment shader, against `--target-env=vulkan1.3`:

| declaration | form |
| --- | --- |
| images | `layout(descriptor_heap) uniform texture2D heap_textures[];` |
| samplers | `layout(descriptor_heap) uniform sampler heap_samplers[];` |
| uniform blocks | `layout(descriptor_heap) uniform Camera { mat4 view; } heap_cameras[];` |
| storage blocks | `layout(descriptor_heap) buffer Cluster { uint counts[]; } heap_clusters[];` |
| storage images | `layout(descriptor_heap, rgba16f) uniform image2D heap_images[];` |
| acceleration structures | `layout(descriptor_heap) uniform accelerationStructureEXT heap_tlas[];` |

Rules that follow from the extension and are already paid for:

- A **combined image sampler cannot be declared**. It is constructed at the use site:
  `texture(sampler2D(heap_textures[i], heap_samplers[s]), uv)`. Every fetch therefore names an image index
  *and* a sampler index - the sampler is no longer implicit in the binding, so the sampler indices become a
  shader-visible convention (the five shared samplers at fixed heap indices, named by `const uint` in a
  shared GLSL header).
- A **variable index** requires `#extension GL_EXT_nonuniform_qualifier : enable` and `nonuniformEXT(i)`.
- Storage images with `descriptor_heap` need a **format layout qualifier** (`rgba16f`, `r16f`, ...), which the
  existing shaders already carry.
- `descriptor_stride` (a power of two) may override the API stride for an array.

### The index space: an index IS a byte offset divided by that kind's descriptor stride

Every `descriptor_heap` array in a shader **aliases the same heap from offset 0**, so
`heap_lights[i]` reads `heapBase + i * stride(lights)`. Two declarations of different names but the same kind
and stride are literally the same memory. The workable convention is therefore:

- the host places a descriptor at an offset that is a multiple of its kind's descriptor stride
  (device properties: buffer 16 B / align 8, image 32 B / align 32, sampler 32 B / align 32);
- a shader indexes with `byteOffset / stride`, i.e. indices are kind-relative **slot numbers**;
- the values that vary per frame or per swapchain image (`frame slot`, `image index`, material id, texture id,
  sampler id, the post chain's per-pass image base) reach the shader through **push constants**.

Push *data* (`vkCmdPushDataEXT`) cannot carry these: it only feeds mapping sources, which this model removes.

## Shader inventory (what each file declares today, from `grep layout(set =`)

| file | set 0 | set 1 |
| --- | --- | --- |
| `shading.glsl` (shared) | 0 CameraUBO, 2 env cube, 3 irradiance cube, 4 BRDF LUT, 7 LightUBO, 8 shadow array, 11/12 cluster | - |
| `surface.glsl` (shared) | 1 `textures[]`, 5 materials | - |
| `pbr.vert` | 0 camera, 6 instance transforms, 13 previous transforms, 9 skin matrices, 10 morph | - |
| `shadow.vert` | 6, 9, 10, 7 | - |
| `shadow.frag` | 1, 5 | - |
| `unlit.frag` | 1, 5 | - |
| `light_cluster.comp` | 0, 7, 11 (write), 12 (write) | - |
| `compute_skin.comp` | 9 | - |
| `mask_bake.comp` | 1, 5 | - |
| `deferred.frag` | 14 rt shadow visibility | 0..3 G-buffer, 7 ml lighting |
| `gbuffer_debug.frag` | - | 0..4 (its own set 0) |
| `megalights_trace.comp` | 16 TLAS | 0..3 G-buffer, 6 `image2D` out |
| `megalights_temporal.comp` | 0..4 (its own set 0) | - |
| `post.frag` | 0 source, 1..4 bloom, 7 depth, 8 normal | - |
| `taa.frag` | 0..3 | - |
| `fxaa.frag` | 5 display colour | - |
| `rt_shadow.rgen` | 0, 7, 16 TLAS, 15 `image2D` out | 1, 3 G-buffer |
| `rt_shadow.rahit` | 1, 5, 17 mask instances | - |

## Host-side work

1. **Heap layout, reserved once** in `core`: per-kind index spaces (frame slot x2; swapchain image xN;
   material table; texture array; the five samplers; TLAS array; storage images; the post chain's per-pass
   images), each offset a multiple of that kind's descriptor stride, recorded in `core` beside the existing
   `heap_*` members.
2. **Every descriptor write gains a heap twin**: each `vkUpdateDescriptorSets` site (scene set, G-buffer
   family, post family, IBL, shadow, cluster, AS, storage images) becomes `write_image`/`write_buffer` into the
   reserved offsets, per frame slot or per swapchain image.
3. **Push constants carry the indices**: each shader's push-constant block gains `frame_slot` and
   `image_index` (and a per-pass image base where needed), and each host push site writes them.
4. **Pipelines**: builders stop taking/creating set layouts, create with `VkPipelineCreateFlags2CreateInfo` +
   `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT` and a null layout - graphics, compute and ray-tracing alike.
5. **Delete**: `vkCmdBindDescriptorSets` everywhere, set layouts, pools, families, and the whole mapping shim
   (`scene_heap_layout`, `scene_heap_stage_mapping`, `map_from_heap`).
6. **Frame**: bind the heaps once (the existing `record_bind`), with no push.

## Acceptance

`BUILD=0`, 8/8 tests, gate 9/9 `changed: 0` (frame byte-identical, `mean|d| ~0.029`), validation silent, and a
deliberate **wrong-index** push as the negative proof (the picture must break, which is what proves the shaders
read the heap).

## The acceleration structure, and the one thing it still needs

The heap's descriptor payload has no acceleration-structure member - the whole union is

```c
typedef union VkResourceDescriptorDataEXT {
    const VkImageDescriptorInfoEXT*        pImage;
    const VkTexelBufferDescriptorInfoEXT*  pTexelBuffer;
    const VkDeviceAddressRangeEXT*         pAddressRange;
    const VkTensorViewCreateInfoARM*       pTensorARM;
} VkResourceDescriptorDataEXT;
```

so an AS in a heap is an **address range** with `type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR`, i.e. the
structure's `vkGetAccelerationStructureDeviceAddressKHR` address - which is what "a heap descriptor for one is its
device address" meant. `descriptor_heap::write_buffer(offset, address, size, type)` already builds exactly that.

WHAT IS STILL OPEN IS THE SIZE, and it is open on purpose rather than guessed: the same call's size ends up in
`VkDeviceAddressRangeEXT::size`, and this renderer already learned (VUID-VkDeviceAddressRangeKHR-address-11365,
on the material table) that a heap range must carry a REAL size rather than `VK_WHOLE_SIZE`. An acceleration
structure is not a buffer, so the candidates are its creation size (`VkAccelerationStructureCreateInfoKHR::size`)
or the size the build sizes query reports - whichever the structures module already knows. The write site is
`runtime::build_rt_structures`'s per-slot block (runtime.cpp, beside `write_rt_structure_binding`), inside the
same `rt_binding_written[frame_slot] != tlas` guard, at grid slot `heap_slots::tlas + frame_slot` - the two slots
the TLAS has BECAUSE it is rebuilt every frame.

THE SIZE IS COMPUTED IN THE RAY-TRACING MODULE, NOT THE ACCELERATION-STRUCTURE ONE - and that correction cost a
build, so it is written down: the TLAS this renderer uses is built by **`ray_tracing::structure_set`**
(`vulkan/ray_tracing/ray_tracing.cppm`, `build()`/`update()` in `ray_tracing.cpp`), whose public surface today is
`handle(frame_slot)`, `instance_table(frame_slot)`, `casters()`, `attempted()`, `ready()`. The
`acceleration_structure` module's `top_level_structure` also creates a top level structure with
`create.size = sizes.accelerationStructureSize`, and that is the module the first version of this note pointed at;
pointing there made the write compile against a class the runtime does not own (the error names the type:
`no member named 'structure_size' in 'vulkan::ray_tracing::structure_set'`).

So the next step is: publish the size from `structure_set` - it is known where the structure is created, next to
`handle()` - and then the write in `runtime::build_rt_structures` is three lines: the address query, that size,
and `write_buffer` with `VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR`, at grid slot `heap_slots::tlas +
frame_slot`, inside the existing `rt_binding_written[frame_slot] != tlas` guard.

## Where the work stands (measured, not claimed)

POPULATED AND VERIFIED IN THE LOG. The grid is reserved at 1 MiB and every write below is proven by its own log
line, because a heap write has no picture to show for itself until the shaders read the heap:

| what | grid slot(s) | evidence |
| --- | --- | --- |
| 6 shared samplers | sampler grid at 65536 | `6 shared samplers written to the sampler grid at 65536` |
| material table (binding 5) | 512 | `material table written (... offset 1081344)` = (16384 + 512) * 64 |
| texture array (binding 1) | 0 + index | `N texture descriptors written` |
| camera / clusters (0, 11, 12) | 514, 518, 520 (+frame slot) | `2 per-frame descriptor(s) written for grid slots 16898..16899` (and 16902, 16904) |
| motion / skin / morph (13, 9, 10) | 524, 526, 528 (+frame slot) | the same lines for 16908, 16910, 16912 |
| instance table (6) | 522 | silent success, no `did not reach` line |
| env / irradiance / BRDF LUT (2, 3, 4) | 532, 533, 534 | silent success, no `did not reach` line |
| top level structure (16) | 703 + frame slot | silent success, validation SILENT with a real range size |
| light UBO (7) | 516 + frame slot | silent success |

WHAT THE HEAP WRITES ALREADY TAUGHT (each cost a build or a validation cycle):

- A heap-bound buffer needs `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` (VUID-VkBufferDeviceAddressInfo-buffer-02601:
  the material table, the light UBO, then motion/skin/morph and the instance table in one round).
- A heap range needs a REAL size, never `VK_WHOLE_SIZE` (VUID-VkDeviceAddressRangeKHR-address-11365).
- `vkGetAccelerationStructureDeviceAddressKHR` is not exported by the loader: resolve it per device.
- Success must be logged, or "wrote it" and "the vector was empty" look the same.

STILL TO POPULATE: the shadow map (binding 8, base slot 535), the per-swapchain G-buffer and post images (base
slots 551..702), the storage images (binding 15 and the megalights outputs), the mask/instance table at binding 17
(its write site is `runtime::write_rt_structure_binding`, which already handles 16 and 17 together), and any
image whose binding is repointed later (the furnace mode).

STILL TO DO, and it is the larger half: (2) heap-native shaders in place of `layout(set, binding)` - untyped
declarations indexed with `descriptor_stride = 64`, combined image samplers CONSTRUCTED at the use site, and the
`frame_slot` / `image index` carried in PUSH CONSTANTS, because push data only feeds mapping sources; (3) every
pipeline created with `VkPipelineCreateFlags2CreateInfo` + `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT` and a
NULL layout; (4) bind the heaps once per frame and delete descriptor sets, set layouts, pools and the mapping
shim (`scene_heap_layout`, `scene_heap_stage_mapping`, `set_scene_heap_layout`, `build_cluster`'s
`map_from_heap` flag, `push_heap_frame_slot`). Then the flip, and the negative proof: push a WRONG index and show
the picture break.

The migration is still one frame-wide switch, for the reason at the top of this file: a frame whose stages do not
all read the heap renders nothing at all.

## The native path's two open questions, and the POC that died answering them

The plan was to prove the heap-native shader path on the MASK BAKE, which looks ideal: it reads exactly two
descriptors (the material table and the bindless texture array, both already on the grid), it is a job rather
than a frame pass, and it only runs under `[render] rt_mask_bake`, which no gate scenario sets - so a mistake
could not reach a reference frame. The conversion was written and COMPILED (glslc accepts it first try):

```glsl
#include "heap_slots.glsl"
layout(descriptor_heap, descriptor_stride = heap_slot_stride) uniform texture2D heap_textures[];
layout(descriptor_heap) uniform sampler heap_samplers[];
layout(descriptor_heap, descriptor_stride = heap_slot_stride) readonly buffer MaskMaterials { MaskMaterial materials[]; } heap_material_tables[];
...
heap_material_tables[heap_slots_materials].materials[pc.material_index]          // ARRAY name = heap index,
texture(sampler2D(heap_textures[heap_slots_textures + material.tex_indices.x],  // block member = record index
                  heap_samplers[heap_sampler_texture]), uv)
```

It was REVERTED, because the probe cannot be isolated: `ray_tracing::structure_set` builds the structures in the
FRAME's command buffer (`runtime::build_rt_structures`, whose gpu marks sit next to the frame's), and
`runtime::structure_record_mask_bake` records the bake into that same command buffer. Binding the heap there is
exactly the measured poison that makes every set-based stage in the frame read the heap - so the bake is only
safe to convert *together with the whole frame*, which is the point this file opens with. A lesson about the
method, not the mechanism: "off the gate's path" is not the same as "isolated".

TWO QUESTIONS THE REAL CONVERSION MUST ANSWER, in this order:

1. **Does a heap-native pipeline need `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT` at all?** The flag's
   validation message is about MAPPINGS being read instead of a layout; a native declaration references the heap
   builtin directly. Test it by converting one stage with the flag OFF and a normal layout.
2. **If the flag IS needed, the layout must be NULL** (measured, VUID above) - and then there are no PUSH
   CONSTANTS, because those live in the pipeline layout. The indices this design planned to carry in push
   constants would have to come from the heap instead (a per-frame-slot index block is already on the grid, and
   which slot is read is itself the chicken-and-egg: it must be derivable from something the shader already
   knows, or the slot's DESCRIPTOR must be rewritten per frame, which frames in flight forbid).

That is the decision to make first, and it is a measurement away: one stage, either answer, both cheap.

## What populating the IMAGE half needs (found by trying)

A heap image descriptor carries a **`VkImageViewCreateInfo`, not a view** - the driver creates the view inside
the descriptor (this is why `descriptor_heap::write_image` takes a create info). The renderer builds its views
through `core::make_image_view`, which throws its create info away, so the heap twin needs that info again.

IT DOES NOT NEED NEW PLUMBING: `vulkan::make_image_view_info(image, format, view_type, aspect, mip_levels,
array_layers)` - a `constexpr` function in **`vulkan.constant_init`**, which `runtime.cpp` already imports - is
already the one place that builds it, and `core::make_image_view` is a call to it plus `vkCreateImageView`. So
the rule for every image is: write the heap descriptor **where the image and its view are created**, from the
same `make_image_view_info` arguments - not in `write_*_bindings`, which only carry handles.

Two consequences worth knowing before doing it:

- The site that WRITES the heap descriptor has to be the site that knows the format and range. `set_ibl` knows
  them (it creates the environment, irradiance and BRDF-LUT images), while `write_ibl_bindings` only sees the
  three views - so the IBL writes belong in `set_ibl`.
- A binding that CHANGES VIEW later needs a heap rewrite beside that change: the furnace mode points slots 0 and
  1 at the constant cube instead of the real environment (`write_ibl_bindings`), and a heap descriptor written
  only at creation time would keep reading the old image. The per-frame slot arrays are the same kind of trap
  seen from the other side: which SLOT is read is a shader index, and it is the host's job to keep the contents
  of every slot that index can name current.

## The grid as it stands (measured at startup)

```
descriptor heap: available (VK_EXT_descriptor_heap, revision 1)
SUCCESS: descriptor heap created (resource 1088 KiB at ..., sampler 128 KiB at ...;
         strides buffer 16 B, image 32 B, sampler 32 B)
descriptor heap: the reserved window ends at 96768 B, so the grid's 1 MiB base is 1048576 B away
descriptor heap: slot grid at 1048576 (1024 slots x 64 B; textures 0 materials 512 tlas 513
                 camera 514 light 516 clusters 518/520 gbuffer 551 env 532 lut 534)
```

- The resource grid's base is **1 MiB** and its stride **64 B**; the slot numbers (relative to that base) are
  the constants in `shaders/heap_slots.glsl` and `core::heap_slots`. 703 of the 1024 slots are named.
- The SAMPLER heap is a second grid at a **64 KiB** base with the device's 32 B stride: the API caps that heap
  at 128 KiB, so it cannot use the resource grid's 1 MiB. Refused unless the device's reserved window fits
  below the base and its sampler stride is 32 B.
- **The heap sizes are derived from the layout, not guessed.** They were 256 KiB and 64 KiB, and both were
  wrong in ways only the log showed: 256 KiB could not fit the 1 MiB base ("a reservation of 951808 B does not
  fit the 262144 B resource heap", after which the grid landed on the reserved window at 96768 B and was
  refused), and 64 KiB of sampler heap is *exactly* the reserved window the embedded-sampler path requires,
  i.e. no usable sampler space at all. They are 1088 KiB and 128 KiB now.
- Still inert: nothing reads the grid, so the frame is unchanged (gate 9/9, `changed: 0`).
