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
