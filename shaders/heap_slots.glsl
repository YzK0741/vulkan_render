// THE SLOT GRID: where every descriptor of this renderer lives, as a HEAP-NATIVE shader sees it.
//
// A shader that declares `layout(descriptor_heap) uniform texture2D t[];` can only say `t[i]`, and
// GL_EXT_descriptor_heap resolves that to `heapBase + i * stride` - every such array aliases the heap from
// offset 0, so an INDEX IS A BYTE OFFSET DIVIDED BY THAT KIND'S STRIDE, and the only way for a shader to find
// its data at a compile-time-known index is for the data to sit on a grid with a FIXED base. Both heaps start
// that grid at 1 MiB (past any reserved window a driver reports here: 94 KiB of resource heap, 64 KiB of
// sampler heap), the stride is fixed at 64 B - the smallest power of two covering the device's largest
// descriptor, 32 B - and each array below is a CONTIGUOUS RUN OF SLOTS.
//
// THESE NUMBERS MUST EQUAL vulkan/core/core.cppm's `core::heap_slots`, which is where the host reserves them;
// the host logs its own table at startup (`descriptor heap: slot grid ...`), which is how a drift is seen.
// See docs/descriptor_heap_migration.md.
//
// The stride override is what makes the grid possible: `layout(descriptor_heap, descriptor_stride = 64)`.
// GL_EXT_descriptor_heap requires that value to be a power of two, and the host writes its descriptors 64 B
// apart, so both sides agree on where slot N is.

#ifndef HEAP_SLOTS_GLSL
#define HEAP_SLOTS_GLSL

// the grid's base, in slots: 1 MiB / 64 B
const uint heap_slot_base = 16384u;
// what a declaration must pass as its layout qualifier (see the note above)
const uint heap_slot_stride = 64u;
// per-swapchain-image arrays are given 8 slots: more than any swapchain this renderer creates (3-4 in practice)
const uint heap_image_capacity = 8u;

// the bindless texture array: binding 1, one slot per scene texture
const uint heap_slots_textures = heap_slot_base + 0u;
// the material table (binding 5), and the top level structure (binding 16) as a TWO-SLOT array: the TLAS is
// rebuilt every frame, so one slot would hold one frame's structure while the other is still in flight (slot 513,
// where it first was, is left unused rather than half-filled; see core.cppm's heap_slots).
const uint heap_slots_materials = heap_slot_base + 512u;
const uint heap_slots_tlas = heap_slot_base + 703u;
// per frame slot (2 of them), the scene set's frame-varying half
const uint heap_slots_scene_camera = heap_slot_base + 514u;
const uint heap_slots_scene_light = heap_slot_base + 516u;
const uint heap_slots_cluster_counts = heap_slot_base + 518u;
const uint heap_slots_cluster_indices = heap_slot_base + 520u;
const uint heap_slots_instance_transforms = heap_slot_base + 522u;
const uint heap_slots_previous_transforms = heap_slot_base + 524u;
const uint heap_slots_skin_matrices = heap_slot_base + 526u;
const uint heap_slots_morph_data = heap_slot_base + 528u;
const uint heap_slots_mask_instances = heap_slot_base + 530u;
// frame-invariant images the shading stage samples
const uint heap_slots_env_cube = heap_slot_base + 532u;
const uint heap_slots_irradiance_cube = heap_slot_base + 533u;
const uint heap_slots_brdf_lut = heap_slot_base + 534u;
// per swapchain image: the shadow map, the ray-traced visibility, and every G-buffer/post image
const uint heap_slots_shadow_map = heap_slot_base + 535u;
const uint heap_slots_rt_visibility = heap_slot_base + 543u;
// the SAME image as a STORAGE descriptor (the visibility pass writes it, the lighting stage samples it, and no
// single heap descriptor is both): see core.cppm's heap_slots
const uint heap_slots_rt_visibility_storage = heap_slot_base + 711u;
const uint heap_slots_gbuffer_albedo = heap_slot_base + 551u;
const uint heap_slots_gbuffer_normal = heap_slot_base + 559u;
const uint heap_slots_gbuffer_material = heap_slot_base + 567u;
const uint heap_slots_gbuffer_depth = heap_slot_base + 575u;
const uint heap_slots_gbuffer_velocity = heap_slot_base + 583u;
const uint heap_slots_ml_trace = heap_slot_base + 591u;
const uint heap_slots_ml_history = heap_slot_base + 599u;
const uint heap_slots_ml_resolved = heap_slot_base + 607u;
const uint heap_slots_ml_lighting = heap_slot_base + 615u;
const uint heap_slots_taa_current = heap_slot_base + 623u;
const uint heap_slots_taa_history = heap_slot_base + 631u;
const uint heap_slots_post_color = heap_slot_base + 639u;
const uint heap_slots_bloom_l0 = heap_slot_base + 647u;
const uint heap_slots_bloom_l1 = heap_slot_base + 655u;
const uint heap_slots_bloom_l2 = heap_slot_base + 663u;
const uint heap_slots_bloom_l3 = heap_slot_base + 671u;
const uint heap_slots_post_depth = heap_slot_base + 679u;
const uint heap_slots_post_normal = heap_slot_base + 687u;
const uint heap_slots_display_color = heap_slot_base + 695u;
// one past the last array: 703 slots are in use, the grid reserves 1024
const uint heap_slot_count = 1024u;

// THE SAMPLER HEAP IS A SECOND GRID, and its own base: the sampler heap is a separate heap in this API, and the
// API caps it at 128 KiB, so it CANNOT use the resource grid's 1 MiB - 64 KiB, the reserved window the
// embedded-sampler path requires, is the largest base it can have, and the host refuses the heap path if a
// device's reserved window were larger or its sampler stride were not 32 B (core.cpp checks both).
// Index 0..4 are the five shared samplers the scene set's declarations choose between, in the order
// core::create_samplers makes them.
const uint heap_sampler_base = 2048u; // 64 KiB / 32 B
const uint heap_sampler_stride = 32u;
// In the order core::create_samplers makes them (see core.cppm's shared_sampler_infos), which is also the order
// they are written onto this grid.
const uint heap_sampler_texture = heap_sampler_base + 0u;        // linear, repeat, mips to 12
const uint heap_sampler_post = heap_sampler_base + 1u;           // linear, clamp, one mip
const uint heap_sampler_gbuffer = heap_sampler_base + 2u;        // nearest, clamp
const uint heap_sampler_post_nearest = heap_sampler_base + 3u;   // nearest, clamp (the same sampler twice)
const uint heap_sampler_taa = heap_sampler_base + 4u;            // linear mag / nearest min, clamp
const uint heap_sampler_shadow = heap_sampler_base + 5u;         // depth compare, clamp

#endif // HEAP_SLOTS_GLSL
