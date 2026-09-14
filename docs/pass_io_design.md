# `vulkan.pass_io`: describing a pass's inputs and outputs the way Vulkan already does

The design for the layer that `docs/runtime_split.md` names as the precondition of per-pass modules. It is
written before any code, and it is a DESIGN: nothing below is implemented yet.

## 1. Why this shape, and what it is not

A pass's inputs and outputs in this renderer are ALREADY descriptor sets - the G-buffer set is the interface
between the G-buffer pass, the lighting stage, the GI chain, the probe cache and post; the scene set is the
substrate; each pass's private family (TAA 4 bindings, post 9, probe 9, the temporal resolve 7) is its own I/O.
What exists today is that interface written TWICE BY HAND and kept in agreement by discipline:

* the layout, in `vulkan/pipelines/pipelines.cppm` (`build_ssgi_temporal` builds a 7-binding layout with a loop
  whose storage case is `b == 4u`, `build_gbuffer_debug` a 16-binding one whose storage cases are enumerated);
* the descriptor WRITES, in `vulkan/runtime.cpp`'s `ensure_*_descriptors()`, as a parallel array of views, a
  parallel array of `VkDescriptorImageInfo`, and ternaries that decide storage-vs-sampler and which sampler.

Both drifts this pair can have are already in this project's history, and both were found by the validation
layer rather than by review: a pool sized for four descriptors per set while the layout asked for five, and a
binding whose type changed without its writer noticing. This design makes the two come from ONE declaration.

IT IS NOT a render graph and not a mini-RHI. It does not allocate images (`core` keeps doing that - it is the
only place the complete image list exists), it does not order passes or derive barriers in its first version,
and it does not renumber a single shader binding. What it does is: one declaration, from which the layout, the
writes and the per-type descriptor counts are generated, checked against the shader that must consume them.

## 2. The model

Three ideas, and everything else follows from them.

1. **A binding is a use.** One record says "at (set, binding) this pass touches THIS resource, this way". The
   `VkDescriptorType` follows from the kind, the image layout follows from the kind, the resource's identity
   follows from the resource - so the three things that are hand-maintained in parallel today are one line.
2. **A resource has an identity, a SCOPE and a LIFETIME, and the scope is the trap made explicit.** This
   project's documented per-image-lifetime trap is the difference between a per-frame-slot resource (shadow
   maps, light/camera/skin buffers, scene sets) and a per-swapchain-image one (TAA history, every GI image,
   the probe's ping-pong), and it has already produced a real bug (one first-use flag guarding a per-image
   resource). In this model a resource cannot be referenced without saying which it is.
3. **The declaration is data, and it lives beside the pass.** Constexpr tables, no allocation, no registration
   order, no discovery: the same style as `gbuffer_formats`, `gpu_timing_labels` and the `static_assert`s that
   pin `light_ubo`'s and `instance_record`'s layouts.

## 3. Naming

The repository's conventions, which this follows: module names are flat peers under `vulkan.`, the namespace
mirrors the module's suffix (`namespace vulkan::bindings`), types are `snake_case`, a builder's result carries
`_owned`, and capacities/constants are lower_snake_case.

    module            vulkan.pass_io                       (doxygen: @defgroup vulkan_pass_io)
    namespace         vulkan::pass_io
    files             vulkan/pass_io/pass_io.cppm + pass_io.cpp   (CMake: FILE_SET + PRIVATE, as every module)

| name | kind | what it is | why this name |
| --- | --- | --- | --- |
| `resource_id` | `enum class : uint32_t` | one enumerator per resource FAMILY that `core` already owns (`gbuffer_albedo`, `gi_trace`, `gi_spec_reproject`, `shadow_map`, ...) | the family is the unit `core` creates, destroys and recreates; UNLIKE `gpu_mark_id`, **the order carries no meaning** - only identity |
| `resource_scope` | `enum class : uint8_t` | `per_frame_slot` / `per_swapchain_image` / `device_wide` | the project's own vocabulary for the trap (`docs/runtime_split.md` C6) |
| `resource_lifetime` | `enum class : uint8_t` | `per_frame` / `persistent` / `imported` (`imported` = the swapchain image, owned outside) | says whether a first-use transition or a clear is needed at all |
| `resource_info` | `struct` | `{ resource_id id; std::string_view name; resource_kind kind; resource_scope scope; resource_lifetime lifetime; }` | mirrors `VkDescriptorSetLayoutBinding`'s role for resources: what it is, not where it lives |
| `resource_kind` | `enum class : uint8_t` | `image2d` / `image3d` / `image_cube` / `buffer` / `accel_struct` | decides the descriptor type together with `binding_kind` |
| `binding_kind` | `enum class : uint8_t` | `sampled_image` / `storage_image` / `sampler` / `uniform_buffer` / `storage_buffer` / `input_attachment` | 1:1 with `VkDescriptorType`; named after the Vulkan concept rather than "read/write" because the DESCRIPTOR is what the layout is built from |
| `binding_access` | `enum class : uint8_t` | `read` / `write` / `read_write` | the access is NOT derivable from the descriptor type (the spatial filter only READS its `gi_input` storage image), and it is what a later barrier stage keys on |
| `sampler_hint` | `enum class : uint8_t` | `gbuffer` / `probe_grid` / `taa` / `post` / `nearest` / `shadow` | this renderer creates SIX samplers today (`gbuffer_sampler`, `gi_probe_sampler`, `taa_sampler`, `post_sampler`, `post_nearest_sampler`, `shadow_sampler`) and which one a binding gets is currently a ternary; naming the choices makes it a field |
| `pass_binding` | `struct` | `{ uint32_t set; uint32_t binding; binding_kind kind; resource_id resource; binding_access access; VkShaderStageFlags stages; sampler_hint sampler; }` | **the heart**: one binding, one use. `binding` alone would collide with the `vulkan.bindings` module, hence the `pass_` prefix |
| `push_block` | `struct` | `{ uint32_t offset; uint32_t size; VkShaderStageFlags stages; }` | the second push range already exists in this codebase (`scene_cascade_push_offset/size`), so the shape is not hypothetical |
| `pass_io` | `struct` | `{ std::string_view name; std::span<pass_binding const> bindings; std::optional<push_block> push; }` | the declaration. `name` is for the error messages the validator produces, not for dispatch |
| `make_set_layout` | function | `expected<VkDescriptorSetLayout, std::string> make_set_layout(core&, pass_io const&, uint32_t set)` | a straight replacement for the hand-written binding loops in `pipelines.cppm` |
| `set_pool_requirements` | function | per-`VkDescriptorType` counts for one set | the number that had to equal the layout by hand and did not once |
| `write_set` | function | `void write_set(core const&, pass_io const&, uint32_t set, VkDescriptorSet, resource_views const&)` | replaces the parallel arrays and ternaries in `ensure_*_descriptors` |
| `resource_views` | `struct` | the owner hands in the actual `VkImageView`/`VkBuffer` per `resource_id` for one (image_index, slot) | the one thing that must stay with the resource's owner; it is why this layer needs no `runtime&` |
| `validate` | function | `expected<void, std::string> validate(pass_io const&, spirv_reflection const&)` | checks the declaration against what the SHADER actually declares, using the existing `vulkan.core.pipeline.spirv_parser` |

Names considered and rejected: `vulkan.graph` (implies order derivation, which is a later stage and not what
this module is), `vulkan.descriptors` / `vulkan.sets` (the repository already has `vulkan.bindings`, which owns
set OWNERSHIP and pool lifetime; this module owns DECLARATION), `vulkan.resources` (it owns no resources),
`vulkan.passes` (reads like pass implementations, which is what it is meant to keep out).

## 4. Implementation overview, in the order it gets built

**Step 1 - the declaration type and the layout generator.** `pass_io.cppm` exports the types above plus
`make_set_layout`. The generator is a loop: `VkDescriptorSetLayoutBinding{ binding = b.binding, descriptorType =
to_vk(b.kind), descriptorCount = 1, stageFlags = b.stages }`, then `vkCreateDescriptorSetLayout`. Converted
first: the passes whose layouts are built in `pipelines.cppm` with a hand-written loop - `build_ssgi_temporal`
(7 bindings) and `build_gi_probe` (9). Acceptance: the generated layout is indistinguishable, which the
byte-exact capture gate proves; plus a startup log of the descriptor counts per set, which step 2 consumes.

**Step 2 - the write generator and the pool counts.** `set_pool_requirements` and `write_set`. The writes become
mechanical: for each binding, an image or buffer info whose layout is `GENERAL` for storage and
`SHADER_READ_ONLY_OPTIMAL` otherwise, and a sampler chosen by `sampler_hint`. This is the step that removes the
`b == 6u || b == 8u || b >= 13u` predicate and its sampler ternary - the two places where adding one binding
this session required hand-editing three parallel decisions. Acceptance: same gate; and `image_set_family`'s
pool sizing is fed the DERIVED counts rather than the fingerprint count.

**Step 3 - the shader check.** `validate(pass_io, spirv_reflection)` inside each `build_*`, before pipeline
creation: every declared (set, binding, kind) must exist in the SPIR-V with a compatible type, and every
statically-used binding in the SPIR-V must be declared. This is where the layer starts paying for itself at
runtime rather than at review time: the failure becomes a startup message naming the pass and the binding
instead of a validation-layer line at submit, or a silently wrong image. The validator's own acceptance test is
that a DELIBERATELY wrong declaration fails - that test is part of the step, not an afterthought.

**Step 4 (a later stage, once 1-3 are green) - barrier and order derivation.** `binding_access` plus
`resource_scope` are what make it possible: a storage WRITE needs `GENERAL`, a sampled READ needs
`SHADER_READ_ONLY_OPTIMAL`, and the FIRST use of each resource in a generation needs the `UNDEFINED ->` form.
The three cases that are NOT mechanical must be expressible as explicit overrides, because they are deliberate
and documented: the tracer skips its hand-off barrier when the glossy lobe will write the same image, and the
lobe owes it back; and the resolve's first-use transition to `SHADER_READ` exists because the multi-bounce
feedback samples last frame's resolve before this frame's writes it.

## 5. Where the declarations live

Beside the pass they describe, not in a central table: `pass_io` is a description OF a pass. While a pass still
lives in `runtime.cpp`, its declaration lives in a small unit of its own next to it (for the first conversion,
`vulkan/pass_io/probe_io.cppm` exporting `gi_probe_io`, the probe cache's 9 bindings and its push block); when
that pass is later extracted into `vulkan.gi_probe` (the first extraction in `docs/runtime_split.md`), the
declaration moves with it and the runtime only passes the `resource_views` in. The resource list itself
(`resource_id` + `resource_info`) belongs to `vulkan.core`, because `core` is the only place the complete image
list exists - and that is the same reason its create/destroy loops can later be driven from it.

## 6. Migration order and the cases that shape the design

    1. probe cache          9 bindings, own family, own push block, no shader change      (the first conversion)
    2. TAA                  4 bindings, own family, per-image history
    3. post                 9 bindings, 5 sets per image, ONE push struct shared by 3 pipelines
    4. the two temporal resolves  SAME layout, TWO resource bundles
    5. the G-buffer set     last: 16 bindings, shared by six passes and seven shaders

Case 4 is the one that proves the model needs a notion this document must state: **the layout is shared while
the resource bundle is not**. `ssgi_spec_temporal_family` was created with `ssgi_temporal_set_layout` and its
own images; the declaration therefore describes the LAYOUT (one `pass_io`), while `resource_views` selects the
bundle per dispatch. Case 3 proves the same for push constants: one `push_block`, three pipelines.

Case 5 is deliberately last and belongs to a different project: the G-buffer set is shared across six passes and
its bindings appear in six shaders, so converting it is a shader change - a behaviour-visible change with its
own gate and its own measurement, not a step of this layer.

## 7. Acceptance, and what would make it fail

Every step: the capture gate 12 scenarios x 2 with 0 changed, Release/Debug/ASan+UBSan clean, ctest 6/6,
doxygen exit 0 with an empty warning stream, every run validation clean. The layer's own success metric, which
is testable and whose baseline is measured: **adding a pass becomes declare-I/O plus implement-record** - no
separate layout loop, no separate write loop, no separate pool count, no parallel storage/sampler ternaries.
Today that list has thirteen entries.

Failure modes to refuse in advance:

* **the declaration drifts from the shader** - the reason step 3 exists, and the reason the validator itself
  needs a test that it FAILS on a wrong declaration;
* **the layer becomes a mini-RHI** - it must never allocate an image, own a queue, or hide a Vulkan type it is
  not generating;
* **`resource_id` becomes a god-enum** - it stays owned by `core` and grows only when a family is added there;
  its order is never a contract (that is the difference between it and `gpu_mark_id`);
* **push-constant size**: declared plus `static_assert`-pinned (as `light_ubo` and `instance_record` already
  are), never computed from the declaration against a shader block whose layout this layer cannot see;
* **over-reach into ordering**: steps 1-3 must not derive a single barrier. The byte-exact gate is what proves
  a step changed nothing, and a step that reorders synchronization can pass it while changing behaviour on
  another driver - so barriers are a separate stage with their own justification, not a side effect.

## 8. What a pass needs from Vulkan, and the three decisions that settled the interface

AT CREATE TIME, once per device generation: the device; the shaders' SPIR-V; for each declared set, a
`VkDescriptorSetLayout` (the pass's own generated from its declaration, the shared ones borrowed from their
owners); a `VkPipelineLayout` from those plus the declared push range; the pipelines themselves; for a graphics
pass the colour/depth formats and blend state; a descriptor family (the pool lives in `vulkan.bindings`, not in
a pass); and one of the renderer's six samplers per sampled binding.

EVERY FRAME: the command buffer, the pass's own `VkDescriptorSet`, the shared sets it declared usage of, one
`VkImageView`/`VkBuffer` per own binding, the resolved pipelines, and the extent this pass works at.

AND DELIBERATELY NOT: instance, physical device, surface, swapchain, queue, fence, semaphore, command pool,
`VkRenderPass`/`VkFramebuffer` (this renderer uses dynamic rendering), and `VmaAllocator` - a pass allocates
nothing, because `vulkan.core` is the single allocator of images. A pass that wanted a `VmaAllocator` would be
taking over an image family, which is a resource-layer change and must be argued separately.

THREE DECISIONS, taken while the interface was being written and recorded because each one closes a gap that
was open in the first draft of this document:

* **shared sets are simply given.** `resolved_io` carries them as they are (one slot per owner: scene, G-buffer,
  post) and a pass that declared usage of one may bind it. No ownership mechanism, no per-set abstraction: the
  runtime already binds the scene set before a draw, and the G-buffer set is shared by six passes by
  construction.
* **the command buffer is handed out per frame, at recording time, and never stored.** That is what lets a pass
  hold no device state between frames - and it closes the first draft's hard gap, in which a pass had nothing to
  record into at all.
* **pipelines are referenced by NAME for now**, and this costs nothing new: `vulkan.runtime` already keys its
  pipelines by name (`make_pipeline` / `set_default_pipeline` / `get_pipeline`), so a pass names what it records
  with and the host resolves those names into `resolved_io::pipelines`, in order. The 787 lines of
  `vulkan.pipelines` stay where they are, with the pipeline layouts still built there.

STILL OPEN, stated rather than implied: push constants are the one input that is neither a resource nor a
behaviour, and they are composed by the runtime today (camera matrices, radii, the frame counter, the instance
table's device address) while the shader that consumes them belongs to the pass - so either the pass pushes them
itself (needing the pipeline layout in `resolved_io`) or the host pushes on its behalf (needing to know each
pass's push layout). And parallel recording (`main_segments` + the task pool) is not expressible in `stage` yet.
