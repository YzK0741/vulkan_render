# Conventions

The rules this repository follows when a name has to be chosen, and the measured reasons behind them.
Everything here was decided against the tree rather than in the abstract; where a rule has exceptions, the
exception list is the measurement.

## Module names

**A module name says what the module IS, in full words.** No abbreviations.

The tree has 56 modules. These keep a short spelling, and each one is a NAME rather than an abbreviation:

| keeps | why |
| --- | --- |
| `vstd` | the project's own STL module name - documented in `vstd/README.md`, imported by 34 files |
| `gltf_loader`, `vulkan.core.pipeline:spirv_parser` | glTF and SPIR-V are **format** names |
| `vulkan.core:vma`, `utility:better_pmr` | VMA and PMR are the library's and the standard's own terms (Vulkan Memory Allocator, polymorphic memory resource) |
| `vulkan.core:init_utils`, `vulkan.init_utils`, `vulkan.constant_init` | `init` was kept by request |
| `vulkan.pass:taa`, `vulkan.pass:fxaa` | the `-aa` pair was kept by request |
| `vulkan.pass.megalights_*` | a feature of this renderer, not a shortening of one |

The rule renamed `app_config` -> `application_configuration`, `vulkan.gui` ->
`vulkan.graphical_user_interface`, `vulkan.pass.rt_shadow` -> `vulkan.pass.ray_traced_shadow` and
`vulkan.pass.gbuffer_debug` -> `vulkan.pass.geometry_buffer_debug`. A module name is not the only thing
wearing one of those words: the `app_config` STRUCT, the `gui` CLASS and the `gbuffer_debug` / `rt_shadows`
CONFIG KEYS kept their spelling, because none of them is a module name.

### The rule that decides whether a submodule becomes a partition

**A partition cannot be imported from outside its module.** That is a hard constraint, not a preference, so
a dotted submodule becomes a `:partition` exactly when **nothing outside its own module imports it** -
measured per module rather than judged:

* it moved: `vulkan.core:init_utils`, `:vma`, `:vma_handles`, `:descriptor_heap`,
  `vulkan.core.pipeline:spirv_parser`, and `utility:better_pmr`, `:bvh`, `:data_block`, `:frame_clock`,
  `:frame_stats`, `:thread_pool`;
* it stayed a module: `vulkan.core.handles` (12 importers outside the family),
  `vulkan.render_resource.shared` (10), `vulkan.core.pipeline` (6), `vulkan.core.filters` (1). Converting
  those would force every consumer to import the whole parent, which is the opposite of what the split is
  for.

## Verb prefixes

Eight verbs lead the names on a module's surface. A public name that starts with none of them is worth a
second look.

| prefix | means |
| --- | --- |
| `make_` | build a value and return it; pure, no GPU work |
| `create_` | create a GPU object (buffer, image, pipeline, pool) at runtime |
| `build_` | construct a pipeline or a structure from a description |
| `write_` | write into something that already exists (a heap descriptor, a buffer) |
| `set_` | change one tunable; no work beyond recording the choice |
| `update_` | refresh per-frame state |
| `ensure_` | idempotent lazy creation: make it if it is not there yet |
| `record_` | record commands into a command buffer |

They are not interchangeable, and the difference is what makes a call site readable at a glance:
`set_shadow_cascades` is a knob, `ensure_shadow_resources` allocates, `record_shadow_content` draws.
Counted when the rule was written: `make_` 150, `create_` 93, `build_` 68, `set_` 174, `write_` 32,
`update_` 21, `ensure_` 24, `record_` 62 across the `.cppm` surfaces.

## Partitions

**One file per partition.** CMake names a partition's dyndep output after the MODULE, so a named partition
that also has a separate `.cpp` implementation unit cannot be expressed: with the `.cpp` in the module file
set ninja reports `multiple rules generate ...<module>-<partition>.pcm`, and with it outside CMake reports
that the file `provides the ... module but it is not found in a FILE_SET of type CXX_MODULES`. Both were
measured. A module's own unnamed implementation unit (`utility.cpp`) is fine either way.

A partition **does not see the primary interface**, and imports are **not transitive**: each partition
imports `:declarations` and whatever else it calls, even when the interface already imports it.

Under this project's `-Werror`, the primary **may not** import its own implementation partitions - clang 22
rejects it with `-Wimport-implementation-partition-unit-in-interface-unit`. Listing those partitions in the
CMake module file set is what gets them compiled, and the linker finds their definitions.

## Where a helper belongs

Beside the thing it is defined in terms of, and published once. The heap-write helper `core::heap_slot_offset`
was copied into three translation units because it multiplies `core::heap_slot_stride`; removing the copies
needed the constant and the helper to live together, and putting the helper in the heap-plumbing partition
instead would have closed a cycle (`core.declarations.cppm` already re-exports `:descriptor_heap`).
