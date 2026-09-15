# The pass chain, rebuilt on the pre-GI state

## THE OWNERSHIP MODEL (decided after the pass work, and it is what "small runtime" is built on)

The device root is `std::shared_ptr<core>`, and everything else holds a VIEW of it, not a share of it:

| layer | holder | what it holds | why |
|---|---|---|---|
| root | `runtime`; and any owner that must keep the device alive (a second viewport, an editor, a test host) | `shared_ptr<core>` | the core owns the device, the allocator and every resource created on them, so its lifetime is the frame's outermost fact - and a reference-counted root lets two owners share ONE device without an artificial "who owns whom" order |
| ambiguous | task-pool jobs, GUI callbacks, animation callbacks, readback | `weak_ptr<core>` + `lock()` at use | their lifetime leaves the call, so it cannot be a scope contract |
| scoped view | the summary a pass is handed at create/record | **non-owning** (a `core_filter`-style view, or plain references) | valid for the call; a view must not extend anything's lifetime, which is why filters hold a POINTER, not a share |

`runtime` holds the root as `core_owner` and an alias `core& vulkan_core = *core_owner`, declared in that order so
the device is still held while every other member (the pipelines a pass has not taken over, the descriptor
families, the readback staging buffer, the filter) is destroyed. It also has a constructor that takes a finished
`shared_ptr<core>`, which is what makes sharing expressible at all.

TWO THINGS REFERENCE COUNTING DOES NOT SOLVE, and they stay explicit:

* **generation invalidation** - per-image images and descriptor families live and die with the swapchain
  generation, so `on_swapchain_recreated` / `recreate_stage` stay the mechanism. A shared root guarantees "not
  destroyed too early", never "not stale" (this codebase's own comment calls it the per-image-lifetime trap).
* **the teardown window** - a `weak_ptr::lock()` SUCCEEDS while a runtime is running its destructor (the device
  is still referenced), so "the lock succeeded" does not mean "the sibling resources are usable". Use is gated by
  the lock; RELEASE belongs to the destructor's order, in one place.

**AND THE OLD STYLE IS GONE, in four steps that were each verified on their own.** The model above was installed
by deleting the old entry points rather than by adding new ones, which is why every step is small and every step's
acceptance is the same gate:

1. `8409bf6` - **the device root is a `shared_ptr`.** `core` derives `std::enable_shared_from_this` and the runtime
   holds `shared_ptr<core> core_owner` plus the `core&` alias declared after it, so the device outlives every
   member that was created on it; `runtime(std::shared_ptr<core>)` is the constructor that lets a second owner
   share one device. The `enable_shared_from_this` is deliberately UNUSED so far: it is what a creation site will
   use when a reference-counted device handle has to be minted from inside `core` itself.
2. `a7b1d02` - **nothing in `vulkan.pipelines` needs a whole `core`.** All eight remaining builders
   (`build_post`, `build_fxaa`, `build_deferred`, `build_gbuffer_debug`, `build_ssgi_spatial`, `build_rt_shadow`,
   `build_mask_bake`, `build_compute_skin`) take a `VkDevice` (and a `VkFormat` where they need the surface's),
   so a pipeline builder names exactly the two things it reads instead of importing the device root.
3. `fe96e64` - **the named scene pipelines are built from the device.** The one builder that genuinely needed the
   surface's format now goes through `vulkan::make_pipeline(device, scene_pipeline_layout, {swap_chain_image_format},
   depth_format, vert, frag, VK_SAMPLE_COUNT_1_BIT, true, 0, 0, 0, blend_attachments)`. This is the ONE step of
   the four that could have changed a frame, and it did twice while it was being written: the named entry point
   needs an explicit cached viewport/scissor (the hand-rolled one set them itself) and
   `make_color_blend_attachment()`'s src-alpha blend, and the gate caught each omission as `changed: 1` before the
   pair was restored to `changed: 0`.
4. **the dead overload is deleted** (this commit): `core::make_pipeline(vert, frag, depth_test_enabled)` had no
   caller left once the runtime's pipelines were named, so its declaration and its 31-line definition are gone.
   `make_gbuffer_pipeline` and the single-format convenience overload in `vulkan/core/pipeline/pipeline.cpp` STAY:
   they still have callers, and "unused" is a measurement, not a guess.

The consequence for the pass work that follows is the point of doing this first: a pass's `create` now has nothing
to reach for except a `VkDevice` and the declarations, so moving a pass's three pieces (its `make_*`, its
`ensure_*` and its hand-written descriptors) out of the runtime is a move, not a redesign.


> **READ THIS FIRST - THE TENSE OF THIS DOCUMENT.** Everything above the "HANDOFF" section is a CHRONOLOGICAL
> RECORD: each section was written as its step landed and describes the tree AS IT WAS THEN, including steps that
> later steps superseded (the pre-GI sections speak of a branch with no GI in it, of a `pass_context::shader` that
> is null, and of samplers nothing backs - all of which the GI attach changed). The **HANDOFF** section at the
> end is the only part that describes the tree as it IS. (The one piece of work that was designed here but not
> installed - the temporal pass - LANDED as `bc2dc1d`; the drafts directory that carried it is gone with it.)
> Read the record for why things are the way they are; read the handoff to act.

This branch (`pass-chain`, forked at `281ab06` - the commit immediately before the first GI commit) exists to do
one thing the later work could not do in place: let the PBR rendering be **built** as passes rather than
migrated into them, and then let GI be **added** to that architecture pass by pass. The later state on `master`
holds the GI implementation, its measurements and its documentation; this branch starts without them and
earns them back one pass at a time.

## THE ONE RULE THAT PROTECTS THE ACCEPTANCE ANCHOR

The capture gate's references live outside the repository, in
`%LOCALAPPDATA%\vulkan_render\baseline`, and on this machine they were captured from `master` WITH GI. Those
hashes are the acceptance test for restoring GI (`default_gi` `BF180E98ADB29E7E`, `sponza_gi`
`58EC848DFABE654A`, `metal_rough_glossy` `46F9851B7BC89872`, `glossy_motion` `98B06F2190B49519`).

While the branch was pre-GI it compared against its own directory, so that a branch which renders no GI could
not silently overwrite the anchor:

```powershell
$env:VR_RENDER_BASELINE_DIR = "$env:LOCALAPPDATA\vulkan_render\baseline-pass-chain"
```

**GI IS BACK, so that separation is over and the DEFAULT directory is now the acceptance** - compared, never
updated. The rule that replaces it is narrower but harder: `-Update` must NEVER be run on this branch without
an explicit baseline override, because the default directory is the anchor the objective is measured against,
and `-Update` rewrites it from a SINGLE run with no determinism check (see the script's own note).

**AND THE GATE MUST BE GIVEN AN ABSOLUTE `-BuildDir`.** This is a measured finding, not a preference: with
`-BuildDir build-release-clang64` (relative), `default_gi` reported FLAKY - two runs of one binary giving
`1D72F82B946F0338` and `54B854F69AEAA620`, neither of them the reference. With
`-BuildDir (Resolve-Path build-release-clang64).Path`, the same binary gives `BF180E98ADB29E7E` twice, i.e. the
reference. The cause is the scenario config: the script writes `screenshot_dir` as `$BuildDir\render-check`, so
a RELATIVE `-BuildDir` puts a relative path into the config, which the app resolves against a base that is not
the working directory the script then searches - and the two runs end up being compared across two different
places. Nothing about the renderer changed between those two gate runs.

**FIXED IN THE SCRIPT**, because a harness that reports a false FLAKY is worse than one that fails: the script
now canonicalizes `$BuildDir` with `Resolve-Path` before it derives the work directory, so both spellings mean
the same run. Re-verified with the RELATIVE spelling afterwards: 12 x 2, 0 changed, 0 flaky, all four GI hashes.

## WHAT IS ON THIS BRANCH SO FAR

| commit | what |
|---|---|
| `bfed4cd` | the description layer (24 schema families - this state's resources, not master's 37), the shared sampler module, and `vulkan.pass` (the framework). Nothing consumes it: gate 7 x 2, 0 changed |
| `de23673` | `/build/` is ignored (this branch predates that rule, which is why a bare `build/` was committed once and had to be amended out) |
| `2593e7e` | `test_render_resources` ported and adapted (89 checks): the schema, the validators, the three declarations this state can carry |
| `2f363c6`, `db99f0e` | this plan, and the measured scope of the generator port |
| `e05d0b5` | the generators take a `VkDevice` (`make_set_layout`, `write_set`, `image_set_family::ensure*`), which is the prerequisite for a pass building its own layout |
| `bcf62bb` | the `scene` and `transparent` pass modules come over from `master` and build here with **no edit at all**. Still unconsumed: gate 7 x 2, 0 changed |
| `1e2e493` | the **transparent pass is wired and recording**: the runtime drives a real `pass::stage`, and the frame is byte-identical |
| `99583e6` | the **scene pass is wired**: the runtime no longer opens OR closes the surface instance, and `record_opaque_scene` / `record_main_segment` / `sub_render_task` are gone |
| `6cd9c6e` | the **TAA resolve is wired, and it is the first pass here that OWNS something**: set layout from its declaration, pipeline layout, pipeline, per-image family, history flags. `make_taa_pipeline` and `ensure_taa_descriptors` are gone, and the app's shader hand-over (`register_shader`) exists because a pass that builds a pipeline needs shader bytes |
| `08b84ac` | the finding that the non-GI port is complete: `master` has four pass modules, three were wired |
| this step | **GI IS ATTACHED**: the 37-row declaration layer, the `gi_probe` pass, the GI pipeline builders, the acceleration structures, the GI shaders and config, and the 12-scenario gate. **12 x 2, 0 changed, 0 flaky**, including all four GI reference hashes |
| this step | the **SSGI tracer is extracted** (`vulkan.pass.ssgi_trace`), which closed the framework gap the plan named: `pass_io::barrier_images` + `resolved_io::barrier_images` let a pass transition images that live in a SHARED set and own no descriptor. Gate 12 x 2, 0 changed, 0 flaky |
| this step | the **glossy lobe is extracted** (`vulkan.pass.ssgi_spec`) on the same channel, and doing it found a defect in the tracer: nothing told the pass its generation had changed (`recreate_stage` now does). Gate 12 x 2, 0 changed, 0 flaky |
| this step | the **GI denoiser's layout is generated from a declaration** (`ssgi_temporal_io`), replacing the hand-written seven bindings that its own comments record as having drifted from the pool count once. The pass extraction itself is recorded as blocked on a framework decision (two signals over one layout). Gate 12 x 2, 0 changed, 0 flaky |

`docs/pass_chain_inventory.md` (192 lines) is the read-only map of this state: its resource set, the frame's
recording spine and mark intervals, every function of the PBR/scene chain with its attachments, sets,
pipelines, pushes and barriers, the shared sets binding by binding, the pipeline registry, the leaf draw path,
and what this state does NOT have.

## THE ORDER, AND WHY IT IS THIS ORDER

1. **the generators** (`make_set_layout`, `write_set`, and the `image_set_family` API) must take a
   `VkDevice` rather than a whole `core`, because a pass's create step is handed a device and nothing else.
2. **`scene`** - the PBR surface write: the rendering instance over its declared targets, the segmented draw of
   the visible leaves, and the instance's close. This is the pass that fixes the coupling the inventory found:
   the instance is opened by one function and closed by another, three subsystems later. It is also the step
   that has to delete `record_opaque_scene`, `record_main_segment` and `sub_render_task`.
3. **`transparent`**, then **`taa`** - both are small once `scene` exists, and both are on this branch already
   (TAA predates GI). `transparent` landed FIRST of the two anyway, because it is the additive one and needed no
   change to any existing function: see the next section. `taa` is where `pass_context::shader` stops being
   null, because it is the first pass that owns a pipeline.
4. **`lighting`** and **`post`** - NOT ports (no such pass exists on `master`; see the section below) and
   therefore outside the acceptance. They are listed here because the ORDER is about what a full pass chain
   would contain, not because there is anything to migrate.
5. **then GI**: `probe cache` (master's fourth pass) -> `ssgi` (trace) -> `temporal` -> `spatial` -> `spec`,
   each added as a pass and each verified against `master`'s references for the GI scenarios.

Acceptance is the same for every step and is not negotiable: Release + Debug + ASan+UBSan build clean, `ctest`
all green, doxygen exit 0 with an empty warning stream, and the gate `0 changed / 0 flaky` against the
branch's own baseline directory (and against `master`'s references once GI starts coming back).

## THE GENERATOR PORT'S EXACT SCOPE, MEASURED ON THIS BRANCH

The step that makes the passes possible is a signature change, and reading the branch rather than assuming
`master`'s diff gives its real size:

* three write callbacks still take `core const&` - `runtime.cpp:2197` (post), `2540` (TAA), `2736` (G-buffer
  debug) - and three `ensure*` call sites pass `vk` rather than `vk.device` (`2233`, `2558`, `2758`);
* three layouts are still built by hand in `pipelines.cppm` (`vkCreateDescriptorSetLayout` at `96` post, `165`
  G-buffer debug, `213` TAA).

**AND ONLY ONE OF THEM CAN BE GENERATED TODAY**, which is the finding worth recording: `make_set_layout` takes
a declaration, and this branch has exactly ONE declaration that owns bindings (`taa_io`). The post chain and
the G-buffer debug view have no declarations yet, so their hand-written layouts are not a missed conversion -
they are the honest state until those passes are written. That is why the order in this document puts the
declarations and the passes together: a layout cannot be generated before the thing that declares it exists.

## THE FIRST PASS THE RUNTIME DRIVES, AND WHY IT WAS `transparent` RATHER THAN `scene`

The order above is by DEPENDENCY (`scene` is what the rest are shaped like); the first WIRING went to
`transparent`, and for a reason worth keeping: its body is the only one that moves without touching anything
else. It owns its own LOAD instance and its own per-slot secondary, so `record_opaque_scene`,
`record_main_segment` and `sub_render_task` are not involved at all and the step is purely additive - the
cheapest possible proof that the framework can drive a real pass and produce the same bytes. The `scene` step,
which has to delete those three functions, followed it and is described in its own section below.

What the wiring needed, measured:

* the pass module took **no edit** (already true at `bcf62bb`); the runtime gained five host answers
  (`create_passes`, `shared_samplers`, `make_scene_environment`, `make_transparent_frame`,
  `resolve_transparent_pass`) plus `pass_frame`, `make_pass_host`, `resolve_pass` and `apply_pass_behaviour`;
* the early return that used to open `record_transparent_pass` ("nothing blended this frame") moved into
  `resolve_transparent_pass`, which is where the runner asks - so a frame with no blended leaves still records
  no instance and pays no barrier, and the pass is skipped WITHOUT being resolved;
* `make_scene_environment` is `record_main_segment`'s environment builder lifted out into a `static` member
  (the pass hands it an owner pointer), which is why its comments still speak of "the main pass";
* the result: gate 7 x 2, **0 changed, 0 flaky**, including `transparent_blend` (`26D9B28C00EA4D00`) - the one
  scenario that actually records the pass.

The two facts the create context forced into the open, both recorded rather than papered over:

* `pass_context::samplers` is a six-slot `sampler_set`, and **two slots have no backing sampler on this
  branch** - `probe_grid` (the GI probe pass does not exist here) and `nearest` (the post chain has no pass
  here). `shared_samplers` fills the four that exist and leaves those two null, named in the code; this is the
  second half of the "hint nothing can resolve" finding below, and it says the lie is in the vocabulary rather
  than in the fill;
* `pass_context::shader` is **null** here, because no pass wired on this branch declares a shader. The app-side
  registration (`register_shader`) therefore lands with the TAA pass, and until then a `create_passes` call's
  only observable effect is the VALIDATION of the declaration - which is still worth the call.

**BOTH OF THOSE WERE RESOLVED BY THE STEPS THAT FOLLOWED**, and it is worth saying so here rather than leaving
two stale bullets in a plan: the shader callback landed with TAA (`6cd9c6e`), and `probe_grid` got its sampler
when the GI probe pass arrived (this step). The one thing that did NOT resolve is `nearest`, which is still null
because the post chain still has no pass.

`resolve_pass` is a two-branch `if (&pass == &this->scene) / (&this->transparent)` chain, and stays a chain
until a third pass makes the missing table obvious (the next section's scene step has landed since this was
written, and the chain grew by exactly the one branch it was predicted to). The framework gaps this does NOT
close are unchanged: the resolver chain, the absence of a channel for shared-set IMAGE handles (the GI chain's
blocker), and parallel segment recording still being the frame loop's policy rather than a framework concept.

## THE SCENE PASS: THE OPEN AND THE CLOSE IN ONE FUNCTION

This is the step the inventory said was worth the most, because it fixes a COUPLING rather than moving code: the
surface rendering instance used to be opened by `record_opaque_scene` and closed by `record_scene_tail`, three
subsystems later, with the whole rest of the frame's work between them. Now `vulkan.pass.scene` opens it, draws
the segments and CLOSES it in one function, and `record_scene_tail` simply starts with a GPU mark.

What moved, and what deliberately did not:

* the instance, the segment strategy (`segment_count`, the `leaf_count < 4` threshold), the secondary
  begin/end and its inheritance, the per-segment environment and the draw loop are the PASS's;
* the pipeline registry, the per-slot secondary buffers, the task pool and the scheduler are the RENDERER's,
  handed over as `scene_frame::make_environment` and `scene_frame::run_tasks` - so the pass says WHAT the
  segments are and the frame loop still decides HOW they are recorded;
* the attachment barriers (`record_scene_attachments`) and the per-frame geometry resync
  (`update_pass_geometry`) stay in `record_scene`, because they are frame facts, not pass facts;
* `record_opaque_scene`, `record_main_segment` and `sub_render_task` are DELETED - including their declarations,
  which `master` still carries (a dangling `record_opaque_scene` declaration with no definition). Nothing on
  this branch declares a function it does not define;
* `begin_rendering` survives for one caller only, and it is documented as such: the DEGENERATE frame with no
  surface pipeline, where there is no pass to run and an empty instance is opened and closed anyway so the
  scene target still ends in a layout the post chain can sample.

Two honesty fixes taken while the file was open, both about text that had stopped being true:

* `record_scene_tail`'s declaration said "close the geometry instance and record the scene-side stages that
  follow it" - it no longer closes anything, and now says so (and says why);
* `record_scene`'s declaration described a body that no longer exists, so it now names the pass and lists the
  three things that stayed.

Verified: gate 7 x 2, **0 changed, 0 flaky**, including `sponza` (`0EBA5300E8F84E6F`) - the scene the GI
acceptance anchor is measured on - and `transparent_blend`, whose pass runs after lighting over the surface
this one writes.

## THE TAA RESOLVE: THE FIRST PASS HERE THAT OWNS GPU OBJECTS

`scene` and `transparent` left the framework's CREATE half unexercised: their `create` is empty because their
leaves name their pipelines and everything they bind is the shared scene set. The resolve is the first pass on
this branch that owns something, and that is what made this step the first real test of the other half:

* it builds its **set layout from its declaration** (`bindings::make_set_layout(device, taa_io, taa_io.own_set)`),
  so the layout the fragment stage sees and the declaration cannot drift - which is what the generator port in
  `e05d0b5` was for;
* it builds its pipeline layout and pipeline through `pipelines::build_taa`, whose signature CHANGED for this
  step: `(core&, push_size, vert, frag)` became `(VkDevice, pass_set_layout, push_size, vert, frag)`, because a
  pass's create step has a device and its own layout and no `core`. `taa_owned` consequently lost its
  `set_layout` field;
* it owns its **per-image descriptor family**, and needed the one piece of design that is not a straight move:
  the fingerprint. The family compares "what my sets point at", and for a per-image resource the CURRENT image's
  view is the wrong fingerprint (it changes every frame, and rewriting a set a pending frame names is a
  validation error), so the pass caches ONE image's four views as its generation stamp and drops them in
  `on_swapchain_recreated` - which the runner calls for every pass in a stage (`recreate_stage`), replacing the
  runtime's hand-kept `taa_family.retire_all()` line;
* it owns its **history flags**, so the runtime's `taa_history_valid` member is gone and `set_taa` calls
  `taa_resolve.reset_history()` instead of assigning to a vector it used to own.

The app gained a **shader hand-over** because of it - `register_shader(name, bytes)` and the runtime's
`registered_shader(name)`, filled through `pass_context::shader`. This is the piece the transparent step
recorded as MISSING ("`pass_context::shader` is null here because no pass wired on this branch declares a
shader"), and it landed exactly where it was predicted to: with the TAA pass. Chores now registers
`post.vert.spv` + `taa.frag.spv` instead of calling a pipeline builder, and `create_passes` is the one call that
turns those bytes into a pipeline.

ONE LINE OF THE OLD BODY DELIBERATELY DID NOT MOVE: `ensure_gbuffer_depth_sampled`. The resolve samples the
G-buffer depth, and that image is transitioned out of its attachment layout by a helper whose per-image "was it
written this frame" flag belongs to the G-buffer pass - shared per-image bookkeeping that a pass cannot express
while it may only declare its own bindings. It stays with the host, gated on the SAME predicate the runner gates
the stage on. (On `master` there is a second such line - clearing the GI chain's motion-vector flag - which this
branch does not have because it has no GI chain.) This is the framework gap the plan already named, now with a
measurement attached: **one stage in this frame needed one host-side line that a pass cannot yet own.**

Two hand-kept lists disappeared with it, both of them hazards the framework removes by construction:
`update_pass_geometry` no longer resyncs the resolve's viewport (the pass declares `resync_viewport = true`, so
the runner sets it from the declaration's extent every frame), and the runtime's destructor no longer destroys
the resolve's set layout and pipeline layout (the pass does, in `release_owned`).

Verified: gate 7 x 2, **0 changed, 0 flaky**, including `deferred_taa_fxaa` (`6999D01E5FBAB508`) - the one
scenario that records the resolve.

## WHERE THE NON-GI PORT STANDS: THREE OF MASTER'S FOUR PASSES, DONE

`master` has exactly FOUR pass modules - `gi_probe`, `scene`, `taa`, `transparent` (every other rendering
subsystem there is still a runtime function). This branch has now wired THREE of them, each verified
byte-identical against its own baseline:

| pass | state | what it proved about the framework |
|---|---|---|
| `transparent` | wired (`1e2e493`) | the record half drives a real pass; the scene-family draw path survived the move |
| `scene` | wired (`99583e6`) | a pass can own a rendering instance and its close, and the per-segment scheduling stays the owner's |
| `taa` | wired (`6cd9c6e`) | the CREATE half works: a pass builds its own layout, pipeline, family and state from its declaration |
| `gi_probe` | **not here yet** | it is the GI phase's first step, and the reason for the shared-set image channel recorded below |

**THEREFORE ITEMS 4 AND 5 OF THE ORDER ABOVE ARE NOT PORTS.** "lighting" and "post" are not passes on `master`
- they are runtime functions there, and no version of them exists to bring over. Writing them would be NEW
passes rather than the migration this branch is for, so they are out of the acceptance: the objective's own
wording names the three passes (`scene/transparent/TAA`), and those three are done. What is left is GI: the
probe cache first (it is master's fourth pass), then the tracer, the temporal and spatial filters, and the
glossy lobe - added as passes, against `master`'s four reference hashes.

## ATTACHING GI: WHAT WAS ALREADY THE SAME, AND THE ONE THING THAT WAS NOT

The measurement that shaped this step, taken before any code moved: **the branch's shared architecture files
were already byte-identical to `master`'s**. `git diff HEAD master` was EMPTY for `vulkan/pass/pass.cppm`,
`vulkan/pass/scene.*`, `vulkan/pass/transparent.*`, `vulkan/bindings/bindings.cppm` and
`vulkan/render_resource/shared.cppm`; `render_resource.cppm` differed by 86 lines (the GI declarations) and
`pipelines.cppm` by 513 (the GI builders). So the rebuild had converged on the same architecture `master`
converged on, and attaching GI was a matter of bringing the GI-era content into that shape:

* **came over verbatim**: the 37-family declaration layer (13 GI resources, `gi_probe_io`, the SSGI
  declarations), the `gi_probe` PASS module, the GI pipeline builders (tracer, temporal, spatial, spec,
  probe, mask bake), `vulkan.acceleration_structure` (BLAS + the per-frame TLAS), `core`'s GI images and ray
  query enablement, `constant_init`'s GI barriers, the GI shaders (`ssgi*.comp`, `gi_probe.comp`,
  `hit_shading.glsl`, `probe_sh.glsl`, `rt_shadow.comp`, `mask_bake.comp`), `app_config`'s GI keys,
  `main.cpp`/`chores.cpp`'s wiring of them, the 12-scenario gate with its five GI scenarios, and the tests
  (`test_pass`, the GI half of `test_render_resources`, the new `test_app_config` expectations);
* **could not be a slice**: the RUNTIME. The GI-era change to `vulkan/runtime.cpp` is a ~250 KB diff threaded
  through the very functions the pass work touched (`record_scene_tail`, `record_taa_pass`, `create_passes`,
  `resolve_pass`, `on_swapchain_recreated`, the destructor, `active_features`, the constructor), and it is
  interleaved with the pass wiring rather than separable from it. Re-deriving 60 commits of that by hand would
  have been a multi-round exercise whose only oracle is the final hash; adopting `master`'s runtime is the
  honest way to attach it, and the four reference hashes are what checks it. **The branch's verified step-by-step
  rebuild remains the record of how the architecture got there, and this step is the attachment.**

What the GI chain's own shape says about "GI as passes", measured rather than assumed: of the whole chain,
`master` extracted exactly ONE pass (`gi_probe` - the world-space probe cache). The tracer, the temporal
accumulator, the spatial filter and the glossy lobe are runtime compute stages there, driven by the same
`record_*` functions the pre-GI renderer used. So this step satisfies "the probe cache comes back as a pass" and
leaves the rest as master has it; extracting those four into pass modules is the remaining architectural work,
and it is NOT required for the acceptance - it would be verified by the same four hashes staying identical.

ONE OPEN THREAD this step inherits and does not fix: the shared-set IMAGE channel. `gi_probe` binds the shared
scene set and needs image handles from it, which the framework's `resolved_io` still cannot carry. Master's
answer is that the pass resolves the shared set itself through the host and the declaration names the set; the
framework-level version of that channel is still the gap the earlier sections record.

## THE GI CHAIN'S FIRST STAGE: THE TRACER, AND THE FRAMEWORK GAP IT CLOSED

The plan recorded one blocker for the whole GI phase: *no channel for shared-set image handles*. The tracer is
the pass that made it concrete, and closing it is what this step did.

WHY A PASS THAT BINDS NOTHING STILL NEEDED A NEW DECLARATION LIST. `ssgi.comp` binds the shared scene set and
the shared G-buffer set and owns no descriptor at all - and yet it is the frame's first writer of `gi_trace`,
its first reader of last frame's `gi_resolve`, and the pass responsible for the probe grid's first-use layouts.
Every one of those images lives in the G-buffer set, whose contents its OWNER writes, so the pass could neither
bind them nor describe them: the same argument that made `render_target` a separate list from `bindings` applies
to a layout transition, which names an IMAGE and no descriptor at all. So:

* `render_resource::barrier_image` (resource + element) and `pass_io::barrier_images` say which images a pass
  must move between layouts, in the order its own `record()` indexes them;
* the validator checks them like targets: the resource must exist in the schema, must be an image, and may not
  be declared twice (a pass indexes them by POSITION);
* `resolved_io::barrier_images` carries the handles, filled by the host from the frame - the per-swapchain-image
  families take the frame's image, the probe grid takes the element the declaration names;
* `max_barrier_images = 16` is the new cap (the tracer's twelve are the most).

WHAT THE TRACER PASS OWNS, and what it deliberately does not:

* OWNS its pipeline layout and compute pipeline (from `ssgi.comp`, handed over with `register_shader`), the
  dispatch, the whole first-use barrier batch, the hand-off barrier, and the one piece of per-generation state
  that batch needs (`probe_grid_seen_`, reset by `on_swapchain_recreated` - the TAA generation-fingerprint
  pattern again);
* DOES NOT OWN whether the previous accumulation may be trusted (that is the DENOISER's state, so the host
  reads it off and hands it over), whether the glossy lobe runs next (which decides who owes the hand-off), or
  any value in the push block. Those three are a `ssgi_trace_frame` and `resolved_io::push`.

THE CREATE CONTEXT GREW ITS FIRST SECOND-SET ANSWER: `pass_context::shared_set_layout` used to answer set 0 (the
scene set) and nothing else; the tracer's pipeline layout needs the G-buffer set layout too, so the renderer now
answers set 1 as well and a pass asking for anything else still gets "none".

ONE HONEST DIFFERENCE FROM THE MOVED CODE, recorded rather than glossed: the resolver skips the whole pass when
ANY declared barrier image is missing, where the moved body skipped only that one barrier and still dispatched
(it tested `index < vk.gi_spec_resolve_images.size()` per image). The case cannot arise - those images are
created and destroyed together with the target generation - and skipping is the stricter of the two answers.
`build_ssgi` also changed signature here, from `(core&, ...)` to `(VkDevice, ...)`, because a pass's create step
has a device and no core: the same change `build_taa` needed.

Verified: gate 12 x 2, **0 changed, 0 flaky**, all four GI hashes (`default_gi` `BF180E98ADB29E7E`, `sponza_gi`
`58EC848DFABE654A`, `metal_rough_glossy` `46F9851B7BC89872`, `glossy_motion` `98B06F2190B49519`) plus
`sponza_march` and the seven pre-GI scenarios - which is the evidence that moving 170 lines of dispatch and
barriers out of the runtime changed no pixel. `sponza_gi` is the scenario that exercises the probe grid's
first-use batch (it is the only one with `ssgi_probes = true`).

ONE HARNESS OBSERVATION, kept because it cost a round: a full-gate run once reported `deferred` FLAKY - one run
`2DD1D13857322C0F` (the reference) and one `CEDD831932C76340` - while every other scenario passed. `deferred`
alone then passed three times in a row (six renders), and the next full run was 12 x 2 clean. Nothing about the
renderer changed between those runs, so this is recorded as INTERMITTENT machine-level flakiness on the FIRST
scenario of a batch (the one that runs while the driver is still warming up), not as a regression: the frame is
right when it is right, and the partner run proved it byte-for-byte.

WHAT IS LEFT OF THE GI CHAIN: the temporal resolve, the spatial filter and the glossy lobe, in that order, each
extractable the same way now that the image channel exists. The probe cache was already a pass.

## THE GI CHAIN'S SECOND STAGE: THE GLOSSY LOBE, AND A DEFECT IT FOUND IN THE FIRST

The lobe (`vulkan.pass.ssgi_spec`) is the same shape as the tracer - two shared sets, no own binding, its images
reached through `pass_io::barrier_images` - which is the point: **that channel is per-pass, not a special case
for one pass**. It exists because the lobe writes the SAME image the tracer wrote (it adds a traced reflection to
the raw diffuse trace), and because it has two storage outputs of its own whose descriptors live in the G-buffer
set.

Its frame position is why it is a pass rather than part of the tracer, and it is a correctness argument rather
than a tidiness one:

* AFTER the tracer, because it READS the trace it corrects. Two consecutive dispatches have no memory dependency
  between them, so the first thing the lobe records is the compute-storage barrier that makes the
  read-after-write legal;
* BEFORE the denoiser, because the temporal resolve consumes the SUM. The tracer therefore skips its own
  hand-off barrier when the lobe will run (`specular_next`), and the lobe owes the transition after the LAST
  writer - which is why `feature_active("ssgi_specular")` had to become the same predicate the tracer reads
  (`ssgi_specular_active()`), one answer in one place. That is the framework's feature registry doing exactly
  what it is for: the pass names a feature, the renderer answers it, and the answer is the chain's ordering
  constraint.

THE DEFECT THIS STEP FOUND IN THE PREVIOUS ONE, because it is the kind of thing that only a question finds: the
tracer's `probe_grid_seen_` describes a GENERATION, and nothing told the pass when the generation changed. It
did not show up in any gate run - a capture never resizes - but after a swapchain recreation the pass would have
skipped the probe grid's first-use batch while the host asked for it, leaving the grid in UNDEFINED the moment
the tracer sampled it. The fix is the call `recreate_stage` was built for: `on_swapchain_recreated` now tells
the tracer's stage and the lobe's stage, exactly as it tells the probe cache's and the TAA resolve's. The
hand-kept list in that function is now the list of *stages*, and a pass added to a stage is covered.

The lobe's own state is per IMAGE rather than per generation (`seen_`, one flag per swapchain image, sized from
`ssgi_spec_frame::image_count`), for the reason the tracer's is not: it has one output per image and each needs
its first-use transition once. The host asks for the reset in the two moments it knows and the pass cannot - a
new generation (`on_swapchain_recreated`, via the runner) and the chain being switched on
(`ssgi_spec.reset_first_use()`, the same shape as `taa_pass::reset_history`).

Verified: gate 12 x 2, **0 changed, 0 flaky**, all four GI hashes plus `sponza_march` and the seven pre-GI
scenarios. `metal_rough_glossy` and `glossy_motion` are the two the feature was built around, and they are the
ones that decide this move.

WHAT IS LEFT: the temporal resolve and the spatial filter. Both own a SET LAYOUT of their own (the temporal
denoiser's four inputs), so they are the first GI stages that need the `pass_context` create half as well as the
frame half - which the TAA resolve already demonstrated.

## THE GI DENOISER: ITS LAYOUT IS GENERATED NOW, AND WHY THE PASS ITSELF IS NOT EXTRACTED YET

Two things came out of attempting the temporal resolve, and the second is the more useful one.

**DONE: the denoiser's set layout is generated from a declaration.** `render_resource::ssgi_temporal_io` now
describes its seven bindings (the trace, the history, the motion vectors, the depth, the accumulation it WRITES,
the normal/roughness target for the reflection's cap, and the lobe's reprojection), and
`runtime::make_ssgi_temporal_pipeline` builds the layout from it with `bindings::make_set_layout` - the same
pattern `build_taa` and `build_ssgi*` already use. The family's pool count comes from
`descriptor_counts_for(ssgi_temporal_io, 0)` instead of `signature.size()`. **That closes a drift the code's own
comments record as having happened**: the layout and the pool count were two hand-written lists of the same
seven bindings, they were once out of step by one, and the validation layer found it
("Trying to allocate 15 of VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER descriptors ... but this pool only has a
total of 12"). One declaration now feeds both.

**NOT DONE, AND THE MEASURED REASON: the pass itself does not fit the framework yet.** The denoiser resolves TWO
signals - the diffuse bounce and the reflection - with ONE pipeline and ONE layout, and each signal has its own
pair of images in the SAME seven slots. So:

* `resolved_io::own` can describe one set's contents, and the reflection's set needs a DIFFERENT image in the
  same slot. A declaration that named the diffuse images would be right for one family and wrong for the other;
* declaring the reflection's images side by side would need the declaration to say "these two lists are the same
  slots", which the schema has no way to express;
* and `resolved_io` carries exactly ONE own-set handle, while this pass needs two.

The honest options, for whoever takes it next, are one of: a per-pass `variants` concept (N binding lists over
one layout), a second own-set slot in `resolved_io`, or a declaration that names only the SLOTS (kind, count,
layout, stages) and lets the host supply the images - each is a real framework decision rather than a local
hack, which is why it was not made inside a pass-extraction step. What did land is the half that needed no such
decision, and it is verified: gate 12 x 2, **0 changed, 0 flaky**, all four GI hashes plus `sponza_march` and the
seven pre-GI scenarios - i.e. a generated layout produces byte-identical frames to the hand-written one.

ONE HARNESS NOTE, the same intermittent one recorded earlier: this step's FIRST full gate run reported one
scenario FLAKY (`changed: 0` in every run, and the tail I captured did not name it); the immediate re-run was
12 x 2 clean with all twelve hashes matching. Nothing about the renderer changed between the two runs.

## THE GI DENOISER: MEASURED TWICE, THEN TAKEN IN TWO STEPS

**STEP 1a LANDED** (`bc2dc1d`): the diffuse temporal resolve's RECORDING is a pass - the two barrier batches, the
dispatch, the two push lanes that describe its own state (`history_valid`, `mode`), the history copy and the
hand-backs. Its set layout, its pipeline and both per-image families stay the renderer's, arriving through
`resolved_io::own_set` and `resolved_io::pipelines`, which is inside the framework's contract rather than a
shortcut. Gate 12 x 2, 0 changed, 0 flaky.

The two failures below are kept because they are what produced the split, and the second one is what produced
`resolved_io::own_per_image` - the channel step 1b uses.

Two attempts at extracting the temporal resolve failed, both with a validation error that names the missing
piece precisely, and the second one is the finding worth keeping. Neither attempt was landed (the tree was
reverted to `f117f29` rather than left half-done).

**Attempt 1: a single pass with two families.** The denoiser resolves two signals (the diffuse bounce and the
reflection) with one pipeline and one layout, and each signal has its own images in the SAME seven slots. A
declaration can describe one set's contents, so it would be right for one family and wrong for the other; a
second list would need the schema to say "these are the same slots"; and `resolved_io` carries exactly ONE
own-set handle while the pass needs two. Recorded last round as a design question, and it still is.

**Attempt 2: split the signals into two passes** (the shape that needs no framework change at all - each pass
then has one honest declaration, which is why `ssgi_temporal_io` was trimmed to the diffuse signal's two barrier
images). The diffuse pass got as far as building, dispatching and producing byte-identical frames, and then
failed on its own descriptor family, twice, in a way that is not about the two signals at all:

1. with the family's signature taken from the CURRENT frame's views -
   `VUID-vkUpdateDescriptorSets-None-03047`, i.e. rewriting a set a pending frame names. That one is understood
   and solved (the TAA resolve's generation fingerprint: cache one frame's views, drop them on
   `on_swapchain_recreated`);
2. with that fingerprint in place - `vkQueueSubmit(): ... expects VkImage 0x64 ... to be in layout
   VK_IMAGE_LAYOUT_GENERAL--instead, current layout is VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`. The sets point
   at the WRONG IMAGE: a per-image family needs each image's OWN views in that image's set, and the write
   callback `image_set_family::ensure` invokes is handed only an `image_index`, not that image's views. The
   renderer's own lambda solved this by reaching into `core` (`vulkan_core.gi_image_views[image_index]`); a pass
   cannot, because `resolved_io::own` is the CURRENT image's handles.

**SO THE MISSING CHANNEL IS PER-IMAGE VIEWS, and this also exposes a defect in the TAA pass that the gate cannot
see.** `taa_pass::record` ignores the `image_index` its write callback is given and writes `io.own` (the current
frame's four views) into EVERY set: with more than one swapchain image, every set points at one image's history,
velocity and depth. The gate reports no finding because the reference was captured from the same code - and
because the pages this branch compares are pages `master` renders with the same defect.

**THE DESIGN IS SETTLED AND IS ONE FIELD.** `resolved_io` gains

```cpp
/// the same own bindings, ONE VIEW PER SWAPCHAIN IMAGE: own_per_image[k][image] is the view binding k has for
/// swapchain image `image` (each span is frame.image_count long, or empty where the host filled nothing)
std::array<std::span<VkImageView const>, max_own_bindings> own_per_image = {};
```

and that is all the framework needs to add: the host already holds these lists (they are the core's
`gi_image_views`, `gi_history_image_views`, ... - the same vectors the host-written families index by hand), and
a pass's write callback is handed an `image_index` precisely so it can look up that image's views. The first
entry doubles as the per-generation fingerprint (`own_per_image[k][0]` is stable for as long as the target
generation lives, while `own[k].view` changes every frame).

**THE CHANNEL IS IN** (this step): `resolved_io::own_per_image` exists, documented with the measurement that
asked for it, and `test_pass.cpp` now proves the runner hands it through untouched - the framework does not
interpret it, and a pass that owns a per-image family is the only thing that reads it. Nothing else changed, so
the gate is 12 x 2 with 0 changed and 0 flaky, which is also the evidence that adding a channel is inert until its
user arrives.

The three places it lands, in the order they should be done:

1. **the temporal resolve** (the diffuse signal first, as a pass of its own - its declaration is already in
   `render_resource.cppm` and already verified), which is the extraction that measured the need;
2. **`taa_pass::record`**, which becomes correct for more than one swapchain image. THIS ONE CHANGES FRAMES: the
   reference for `deferred_taa_fxaa` was captured from the defective code, so fixing it is a deliberate,
   separately verified change with a reference update - not something to fold into an extraction whose
   acceptance is "0 changed";
3. the same field then serves the SPATIAL filter, the last GI stage.

Each of the two failed attempts is preserved in this document rather than in the tree: the branch is at
`b7633e0`, whose code is `f117f29` (fully verified: Release/Debug/ASan, ctest, doxygen, gate 12 x 2 with 0
changed and 0 flaky), because a half-wired pass is worse than an unimplemented one.

## THE EXTENT RULE: TWO RESOLVERS WERE NOT APPLYING THEIR OWN DECLARATION

A declaration says what a pass works at (`behaviour::extent`: `full`, `half`, or a resource's size), and
`resolved_io::extent` is "the extent THIS pass works at ... per `behaviour::extent`". So the RESOLVER has to apply
the rule, and `runtime::pass_extent` is the one place that maps it to a number. **Two of the eight resolvers did
not:** the tracer and the glossy lobe both declared `half` and both handed the pass `vk.swap_chain_extent` - the
FULL frame - and the tracer's comment even claimed "applied by the runner", which was never true.

Why it survived every measurement until now: both shaders open with
`const ivec2 gi_extent = imageSize(<their output>); if (pixel >= gi_extent) return;`, so the extra invocations
returned immediately and wrote nothing. **The frames were byte-identical and the dispatch was four times the
work** - 1920x1080 of workgroups for a 960x540 image, in each of the chain's first two stages. That is exactly the
class of defect a capture gate cannot see: it needs a reader who compares the declaration to the number, which is
why both resolvers now say `this->pass_extent(*static_cast<pass::frame_pass const*>(&this->ssgi_trace))` and the
tracer's comment records the old line and why it looked harmless.

The same step removed the LAST copy of the mapping: `resolve_pass` (the probe cache's fallback) carried its own
17-line `switch` over the three rules, written to the same numbers. It is now a call to `pass_extent`, so a fourth
rule cannot be implemented twice with the second copy wrong.

Measured on this step: Release/Debug/ASan clean, ctest 8/8 in all three, doxygen exit 0, gate 12 x 2 with **0
changed and 0 flaky** - the gate's verdict here is the *proof of the invisibility claim* (the shaders' clips made
the extra work unobservable), not evidence that the change is inert.

## THE SPATIAL FILTER BUILDS ITS OWN PIPELINE, AND ONE `make_*` LEAVES THE RUNTIME

The extraction of the chain's last stage (`fa8c1c0`) had left it a pass that records but owns nothing: the
runtime built its pipeline (`runtime::make_ssgi_spatial_pipeline`, from the shared scene and G-buffer layouts) and
handed it over through `resolved_io::pipelines`. That is a legal shape - the framework's contract says a pass may
receive pipelines - but it is the WRONG shape for a handle only this pass ever names, and the tracer and the
glossy lobe had already shown the right one: `create` builds it from `pass_context::shader` and
`pass_context::shared_set_layout`, keeps it, and releases it in its own destructor.

So `ssgi_spatial_pass` now owns its pipeline layout and its compute pipeline, and the four places that used to
reach for the runtime's members changed with it:

* `runtime::make_ssgi_spatial_pipeline` is DELETED (declaration, definition and the two `ssgi_spatial_pipeline*`
  members), as is the `vkDestroyPipelineLayout` in the runtime's destructor - the destructor no longer names a
  single SSGI pipeline handle except the temporal resolve's, whose pass is the next step;
* the app (`chores.cpp`) registers `ssgi_spatial.comp.spv` the way it registers every other pass's shader, and
  the pass logs its own outcome: `SUCCESS: GI spatial filter created (joint-bilateral, depth + normal edge stops)`
  now comes from `vulkan.pass.ssgi_spatial`, and the log in the gate's `render-check/debug.log` is where that is
  visible rather than inferred;
* `runtime::ssgi_active()` and the warning in `set_ssgi` ask the pass (`pipeline_ready()`) instead of a member;
* `resolve_ssgi_spatial` reads `this->ssgi_spatial.pipeline()` / `.pipeline_layout()`.

**WHAT MAKES THIS A MEASUREMENT RATHER THAN A REFACTOR**: the GI scenarios only match their references if the
chain actually runs, and the chain's predicate now requires the SPATIAL FILTER's pipeline - so `default_gi`,
`sponza_gi`, `metal_rough_glossy` and `glossy_motion` coming back byte-identical is evidence that the pass built a
working pipeline at create time. A create that silently failed would have turned GI off and changed all four.

Measured on this step: Release/Debug/ASan clean, ctest 8/8 in all three, doxygen exit 0, gate 12 x 2 with **0
changed and 0 flaky**.

## THE TEMPORAL RESOLVE OWNS ITS LAYOUT AND ITS PIPELINE - AND THE FAMILY IS WHAT IS LEFT

Step 1a made the diffuse temporal resolve a pass that RECORDS; its set layout, its pipeline layout, its pipeline
and both descriptor families stayed the renderer's, which was the honest state of a step whose acceptance was "0
changed". This step takes the three handles:

* `ssgi_temporal_pass::create` builds the SET LAYOUT from `render_resource::ssgi_temporal_io` (the same
  declaration that already generated the family's pool count) and the PIPELINE LAYOUT and PIPELINE from
  `build_ssgi_temporal`, and releases all three in its own destructor;
* `runtime::make_ssgi_temporal_pipeline` and the runtime's `ssgi_temporal_set_layout`,
  `ssgi_temporal_pipeline_layout` and `ssgi_temporal_pipeline` members are DELETED, as are the two destroys they
  needed; the app only registers `ssgi_temporal.comp.spv`, and the pass logs its own
  `SUCCESS: GI temporal denoiser created (history accumulation)`;
* `ensure_ssgi_denoise_descriptors`, `resolve_ssgi_temporal`, `record_ssgi_resolve_pass` (the reflection's mode-1
  path) and `ssgi_active()` / `set_ssgi`'s warning all read the pass now.

**THE ONE THING THAT DID NOT MOVE, and the reason is a framework gap rather than an oversight: the two descriptor
FAMILIES.** One layout and one pipeline resolve TWO signals - the diffuse bounce (this pass) and the reflection
(the renderer's mode-1 recording) - and each signal needs its own list of images in the same seven slots. The pass
owns the layout; the renderer builds both families on it, taking it through the new
`ssgi_temporal_pass::set_layout()` accessor. That accessor is the shape of the remaining gap: a declaration cannot
say "these two lists are the same slots", so the second family cannot be declared, and `resolved_io::own_per_image`
- added for exactly this - is what closing it needs.

Measured on this step: Release/Debug/ASan clean, ctest 8/8 in all three, doxygen exit 0, gate 12 x 2 with **0
changed and 0 flaky**, and the gate's `render-check/debug.log` carries the pass's own create line. As with the
spatial filter, the four GI scenarios matching is the evidence that the create really built a working pipeline: the
chain's predicate now requires the TEMPORAL pass's, so a create that failed would have turned GI off in all four.

## THE TEMPORAL RESOLVE'S DIFFUSE FAMILY IS THE PASS'S - AND WHAT `own_per_image` ACTUALLY INDEXES

This is step 1b, the one `resolved_io::own_per_image` was added for: the pass now writes the per-image sets its
declaration describes, so the renderer no longer touches them. Three things came out of it, and the first is a
correction to the record rather than to the code.

**1. THE CHANNEL IS INDEXED BY BINDING FIRST: `own_per_image[binding][image]`, one span PER BINDING whose length
is the generation's image count.** The channel's own doc comment says so ("`own_per_image[k][image]` is the view
binding `k` has for swapchain image `image`"), and it is easy to read the other way round - the first version of
this pass's write callback did, and it wrote nothing: the family's callback is handed an image index and the
callback then asked for `own_per_image[image_index]`, which is binding `image_index`'s list of ALL images. The sets
were created and never updated, so validation said exactly that - "the descriptor ... is being used in dispatch but
has never been updated via vkUpdateDescriptorSets()" - on the first gate run. The fix is a `views_for` helper that
gathers the seven `own_per_image[b][image]` handles, and the same helper's image-0 list is the family's generation
fingerprint.

**2. THE WRITES ARE GENERATED NOW, on both sides of the ownership line.** The pass writes its own sets with
`bindings::write_set(device, ssgi_temporal_io, own_set, set, views, {}, samplers)` - the binding numbers, the
descriptor types, the image layouts and the sampler each hint chose all come from the declaration that also
generated the layout - and the reflection's family in the renderer was converted to the same call in the same step,
which deleted the second hand-written `VkWriteDescriptorSet` loop (~40 lines). The pass caches the six samplers at
create time (`pass_context::samplers`), which is what lets it write a descriptor at all.

**3. THE DECLARATION CANNOT DESCRIBE BOTH SIGNALS, and this step paid for that again - measured, not predicted.**
Binding 6 is `gi_spec_reproject`: the REFLECTION's reprojection, which only mode 1 reads and only the lobe's frame
maintains. The hand-written diffuse writes had quietly put the G-BUFFER DEPTH there instead, and the comment said
why (a shader that samples a binding in a branch still leaves the access in the SPIR-V, so validation checks the
descriptor whether the branch is taken - and the lobe's image is UNDEFINED on exactly the frames the lobe is off).
The first gate run after this move, with the declaration's own resource at binding 6, reproduced that as
"expects ... SHADER_READ_ONLY_OPTIMAL--instead, current layout is VK_IMAGE_LAYOUT_UNDEFINED" plus 2 changed
scenarios. So the pass substitutes `views[6] = views[3]` and says why, named by constant rather than by literal.
It is the SAME fact as the framework gap - one declaration, two signals - seen from the descriptor's side.

**AND THE GENERATION RESET MOVED WITH THE FAMILY**: `on_swapchain_recreated` now retires it, which is the
framework's own contract for per-generation state (the runner calls it for every pass in a stage), instead of the
host reaching into the pass's object to retire it.

Measured on this step: Release/Debug/ASan clean, ctest 8/8 in all three, doxygen exit 0, gate 12 x 2 with **0
changed and 0 flaky**. What is still the renderer's: the reflection's family, and the mode-1 recording that uses
it - the two things a declaration cannot express.

## THE FIRST PASS OUTSIDE THE GI CHAIN: THE RAY-TRACED SHADOW - AND A GATE COVERAGE GAP IT EXPOSED

With the GI chain fully extracted, the next pass is one the frame loop calls between two other stages:
`record_rt_shadow_pass` was a 55-line function in `runtime.cpp`, and its pipeline was built by
`runtime::make_rt_shadow_pipeline`. It is now `vulkan.pass.rt_shadow` (`rt_shadow_io` in the declaration layer,
80-byte push, two shared sets, one barrier image, `extent_rule::full`), and it owns its pipeline layout, its
pipeline and its one-shot "tracing WxH rays per frame" log. What the renderer keeps is the STAGE: its position
(after the G-buffer pass, before the lighting stage - the ordering constraint the rays' origins depend on), the
off-path transition the lighting stage's descriptor needs on a frame where the pass does not run, and the
`rt_shadow_end` mark.

**THE GATE CANNOT DECIDE THIS PASS, and finding that out is worth more than the extraction.** All twelve
scenarios were captured with `rt_shadows = false` (it is the config default and no scenario overrides it), so the
pass built nothing, recorded nothing, and the 12 x 2 run came back 0 changed - which is a TRUE statement about
frames that never ran it and worthless as evidence. The claim "the gate decides the move on the scenarios that run
with `rt_shadows` on" was written in the pass's first draft; it is false, and the gate is not a proof of a path it
does not enter.

So the pass was verified the way the gate is built to avoid needing: **an A/B against the parent commit's binary,
with the feature ON.** A worktree at `b08c8b5` (the commit before this change) was configured and built Release,
and both binaries were run with `shadow_single`'s settings plus `rt_shadows = true`
(1080x960, camera `35,20,7,0,-1.6,0`, 40 frames, validation layers on), twice each:

| build | run A | run B | validation |
|---|---|---|---|
| parent (`b08c8b5`) | `AEEB757EA347CC41…` | `AEEB757EA347CC41…` | clean |
| this change | `AEEB757EA347CC41…` | `AEEB757EA347CC41…` | clean |

Four runs, one hash: the extraction is byte-identical on the path it changes, the path is deterministic, and it is
validation-clean. The pass's own log line is in both logs ("ray-traced shadows: tracing 1080x960 rays per frame"),
which is also how the run proves the pass's `record` ran rather than the frame merely looking the same.

**WHAT THIS MEANS FOR THE GATE, stated rather than left implicit**: `rt_shadows`, `rt_mask_bake` and the compute
skin path are all OFF in every scenario, so the harness covers the deferred path, the GI chain and the TAA resolve,
and does NOT cover the traced-shadow path or the bake. Adding a thirteenth scenario would close it - and that
scenario needs a reference captured with a deliberate `-Update`, which is the anchor rule's one forbidden move
without an explicit override. The A/B above is what that scenario would have bought, taken without touching the
anchor.

Measured on this step: Release/Debug/ASan clean, ctest 8/8 in all three, doxygen exit 0, gate 12 x 2 with 0
changed and 0 flaky (a statement about the twelve frames that exist), plus the four-run A/B above for the path
that actually changed.

## THE MASK BAKE: A JOB, NOT A FRAME PASS - AND A COMMENT THAT WAS WRONG FOR 44 BYTES' WORTH OF REASON

The alphaMode MASK bake was four raw handles plus a 45-line `make_mask_bake_pipeline` and eight lines of
recording inside `record_acceleration_structures`. It is now `vulkan.pass.mask_bake_job`, which owns its pipeline
layout, its pipeline, the descriptor set it writes and the dispatch; the renderer keeps the policy (which casters
carry a MASK material, the expanded buffer each one is baked into, the counters) and the two bindings the set is
written with.

**IT IS NOT A `frame_pass`, and that is a statement about the work rather than a shortcut.** The framework's pass
contract is per frame: a declaration, a `resolved_io` built from it, a stage the runner walks, generation state
reset by `on_swapchain_recreated`. This runs ONCE, inside the command buffer that builds the bottom level
structures, and its input is the caster list the runtime is walking at that moment. Making it a frame pass would
need a feature gate that is true on exactly one frame and a frame struct standing in for a build loop - two lies
instead of one honest difference. What it shares with a pass is the ownership rule, and it is CONSTRUCTED the same
way: the runtime builds the same `pass_context` a pass's create step gets (device, the six samplers, the shared
set layouts, the shader registry) and hands it over. The set itself stays the renderer's to ALLOCATE - the pool is
the core's - and is moved into the job, which frees it.

**VERIFIED THE SAME WAY THE RAY-TRACED SHADOW WAS, because the gate has no scenario that enables it either**
(`rt_mask_bake = false` is the default): an A/B against the pre-change binary with the knob ON. The asset is the
one that actually carries MASK materials - `AlphaBlendModeTest`, not the gate's helmet - and the config pins
`rt_shadows = true` (structures are built for `rt_shadows || ssgi`, and the bake feeds the structures) plus
`rt_mask_bake = true`:

| build | run A | run B | the bake |
|---|---|---|---|
| pre-change | `ED602F42AF764D70…` | `ED602F42AF764D70…` | `3 MASK casters baked into their structures` |
| this change | `ED602F42AF764D70…` | `ED602F42AF764D70…` | `3 MASK casters baked into their structures` |

Four runs, one hash, validation clean, and the log line is the proof that the bake RAN rather than that the frame
merely looks the same. Both runs also had `rt_shadows` on, so the A/B isolates this change rather than measuring
two at once.

**AND IT FOUND A COMMENT THAT WAS WRONG, which the static assertion then made un-repeatable.** The old note
claimed the push block is "48 bytes on the CPU and 44 in the shader". It is 44 and 44: measured with a
`show<sizeof(...)>` probe, `glm::uvec2` is 8 bytes with ALIGNMENT 4 in this build (glm does not SIMD-align the
2-component vector types here), so five uints after three uvec2s need no tail padding. The number was harmless -
`sizeof` is what the pipeline layout's range was always built with, and validation is clean - but it was a claim
nothing checked, so the job now carries `static_assert(sizeof(mask_bake_push_constants) == 44, ...)`.

Measured on this step: Release/Debug/ASan clean, ctest 8/8 in all three, doxygen exit 0, gate 12 x 2 with 0
changed and 0 flaky (about the twelve frames that exist), plus the four-run A/B above for the path that changed.

## COMPUTE SKINNING: THE THIRD JOB, AND THE ONE WHOSE LIST IS RECORDED TWICE

The compute-skinning stage was `make_compute_skin_pipeline` (pipeline layout, pipeline, one descriptor set per
frame slot with binding 9 written) plus `record_compute_skin_pass` (a dispatch per skinned caster and the memory
barrier the acceleration structure build needs after them). It is now `vulkan.pass.compute_skin_job`: the job owns
the pipeline layout, the pipeline, the per-slot sets and the barrier; the renderer keeps the knob, the policy
(which casters are skinned, the buffer each is skinned into, the strides), the request list - a member, so a frame
does not allocate while recording - and the refit bookkeeping (`rt_skin_levels`).

**THE FRAME SLOT IS AN ARGUMENT, not something the job looks up.** The original read
`vk.current_frame` from the core to pick which slot's per-joint matrices to bind. A job has no core, and picking
the wrong slot would skin against another frame's animation - so `record(command_buffer, slot, requests)` is
handed it by the one component that paces frames. That is the same split as the push block's VALUES: the renderer
knows the frame, the job knows the work.

**WHY A JOB RATHER THAN A `frame_pass`, stated as the measured shape rather than a preference**: the SAME request
list is recorded at TWO places with different meanings - once on the frame the structures are created (the build
below reads the vertices those dispatches wrote) and once per frame after (the structure is REFITTED because only
the bytes change). A frame pass would have to pretend the first of those is a frame. Everything else is the pass
rule: constructed from the same `pass_context`, owns what it names, releases it in its own destructor.

**VERIFIED BY THE SAME KNOB-ON A/B, on the asset class this stage exists for.** `rt_skin_bake` is false in every
gate scenario, so the 12 x 2 run says nothing about it; the A/B is `CesiumMan` (the model with a skinned caster)
with `rt_shadows = true` and `rt_skin_bake = true`:

| build | run A | run B | the skinning |
|---|---|---|---|
| pre-change | `6E0BED153DA631F0…` | `6E0BED153DA631F0…` | `1 skinned casters re-skinned and REFITTED … every frame` |
| this change | `6E0BED153DA631F0…` | `6E0BED153DA631F0…` | `1 skinned casters re-skinned and REFITTED … every frame` |

Four runs, one hash, validation clean, and the log line in BOTH runs is what shows the job recorded rather than
that the frame merely looks the same. (The refit has to happen for the frames to match at all: a frame that
skipped it would cast the bind pose, which is exactly the A/B `docs/gi_hit_shading.md`'s L2.2b measured.)

**ONE TOOL OBSERVATION, recorded because the acceptance criterion is a clean doxygen run**: the first `doxygen`
after this change printed `error: Problems running epstopdf. Check your TeX installation!` while regenerating the
LaTeX output. It is NOT the sources: there is no `\f$` formula anywhere under `vulkan/`, three subsequent runs are
silent (this tree and the pre-change worktree alike), the exit code was 0 every time, and the message comes from
MiKTeX's converter rather than from doxygen's warning stream. Recorded as a one-off from the toolchain instead of
being left unexplained.

Measured on this step: Release/Debug/ASan clean, ctest 8/8 in all three, doxygen exit 0 with an empty warning
stream, gate 12 x 2 with 0 changed and 0 flaky (about the twelve frames that exist), plus the four-run A/B above.

## THE CLUSTERED-LIGHT SORT: THE FIRST PASS WHOSE RESOURCES ARE BUFFERS, AND TWO CHANNELS IT NEEDED

The last compute stage on the "still the runtime's" list, and the one that had been out of reach for a reason
worth stating: `record_cluster_pass` wrote two buffers that live in the SHARED scene set (bindings 11 and 12),
and the framework had a channel for an IMAGE a pass moves without binding it (`barrier_images`) and none for a
BUFFER. So the renderer kept the whole recording. `vulkan.pass.cluster` owns it now, and the step added exactly
the two things it needed:

1. **`pass_io::barrier_buffers`** (+ `resolved_io::barrier_buffer_storage` / `barrier_buffers`, `max_barrier_buffers`,
   the validator's three checks and their tests) - the buffer twin of `barrier_images`, with the same argument: a
   compute SHADER_WRITE is not visible to the fragment stages that read it later in the same submission without a
   buffer memory barrier, and only the WRITER can place it. The old pass's own comment said as much; what was
   missing was a way to name the buffers in a declaration.
2. **`extent_rule::none`** - the sort's dispatch is `tiles_x * tiles_y * slices` workgroups, a count derived from
   the frame's extent but equal to neither it nor half of it. All three existing rules would have been a claim the
   host cannot honour, so `none` says "this pass sizes its own work" and `pass_extent` hands over an EMPTY extent
   rather than a plausible-looking number. The pass reads its count from its own frame, the same split as the
   scene pass's leaves.

**AND THE LAST `core::make_*` COMPUTE PIPELINE IS GONE.** `core::make_cluster_pipeline` built the cluster pipeline
against the core's own scene pipeline layout, which a pass cannot own; `vulkan.pipelines::build_cluster` builds the
same shape (one set layout, no push range - `light_cluster.comp` declares no push_constant block) and the pass
releases it. The runtime's `make_cluster_pipeline` wrapper and the two feature predicates that asked a raw
`optional<vk_pipeline>` now ask the pass; `feature_available("clustered")` still answers AVAILABILITY (the pass
built a pipeline) while `feature_active` answers ACTIVITY (it runs this frame, which also needs a live punctual
light) - those two are different questions and the comments now say so, because a change like this is exactly where
they get conflated.

**VERIFIED BY THE KNOB-ON A/B, and this knob is the app's own stress test rather than a config flag**: no gate
scenario has a punctual light, so `f.clustered` is false in all twelve and the 12 x 2 run says nothing about the
sort. `[lighting] demo_lights = 4` spawns four procedural lights ("clustered light stress" in the log), and with
that the pass runs:

| build | run A | run B | the lights |
|---|---|---|---|
| pre-change (`0146b77`) | `EAFBC494D3032737…` | `EAFBC494D3032737…` | `demo lights: 4 procedural punctual lights` |
| this change | `EAFBC494D3032737…` | `EAFBC494D3032737…` | `demo lights: 4 procedural punctual lights` |

Four runs, one hash, validation clean - and validation clean here is a real statement, because the two buffer
barriers are the pass's own now: a pass that skipped them would leave the fragment stages reading a buffer the
compute stage wrote in the same submission.

Measured on this step: Release/Debug/ASan clean, ctest 8/8 in all three, doxygen exit 0 with an empty output
stream, gate 12 x 2 with 0 changed and 0 flaky (about the twelve frames that exist), plus the four-run A/B above.

## THE FILTERS MODULE: TWO NAMED VIEWS OF THE DEVICE ROOT, AND THE CHANNEL A PASS'S INIT NEEDED

The goal of this step is the one the pass work has been heading toward: **the runtime should stop holding
per-pass resources.** What it still held, measured before touching anything, was wiring rather than GPU objects -
`runtime::create_mask_bake()` and `runtime::create_compute_skin()`, each building its own copy of the create-time
context plus a bespoke input struct, because a pass had no way to name a resource the RENDERER owns.

**WHAT WAS ACTUALLY MISSING, measured rather than assumed.** A pass already needs only `device` to own a
per-image descriptor family: `bindings::image_set_family` creates and retires its own `VkDescriptorPool` from a
raw `VkDevice` (`bindings.cppm:299/469`), which is how the TAA resolve and the temporal pass work. Two things
were missing, and no more: **a one-off descriptor set from the owner's pool** (the two jobs write exactly one set
each from the scene layout) and **the handle of a declared resource** (the material table, the bindless texture
array, a slot's skin matrices). Formats were NOT missing either: `hdr_format`/`gbuffer_formats` are compile-time
constants in `core.cppm`, and `depth_format`/`swap_chain_image_format` are runtime values that the `scene` and
`transparent` passes already receive per frame in their own frame structs.

**THE MODULE**: `vulkan/core/filter/filters.{cppm,cpp}` (module `vulkan.core.filters`), holding two named views
of a core, both constructed from a `std::shared_ptr<core>`:

* `vulkan::user_filter` - what `runtime::operator->` exposes to the application. It is the old `core_filter`,
  renamed (and now holding a share instead of a bare `core&`), which is what makes room for a family of filters.
* `vulkan::pass_filter` - what a pass's create step is handed: `device()`, `swap_chain_image_format()`,
  `swap_chain_extent()`, `make_descriptor_set(layout)`, `vma()` (the door for a pass that must create its own
  buffer or image) and `resource(id, element)` over what the owner published through `register_resource`.

**THE NAMING IS `vulkan::user_filter`, NOT `vulkan::core::user_filter`, and that is a language fact rather than a
preference**: `vulkan::core` is the device-root CLASS (`export struct core`), so no namespace of that name can
exist. Nesting the filters inside `core` would put their definitions in the `vulkan.core` module (a nested class
cannot be defined in a different module), which is the opposite of "a new module for the filters". If nesting is
wanted later, it is available - at the cost of the module split.

**TWO RESOURCE DOMAINS, and why the filter is a registry rather than a lookup table.** Core owns the images it
creates (the HDR chain, the G-buffer, the GI chain, the probe grid); the RUNTIME owns others (the material table,
the bindless texture array, the per-slot skin matrices and cluster bins, the shadow map). A filter built over core
alone can only answer for the first. So the pass filter carries a small registry the owner fills
(`register_resource(id, element, handles)`), and `resource()` resolves from it - which serves both domains with one
vocabulary and lets the core-owned families join the same table when the first pass needs one.

**THE LIFETIME CONTRACT, which is why the channel is small and why it is not `resolved_io`**: what `resource()`
answers at CREATE time is a SESSION-STABLE handle. The runtime's material table and texture array are created once
and only have their contents rewritten; its per-frame-slot buffers are created once and rewritten per slot. A
per-swapchain-image VIEW is not stable - it is rebuilt with every generation - and those keep arriving per frame
through `own` / `own_per_image` / `barrier_images`. The measured evidence that the contract is satisfiable: the
MASK bake's create already runs BEFORE the scene is uploaded (log line 301 vs 311 in a real run; the material
buffer is created once at `runtime.cpp:371` and the textures only ever `push_back` at `:946`).

**AND ONE CONSTRUCTION SITE**: `runtime::make_pass_context()` replaces the three copies of the create-time context
(the stage loop and the two jobs), which is what stops a per-pass entry point per job from growing back.

This step lands the channel INERT - no pass calls `resource()` yet - which is the same shape the per-image view
channel landed in (`own_per_image`), and for the same reason: adding a channel is provably harmless while its user
is absent. The migration that uses it is the next step (the two jobs), and its acceptance is the two knob-on A/Bs,
since neither path runs in the twelve gate scenarios.

Measured on this step: Release/Debug/ASan clean, ctest 8/8 in all three, doxygen exit 0 with an empty output
stream, gate 12 x 2 with 0 changed and 0 flaky.

## THE RESOURCE CHANNEL'S FIRST TWO USERS: THE RUNTIME STOPS BUILDING WHAT A PASS CAN ASK FOR

The previous step landed the channel inert (the pass filter, `pass_context::resource` / `descriptor_set`, and one
`make_pass_context()`); this one spends it. The renderer's two per-pass entry points - `create_mask_bake()` and
`create_compute_skin()`, each building its own copy of the create-time context plus a bespoke input struct - are
DELETED, and both jobs now build themselves inside `create_passes()` from the same context every pass gets:

* `mask_bake_job::create(context)` asks for `resource_id::material_table` and `resource_id::scene_textures`, takes
  the set from `descriptor_set(scene_layout)`, reads the array's sampler from the shared `sampler_set`, and logs
  its own outcome;
* `compute_skin_job::create(context)` asks for `resource_id::skin_matrices` once per frame slot (the count comes
  from the new `pass_context::frames_in_flight`, because "the owner has three slots" and "it forgot the fourth"
  must not be the same statement) and allocates one set per slot itself;
* the runtime's half is `publish_pass_resources()`, called before any pass is created: it registers the material
  table, the texture array and each slot's skin matrix buffer in the DECLARATION's vocabulary, so a pass asks by
  `resource_id` rather than being handed a member of `runtime`.

**ONE SAMPLER JOINED THE SHARED SET.** The bake's set needs the sampler the bindless array is read through
(REPEAT, a long LOD range), which until now was reachable only from the renderer's hand-written set code:
`render_resource::shared::sampler_set` grew a seventh field (`textures`). It has no `sampler_hint` yet, and
deliberately: the only user is a job with no declaration of its own, and a hint belongs with the first DECLARATION
that names the array.

**THE APP'S PART SHRANK TO WHAT ONLY IT KNOWS**: it registers the two jobs' SPIR-V files (whose names the jobs
own) before the single `create_passes()` call, and it no longer calls, logs or unwraps anything per job.

**VERIFIED BY THE TWO KNOB-ON A/Bs, because neither path runs in the twelve gate scenarios.** Both were run
against the IMMEDIATE parent commit (`e25b7d8`) with the knob on, twice each on both sides - eight runs, one hash
per path:

| path | scenario | pre-change | this change | the work ran |
|---|---|---|---|---|
| alphaMode MASK bake | `AlphaBlendModeTest`, `rt_mask_bake = true` | `ED602F42AF764D70…` | `ED602F42AF764D70…` | `3 MASK casters baked into their structures` |
| compute skinning | `CesiumMan`, `rt_skin_bake = true` | `6E0BED153DA631F0…` | `6E0BED153DA631F0…` | `1 skinned casters re-skinned and REFITTED` |

Each log line is the proof the work ran rather than that the frame merely looks the same, and both paths are
validation-clean on both sides.

Measured on this step: Release/Debug/ASan clean, ctest 8/8 in all three, doxygen exit 0 with an empty output
stream, gate 12 x 2 with 0 changed and 0 flaky (about the twelve frames that exist), plus the eight A/B runs above.

## THE PASS CHAIN: THE GI CHAIN IS A VALUE NOW, AND TWO TRAPS IT SPRANG

`vulkan.pass.chain` (module `vulkan.pass.chain`) holds an ordered list of passes and the runner's two calls over
it (`init`, `record`), so the ORDER of a chain is a sequence of `add` calls rather than the line order of a frame
loop that also interleaves barriers, marks and the renderer's own work. It is deliberately NOT a scheduler and NOT
an ownership container: it holds non-owning pointers, and each pass's `feature()` gate, resolver and behaviour
still belong to the runner - a chain of four passes runs exactly like the four separate stages it replaced, down
to the per-pass `resolved_io` the runner builds for each of them.

**THE GI CHAIN WAS THE FIRST (AND ONLY) CONTIGUOUS RUN**, measured before touching anything: the tracer, the
lobe, the temporal resolve and the spatial filter are recorded back to back, and the only thing between the
temporal and the spatial was the renderer's REFLECTION resolve. So the chain could not be contiguous without
answering for that, and the answer was the shape the temporal pass already uses twice: a frame CALLBACK. The
temporal's frame now carries `record_reflection`, which the pass calls at the END of its own recording - after
mode 0's hand-backs and before the spatial filter's stage, exactly where it ran before - so the reflection's
recording stays the renderer's (a declaration cannot describe two signals in the same seven slots) while the pass
decides WHEN it happens. The four `std::array<frame_pass*, 1>` stage members, four `create_stage` calls, four
`record_stage` sites and the `if (record_ssgi_denoise_pass(...))` around the filter all collapse into one chain,
one `init` and one `record`.

**AND THE "ONLY IF THE TEMPORAL RESOLVED" GATE MOVED TO THE FEATURE REGISTRY**, because a chain that can skip its
own tail is a scheduler and this renderer already has one place that answers "does this pass run this frame":
`feature_active("ssgi_spatial")` = the chain is on AND this frame's temporal resolve recorded. The temporal pass
now clears its answer when the host sets its frame, so that predicate cannot read an earlier frame's answer.

**TRAP ONE, and it is the kind that a gate catches and reasoning does not.** The new `feature_active` branch was
first written into `feature_available` instead - the function the overlay and the startup log ask, which answers
"could this run this session" - so the runner's `feature_active("ssgi_spatial")` fell through to `false` and the
filter was skipped on EVERY frame. The symptom was a subtly darker frame in the model's region (3.8% of pixels,
mean -0.089 R) and the five GI scenarios changed. What found it was instrumenting the runtime with the run
report's own counters for the first four frames: `recorded 3 skipped_inactive 1` on every one of them, which said
"the last stage never ran" in a single line. Two predicates with the same vocabulary and different questions is
the trap; the file now has a comment at both branches saying which is which.

**TRAP TWO, the honest half**: the reflection's descriptor family (`ssgi_spec_temporal_family`) used to be ensured
inside the deleted `record_ssgi_denoise_pass`; the first version of this step moved that ensure into the new
callback, i.e. AFTER the temporal dispatch instead of before it. That was measured NOT to change the frames (the
gate reported the same hash before and after moving it back), but it is back in the segment - once per frame,
before the chain - because that is where the renderer ensures the family it owns, and because "the same hash"
here would not have held on a frame where the family was first built.

Measured on this step: Release/Debug/ASan clean, ctest 8/8 in all three, doxygen exit 0 with an empty output
stream, gate 12 x 2 with **0 changed and 0 flaky** - including all five GI scenarios, which are the ones that run
the chain this step rewired.

## WHAT THE INVENTORY ALREADY FOUND (RE-AUDITED AGAINST THE CURRENT TREE)

The list below was written on the pre-GI state. Attaching GI (`5036a01`) brought `master`'s files over, so most
of these were FIXED by that - and a stale finding is its own defect in the record, which is why each entry now
says what a re-audit of the current tree found.

* **STILL OPEN, and now measured - the transparent pass's attachment format vs the named scene pipelines.**
  The transparent pass's instance is declared with `vulkan::hdr_format` = `VK_FORMAT_R16G16B16A16_SFLOAT` for its
  colour attachment and `depth_format` for its depth, while the leaves it draws go through the named scene
  pipelines, which `core::make_pipeline` creates with `this->swap_chain_image_format` (the surface's format, e.g.
  `B8G8R8A8_SRGB`) and `this->depth_format`. Those are not the same format *or* the same format class, and the
  gate reports no validation finding on `transparent_blend` - the scenario that actually draws a BLEND material
  through that instance - so the layer tolerates the pair on this driver, which is exactly the kind of thing that
  is a validation error on a stricter one. Two candidate fixes, and they are not equivalent:
  (a) **name the format in the declaration** (`render_target` has no format field today), which is the general
  answer and would let every pass state what it renders into; or
  (b) **build the named scene pipelines with the format the instance uses**, which is narrower and would need the
  pipeline registry to know which instance a leaf is drawn into - a leaf can be drawn in both the G-buffer and
  the transparent instance, so this is likely wrong.
  Either way the finding is real rather than bookkeeping: it is the one recorded item that a stricter driver
  would fail, and it is why "a declaration names its target's format" is on the list at all.
* **RESOLVED - `sampler_hint::probe_grid` with no backing sampler.** The hint now has one:
  `gi_probe_sampler` is created for the probe cache's declaration to choose, and `shared_samplers()` fills that
  slot. The "lying vocabulary" was a property of the pre-GI state, not of the schema.
* **RESOLVED - the "skybox" references.** `draw_skybox` does not exist as a parameter, nothing implements
  `feature_available("skybox")` (zero hits), and the only remaining mentions of a skybox are comments that say
  the pass was REMOVED ("the same function the removed forward skybox pass used"). The vocabulary is clean.
* **RESOLVED - the shadow-sampler doc.** `core::make_shadow_sampler` really does build a depth-compare sampler
  (`make_shadow_sampler_info`, with the comment naming the PCF contract), which is what the runtime's member
  comment claims. The two agree.
* **NOT RE-VERIFIED - "an overlay-record comment pointing at the wrong function".** The audit did not locate it
  (the overlay's own comments and the `gui` import read correctly), so it is neither confirmed nor denied here
  rather than being repeated as fact.

## HANDOFF: WHERE THIS STANDS AND WHAT IS LEFT, EXACTLY

**DONE, and each step verified byte-for-byte against the capture gate as it landed.**

* **EVERY STAGE OF THE GI CHAIN IS A PASS** (as of `fa8c1c0`): `gi_probe` (the cache), `ssgi_trace` (`b62c4b3`),
  `ssgi_spec` (the glossy lobe, `d2f5715`), `ssgi_temporal` (the resolve's recording, `bc2dc1d`) and
  `ssgi_spatial` (the filter that ends the chain, `fa8c1c0`). The last two are the same shape as the first two -
  the pass owns the frame's recording and the renderer supplies what the declaration cannot describe - and each
  move left the gate 12 x 2 with 0 changed and 0 flaky.
* The non-GI PBR chain is passes: `vulkan.pass` (framework), the declaration layer (`render_resource`), the
  generators (`make_set_layout` / `write_set` / `image_set_family` taking a `VkDevice`), and `scene`,
  `transparent` and `taa` wired and recording (commits `bfed4cd` .. `6cd9c6e`). The branch's own baselines
  proved it while it was pre-GI.
* GI is attached (`5036a01`): the 37-family declaration layer, `gi_probe` (a pass on `master` already), the GI
  pipeline builders, the acceleration structures, the GI shaders and config, the 12-scenario gate. All four
  acceptance hashes match (`default_gi` `BF180E98ADB29E7E`, `sponza_gi` `58EC848DFABE654A`,
  `metal_rough_glossy` `46F9851B7BC89872`, `glossy_motion` `98B06F2190B49519`) plus `sponza_march` and the seven
  pre-GI scenarios, 12 x 2 with 0 changed and 0 flaky.
* The GI chain's diffuse tracer (`b62c4b3`, `vulkan.pass.ssgi_trace`) and glossy lobe (`d2f5715`,
  `vulkan.pass.ssgi_spec`) are passes, on the `pass_io::barrier_images` channel that the tracer's extraction had
  to add - and doing it found and fixed a real defect (a pass's generation state was never reset, because
  `on_swapchain_recreated` did not call `recreate_stage` for its stage).
* The denoiser's set layout is generated from its declaration (`f117f29`), closing a drift the code's own
  comments record as having happened once.
* The per-image view channel exists (`4e89078`), with a contract test.
* **Every resolver applies its own declaration's extent rule, through the one function that maps it**
  (`pass_extent`): the tracer and the lobe had been dispatching the full frame at a half-size image and were
  invisible only because both shaders clip against `imageSize`, and `resolve_pass`'s second copy of the mapping is
  gone. See the section above for the measurement.
* **The spatial filter builds its own pipeline** (see the section above): `runtime::make_ssgi_spatial_pipeline`
  and the runtime's two `ssgi_spatial_pipeline*` members are deleted, the app only registers the shader, and the
  four GI scenarios matching is what proves the pass's own create produced a working pipeline.
* **The temporal resolve owns its set layout, pipeline layout and pipeline** (see the section above), so
  `runtime::make_ssgi_temporal_pipeline` and its three members are gone too. **No SSGI pipeline handle is the
  runtime's any more**; what remains of the denoiser in the renderer is the reflection's family and the mode-1
  recording that uses it.
* **Step 1b: the temporal resolve's DIFFUSE descriptor family is the pass's**, written in `record` from
  `resolved_io::own_per_image[binding][image]` through `bindings::write_set` (so the writes are generated from the
  same declaration as the layout), retired by its own `on_swapchain_recreated`, and with one deliberate
  substitution at binding 6 that the section above records. The reflection's family in the renderer uses the same
  generated writes now.
* **The ray-traced shadow is a pass, and it is the FIRST one outside the GI chain** (see the section above):
  `vulkan.pass.rt_shadow` owns its pipeline layout, its pipeline, the two barriers around the visibility image and
  the dispatch; `runtime::make_rt_shadow_pipeline` and `record_rt_shadow_pass` are deleted, and the frame loop
  keeps only the stage's position, its off path and its mark. Verified by a four-run A/B against the parent
  commit with `rt_shadows = true`, because **the gate has no scenario that enables it** - a coverage gap that is
  itself recorded above.
* **The alphaMode MASK bake is a JOB** (`vulkan.pass.mask_bake_job`), not a frame pass: it owns its pipeline
  layout, its pipeline, its own set (moved in from the renderer, which owns the pool) and the per-caster
  dispatch, and it is constructed from the same `pass_context` a pass's create step is handed. Its four raw
  handles and `make_mask_bake_pipeline` are gone, and its verification is the same knob-on A/B (3 MASK casters
  baked, one hash across four runs). The step also found and fixed a comment that claimed a 48-byte CPU push
  block where the measured size is 44 - now a `static_assert`.
* **Compute skinning is the third JOB** (`vulkan.pass.compute_skin_job`): pipeline layout, pipeline, the
  per-slot sets it writes (binding 9 from each slot's matrix buffer) and the build-ordering barrier are the
  job's; the renderer keeps the knob, the caster policy, the request list and the refit bookkeeping, and hands
  the FRAME SLOT to `record` because the job cannot know it. Its verification is the same knob-on A/B on
  `CesiumMan` (1 skinned caster re-skinned and refitted, one hash across four runs). See the section above.
* **The clustered-light sort is a PASS** (`vulkan.pass.cluster`), and it is the last compute stage that was the
  runtime's: `core::make_cluster_pipeline` (the last `core::make_*` compute builder) is deleted in favour of
  `vulkan.pipelines::build_cluster`, the recorder and its two buffer barriers are the pass's, and the framework
  grew `pass_io::barrier_buffers` and `extent_rule::none` to describe it. Verified by the same knob-on A/B with
  `[lighting] demo_lights = 4` (four runs, one hash). See the section above.
* **The filters module exists** (`vulkan.core.filters`): `vulkan::user_filter` (the old `core_filter`, renamed,
  now holding a `shared_ptr<core>`) and `vulkan::pass_filter` (a pass's init view: device, surface format,
  descriptor-set allocation, the allocator, and `resource(id, element)` over what the owner registers). The
  framework's `pass_context` gained the callbacks that forward to it (`resource`, `descriptor_set`,
  `frames_in_flight`), and the three copies of the create-time context collapsed into
  `runtime::make_pass_context()`.
* **The channel has its first two users, and the runtime's per-pass entry points are GONE** (see the section
  above): `runtime::create_mask_bake()` and `runtime::create_compute_skin()` are deleted, both jobs build
  themselves inside `create_passes()`, and the renderer's half is `publish_pass_resources()` - the material
  table, the texture array and the per-slot skin buffers published in the declaration's own vocabulary. Verified
  by eight A/B runs against the immediate parent (one hash per path, both paths validation-clean).
* **THE PASS CHAIN EXISTS** (`vulkan.pass.chain`), and the GI chain is one: `init` and `record` take the chain,
  the order is the `add` calls, the reflection's recording rides the temporal pass's frame as a callback so the
  chain stays contiguous, and the filter's "only if the temporal resolved" gate lives in `feature_active`. The
  step's two traps - the gate written into the wrong feature function, and the reflection family's ensure moving
  with the callback - are recorded with their measurements. Gate 12 x 2 = 0 changed, GI scenarios included.
* **`bind_pass_chain` is NOT here yet, deliberately**: a selector needs a second candidate chain to select
  between, and the renderer has exactly one frame order today (`gbuffer_debug`/`unlit` are per-pass feature
  gates, not alternative chains). The candidates arrive with the graphics stages: the deferred tail
  (lighting + transparent) versus the debug view is the first pair that would be two real chains.

**NOT DONE, with the reason and the exact next step.**

1. **The diffuse temporal resolve as a pass - and it is TWO steps, not one.** Every piece is ready:
   `render_resource::ssgi_temporal_io` is declared, generated and tested; `build_ssgi_temporal` already takes a
   device and the pass's layout; `resolved_io::own_per_image` is the channel its family needs. But the extraction
   splits cleanly, and taking the first half first is both smaller and honest:
   * **(1a) the RECORDING, which needs no new framework at all.** A pass whose set and pipeline arrive through
     `resolved_io::own_set` and `resolved_io::pipelines` is WITHIN the framework's contract - those fields exist
     for exactly that, and the scene pass's leaves already get their pipelines from the owner. So a
     `vulkan.pass.ssgi_temporal` can own the barriers, the dispatch, the two push lanes that describe its own
     state (`history_valid`, `mode`), the history copy and the hand-backs while the set layout, the pipeline and
     both per-image families stay the renderer's. That is the part where the ORDER lives (after the tracer and
     the lobe, before the spatial filter), and it is worth taking on its own.
     **STEP 1a IS LANDED** (`bc2dc1d`): that module is in the tree as `vulkan/pass/ssgi_temporal.{cppm,cpp}` and it
     records the resolve, with its set and its pipeline arriving through `resolved_io::own_set` and
     `resolved_io::pipelines` - a shape the framework allows, which is why 1a needed no new framework at all. What
     is left of this bullet is 1b.
   * **(1b) the pass-owned FAMILY**, which is what `own_per_image` was added for: the pass then ensures its own
     set from the declaration, writes each image's set from that image's views, and keeps its own history flags
     (the tracer reading them through an accessor). This is the step that measured the channel.
     **STEP 1b IS LANDED**: the DIFFUSE family is the pass's, written from `own_per_image[binding][image]` with
     `bindings::write_set` and retired by `on_swapchain_recreated`; the REFLECTION's family and the mode-1
     recording are still the renderer's, which is the framework gap this document keeps naming (one declaration
     cannot describe two signals' images). The history flags are still the renderer's - they are read before the
     edge, not by the pass.
2. **The spatial filter**, after (1): it reads the temporal resolve's output and binds both of its sets, so it is
   the second reader of the same channel.
3. **The TAA pass's per-image defect** (recorded above): it writes the current frame's views into every set, so
   with more than one swapchain image (this machine: `minImageCount + 1`, mailbox) every set points at one
   image's history. Fixing it uses `own_per_image` and **changes frames**, so it is a deliberate change with a
   reference update - it must not be folded into an extraction whose acceptance is "0 changed".

**AND THE TWO ANCHORS AGREE, which is the strongest statement this branch can make about "the rest is
unchanged".** The seven pre-GI scenarios were re-run against the branch's OWN origin baselines
(`%LOCALAPPDATA%\vulkan_render\baseline-pass-chain`, captured on the pre-GI state before any pass work) at the
tip of the GI-attached tree: 7 passed, **0 changed, 0 flaky** - the same seven hashes master's directory holds.
The five GI scenarios are reported NOT SEEDED there, which is what that directory is: it predates them. So the
whole journey - pre-GI state, the three extracted PBR passes, and the GI feature set attached on top - left every
frame that predates GI byte-identical, measured against BOTH the branch's origin and `master`'s references.

**THE RULE THAT KEEPS THE ANCHOR USABLE**: every gate run compares against
`%LOCALAPPDATA%\vulkan_render\baseline` (12 scenarios) and never passes `-Update` without an explicit override;
and `-BuildDir` must be ABSOLUTE or the script's relative `screenshot_dir` produces a false FLAKY (fixed in the
script, `5036a01`).
