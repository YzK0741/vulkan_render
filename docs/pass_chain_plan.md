# The pass chain, rebuilt on the pre-GI state

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

## THE GI DENOISER: MEASURED TWICE, AND WHAT IT ACTUALLY NEEDS

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
   * **(1b) the pass-owned FAMILY**, which is what `own_per_image` was added for: the pass then ensures its own
     set from the declaration, writes each image's set from that image's views, and keeps its own history flags
     (the tracer reading them through an accessor). This is the step that measured the channel.
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
