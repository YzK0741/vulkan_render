# The pass chain, rebuilt on the pre-GI state

This branch (`pass-chain`, forked at `281ab06` - the commit immediately before the first GI commit) exists to do
one thing the later work could not do in place: let the PBR rendering be **built** as passes rather than
migrated into them, and then let GI be **added** to that architecture pass by pass. The later state on `master`
holds the GI implementation, its measurements and its documentation; this branch starts without them and
earns them back one pass at a time.

## THE ONE RULE THAT PROTECTS THE ACCEPTANCE ANCHOR

The capture gate's references live outside the repository, in
`%LOCALAPPDATA%\vulkan_render\baseline`, and on this machine they were captured from `master` WITH GI. Those
hashes are the acceptance test for restoring GI later (`default_gi` `BF180E98ADB29E7E`, `sponza_gi`
`58EC848DFABE654A`, `metal_rough_glossy` `46F9851B7BC89872`, `glossy_motion` `98B06F2190B49519`).

**Therefore every gate run on this branch must use its own baseline directory:**

```powershell
$env:VR_RENDER_BASELINE_DIR = "$env:LOCALAPPDATA\vulkan_render\baseline-pass-chain"
pwsh -NoProfile -File scripts/windows/check_render.ps1            # compare (never -Update on the default dir)
```

`-Update` may only ever be used with that variable set. Running it without is how the anchor would be
destroyed, and it would be destroyed silently: the next gate run would compare against the wrong references and
report "0 changed" for a branch that changed everything.

This branch's gate has **7 scenarios**; the five GI-era scenarios and their configs live on `master` with the
GI work. `sponza` is the useful overlap: its hash on this branch (`0EBA5300E8F84E6F`) is IDENTICAL to
`master`'s, which is the evidence that the frames this branch renders are the frames `master` renders whenever
GI is not involved.

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
| this step | the **TAA resolve is wired, and it is the first pass here that OWNS something**: set layout from its declaration, pipeline layout, pipeline, per-image family, history flags. `make_taa_pipeline` and `ensure_taa_descriptors` are gone, and the app's shader hand-over (`register_shader`) exists because a pass that builds a pipeline needs shader bytes |

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
4. **`lighting`** and **`post`** - the rest of what "PBR as a pass chain" means.
5. **then GI**: `ssgi` (trace) -> `temporal` -> `spatial` -> `spec` -> `probe cache`, each added as a pass and
   each verified against `master`'s references for the GI scenarios.

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

## WHAT THE INVENTORY ALREADY FOUND (recorded, not yet fixed)

* The named scene pipelines are built with `swap_chain_image_format` (`B8G8R8A8_SRGB`) while the transparent
  pass's instance declares `vulkan::hdr_format` (`R16G16B16A16_SFLOAT`) for its single colour attachment. The
  gate reports no validation finding on this state, so it is either tolerated or the formats are in fact
  compatible - but a declaration that names its target's format would settle it, which is one of the reasons
  the transparent pass gets a declaration.
* `sampler_hint::probe_grid` is part of the hint vocabulary on this branch, and no sampler backs it here (the
  six-hint set is `master`'s). A hint nothing can resolve is a lie in the vocabulary.
* Stale references in comments and docs: a "skybox" pipeline that does not exist (the sky is evaluated inside
  the lighting stage's shader), a `draw_skybox` parameter that no signature has, an implemented-but-undocumented
  `feature_available("skybox")`, an overlay-record comment pointing at the wrong function, and a shadow-sampler
  doc that contradicts `constant_init`.
