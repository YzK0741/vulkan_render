# Level 2 handoff: state, plan, traps

This is a handover note for an agent continuing the GI work in this repository. It assumes no memory of the
session that produced it. Everything here is either a verified number from that session or a pointer into the
repository; where something is unverified it says so.

## 0. Read these first, in this order

* `docs/gi_hit_shading.md` - the full record. Level 1 (leak fix, view independence, reset) with the measurement
  that closed each; Level 2's four-step plan; the whole history of the furnace verification mode up to the
  5.7% it currently reports; and every trap that cost time.
* `docs/reference/lumen_radiance_cache.md` and `docs/reference/lumen_surface_cache.md` - studies of the UE 5.8.2
  source (available at `C:\UnrealEngine-5.8.2-release`). They are mechanism references, not numbers to copy.
* `config.example.toml` - every `[render]` knob with its reasoning, including the furnace key, which is marked
  as HALF wired in the file (see section 2: the mechanism is now complete, so that warning needs updating when
  the acceptance passes).
* `docs/shaders.md` and the module banners - the project's conventions for shaders and module versions.

## 1. Where the work stands (verified)

Level 1 is complete except for one documented limit:

* the probe cache's propagation is a six-neighbour gather gated by a bidirectional segment-versus-surface test,
  and a failed pair is dropped rather than redistributed (the unoccluded spread it replaced was measured at
  0.81% / 0.75% / 0.06% of the frame's brightness thirds; the gated version is 0.13% / 0.11% / 0.02%);
* the probe cells trace their own rays against the top level structure and shade what they hit, through
  `shaders/hit_shading.glsl`, which both the tracer and the probe pass include. `gi_probe.comp` declares no
  camera at all, so its cells cannot depend on the view - the property is compiler-enforced;
* the cache REPLACES the far-field probe where it is trusted (it used to add to it, which counted the sky
  twice). Measured effect at 180 frames: -3.93% / -4.28% / -0.42% by brightness third, 428645 pixels darker;
  in a 4x4 spatial table the effect spans -0.14% to -6.78% and is ordered by the scene (interior tiles lose
  4-7%, sky-facing tiles 0.1-0.6%), which is what a correct cache looks like;
* cost: the probe pass is 0.15 ms for 32768 cells at four rays each (131k traced-and-shaded rays); the GI pass
  is 0.58-0.9 ms;
* the one limit: arbitrating "is the shaded model closer to the truth than the screen model" needs an
  independent estimator, and every one that can be built from this engine's own pieces either reuses the code
  under test (circular) or shares one model's bias. The check that IS possible was made: against the engine's
  own lighting stage, at surfaces the camera sees, the shaded model agrees to 0.36% of mean brightness.

Level 2's plan (unchanged, in dependency order):

    L2.0  an analytic reference: the furnace verification mode
    L2.1  directional probes (SH-2 or a small octahedral map)
    L2.2  dynamic geometry: bake alphaMode MASK into the BLAS; compute skinning + BLAS refit for skinned meshes
    L2.3  specular / glossy GI (its own objective; it brings a denoiser problem with it)

NOT doing: a surface cache. The reasoning is measured and recorded: UE needs one because it traces 1024 rays
per probe over 16384 probes; this renderer traces 131k rays in total and shades each hit in 0.15 ms, so
"shade the hit" IS its cache. Revisit only if the ray count rises by an order of magnitude.

## 2. The immediate task: make the furnace acceptance pass

### What the mode is, and why it is the reference this project could not otherwise have

With the sun off and the environment a uniform level L, a diffuse surface's outgoing radiance is exactly
`albedo * L`, and a bounce has nothing to add because the environment already is the light. So the GI-on and
GI-off frames must agree, and any difference is an energy error rather than a preference between two of this
project's own estimators.

### What it reports today, and the attribution already done

Measured in the Sponza interior, 180 frames, validation clean, non-degenerate level:

    GI off, furnace                134.7686
    GI on,  furnace, intensity 1   127.1371        -7.63, i.e. 5.7% darker
    GI on,  furnace, intensity 0    49.4407        -85.33

So the chain's ambient SUBTRACTION removes about 85.3 on its own (intensity 0 scales the traced estimate to
nothing while the subtraction still runs), and at intensity 1 the traced estimate puts back about 77.7. The
5.7% is that gap: the traced chain restores roughly 91% of the ambient the lighting stage drops on a traced
frame. The remaining 9% is what to hunt.

### Hypotheses, in the order worth testing (each is measurable)

1. **The AO definition mismatch.** The tracer's tail multiplies its result by the G-buffer's baked AO
   (`texture(gbuffer_material, uv).b` in `shaders/ssgi.comp`), while the spatial filter's subtraction uses the
   material's AO (`shaders/ssgi_spatial.comp`, `ambient_removed_at`: `albedo_metallic.rgb * ao *
   texture(irradiance_sampler, normal).rgb * (1.0 - albedo_metallic.a)`), and the lighting stage's ambient
   carries `s.ao` as well. In the furnace the lighting stage's ambient is deliberately AO-FREE (see the
   furnace branch in `shaders/shading.glsl`), so the three definitions cannot all be right at once.
   CHEAPEST TEST: temporarily drop `* traced_ao` from the tracer's tail and re-run the acceptance. If the gap
   closes, the AO multiplication is the mechanism - and the correct fix is then to make the ambient the
   subtraction removes and the ambient the lighting stage drops the SAME quantity, which may mean the tracer
   should not apply the baked AO when it is replacing an AO-free ambient.
2. **The specular IBL a hit returns.** In furnace mode a shaded hit returns
   `vec3(furnace_level) * (fssess + fmsems)` on top of its diffuse ambient, so the traced mean exceeds L. That
   would make the frame BRIGHTER, not darker, so it cannot be the reported sign - but it is worth knowing it is
   there, because once (1) is fixed this term may become the dominant error.
3. **Half resolution and the denoiser.** The traced estimate is a half-resolution, temporally and spatially
   denoised, joint-bilaterally upsampled image. A systematic shortfall could come from the upsampler or the
   spatial filter's normalisation. TEST: the two knobs that disable them are documented measurement settings -
   `ssgi_spatial_sigma = 0` (pass-through) and `ssgi_upsample = false` (plain bilinear) - so the same furnace
   capture with each isolates their share.

### Acceptance for this task

The furnace acceptance passes: GI-on and GI-off agree within the noise of two 180-frame captures, at a
non-degenerate level, both runs validation clean. State the reason for whichever change achieves it. The
default (furnace off) must stay byte-exact - see the gates in section 5.

## 3. Then L2.1: directional probes

The measured gap this closes: a cell stores ONE RGB, so "a bright window to the left and a dark wall to the
right" average away. UE stores a 32x32 octahedral radiance map plus a depth map per probe; the study notes this
as the main structural divergence from this renderer.

Shape that fits here: **SH-2 (four coefficients per channel)** - cheap to store, cheap to project in the
injection (the cell's rays each contribute to four basis values), and the tracer's lookup becomes a dot product
with the direction it already has. The storage is the interesting part: the current grid is one RGBA16F 3D
image plus a geometry image, ping-ponged for propagation, so four coefficients need either four images or two
RGBA images per coefficient pair - and the ping-pong must carry all of them.

ACCEPTANCE, and it is worth building the test before the change: sample the SAME cell in two opposite
directions and require the two values to DIFFER. Today the representation answers identically for both by
construction, so the test must fail before the change and pass after - the cleanest before/after this project
has. The 4x4 spatial table must stay structured, and the furnace acceptance of section 2 must still pass.

## 4. Then L2.2 and L2.3

L2.2 alphaMode MASK: with inline ray queries there is no any-hit stage, so the mask is baked into the
acceleration structure - a compute pass evaluates the mask and collapses masked-out triangles to degenerate
ones in the position buffer the BLAS reads, which requires masked geometry to be expanded to three unique
vertices per triangle and made non-indexed. UE's own fallback when no any-hit shader is available is the
silent solidity this renderer has, so this is a limitation with a known fix.

L2.2 skinned meshes: genuinely larger than it looks. UE's zero-copy path depends on having a compute skinning
cache whose float3 position buffer IS the BLAS vertex buffer; this engine skins in the VERTEX shader
(`shaders/pbr.vert`), so a compute skinning pass is a prerequisite, then `ALLOW_UPDATE` at build and a per-frame
`MODE_UPDATE` refit under a triangle budget. Worth noting for morale: UE itself keeps `bRenderStatic` and
instanced-skinned meshes permanently in BIND POSE in its ray tracing scene, so the current limitation is a
documented mode in the reference.

L2.3 specular GI: currently the bounce chain is diffuse-only (the specular in a hit's shading is the direct and
IBL term). A glossy ray per pixel is the feature, and it brings a denoiser problem with it - treat it as its own
objective rather than a step.

## 5. Working discipline (non-negotiable; every item was learned the hard way here)

Gates before any commit:

* Release, Debug and ASan+UBSan builds clean (`-Werror` is on everywhere);
* `ctest` in the release build: 6/6;
* the capture harness `scripts/windows/check_render.ps1`: 7 scenarios, each run twice, 0 changed - or the
  change recorded deliberately with its reason and the baseline re-recorded;
* `doxygen Doxyfile`: exit 0 and an EMPTY warning stream. Capture the real exit code; piping doxygen into
  anything makes `$LASTEXITCODE` the pipeline's, and this was mistaken for a pass once;
* every measurement run validation clean (the harness greps for `VUID-`, `Validation Error`, `[ERROR]`,
  `panic`, `recorded out of order`).

Habits that caught real errors here:

* verify with hashes and numbers, never with assertions in prose; a capture hash, a mean, a per-tile table;
* keep every feature behind an A/B toggle so its effect is measurable, and report NEGATIVE results - several
  slices in this session ended "the measurement disproved the hypothesis", and those are the valuable ones;
* never commit what has not been verified: REVERT it and record why. Two rounds of this session ended in
  reverts, and that was the right call both times;
* when a change is supposed to be invisible, say so and check it (all seven scenarios, 0 changed).

Edit mechanics this repository punishes:

* anchors must be WHOLE LINES. Multi-line anchors silently match nothing because the working copy mixes CRLF
  and LF, and a silent miss means the build keeps using the previous binary while you read the run's numbers as
  if they described the change. Prefer replacing a unique single-line string, and ALWAYS confirm the
  replacement changed something (`if ($fixed -eq $cfg) { 'no change' }`) before reading any result. This
  specific mistake happened three times in the session that wrote this note;
* a definition appended to a `.cpp` lands OUTSIDE `namespace vulkan` (the namespace closes before EOF). Anchor
  the insert inside the namespace, on a unique function that follows the insertion point, rather than appending;
* a forced member initialiser is OVERWRITTEN by the config-driven setter that `main.cpp` calls at startup, so
  force a behaviour through the CONFIG, not the header;
* do not restructure expressions in `shaders/shading.glsl`, even with arithmetically neutral changes: TAA
  amplifies a one-ULP code-generation difference over forty frames, and it changed 2 of the 7 scenarios once.
  Force INPUTS instead (the light UBO lanes, the descriptors), which leaves the default byte-exact by
  construction;
* `git` will warn about LF/CRLF on nearly every edit; that is normal and no line-ending fixing should be run.

Commit and reporting style:

* `prefix: lowercase summary`, then a prose body that explains WHY, states the numbers, and says what failed;
  ASCII only, no BOM;
* bump the `// module version:` banner of every module whose interface changed;
* document a knob in `config.example.toml` next to the others, with its reasoning and its limits, and mirror it
  in `scripts/make_config.py`, `tests/fixtures/config_generated_defaults.toml`, `tests/fixtures/config_full.toml`
  and `tests/test_app_config.cpp`.

## 6. A note on the furnace mode's own honesty

`config.example.toml` currently says the mode is HALF wired and that turning it on gives a dark frame rather
than a reference. That was true when it was written; the mechanism is now complete (the constant cube is bound
to the IBL's two cube slots and the sun lane is off), so the warning should be replaced by the measured state:
the mode runs, and it reports a 5.7% energy error that is not yet fixed. A verification mode whose file
misdescribes it is worse than one that is absent.
