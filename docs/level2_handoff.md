# Level 2 handoff: state, plan, traps

This is a handover note for an agent continuing the GI work in this repository. It assumes no memory of the
session that produced it. Everything here is either a verified number from that session or a pointer into the
repository; where something is unverified it says so.

## 0. Read these first, in this order

* `docs/gi_hit_shading.md` - the full record. Level 1 (leak fix, view independence, reset) with the measurement
  that closed each; Level 2's four-step plan; the whole history of the furnace verification mode, ending in the
  section that records what it measured, the energy error it found, and why its acceptance is an exactness test
  only a scene its premise holds in can pass; and every trap that cost time.
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

## 2. The furnace acceptance: what was asked, what it measured, where it passes

### What the mode is, and why it is the reference this project could not otherwise have

With the sun off and the environment a uniform level L, a diffuse surface's outgoing radiance is exactly
`albedo * L`, and a bounce has nothing to add because the environment already is the light. So the GI-on and
GI-off frames must agree, and any difference is an energy error rather than a preference between two of this
project's own estimators.

That identity has a premise, and the rest of this section is what it costs: it holds when the incident
radiance really is L from EVERY direction, which a convex object satisfies and a building's interior does
not.

### What it reports, and what the gap turned out to be

Measured in the Sponza interior, 180 frames, validation clean, non-degenerate level - and reproduced exactly
in this step before anything was changed:

    GI off, furnace                134.7686
    GI on,  furnace, intensity 1   127.1371        -7.63, i.e. 5.7% darker
    GI on,  furnace, intensity 0    49.4407        -85.33

The first reading of that - "the traced chain restores 91% of the ambient, so the missing 9% is an energy
error" - is WRONG, and the three hypotheses below were the wrong three. The gap is the geometry the rays
reach: a ray that lands on a wall comes back with the wall's radiance, and in a furnace that is
`albedo_wall * L`, below L. The chain is reporting an occlusion-corrected ambient, against a reference that
has no occlusion in it at all (Sponza carries no occlusion textures, so the lighting stage's ao lane is 1
everywhere). Two things follow, both measured. The mode's acceptance can only pass on a scene whose traced
rays cannot reach absorbing geometry - and the chain's first real finding is somewhere else entirely: a
missing `(1 - metallic)` on the traced diffuse estimate, which the `ssgi_radius = 0` configuration isolated
to the last digit. `docs/gi_hit_shading.md` has the numbers, the fix's before and after, and the restated
acceptance.

### The three hypotheses, and what each measured

1. **The AO definition mismatch** - VOID, by inspection rather than by test. Sponza has no occlusion
   textures, so `texture(gbuffer_material, uv).b` in the tracer and `ao` in the filter's subtraction are the
   same 1.0 at every pixel and the factor cancels. Dropping `* traced_ao` could not have moved anything on
   this scene. (The handoff's own text was wrong about `shading.glsl` too: there is no furnace branch in it,
   and `furnace_level` is declared there and never read - the mode is forced entirely by the descriptor
   layer and the sun lane.)
2. **The specular IBL a hit returns** - real, and in the wrong direction: it makes a traced frame BRIGHTER,
   while the measured sign is darker. It matters only through the hit-shading A/B, where a shaded hit
   returns `albedo * L + its own specular` and a screen-sampled confirmed hit returns the lighting stage's
   value for the same surface, which is the same two terms.
3. **Half resolution and the denoiser** - REAL and measured to be small. On the convex Cube the whole
   residual is the denoiser (`ssgi_spatial_sigma = 0` takes +0.3278 to -0.0045, and the bilateral upsample
   moves the remainder), and on Sponza the spatial filter contributes +0.52 of the -7.63.

### Acceptance, restated where it is well posed

The original bar - GI-on and GI-off agree "within the noise of two 180-frame captures", at a non-degenerate
level - is in practice an EXACTNESS test, because two runs of one config are byte-identical (verified by
SHA256). It is met on a scene the traced rays cannot reach absorbing geometry in, which is where the mode's
identity really holds:

    convex scene, real rays, 180 frames:  GI-on against GI-off  -0.0001 of mean brightness,
                                          122 of 1036800 pixels off by one 8-bit step.

Before this step's change the same scene read +1.4005, so the bar is met by a fix rather than by a widened
tolerance. In an interior it cannot be met, and the number there measures the scene's occlusion rather than
the chain's energy. The default (furnace off) stayed byte-exact - see the gates in section 5.

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
  `[WARNING]`, `panic`, `recorded out of order` - `[WARNING]` was added after a missing descriptor-pool
  type turned out to be reported on EVERY run while the gate read clean).

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

## 6. The furnace mode's own honesty, now settled

The stale "HALF wired / dark frame" warning this section asked for had already been replaced by the time
this note was written (the cube's level was wired in f06c83a): `config.example.toml` describes the mode as
what it is, and it now carries the measured state as well - where the identity holds, where an interior
legitimately differs, and the bug the mode found. The 5.7% is no longer called an unfixed energy error,
because it is not one. A verification mode whose file misdescribes it is worse than one that is absent; so is
a handoff that describes a finished investigation as pending.

One measured residual is deliberately left alone rather than "fixed": the spatial filter's weighted average
is not the identity on a field that varies (+0.33 of mean brightness on the convex Cube, +0.52 on Sponza).
That is a property of the denoiser - a bilateral average whose subtraction belongs to the CENTRE pixel - and
not an energy error in the ray path, which the `ssgi_spatial_sigma = 0` bypass proves by making the convex
furnace exact to one 8-bit step. Recording it with its A/B is the honest treatment; redesigning the filter
to make a test pass would not be.
