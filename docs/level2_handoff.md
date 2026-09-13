# Level 2 handoff: state, plan, traps

This is a handover note for an agent continuing the GI work in this repository. It assumes no memory of the
session that produced it. Everything here is either a verified number from that session or a pointer into the
repository; where something is unverified it says so.

## 0. Read these first, in this order

* `docs/gi_hit_shading.md` - the full record. Level 1 (leak fix, view independence, reset) with the measurement
  that closed each; Level 2's four-step plan; the whole history of the furnace verification mode, ending in the
  section that records what it measured, the energy error it found, and why its acceptance is an exactness test
  only a scene its premise holds in can pass; the L2.1 section, which records the SH-2 probes, the direction A/B
  that proves they are directional, and their stated limits; and every trap that cost time.
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

L2.2 is DONE, both halves, and one of them is a negative result: the MASK bake is measured and OFF by default,
the skinned refit is measured, works, and is also OFF by default (a knob, so its cost and its effect stay
measurable). L2.3's first slice is done too - the glossy reflection works and is measured, OFF by default -
and what remains of it is the subtraction-site/denoiser work that its own measurement pointed at. See the
L2.3 paragraph in section 4.

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

## 3. L2.1: directional probes - DONE, with the planned shape and the planned test

The gap it closed: a cell stored ONE RGB, so "a bright window to the left and a dark wall to the right"
averaged away. UE stores a 32x32 octahedral radiance map plus a depth map per probe; the study calls that the
main structural divergence from this renderer.

WHAT WAS BUILT, exactly the shape this section predicted: **SH-2, four coefficients per channel**, in four
RGBA16F 3D images per ping-pong side (eight images, 2 MB, 32^3 cells). The basis is one shared include
(`shaders/probe_sh.glsl`) so the projection and the reconstruction cannot disagree; the tracer's lookup is a
dot product with the direction it already had; the probe pass's projection is four basis values per ray. The
4*pi in the projection is what makes a uniform field reconstruct as itself, i.e. the furnace's identity.

THE TEST WAS BUILT FIRST, and it is the cleanest before/after this project has, because it needed an A/B that
changes nothing but the cache's direction and the probe gain's SIGN provides one: |gain| is the gain, a
negative gain looks the same cell up along the opposite ray direction, and the far-field term stays sampled
along the ray. Two captures at +1 and -1 are therefore identical unless the cache is directional:

    before the change   byte-identical (same SHA256, mean green 60.0420), as the representation required
    after the change    hashes differ; 234921 pixels (22.66%) differ, up to 14 of 255

The 4x4 spatial table stayed STRUCTURED (-0.03% to -7.53%, ordered by the scene, against the recorded -0.14%
to -6.78%), the furnace acceptance still passes (-0.0001 on the convex scene), the captures are deterministic,
and the gates are green. `docs/gi_hit_shading.md`'s L2.1 section has the tables, the numbers and the stated
limits - the directional terms are four rays per frame and converge over ~100 frames, the reconstruction
clamps at zero, and a cell still holds one surface offset rather than a per-direction depth map.

NOT DONE, and worth knowing before building on it: the gate now runs ONE scenario over this path
(`sponza_gi`), which catches a break but not a subtle regression - the L2.1 measurements themselves are the
evidence for a GI change, and `docs/gi_hit_shading.md`'s tables are the instrument they are read with.

## 4. Then L2.2 and L2.3

L2.2 alphaMode MASK: BUILT AND MEASURED, AND IT IS OFF BY DEFAULT because the measurement was negative.
`shaders/mask_bake.comp` resolves the mask before the build and writes an expanded, non-indexed copy of the
masked geometry with the empty triangles collapsed, exactly the mechanism this section described; the
plumbing is proven exact (baking with the rule disabled is byte-identical to not baking), but the
per-triangle RULE moves a ray-traced shadow 1.29 of mean brightness AWAY from the raster reference on a
MASK-heavy sample asset - it deletes triangles the raster path still shadows, because a triangle cannot
represent a per-pixel mask. `[render] rt_mask_bake` defaults to false and exists so the next attempt can be
measured; `docs/gi_hit_shading.md`'s L2.2 section has the numbers, the three rules compared, and what a
better answer needs (subdivision along the mask boundary, or opacity micromaps).

L2.2 skinned meshes: DONE AND MEASURED, and it works. `shaders/compute_skin.comp` is the compute skinning
pass this section called a prerequisite - one invocation per vertex, the same deformation `shaders/pbr.vert`
does, writing a 32-byte object-space record that keeps the primitive's vertex order. `ALLOW_UPDATE` at build
plus a per-frame `MODE_UPDATE` refit (`acceleration_structure::record_update`) is the second half, and the
refit is legal for exactly the reason the plan predicted: a skinned mesh's positions change every frame while
its triangle count, vertex order and index buffer do not. The acceptance written down before the fix was that
the baseline table collapse, and it does: the traced shadow's pose dependence goes from 0.7296 of mean
absolute green (worst tile +3.780, 1953 pixels where the raster shadow moved and the traced one did not) to
0.0039 (worst tile -0.008, 103 pixels), i.e. to the ordinary traced-versus-cascade difference, and the traced
path's own pose dependence now equals the raster path's to four decimals (+0.4845 against +0.4845). A third
pose pair reproduces it independently. Cost, measured on the engine's own GPU pass timings: +0.04 ms on Fox -
ONE skinned caster of 576 triangles and 24 joints - so the number is a floor, not a budget; a scene with many
or heavy skinned casters is unmeasured. `[render] rt_skin_bake` defaults to false (the pass runs every frame,
so the flag is read every frame and can be flipped at any time), and the knob-off frame is byte-identical to
the baseline capture, which is the control that proves the addition is inert when it is not asked for. Morph
targets are NOT covered: they are the step before skinning and the pass does not read their deltas.
`docs/gi_hit_shading.md`'s L2.2b section has the tables, the cost, and the four traps this cost (a leaked
pipeline layout, a missing `COMPUTE` stage flag on binding 9, a false-positive no-op test caused by the wrong
camera, and a config key appended into the wrong TOML table).

L2.3 specular GI: THE FIRST SLICE IS DONE AND MEASURED, and it came out well. `shaders/ssgi_spec.comp` is a
pass of its own between the tracer and the denoiser: a GGX-sampled reflection ray per pixel, written into the
TRACER'S image (read, add, store back), with its hit shaded from its own geometry - so it needs
`ssgi_hit_shading`. `shaders/ssgi_spatial.comp` then removes the lighting stage's specular ambient for those
pixels, which makes the traced reflection a REPLACEMENT rather than an addition; both sides of that
subtraction come from a new shared `shaders/ibl_specular.glsl`, which also collapsed the two copies of those
expressions that already existed. `[render] ssgi_specular` (off by default) and `ssgi_specular_rays` (1-8,
default 1) are the knobs.
WHAT IT MEASURED. The control is the sharpest instrument in this whole document: where a ray finds nothing the
estimate IS the lighting stage's term, so with the filter bypassed and a ray length too short to reach
anything, the feature on and off must agree - and on an isolated model they do to ONE 8-bit step on 0.08% of
pixels and nothing beyond it (the half-float round trip through the trace image). On Sponza the effect is
-1.3860 of mean green (-2.5%) in a 4x4 table ordered by the SCENE (interior tiles -7% to -17%, sky-facing
0.1-0.3%), which is what a correct local reflection looks like; isolating the specular channel alone gives
-20% to -33% in the interior against -0.4% at the sky. One ray is already converged (1 vs 4 rays differ by
+0.0028 against the feature's -1.3860), and the cost is +0.13 ms at one ray / +0.81 ms at four on the GPU
timings' `gi` interval. `docs/gi_hit_shading.md`'s L2.3 section has all of it.

WHAT IS LEFT OF L2.3, and it is the denoiser problem the plan warned about, now with a number. The
subtraction happens after the joint-bilateral filter has averaged the estimate while the removed value is the
centre pixel's own, so "a ray that misses changes nothing" is exact only with the filter bypassed: with it on
and still nothing reachable, an isolated model moves by mean +0.027 with 1.0% of pixels beyond 4/255 and a
worst pixel of 109. The DIFFUSE subtraction has carried the same artifact since it was written (the recorded
+0.33 convex / +0.52 Sponza note). TWO FIXES ARE ON THE TABLE and `docs/gi_hit_shading.md`'s L2.3 section has
the trade-off: filter the removed value with the same weights as the added one (architecturally consistent,
costs 25 gathers per pixel), or have the glossy pass write the NET correction `E - ibl_specular` and drop the
subtraction (exact for zero cost, but puts a bookkeeping term into an image the multi-bounce feedback re-emits
- the property the L1 work fought for). Either way it re-baselines every GI capture, so it is its own step.
Also open: the reflection is a point sample of the roughness cone (a low-roughness reflection aliases at half
resolution), and its ray length is the diffuse bounce's radius rather than a reflection's own reach.

## 5. Working discipline (non-negotiable; every item was learned the hard way here)

Gates before any commit:

* Release, Debug and ASan+UBSan builds clean (`-Werror` is on everywhere);
* `ctest` in the release build: 6/6;
* the capture harness `scripts/windows/check_render.ps1`: 8 scenarios, each run twice, 0 changed - or the
  change recorded deliberately with its reason and the baseline re-recorded. Until the L2.1 step every
  scenario ran with `ssgi = false`, so the whole GI path (the screen-space chain, its denoisers, the probe
  cache, everything ray-traced) had no coverage and a break in it would have passed this gate; `sponza_gi`
  exists for that, and a new subsystem should get its own scenario rather than a note. A new scenario's
  reference is seeded once per machine with `-Update`;
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
* when a change is supposed to be invisible, say so and check it (all eight scenarios, 0 changed).
* A CAPTURE OF AN ANIMATED SCENE IS NOT REPRODUCIBLE BY DEFAULT: playback is driven by the wall clock
  (`frame_clock::delta_seconds`), so two runs of one animated config differ (measured: identical means to
  four decimals, different pixels). `[render] animation_time = <seconds>` pins the pose - it sets the time
  and pauses, the same thing the overlay's time slider does - and with it two runs are BYTE-IDENTICAL
  (verified). Any measurement of skinned or morphed geometry needs it, which is what the L2.2 skinned-mesh
  work runs into first. The knob is documented in `config.example.toml` and mirrored through the config
  chain like every other key.
* ONE UNEXPLAINED FLAKE IS OPEN, and it is recorded rather than explained away: on the first full gate run
  after the L2.2b build, `deferred`'s two runs disagreed (`EAA61CED585471B7..` vs `2DD1D13857322C0F..`, the
  second being the reference), while the next full gate run was clean and five consecutive direct runs of that
  exact config hashed `2DD1D13857322C0F` every time. That scenario runs with ray-traced shadows and GI OFF, so
  nothing the L2.2b change adds is on its path; the harness's own verdict stands ("this scenario cannot be a
  regression check until it is deterministic"), and the next session to see it should capture the differing
  image rather than re-run until it passes.

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
