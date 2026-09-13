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

WHAT A STOCK CONFIG RENDERS NOW (the default flip; `app_config` 0.26.0, `vulkan.runtime` 0.52.0):
`ssgi = true`, `ssgi_ray_tracing = true`, `ssgi_hit_shading = true`, `ssgi_intensity = 1.0`. Those are ONE
decision and not four knobs that happened to move together: the traced path is what lets a shaded hit be
reached at all (a marched ray never leaves the frame), and it is also what turns the chain from an ADDITION
to the probe into a REPLACEMENT of the lighting stage's ambient - so the intensity had to move with it (1.0
means "use the traced estimate"; at 0.7 the ambient the spatial filter subtracts would be 30% larger than
the estimate replacing it, i.e. a systematic darkening of every frame). The MARCHED chain keeps 0.7 as its
own reconciling value, and `ssgi_radius` stays at 0.12 because it is the marched fallback's step size
(radius / `ssgi_steps`) that pins it - a reach that suits the traced path would make the fallback step over
the detail between two samples. `ssgi_specular` is ON as of this note: its blocker (the denoiser item) was
closed when the reflection got an accumulation of its own, and its default-on case was then measured on the
stock path - +1.50 of mean green over 8.67% of the material sweep, with the shipped single ray's per-pixel
estimate differing from an eight-ray convergence on 7.37% of pixels (a bias, not flicker: TAA does not move it).
`ssgi_probes` remains OFF deliberately: the cache is a coarse SH-2 approximation whose contribution is measured
in BOTH of its roles now (L2.1's fallback tables, and the hit-ambient correction, whose -0.47% interior
darkening is ordered by the scene but is a judgement rather than a win). `ssgi_bounce` also stays at 0.0, for
the same measured reason.
THE COST IS MEASURED, not estimated: on Sponza at 1080x960 the shipped default takes the frame from 1.00 ms
to 1.81 ms of GPU time (+0.81 ms, +81%), of which the chain's own interval is 0.86 ms and the lighting stage
gives 0.16 ms back (with `gi_replaces_ambient` set it no longer adds an ambient the spatial filter is about
to remove). The per-interval table and the arm identities are in `docs/gi_hit_shading.md`'s L2.4 section.
FLIPPING A DEFAULT CHANGES EVERY FRAME, so the gate grew a `default_gi` scenario (the compiled defaults with
nothing overridden - until it existed, every scenario either pinned GI off or spelled out the traced keys, so
nothing rendered the configuration a user actually gets) and every non-GI scenario now pins `ssgi = false` so
its reference stays exactly what it was. The one exception is deliberate: `unlit` gets NO pin, because the
flip exposed that the chain could run on that render mode at all - the flat mode's shading stage outputs the
stored albedo, so the image the tracer would average is not radiance and the GI it added was a product of two
albedos rather than a transport term. `runtime::ssgi_active()` now excludes it, and that scenario's unchanged
reference is the proof. The cost table, the two defects and the verification are in
`docs/gi_hit_shading.md`'s L2.4 section.

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
expressions that already existed. `[render] ssgi_specular` (ON since the default flip; it was off while its
denoiser item was open) and `ssgi_specular_rays` (1-8, default 1) are the knobs.
WHAT IT MEASURED. The control is the sharpest instrument in this whole document: where a ray finds nothing the
correction the pass writes is exactly zero, so with a ray length too short to reach anything the lobe-on frame
and the lobe-off frame are the SAME SHA256 - 0 pixels differing, with the denoiser in the loop and with it
bypassed. That is a bit-exact invariant rather than a tolerance, and it is the RESULT of this step, not its
starting point: the first version subtracted the lighting stage's term in the spatial filter, which subtracts a
centre-pixel value from a filtered one, and the same control read mean +0.0255 with 4.4% of pixels beyond
4/255. Both designs were built and measured; the second won on every axis and is what shipped. On Sponza the
effect is -1.3264 of mean green (-2.4%) in a 4x4 table ordered by the SCENE (interior tiles -7% to -16%,
sky-facing 0.1-0.3%); isolating the specular channel alone gives -20% to -33% in the interior against -0.4% at
the sky. One ray is already converged (1 vs 4 rays differ by +0.0028 against the feature's -1.33), and the cost
is +0.13 ms at one ray / +0.81 ms at four on the GPU timings' `gi` interval. `docs/gi_hit_shading.md`'s L2.3
section has all of it.

WHAT IS LEFT OF L2.3, and it is NOT the denoiser problem the plan predicted - that hypothesis was measured
and killed. At a ray length that reaches anything (radius 0.5 on the material sweep) the reflection's structure
survives the shared joint-bilateral filter; the earlier "no reflection visible" reading was the RAY LENGTH,
because `ssgi_radius` is a fraction of the scene radius and the 0.12 that reaches 2.23 units inside Sponza
reaches 0.84 on a compact scene (measured curve: +0.47 / +0.71 / +0.89 / +1.06 at radius 0.12 / 0.25 / 0.50 /
1.00). What is actually open:
(1) THE FEATURE IS NEARLY INVISIBLE ON THE SCENE EVERY OTHER GI MEASUREMENT USES (Sponza is roughened stone,
its roughness channel averages 217/255), which is why the gate grew a `metal_rough_glossy` scenario and why the
L2.3 evidence that matters is a MATERIAL-ordered table - smooth metal +10.33, rough metal +7.61, smooth
dielectric +1.23, rough dielectric +0.13 - rather than a tile table.
(2) the reflection is a point sample of the roughness cone AND it is accumulated by the DIFFUSE temporal
resolve, which reprojects its history by the SURFACE's motion - correct for a bounce, structurally wrong for a
reflection, whose image slides at its own rate. The instrument for that now exists (the camera sweep and the
`glossy_motion` scenario: every capture in this repository used to have a still camera, which exercises every
reprojection path in its trivial case only) and the baseline is a difference of differences: the reflection
loses **0.4029 of mean|.|** - about 40% of its own magnitude - to a 20-degree camera orbit, against 0.5198 for
the whole rest of the frame. Read that number with its caveat: a moving accumulation differs from a converged
static one even with perfect reprojection. HALF OF IT IS NOW CLOSED, and measured: the CLAMP half of that
mechanism is in `shaders/ssgi_temporal.comp` (a smooth pixel may keep ~2 frames, loosening with roughness,
keyed on the roughness the G-buffer already carries), and it took the number from **0.4029 to 0.3492** and its
worst 4x4 tile from 1.942 to 1.488 - with the residual now confined to the sweep's SMOOTH columns and the
rough ones BIT-IDENTICAL, and with the whole procedure run on the build without it as the control, which
reproduced 0.4029 exactly. The other half is open: the reflection still has no history of its own, so it rides
the surface-motion reprojection (shortened rather than corrected) and the clamp also shortens the DIFFUSE
signal on those pixels. The instrument is `scripts/measure/motion_dd.py` now, and its pose rules are in the
script's own docstring, because an arm posed by hand yields a plausible number rather than an obviously wrong
one. MECHANISM 1 IS IN, so the reflection's motion handling is the reference's mechanisms 1+2 together: the lobe
publishes its reprojection (`gi_spec_images` and `gi_spec_reproject_images`, G-buffer bindings 13/14, from
`camera.prev_view_proj`), and the temporal resolve runs a SECOND time in `mode 1` with a history of its own -
`previous_uv` from that reprojection, the disocclusion test comparing the depth of the point the reflection FOUND
(carried in the history image's alpha), the reflection's own screen speed as the blend signal, and the roughness
cap binding there ALONE. Measured with `scripts/measure/motion_dd.py`, three builds side by side: the
reflection's own motion loss 0.4029 (no cap) -> 0.3492 (cap on the shared resolve) -> **0.3068** (its own
history), worst 4x4 tile 1.942 -> 1.488 -> 0.848, and the lobe-off arms back to their uncapped 0.6131. The
verification worth quoting is a hash rather than a mean: the two traced lobe-OFF gate scenarios came out at their
PRE-CAP reference hashes byte-identically, which proves the cap left the diffuse path and that nothing else in the
split perturbed them. `docs/gi_hit_shading.md`'s L2.3 section has the tables, the three traps it cost (a single
first-use flag for a per-image resource - third occurrence; a descriptor bound to an image whose layout only one
path maintains; and `0.0 * undefined` not being zero), and what is still open: mechanisms 3 and 4, which no
measurement has asked for yet. Its REACH is no longer on the list: it has its own
knob now (`ssgi_specular_radius`, default 0.5
of the scene radius), because the shared `ssgi_radius` was pinned low by the marched path's step size and the
lobe was realized only 39% of the signal available - half of the remainder comes back at 0.5 for +0.12 ms
isolated (+45% effect: -1.4756 -> -2.1411), and past it the curve is flat in cost and in effect both.
(3) the DIFFUSE subtraction still carries the non-mean-preserving-average artifact (+0.33 convex / +0.52
Sponza) and cannot use the mechanism that fixed the specular one, because its image has to stay a radiance for
the live bounce - fixing it means filtering the removed term with the same weights as the added one (25 gathers
a pixel) and re-baselining every capture. Two things LEFT this list in the same step: SSAO no longer leaks an
`ambient * (ssao - 1)` term into traced frames (-1.90 of mean green before the fix; an SSAO-on and an
SSAO-off traced frame are now bit-identical), and neither lobe issues a ray query with an empty interval any
more (`RAY_TMIN` is a named constant in both, used by the guard and by `rayQueryInitializeEXT`, so the
"nothing reachable" identity configuration is deterministic instead of relying on what an invalid query
happens to return).
(4) the ray-origin bias is now fixed on every ray this work could reach - the two traced lobes' own origins and
the shadow ray a shaded hit fires - and the MARCHED path keeps its fraction-of-the-ray-length form on purpose
(there it is the step size's own scale). Those fixes were among the largest errors the traced path had: the
diffuse origin alone moved 35.57% of `sponza_gi`'s pixels and its removal made the traced GI 0.27 of mean green
DARKER, i.e. it had been over-bright because its rays started past the geometry beside them and fell back to the
environment probe. THE DISTANCES ARE SMALLER THAN AN EARLIER DRAFT OF THIS NOTE SAID and the measurements are
the same ones: Sponza's scene radius is 18.548 (not Fox's 87.775), so the bias was 0.045 world units at the
traced default, not 0.21. That a sub-decimetre change in where rays START moves a third of the frame is the
finding; quoting the wrong scene's radius was a trap, and it is recorded in section 5.

## 5. Working discipline (non-negotiable; every item was learned the hard way here)

Gates before any commit:

* Release, Debug and ASan+UBSan builds clean (`-Werror` is on everywhere);
* `ctest` in the release build: 6/6;
* the capture harness `scripts/windows/check_render.ps1`: 12 scenarios, each run twice, 0 changed - or the
  change recorded deliberately with its reason and the baseline re-recorded. A new subsystem gets its own
  scenario rather than a note: `sponza_gi` exists because until L2.1 every scenario ran with `ssgi = false`,
  so the whole GI chain (its denoisers, the probe cache, everything ray-traced) had no coverage and a break
  in it would have passed this gate; `metal_rough_glossy` is the L2.3 one (Sponza is roughened stone
  everywhere, so the lobe is only visible on the material sweep - and `sponza_gi` does not even enable hit
  shading, so the lobe would otherwise have been exercised only in its OFF state, which is byte-identical by
  construction and therefore covered by nothing); `sponza_march` covers the MARCHED chain and its "addition,
  not replacement" semantics, which the traced scenarios cannot; and `default_gi` is the L2.4 one - the
  compiled defaults with nothing overridden - which exists because every scenario either pinned GI off or
  spelled out the traced keys, so NOTHING rendered the configuration a user actually gets, and a default that
  had become unreachable or had silently drifted would have passed. Its A/B twin is `deferred`: same model,
  same camera, same frame count, one key apart.
  WHEN A DEFAULT MOVES, every scenario that is not about it PINS THE OLD VALUE (`ssgi = "false"` on the
  non-GI scenarios since the L2.4 flip), because a reference frame is a claim about the path the scenario
  names - letting an unrelated default change into it invalidates eleven references at once for a reason none
  of them is about. A scenario's `extra` should also not restate a compiled default: a key equal to the
  default is a claim of a difference that does not exist, and it hides the day the default moves. The
  deliberate exception is `unlit`, left UNPINNED so that its UNCHANGED reference is what proves the L2.4
  render-mode gate (`runtime::ssgi_active()` excludes the flat mode, whose shading stage outputs albedo
  rather than radiance). A new scenario's reference is seeded once per machine with `-Update`, and with
  `-Only <name>` so the blanket `-Update` cannot silently re-seed every other reference;
* `doxygen Doxyfile`: exit 0 and an EMPTY warning stream. Capture the real exit code; piping doxygen into
  anything makes `$LASTEXITCODE` the pipeline's, and this was mistaken for a pass once;
* every measurement run validation clean (the harness greps for `VUID-`, `Validation Error`, `[ERROR]`,
  `[WARNING]`, `panic`, `recorded out of order` - `[WARNING]` was added after a missing descriptor-pool
  type turned out to be reported on EVERY run while the gate read clean).

Habits that caught real errors here:

* verify with hashes and numbers, never with assertions in prose; a capture hash, a mean, a per-tile table.
  The instruments are in the repository now (`scripts/measure/`, indexed by its README): `mean.py` for the
  per-channel means every number is quoted in, `diff.py` for a per-pixel difference and its 4x4 tile table,
  `tiletab.py` for that table as a percentage of the frame it is measured against, `corr.py` for whether two
  estimators moved in the same PLACES, `spec_material.py` for an A/B bucketed by G-buffer material,
  `shadow_pose.py` for the animated-pose difference-of-differences, and the `mask_*.py` three for the
  alphaMode MASK question. They used to live in the build directory, which meant a `-Clean` took the ability
  to re-derive every recorded table with it;
* keep every feature behind an A/B toggle so its effect is measurable, and report NEGATIVE results - several
  slices in this session ended "the measurement disproved the hypothesis", and those are the valuable ones;
* A FIXTURE THAT CLAIMS TO BE A COPY OF A TOOL'S OUTPUT HAS TO BE DIFFED AGAINST THE TOOL, because a test that
  reads the fixture cannot see the tool at all. `tests/fixtures/config_generated_defaults.toml` says it is
  verbatim `scripts/make_config.py` output with every question answered by its default, and the test that
  reads it is the guard that the generator and the parser still agree - but when it was finally diffed
  against the generator (while the L2.4 defaults moved; `python scripts/make_config.py <tmpdir>` with blank
  answers, then compare) FOUR values had drifted: `max_fps` 240 against the generator's 0, `taa` and `fxaa`
  true against its false, and `camera_fit` missing from the fixture entirely - so the guard had a hole for
  the one key, and its two boolean checks could not have failed. Regenerating it is one command; doing that
  belongs in the checklist for any change to the generator, and the fixture's own header now says so;
* never commit what has not been verified: REVERT it and record why. Two rounds of this session ended in
  reverts, and that was the right call both times;
* when a change is supposed to be invisible, say so and check it (all ten scenarios, 0 changed) - and when a
  change is supposed to be invisible on ONE PATH of the renderer, say which scenario proves it
  (`sponza_march` came out byte-identical through two changes that moved every traced reference, which is a
  stronger statement than the whole gate passing).
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
* `git` will warn about LF/CRLF on nearly every edit; that is normal and no line-ending fixing should be run;
* THE SCENE RADIUS IS PER ASSET AND IT IS EASY TO QUOTE THE WRONG ONE: it is `scene_radius`, printed both as
  "scene bounds (aabb): ... radius N" and as the light frustum's radius, and it is what `ssgi_radius`
  multiplies. Sponza's is 18.548, the metal/roughness sweep's is 6.995, and Fox's is 87.775 - because that
  sample asset carries a huge ground plane. An earlier draft of the L2.3 origin section took Fox's figure for
  Sponza and reported a 0.045-unit bias as 0.21, i.e. off by 4.7x, in a document whose whole point is that its
  numbers can be re-derived. The A/B measurements were unaffected (same code, two values), but every
  "world units" claim in that section had to be recomputed - so when a number is a CONVERSION, print the
  quantity it was converted FROM next to it.

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
