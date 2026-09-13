# Hit shading: where the GI chain stands and what is left

This records the state of the work that made a traced GI ray able to shade the surface it lands on
instead of sampling the screen, the measurements that decided each step, the traps that cost time, and
exactly what the next step needs. It exists because the findings were spread across commit messages and
code comments, and the code comments are the only part a reader of the source would find.

## Status

| Step | State |
| --- | --- |
| 0. Measure the probe cache's contribution | done, committed, recorded (negative result) |
| 1. Shade a hit from its geometry | done and measured; one question open (a model difference, see below) |
| 2. Wire it into the probe grid's injection, gate the propagation with shadow rays | NOT STARTED |
| 3. Reference frame and per-pixel error metric | instrument done and used; an independent reference is missing |

## What is built

* The acceleration structures' instance table carries what a hit needs: the vertex and index buffers'
  device addresses, the object-to-world matrix, the stride, the index type and the material index
  (`vulkan.acceleration_structure::instance_record`, 96 bytes, with a static_assert pinning the layout
  the shader declares).
* `shaders/ssgi.comp` shades a hit when the runtime publishes that table's device address: it takes the
  instance from `instanceCustomIndex`, fetches the triangle's three vertices through
  `GL_EXT_buffer_reference`, interpolates position, normal and UV with the hit's barycentrics, solves the
  tangent frame analytically from the triangle's own edges and UV deltas, fetches the material and its
  textures from the scene set, and evaluates the sun with a shadow ray plus the split-sum IBL ambient and
  the material's emissive.
* The switch is the table's own address, riding two push-constant lanes freed for it. Zero means "sample
  the screen", which is the default, the A/B, and what a marched GI path or a device without ray queries
  keeps. No descriptor had to be added: the geometry is reached through addresses the acceleration
  structure build already required.
* Scene set bindings 1, 2, 4 and 5 (the bindless texture array, the prefiltered environment, the BRDF LUT
  and the material table) now name `COMPUTE` as well as `FRAGMENT`.

## Measurements, and what each one settled

All on the Sponza interior, static camera, traced GI, `ssgi_intensity` 1.0, mean green over the frame
unless stated otherwise.

* Shading a hit changes the frame by a mean of **-1.115** (61.3964 to 60.2814), first-order rather than a
  small bias, and bucketed by the baseline's own luminance it lands hardest on the middle third
  (-5.47%), which is where surfaces lit by geometry the camera cannot see live.
* **On hits the camera CAN see, shading agrees with the lighting stage to +0.22** (0.36%). That is the
  result that validates the surface evaluation, the material and texture fetch, the analytic tangent
  frame and the IBL composition. It was obtained by shading only the hits the screen confirms - one block
  moved below the depth guard, nothing else changed, then reverted.
* The rest of the difference (-1.29) is therefore **entirely** in the hits the screen path refuses to
  answer: it discards them and hands them probe ambient, while the shaded path answers them with their own
  shading - shadowed, interior, and legitimately darker than a sky estimate. This is a MODEL difference,
  not noise, and it is not established which model is right.
* Three explanations for that difference were tested and **disproven by measurement**, which is worth
  recording as much as the surviving one: the shadow bias was quadrupled (+0.045 of the 3.23 the ray
  removes), the bias was aimed along the ray's side (-0.03), and hits on back faces were rejected
  (+0.08 of the 1.29). None of them is the population responsible.
* Cost is not resolvable at the 60-frame window: the GI pass reads 0.77 ms with hits read from the screen
  and 0.73-1.02 ms with them shaded, against 1.06-1.22 ms for one shadow ray per FULL-resolution pixel in
  the ray-traced shadow pass. A longer window is needed to say anything.
* Against a **converged shaded estimator** (16 rays per pixel over 600 frames): a 2-ray screen-path frame
  differs by MAE 1.55 / RMSE 3.04, a 2-ray shaded frame by MAE 0.42 / RMSE 0.76. This is NOT ground truth
  - the reference is one of the two variants - but it splits the shaded path's own noise (0.42, small)
  from the screen path's model difference (1.55).

## Step 2: what it needs

The goal is that the world-space probe cache stop being injected from the screen, so its contents no
longer depend on what the camera happened to show.

1. **Share the shading code.** The shaded surface evaluation lives in `shaders/ssgi.comp` and needs to be
   available to `shaders/gi_probe.comp`. The blockers are structural, not cosmetic: `ssgi.comp` declares
   the camera UBO and the irradiance cube that `shaders/surface.glsl`/`shading.glsl` also declare, so the
   three cannot simply include each other today. The shapes that would work: extract the hit-shading
   block plus its bindings into its own include unit that both passes include (which means moving the
   shared declarations out of `shading.glsl` and `ssgi.comp` into it), or give the probe pass its own
   copy (which duplicates the material and IBL evaluation - the thing that made this project keep
   `shading.glsl` shared in the first place).
2. **Give the probe pass what shading needs.** Its pipeline layout currently has ONE set (its own) and no
   scene set, no TLAS and no push block for the instance table address. Injecting from shaded rays means:
   a second set layout (the shared scene set), the TLAS binding (16, already `COMPUTE`), the instance
   table address in its push block (it has room), and a ray-query-capable shader variant.
3. **Occlusion-gate the propagation.** The 3x3x3 gather is currently unoccluded, which is what leaks
   light through thin geometry and spreads uniformly instead of following the light (measured: the
   propagation delivers about 84% of the cache's contribution, with the same relative weight in the
   darkest and middle thirds of the frame). A shadow ray between adjacent cell centres is the fix, and
   the probe pass would then need the TLAS anyway - so steps 2 and 3 share their plumbing.
4. **Acceptance.** The tertile signature is the instrument: a cache injected from shaded rays should stop
   adding a fixed percentage of every region's brightness and start adding light where the off-screen
   geometry illuminates. The measured baseline for the current (screen-injected, unoccluded) cache is in
   `config.example.toml` next to `ssgi_probes`.

## Step 3: what is missing

The instrument exists - per-pixel MAE/RMSE against a converged reference, at 16 rays per pixel over 600
frames. What it does not have is a reference that is neither of the two variants being compared, so it
currently separates noise from model difference rather than measuring either against the truth. A
brute-force or path-traced answer is what would close it, and it is the only way to settle the open
model question above.

## Traps worth not re-learning

* **`VkIndexType` names `UINT16` as 0 and `UINT32` as 1.** The first version tested for 1 as the 16-bit
  case, so Sponza's 32-bit indices were read as half-words: that does not mis-shade a triangle, it reads
  a different index whose vertex lies outside the buffer. The symptom was not a validation error or a
  crash - the process exited 0, the log was clean, and no frame was ever produced, three runs in a row.
* **glslc rejects a trailing `const` on a declaration** (`float const x`), which `shaders/fxaa.frag`
  already recorded. GLSL also refuses `const` on a buffer-reference-typed local.
* **Buffer references need an aligned address.** A buffer's base address is aligned; an offset into one
  need not be, so the instance record stores bases and the shader does the arithmetic in bytes.
* **Widening a stage mask is how a compute pass reaches the scene's materials and textures.** A binding a
  shader statically uses has to name that shader's stage in the layout, and five of them named only
  `FRAGMENT`.
* **A chunked `static_draw_primitive` carries one material per instance** while its chunks have their
  own. `runtime::make_static_draw` has no caller in this repository, so nothing exercises it today, but a
  hit into one would be shaded with the wrong material.

## Step 2, revised against UE 5.8.2's Lumen (reference study)

The reference is `C:\UnrealEngine-5.8.2-release`, read for mechanism only; no numbers are copied, because
they belong to UE's scale. Three findings change the design, and one of them explains a measurement this
project already made.

### 1. The measured "environment-shaped contribution" has a name: unity DC gain

Step 0 measured the probe cache adding a nearly identical FRACTION of every region's brightness (0.81% of
the darkest third, 0.75% of the middle), and concluded "ambient-shaped, not structured". UE's filter
comments name the mechanism exactly: an unoccluded, NORMALISED blur has unity DC gain and no notion of
whether two cells can see each other, so its relative spread is independent of the light field. That is
what `Σw·L / Σw` with `w = kernel × trust` is. The fix is not a better kernel, it is:
**a failed visibility test must DROP a contribution, lowering the total weight, instead of being
redistributed over the survivors** (UE: `TotalWeight` starts at 1.0 for the centre cell, neighbours add
`AngleWeight * OcclusionWeight`, and a rejected neighbour adds nothing to either sum).

### 2. Per-cell depth is the entire leak defence

UE's radiance cache stores, per probe, a 32×32 equi-area-octahedral radiance map AND a 32×32 depth map
(16-bit distance, sign bit = front face, one mantissa bit = two-sided, 0xFFFF = miss). The filter's
occlusion test is BIDIRECTIONAL and needs nothing else:
`OcclusionWeight = neighbour.bFrontface || neighbour.bTwoSided ? 1 : 0`, then "can this cell see the
neighbour's ray start, and can the neighbour see ours" - each a lookup in the OTHER cell's depth map along
the direction between them, at an offset of `2 * cellSize * sqrt(3)` - plus clamping the neighbour's hit
distance to the receiver's own, and an angular weight from where the neighbour's ray actually landed
(UE: 0.2 rad). Trust decay cannot substitute for any of it; it is a scalar that says how much a cell was
seen, not whether two cells can see each other.

For this renderer that means the grid gains a second 3D image (a distance per cell, or the view depth the
injection already has plus a front-face flag), and the propagation becomes a 6-neighbour bilateral gather
with an axis-only kernel instead of the 3×3×3 blur.

### 3. The cache's feedback loop is bounded by construction, not by a clamp

UE has no energy clamp on the cache. What bounds it: a blind radius per probe (`ProbeTMin = cellSize *
sqrt(3)`) with radiance zeroed when a ray starts inside geometry; an INCREMENTAL refresh (about 100 of
16384 probes per frame, staleness-prioritised, so every probe holds its old value until its turn - a damped
Jacobi iteration); no write-back from the cache into the source it reads; and a hard reset when global
lighting changes materially (a 4x / 0.25x ratio on the light or skylight colour). This project reached the
same discipline by measurement instead - the cache's convergence is judged by whether the increments
between distant frame counts shrink - and the one mechanism worth adopting is the reset-on-change trigger,
which we do not have (our only reset is a new swapchain generation).

### 4. What they do NOT do, which matters for Step 2's direction

* Their world cache's INTERPOLATION has no per-probe depth rejection at all - the depth test in
  `LumenRadianceCacheInterpolation.ush` is `#define`d out, because rejection happened in the filter. The
  consumer that does reject by geometry is the irradiance-field one (Chebyshev from a per-probe
  (mean, mean²) occlusion pair, a normal-wrap weight, a probe validity mask, and a weight crush below 0.2).
* Their probe traces never read the radiance cache: the loop is closed through the SURFACE CACHE instead
  (surface cache at frame N-1 feeds the probes at frame N, the probes feed the screen-probe gather, which
  writes the surface cache). Their probes do not shade materials either - a trace resolves geometry
  against a distance field and reads the surface cache's final-lighting atlas.
* That is the structural answer to Step 2's original goal: **the thing that makes Lumen's cache
  view-independent is the surface cache, not the probe grid.** This renderer has no surface cache, and
  building one (mesh cards, a virtual atlas, capture passes, per-frame budgets) is a different scale of
  work. What it has instead is the ability to shade a hit from its real geometry - so the smallest
  faithful step here is to TRACE from the grid's cells and shade the hits, which makes the cache's
  contents depend on the geometry and the lights rather than on the frame the camera happened to show.
  That is a probe-grid-sized version of Lumen's radiosity pass, and it replaces the screen projection
  rather than supplementing it.

## Step 2, part 2: the surface cache is the actual answer, and what it means at this scale

The second reference study (both are under `docs/reference/`) settles the direction. Every Lumen
world-space ray resolves a hit as **geometry from a distance field or the TLAS, plus radiance from a
persistent per-surface texel atlas** - the surface cache - which stores pre-shaded light:
`(DirectLighting + IndirectLighting) * Albedo/PI + Emissive`, baked by its own pass on a cadence that has
nothing to do with what the camera is showing. The screen contributes only probe PLACEMENT and a BRDF
probability function, never radiance. The radiance cache's own probes are filled by tracing the distance
field and sampling that atlas.

So the thing that makes Lumen's cache view-independent is **not** the probe grid. It is that radiance stops
being a per-frame quantity at all: it is stored per surface texel, keyed by hit identity and local UV, and
refreshed on a budget. Their own comment states why the discipline is load-bearing: *"Secure against
strange values, as we are writing it to a persistent atlas with a feedback loop."*

WHY THIS RENDERER SHOULD NOT BUILD ONE (yet): the atlas exists because Lumen traces a great many rays
(1024 per radiance-cache probe, 64 per screen probe) and cannot afford to shade a material at each hit, so
it caches shaded radiance instead. This renderer traces 2 rays per half-resolution pixel and CAN shade the
hit - measured, and the cost was inside the noise floor. Its equivalent of "cache the shaded value" is
"shade the hit", which it already has, and its problem is a different one: the probe grid is INJECTED from
the screen, so a cell holds what the camera saw. The smallest faithful step is therefore to make the grid
trace its own rays and shade what they hit, which is a probe-grid-sized version of Lumen's radiosity pass,
and to give the grid the per-cell depth that its filter needs.

WHAT TO COPY, IN DEPENDENCY ORDER, AND WHAT TO SKIP (from the studies' own recommendation):
* Copy: a per-cell DEPTH so the filter can test visibility; the bidirectional occlusion test (both
  directions, because one direction misses thin walls); the rule that a rejected neighbour lowers the
  total weight instead of being redistributed; a blind radius of one cell diagonal with radiance zeroed
  when a ray starts inside geometry; amortised refresh with staleness priority; a reset when global
  lighting changes materially (their trigger is a 4x / 0.25x ratio on the light or skylight colour, and it
  is the one mechanism of theirs this project lacks).
* Skip deliberately, with the consequence stated: mesh cards and a sub-allocated virtual page atlas (needs
  another surface parameterisation - this renderer would key an atlas by triangle or mesh if it ever built
  one); virtual-texture feedback for hi-res pages (distant receivers keep only the coarse pages, which is
  what UE's bias terms exist to hide); the radiosity bounce (without it, GI is flat ambient with no colour
  bleeding); the clipmap pyramid with adaptive trace tiles (long rays fall back to sky, energy loss in
  enclosed interiors).

NUMBERS WORTH HAVING IN VIEW, AS SCALE REFERENCE ONLY (they are UE's, in cm, and each would still have to
be A/B'd here): cells of ~104 units doubling over four clipmaps of 48^3, with only the 16384 probes a
consumer marks actually allocated; 1024 rays per probe but only ~100 probes retraced per frame (~164
frames per full refresh), chosen by a 16-bucket staleness histogram; a probe's rays start at one cell
diagonal so a probe cannot hit its own cell; the filter is a single-pass 6-neighbour axis gather, not a
blur; an angular weight of 0.2 rad on where the neighbour's ray landed; a minimum trace distance of two
cell diagonals before the cache may be consulted at all.

## Step 1 revisited: UE's default is NOT to evaluate a material at a hit, and why this renderer still does

The third study (all three are under `docs/reference/`) contains a finding that presses directly on what
Step 1 built. Lumen's inline ray-tracing path - the one whose structure matches this renderer, no SBT, one
compute shader with inline ray queries - evaluates NO material at a hit by default. Its payload carries a
distance, an instance index, a material id, a few flag bits and one geometric normal
(`FLumenMinimalPayload`), and the radiance comes from a surface-cache lookup. UE's own switch text for the
alternative is the reasoning: reading the surface cache "gives the best GI and reflection performance",
while calculating lighting at the hit point "greatly increases GPU cost, as full material and lighting will
be evaluated at every hit point". Their material hit shader exists, is bound per material with the whole
generated material graph compiled into it, and is used only in the opt-in hit-lighting modes - and even
there a second trace against a material SBT is fired rather than paying for it in the main ray.

THIS RENDERER'S CHOICE IS STILL THE RIGHT ONE AT ITS SCALE, for a reason that has to be stated so it is not
mistaken for a claim about the technique: UE avoids material evaluation because it traces an enormous
number of rays (1024 per radiance-cache probe, 64 per screen probe) AND has a surface cache to read
instead. This renderer traces 2 rays per half-resolution pixel and has no surface cache, so shading the hit
IS its replacement for the cache - and it measured inside the noise floor. The finding is therefore a
statement about WHEN this design stops scaling, not about whether it was right: if the ray count rises (a
reference-frame mode, more probes, a path-traced comparison), the scalable shape is UE's, namely keep the
inline hit minimal, record (ray, materialId, distance), bin the rays by material, and only then read
material records and textures coherently. The register pressure that UE names as the wall
("too expensive (as in uses too many registers)") is the reason, and it is a wall this renderer will hit at
the same place.

TWO VALIDATIONS OF WHAT WAS BUILT, both from UE's own constraints:
* A ray-tracing closest-hit shader is FORBIDDEN from reading the scene/G-buffer textures; only uniform
  buffers and loose data may be bound. So material data reached through a structured buffer plus a
  descriptor array of textures is the sanctioned shape, not a workaround.
* UE stores a material ID and flag bits per hit and reaches vertex/index data through per-record
  constants, where this renderer stores the buffer DEVICE ADDRESSES in its instance table. Equivalent, but
  heavier per instance - and UE's indirection is what lets one geometry serve many instances. Worth
  revisiting if instancing of the same mesh ever matters here.

## The two documented limitations now have concrete answers

* ALPHA MASK, and the answer is NOT an any-hit shader: UE resolves the mask in a compute pass and bakes
  the result into the acceleration structure. Masked-out triangles are collapsed to degenerate ones
  (all three vertices written to the same position) in the position buffer the BLAS is built from, which
  requires masked geometry to be expanded to three unique vertices per triangle and made non-indexed, so
  hiding one triangle cannot corrupt a shared vertex. UE's own documentation of the tradeoff is blunt: if
  no any-hit shader is available, masked geometry is SILENTLY SOLID - which is exactly this renderer's
  symptom, and it is a limitation with a known fix rather than a dead end.
* SKINNED AND MORPHED MESHES: the fix is zero-copy, and UE calls it the fast path. The GPU skinning pass
  already produces positions; if it writes them as float3 into the very buffer the BLAS reads (morph deltas
  fold into the same pass for free), then the acceleration structure only needs a refit
  (`ALLOW_UPDATE` at build, `MODE_UPDATE` per frame) with a per-frame triangle budget and round-robin
  skipping, so a heavy scene degrades to a frame or two of lag instead of stalling. Worth recording that
  UE itself keeps `bRenderStatic` and instanced-skinned meshes permanently in BIND POSE in its ray tracing
  scene (a separate `StaticRayTracingGeometry` built from the bind-pose buffer and never refit) - this
  renderer's stated limitation is a documented mode in the reference implementation, not a shortcut only
  it takes.
