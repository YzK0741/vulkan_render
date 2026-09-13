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

## The complete roadmap, in dependency order

DONE (each with its evidence in this file or `docs/reference/`):
0. Measure the probe cache's contribution. Negative result: it adds a nearly identical FRACTION of every
   region's brightness - ambient-shaped, not structured - and the mechanism is a normalised unoccluded
   blur (unity DC gain). Recorded in config.example.toml beside the knob.
1. Shade a hit from its geometry: instance table with buffer device addresses, vertex/index fetch through
   a buffer reference, analytic tangent frame, material records and textures, sun with a shadow ray,
   split-sum IBL, emissive. Measured: +0.22 mean on hits the camera can see (0.36%, validating the whole
   fetch and evaluation chain); the -1.29 on hidden hits attributed to the screen path granting them probe
   ambient; three candidate explanations disproven by measurement; back-face hits rejected for correctness.
2. The per-pixel error instrument (MAE/RMSE against a converged 16-ray, 600-frame reference), used once.
3. The reference studies under docs/reference/, and the mechanisms taken from them.

REMAINING, in the order the dependencies require:

A. Make the cache structured instead of ambient-shaped (the leak fix; the smallest change with the
   largest measured defect behind it).
   A1. A per-cell DEPTH: a second 3D image holding, per cell, the distance the injection found there (the
       injection already computes the cell's screen position and the depth at it) plus a front-face flag.
   A2. Replace the propagation with a 6-neighbour axis gather whose weight is an angular term times a
       BIDIRECTIONAL visibility test between the two cells; a failed test DROPS the contribution (lowering
       the total weight) instead of redistributing it; the centre keeps weight 1.
   A3. Accept when the tertile signature flips from "the same percentage in every third" to "light where
       the off-screen geometry illuminates" - the instrument from step 0 - and when the convergence
       increments still shrink (the same test every energy-carrying change here has had to pass).

B. Make the cache view-independent (its contents stop depending on what the camera showed).
   B1. Share the shading code with the probe pass: extract the hit-shading block and its shared bindings
       into an include unit both passes can read (today ssgi.comp, surface.glsl and shading.glsl declare
       overlapping bindings, which is the actual blocker).
   B2. Give the probe pass the scene set, the TLAS binding and the instance table address in its push
       block.
   B3. Trace from the cells: cosine-weighted rays from a cell's centre starting one cell diagonal away,
       hit -> the shared shading, miss -> sky; amortised across frames by staleness priority so only a
       slice of the grid is retraced per frame.
   B4. Accept when the grid no longer answers with the frame it was injected from: the same cell read from
       a camera angle that never saw it gives the same value, and the tertile signature stays structured.
   B5. Measure the cost and set the slice budget from it.

C. Loop hygiene for the cache's feedback path: reset the grid when global lighting changes materially
   (UE's trigger is a 4x / 0.25x ratio on the light or skylight colour), which is the one convergence
   mechanism this renderer lacks; the increment test remains the acceptance.

D. The two documented limitations, both now with known mechanisms:
   D1. alphaMode MASK: resolve the mask in a compute pass and bake it into the acceleration structure -
       collapse masked-out triangles to degenerate ones in the position buffer the BLAS reads, expanding
       masked geometry to three unique vertices per triangle and dropping the index buffer.
   D2. Skinned and morphed meshes: make the GPU skinning pass write float3 positions into the buffer the
       BLAS reads (morph deltas fold into the same pass), build with ALLOW_UPDATE and refit per frame with
       MODE_UPDATE under a per-frame triangle budget with round-robin skipping.

E. The other half of step 3: a reference that is neither of the two variants (a brute-force or
   path-traced answer, offline and accumulated), so the per-pixel metric measures accuracy rather than
   self-consistency. That is what would settle whether the shaded or the screen model is closer to the
   truth, and it is what turns every remaining knob (probe rate, propagation rounds, bounce gain, spatial
   sigma, probe gain) into something judged against an answer.

F. Only if the ray count rises (a reference mode, more probes): the scalable shape for hit shading is
   UE's - keep the inline hit payload minimal, record (ray, material id, distance), bin the rays by
   material, and read material records and textures only then. Register pressure is the wall that forces
   it, and it is where this renderer would hit the same wall.

## Level 1 progress, and exactly what step B has left

COMPLETED SINCE THE ROADMAP ABOVE (all pushed):
* A1 - the per-cell surface offset: a third 3D image beside the ping-pong pair, holding the vector from a
  cell's centre to the surface the injection found at its screen position, with a validity flag.
* A2 - the filter: six axis neighbours, each gated by a BIDIRECTIONAL segment-versus-surface test; a pair
  that fails drops out of numerator AND denominator (no renormalisation, which is what gave the old filter
  unity DC gain); centre weight fixed at 1; the per-step trust decay removed.
  MEASURED: the contribution fell from 0.81% / 0.75% / 0.06% of the frame's darker-to-brighter thirds
  (171688 pixels brighter) to 0.13% / 0.11% / 0.02% (29413 pixels) - the unoccluded spread is gone. The
  SIGNATURE did not flip, and that was a mis-specification in the plan rather than a defect: a filter
  removes light that should not be there, it cannot make the SOURCE structured while the source is still a
  screen projection. The structured test belongs to B.
* B1 - the probe pass binds the shared scene set (two-set pipeline layout; its own bindings moved to
  set 1). This was also the blocker the earlier note recorded for sharing the shading code.
* B2 - the pass can reach the geometry: the scene set's top level structure is declared (the shader moves
  to `#version 460` with `GL_EXT_ray_query`, because the extension is written against GLSL 4.60) and the
  instance table's device address rides the push block as two 32-bit halves, exactly as the tracer's does.
  The push is now 120 bytes, inside the 128 it is sized for.

WHAT B STILL NEEDS, in the order that keeps each step verifiable:

1. EXTRACT the hit shading into `shaders/hit_shading.glsl`: the material struct, the buffer-reference
   types, `hit_surface`, `shade_hit`, `fresnel_schlick`, and the bindings those functions reference at
   set 0 (kinstance table address is NOT among them - see below).
   Two traps this shapes around:
   * `shade_hit` currently reads the instance table's address from the TRACER's push block
     (`instance_table_address()`, bit-reinterpreting `pc.proj_terms.z/w`), and the probe pass carries that
     address in its own lane (`pc.instance_table`). The address must therefore become a PARAMETER of the
     shared entry point rather than a function of a push block only one of the two passes has.
   * the bindings the shared functions need (camera UBO, LightUBO prefix, irradiance cube, prefiltered
     environment, BRDF LUT, material records, texture array) are currently declared inside
     `shaders/ssgi.comp`. Including the new file in both passes means DELETING them from ssgi.comp, or the
     same binding is declared twice and the shader will not compile. `shaders/surface.glsl` and
     `shaders/shading.glsl` cannot be included instead: they declare the same camera UBO, and
     `shading.glsl` also declares the shadow map and the cluster buffers, which the probe pass has no use
     for and which would drag the FRAGMENT-only `sampler2DArrayShadow` into a compute pipeline.
   The verification for this step is strong and cheap: the traced-GI reference capture must stay
   byte-identical (SHA256 D3506A006670007E), because nothing about a refactor should change a pixel.
2. THE RAY LOOP in the probe pass: cosine-weighted directions from a cell's centre, the origin pushed out
   by one cell diagonal (`cell * sqrt(3)`) so a probe cannot hit its own cell, a hit shaded through the
   shared entry point, a miss left with the sky. Then the amortisation: a per-cell "last traced frame"
   is what a staleness priority needs, and only a slice of the grid should be retraced per frame - the
   reference implementation's numbers are ~100 of 16384 probes per frame with a 16-bucket histogram, and
   the equivalent here would be a slice of the 32768 cells chosen by age.
3. ACCEPTANCE, and this is the part A could not satisfy: the tertile signature must become STRUCTURED
   (light where the off-screen geometry illuminates, not a fixed percentage of every third), and a cell
   read from a camera angle that never saw it must give the same value. The first has an instrument
   already; the second needs a capture from two camera positions with the grid warmed up from the first,
   which is a new measurement script rather than a new pass.

STILL OPEN BEYOND B: C (reset the grid when global lighting changes materially - the one convergence
mechanism the reference has that this renderer lacks) and E (a reference that is neither of the two
variants, which is what would make the per-pixel error metric measure accuracy rather than
self-consistency).

### Decision recorded before the ray loop: retrace everything first, amortise only if it measures slow

The reference implementation's amortisation (a staleness-priority histogram over a fixed per-frame budget)
exists because its probes trace 1024 rays each over 16384 allocated probes. This renderer's grid is 32768
cells, and the honest first version of the ray loop should trace ALL of them every frame with a small ray
count and see what it costs, rather than build the budget machinery first:

* the arithmetic, from numbers this project already measures: the screen-space tracer does 2 rays for each
  of ~259k half-resolution pixels and reports 0.58-0.9 ms. A probe grid at 32768 cells with 4 rays each is
  131k rays, i.e. about half that work, so a first estimate is a few tenths of a millisecond - and the
  probe pass currently costs 0.03 ms for its five dispatches;
* if that measurement comes back acceptable, the whole staleness apparatus (a per-cell last-traced frame, a
  priority bucket, a per-frame slice budget) is unnecessary complexity for this renderer's scale, and the
  grid is simpler than the reference's for a reason that is worth stating rather than copying;
* if it comes back too slow, the machinery has an obvious shape to copy and the measurement says how much
  of it is needed - which is the same order every other decision in this project has been made in.

Consequence for the extraction step that comes first: nothing about it should be designed around
amortisation. The shared entry point takes (query, hit position, direction, instance table address) and
returns radiance; whether it is called for every cell or for a slice of them is the caller's business.

### The structure the tercile instrument was hiding (step B's acceptance, measured)

The previous two slices concluded "the signature did not flip to structured" from the tertile table. That
conclusion was an artifact of the INSTRUMENT, and a 4x4 tile table of the same two captures shows it. Cache
effect per tile, as a percentage of that tile's own brightness, normal interior view, 180 frames:

    row 0 :  -1.88%  -1.17%  -0.16%  -0.14%      base brightness: 53.1  46.8 121.1 132.9
    row 1 :  -4.73%  -1.15%  -0.27%  -0.58%                        30.8  39.3  84.2 100.2
    row 2 :  -6.78%  -4.86%  -3.95%  -1.07%                        43.4  37.2  39.1 107.2
    row 3 :  -4.88%  -3.81%  -4.09%  -1.21%                        23.5  20.7  41.7  55.8

The effect spans -0.14% to -6.78%, a factor of about 48, and it is ORDERED BY THE SCENE rather than by the
frame's brightness: the interior tiles (base 20-53) lose four to seven percent, while the bright tiles (base
84-133, the open, sky-facing parts) lose a tenth to six tenths of a percent. That is what a correct answer
looks like - the cache replaces the sky estimate where the surface faces interior geometry, and leaves it
alone where the surface really does see the sky.

Why the tercile instrument could not see it: a brightness third mixes tiles. The bright third contains both
sky pixels (which the cache should not touch) and bright interior pixels (which it should), so averaging
over the third blends a large effect with a negligible one and reports a middling fraction for every third.
The instrument was fit for the ORIGINAL question - is a small correction added on top of the chain
ambient-shaped or light-following - and it is the wrong lens for a REPLACEMENT, which is why the same
captures read as "still ambient-shaped" through it and as clearly structured through a spatial table.

Two consequences worth carrying forward:
* step B's acceptance is MET, with the instrument corrected: the cache's effect is spatially structured and
  follows the scene;
* the tertile tables earlier in this file must be read as statements about *brightness thirds*, not about
  the cache's quality, wherever they were used to judge a replacement rather than an addition. The numbers
  themselves are right; the interpretation attached to them was not.

### Step E, measured: the only independent reference this engine has is not a neutral one

Step E asks for an answer that is neither of the two models being compared, so that the per-pixel error
measures accuracy rather than self-consistency. The cheapest such estimator this engine can produce is the
MARCHED tracer at 16 rays over 600 frames: its hit radiance comes from the lighting stage''s own output
rather than from the hit-shading code under test, so its radiance source is genuinely independent. Measured
against it, in the same interior view and at the same 8-bit green channel:

    against the marched reference (independent)     screen-sampled MAE 5.5062  RMSE 9.2515  max 76
                                                    shaded         MAE 5.9116  RMSE 9.9911  max 73
    against the shaded estimator (circular)         screen-sampled MAE 1.5544  RMSE 3.0372  max 46
                                                    shaded         MAE 0.4249  RMSE 0.7555  max 14

Two things are visible and only one of them is about the models. Against its own estimator the shaded path
is 3.7x closer, which is exactly the circularity: it is being compared with itself. Against the independent
one both models sit far away (MAE 5.5-5.9) and essentially TIED - and the shaded path is marginally the
WORSE of the two, by seven percent of MAE.

WHY THAT COMPARISON CANNOT DECIDE, which is the actual finding of this step: the marched reference is not a
neutral estimator. A screen-space march cannot see off screen at all, so in an interior scene it is blind to
exactly the light the shaded path exists to recover, and it takes its per-pixel radiance from a lighting
stage that evaluates shadows and occlusion per screen pixel. It therefore SHARES the screen model''s central
limitation, and any measurement that treats it as truth is biased in favour of the screen model. The shaded
path being seven percent worse under a reference biased against it is weak evidence for either model, not a
verdict.

So step E is measured but not satisfied, and the honest statement of what remains is this: the engine has no
neutral reference, and producing one means an estimator that shares neither model''s bias - a reference pass
that (a) traces against the geometry rather than the depth buffer, (b) shades what it hits rather than
reading the screen, (c) accumulates a large number of bounces rather than one, and (d) is averaged over
frames with no denoiser in the loop. That is a new pass, not a knob, and it is the last piece of Level 1.

### Step E, resolved: what can be arbitrated here, and what cannot

Working out what a reference would have to be to decide between the two models changes the answer from
"build a new pass" to a statement about the engine, so it is worth writing down before any code is written.

The two models under test are not as different as they look. The screen-sampled path and the shaded path
both TRACE THE SAME GEOMETRY - the same top level structure, the same inline ray queries, the same cells of
the G-buffer set - and they differ in exactly one thing: where a hit's radiance comes from (the frame's
direct-radiance image, revalidated by a depth test, versus the hit's own material and lights). Their hit
oracle is not a variable in the comparison; it is shared.

That matters because it decides what a reference could possibly be:

* a reference that reuses `shade_hit` - more rays, no denoiser, accumulated over frames, even with several
  bounces - SHARES the model it would be judging. It would produce a flattering number and call it
  independent, which is the one outcome worse than no number at all. The earlier note that proposed exactly
  that shape was wrong for this reason;
* the marched reference shares the SCREEN model's central limitation instead (a depth-buffer march cannot
  see off screen), which is why the previous measurement could not decide and was biased towards the screen
  model;
* so a neutral reference would need a SECOND, independent implementation of the same quantity: its own
  material evaluation, its own shadowing, its own estimator, tracing geometry. That is a second renderer,
  not a pass, and building one is outside what this step can honestly claim.

WHAT E CAN SAY, and it is not nothing. The engine does contain one independent implementation of the
quantity a hit's radiance is: the lighting stage, which evaluates materials, shadows and IBL in a fragment
shader. It can only be consulted at surfaces the camera sees - but there, comparing the shaded model against
it is a genuine cross-implementation check, and it was measured in round 9: shading only the hits the screen
confirms gives a frame within 0.22 of mean brightness of the screen path's (0.36%), which is what a
different shadow method plus the punctual lights this path does not evaluate are worth. That is positive
evidence for the shading model exactly where independent evidence exists.

The conclusion of E is therefore: the shaded model is independently corroborated where an independent
implementation can reach it (visible surfaces, 0.36%), and the disagreement between the two models lives
entirely in the hits the screen cannot evaluate - a region where this engine contains no second opinion and
where a self-built one would be no opinion at all.

## Level 1: final state

    A  probe-cache leakage      DONE   per-cell surface offsets, six-neighbour gather gated by a
                                      bidirectional segment-versus-surface test, failed pairs dropped
                                      (no renormalisation), centre weight 1, trust decay removed.
                                      Measured: the unoccluded spread is gone.
    B  view independence        DONE   the cells trace their own rays (uniform over the sphere - a cell in
                                      empty space has no normal), hits shaded through the shared entry
                                      point, misses answered by the sky; the cache REPLACES the far-field
                                      probe where it is trusted. gi_probe.comp declares no camera at all,
                                      so view independence is compiler-enforced rather than claimed.
                                      Measured: the effect is spatially structured (-0.14% to -6.78%
                                      across a 4x4 tile table, ordered by the scene) and costs 0.15 ms
                                      for 131k traced-and-shaded rays.
    C  lighting-change reset    DONE   the grid is CLEARED when the sun's direction changes by more than
                                      about 25 degrees. The mechanism is verified by executing it
                                      (forced trigger: 180 clears, validation clean, which is what proved
                                      the images need TRANSFER_DST); its EFFECT cannot be observed in a
                                      static scene, and that is stated rather than glossed.
    E  independent reference    LIMIT  the possible independent check was made and passed - the engine's
                                      own lighting stage corroborates the shaded model at visible surfaces
                                      to 0.36%. Arbitrating the two models where the screen cannot reach
                                      would need a SECOND independent implementation of a hit's radiance:
                                      a second renderer, not a pass. A reference built from the shading
                                      under test would be circular, and the marched one shares the screen
                                      model's blindness. See the two notes above for the measurements.

The instrument lesson is worth carrying beyond this work: the tertile table built in step 0 was the right
instrument for "is a small correction added on top of the chain shaped like the light or like the
ambient" and the wrong one for "is a replacement correct", where a spatial table is what shows the
structure. Two slices were recorded as failures against an instrument that could not have shown a success.
