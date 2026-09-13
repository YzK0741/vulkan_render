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

### L2.0 slice 2, designed: force the environment by DATA, not by code

The isolation in slice 1 settled how the furnace must be built: the lanes and the sun multiply are
byte-exact, and restructuring `shading.glsl`'s ambient/specular expressions is not, so the environment can
never be forced by branching in the lighting stage. The remaining question was where to force it instead,
and the answer discards the search for the IBL's fill path altogether:

**bind a constant environment instead of changing how one is produced.** `irradiance_sampler` (set 0
binding 3) and `env_sampler` (binding 2) are descriptors the runtime already writes, once per frame slot,
through `write_ibl_bindings()`. In the furnace mode they point at a constant cube of radiance L instead of
the real environment. That is a DATA change in the descriptor layer:

  * no shader in the frame path changes at all, so the default path stays byte-exact BY CONSTRUCTION - not
    by inspection, and not by hoping a multiply by 1.0 does not reschedule anything;
  * it does not matter how the environment is produced (a precompute pass, a sky render, an uploaded
    asset) because the furnace never consults the produced one;
  * the specular BRDF LUT is deliberately left alone. It is a DIRECT term - the same in the GI-on and GI-off
    frames - so it cannot affect the acceptance, which is the analytic statement that a furnace has nothing
    for a bounce to add. Leaving it bound also keeps the test honest about what it is testing.

What the cube has to be: one texel per face with all six faces at L, no mip chain (`textureQueryLevels`
then returns 1 and the prefiltered lookups collapse to LOD 0, which is what a uniform environment means
anyway). It is created with the other targets, cleared once, and referenced only while the mode is on.

THE ACCEPTANCE, unchanged from the plan: with the sun off (the lane from slice 1) and the environment
uniform, a diffuse surface's outgoing radiance is exactly albedo * L and a bounce has nothing to add, so
the GI-on and GI-off frames must agree. The first attempt at this, before the lane isolation, measured GI
on at +0.90 of mean brightness over GI off (0.65%) - and traced the excess to a real invariant: the spatial
filter's `subtract_ambient` reads `irradiance_sampler` directly to remove what it believes the lighting
stage added, so the two must be kept in agreement. With both reading the same constant cube in furnace mode
that divergence disappears by construction too, which is the second reason this shape is the right one.

### L2.0 slice 3d, where it stands: the clear needs a frame-start insertion point, and that is the blocker

The cube exists (slice 3c). Two pieces remain before the analytic acceptance can run, and the first one is
the reason this slice stopped here rather than being written:

**④ the clear that gives the cube its level.** `vkCmdClearColorImage` needs a command buffer, and the clear
has to happen before the FIRST reader of the environment in a frame - the skybox, not the lighting stage -
because in a furnace the background is part of the environment too. The frame's first recording point is
where `gpu_mark_id::frame_begin` is written, and that mark is at `vulkan/runtime.cpp:1517` inside a function
that the obvious greps for `runtime::record*` / `runtime::render*` do NOT name (the nearest definitions those
patterns find are at L127-139 and then L1734), which means the definition is formatted in a way those
patterns miss. So the next attempt should READ around L1500 rather than grep for it - and a note is worth
more than a guess here, because guessing an insertion point is what produced the mangled append two slices
ago.

A fallback exists if that turns out to be awkward: do the clear in `record_ssgi_pass` (L3100), which is a
function whose shape is already known and which has a command buffer. It costs a frame-1 artifact - the
skybox and the lighting stage read the cube before the GI chain clears it, so the first frame of each target
generation sees undefined texels while every later frame is correct, and a captured frame 180 is unaffected.
For a diagnostic mode that is acceptable if it is stated; it is not acceptable silently.

**⑤ the binding.** `write_ibl_bindings()` (a `const` method that early-returns when the scene sets are not
created) is the single place `irradiance_sampler` and `env_sampler` are written, so it is where the flag goes.
It must NOT be wired before ④: with the cube's contents still undefined, pointing the IBL at it would turn
the mode from "a dark frame" into "a frame of undefined texels", which is worse than half-wired and would
make the config file's warning untrue.

**⑥ the acceptance** is then the same measurement the first attempt made (GI on versus GI off in the furnace),
which already proved it can detect an energy error: it read +0.90 of mean brightness (0.65%) when the mode
was half wired, and traced that excess to the spatial filter's `subtract_ambient` reading `irradiance_sampler`
directly while the lighting stage used the analytic constant. With both reading the same constant cube that
divergence disappears by construction - the second reason the data-shaped design was the right one.

### L2.0, measured: what the furnace reports, and the one error it found

The mode runs and reports, and what it reports is not the energy error the plan expected. Establishing
that is the result of this step. Every number is the mean of the 8-bit green channel over one 1080x960
capture of 180 frames at a static camera; **two runs of ONE config are byte-identical** (verified by
SHA256), so a difference between two configs is signal - which also means the acceptance's "within the
noise of two 180-frame captures" is in practice an exactness test, not a tolerance. The Sponza captures
use the pinned GI camera of the `sponza` scenario in `scripts/windows/check_render.ps1` (yaw 90, pitch 0,
distance 6.41, target (0, -18.55, 0), `taa = false`, `[gui] show = false`, `vsync = false`,
`max_fps = 240`); the convex ones use the model's own `camera_fit`, which is deterministic and is printed
in the startup log. Nothing else about the runs differs from the configs quoted here and in
`config.example.toml`.

THE REPLACEMENT CHAIN IS EXACT, and the test that shows it needs no geometry at all: set
`ssgi_radius = 0`. No ray can reach anything, every ray returns the constant cube, and the traced
estimate and the ambient the spatial filter subtracts are then algebraically the same expression - so
the chain must be the identity and GI-on must equal GI-off.

    Sponza, furnace, radius 0        GI off              134.7686
                                     GI on, before       136.5186    +1.7500
                                     GI on, after        135.2395    +0.4709
                                     GI on, after, sigma 0 134.7419  -0.0267
    and that last one is 27764 pixels (2.68%) off by exactly ONE 8-bit step, max 1 - the identity up to
    the float rounding of two algebraically equal expressions evaluated in two different passes.

THE ERROR THE MODE FOUND is most of that +1.75: removing it lowers the radius-0 frame by +1.2791 (with
the denoiser) and by +1.2494 (with the filter bypassed), which is the size of the term. Sponza's ao is 1
everywhere (it has no occlusion textures, so the G-buffer's ao lane is 1 by construction) and both sides
read the same uniform cube, which leaves `albedo * metallic * L` as the only term that can separate the
two frames at radius 0. The tracer's tail multiplied its diffuse estimate by the receiver's albedo and its
baked AO, but not by the `(1 - metallic)` that `shading.glsl`'s ambient, the spatial filter's subtraction
and `shade_hit`'s own ambient all carry. A metal's kd is zero, so the traced estimate was GRANTING every
metallic surface a diffuse indirect the lighting stage never adds.

That the metallic factor is the whole of the term, and the filter the whole of the rest, is confirmed
twice:

* the +0.4709 that remains after the fix, with the denoiser in the loop, is the spatial filter's own
  weighted average - a weighted average is not the identity on a field that varies - and its size was
  measured independently on the same scene: +0.52 for sigma 2.0 against a pass-through;
* on a CONVEX scene the mode runs with real rays and no geometry within their reach, and there the fix
  is the difference between failing and passing:

        SimpleMaterial (one triangle, metallic 0.5, every ray escapes)
            GI off 138.7898 | GI on before 140.1903 (+1.4005) | GI on after 138.7897 (-0.0001)
            122 of 1036800 pixels differ, each by one 8-bit step
        Cube (one convex cube, metallic 0)
            GI off 151.0392 | GI on 151.3670 (+0.3278) - byte-identical before and after the fix,
            because its metallic is 0 and the fix IS the metallic term
            with ssgi_spatial_sigma = 0: -0.0045 (0.003%), so the +0.33 is the denoiser; its largest
            single-pixel difference is 82, on the cube's silhouette, where a half-resolution result is
            being reconstructed at full resolution

WHY SPONZA CANNOT PASS, and should not. The analytic statement - a diffuse surface's outgoing radiance
is exactly `albedo * L` and a bounce has nothing to add - is the classic furnace identity, and it holds
when the incident radiance really is L from EVERY direction. That is true of a convex object and false
in an interior: a ray that lands on a wall comes back with the wall's radiance, which in a furnace is
`albedo_wall * L < L`. So the traced chain reports an occlusion-corrected ambient, which is precisely
what it exists to report, while the lighting stage's `irradiance_sampler` lookup - an unoccluded
environment - is the approximation it replaces. The two cannot agree in an interior, and the chain
being darker there is the same measured behaviour Level 1 recorded as correct: interiors lose 4-7% and
sky-facing tiles 0.1-0.6%.

The deficit IS the hit population, established three ways:

* it scales with how far the rays reach (same scene, same camera, before the fix): radius 0.185 (0.01 of
  the scene radius) -2.5645; 2.22 (0.12, the default) -7.6316; 18.5 (1.0) -13.4722;
* it disappears when the rays cannot reach geometry (radius 0 above): +1.75 before the fix, and after
  it +0.47, of which the filter is +0.52 and the remainder is 1-LSB rounding;
* it does not depend on which hit model runs. With hit shading off, a screen-confirmed hit returns
  `texture(direct_radiance, hit_uv)` - the lighting stage's OWN value for that surface, `albedo * L +
  spec`, below L - and a miss or an unconfirmed hit returns the probe, which with the constant cube and
  `ssgi_probes = false` is L exactly. The traced mean incoming radiance is therefore L minus what the
  geometry the rays reach absorbs, by construction rather than by accident.
* an unrelated estimator of the same quantity agrees about WHERE: SSAO at the matching world radius
  (2.22, 16 samples) darkens the same furnace frame by -9.4659 against the chain's -8.8250, and its 4x4
  tile table has the same shape - darkest in the interior tiles, lightest at the frame's left and right
  edges. Per pixel the two correlate only +0.27, which is what a 16-sample screen-space heuristic
  against a traced chain should look like; the spatial table is the part that means something.

WHAT CHANGED, and what it was verified against:

* `shaders/ssgi.comp`: the traced diffuse estimate carries `(1.0 - metallic)`, read from the same
  G-buffer lane (`gbuffer_albedo.a`) that `deferred.frag` reads as `s.metallic`. No binding, no push
  constant and no interface changed, so no module version moves with it;
* the GI-off path is byte-identical before and after (same SHA256), and so is a metallic-free GI-on
  capture (the Cube) - so the change is exactly that term and nothing else;
* `scripts/windows/check_render.ps1`: 8 scenarios, each run twice, 0 changed (the new `sponza_gi` scenario
  is one of them - see the note at the end of this section); `ctest` 6/6; Release,
  Debug and ASan+UBSan builds clean; `doxygen Doxyfile` exit 0 with an empty warning stream;
* the traced frame itself moves by -1.1935 of mean brightness on the Sponza GI capture (34% of pixels,
  one-signed), which is the spurious metallic diffuse leaving it.

THE ACCEPTANCE, where it is well posed. The mode is the right instrument and it now passes:

    GI-on and GI-off agree in a furnace whose incident radiance is L in every direction - a scene the
    traced rays cannot reach absorbing geometry in. Convex, one material, real rays, 180 frames: agree
    to -0.0001 of mean brightness, 122 pixels of 1036800 off by one 8-bit step. The same scene read
    +1.4005 before this step's change, which is exactly the error the mode was built to find. In an
    interior the frame is legitimately darker by the hit population, and that number measures the
    scene's occlusion rather than the chain's energy.

One honest note on the tolerance, because it is easy to misread: the acceptance as written asks the two
frames to agree "within the noise of two captures", and the noise is zero. The convex measurement meets
that bar to a single 8-bit step; the interior one cannot meet it at all, for the reason above.

### L2.1, measured: directional probes (SH-2), and the test that was built before them

WHAT CHANGED. A cell used to hold ONE RGB, so a cell between a bright window and a dark wall averaged the two
into one value and answered identically for a ray arriving from either side. It now holds real SH-2 - four
coefficients per channel, the DC term and the three first-order terms - in FOUR RGBA16F 3D images per
ping-pong side (binding order sh0..sh3, with the trust the fourth channel always carried living in the DC
image's alpha). 32^3 cells, two sides, four coefficients: eight images, 2 MB. The basis lives in ONE shared
include (`shaders/probe_sh.glsl`), because the projection in the probe pass and the reconstruction in the
tracer have to agree exactly and a disagreement there would look like a slightly wrong image rather than like
a bug. The projection carries the 4*pi that makes a uniform field reconstruct as itself - which is the
furnace's identity, so that constant is load-bearing rather than conventional.

THE ACCEPTANCE TEST WAS BUILT FIRST, and it needed an A/B that changes NOTHING but the cache's direction. The
probe gain's sign is where it fits: |gain| is the gain, and a negative gain looks a cell up along the OPPOSITE
direction of the ray, while the far-field environment term is still sampled along the ray itself. So the two
captures differ only through the cache - and while a cell held one RGB they were byte-identical BY
CONSTRUCTION. Measured before any of the storage changed (Sponza interior, traced GI, probes on, 180 frames,
the check_render sponza camera):

    gain +1 and gain -1    byte-identical, mean green 60.0420
                           SHA256 1C6AF8F1B4D2E14F77B0A9FC7B615BA317072DC8D341275BF8AB5AED02FEE5C1

and after the change, in the same configuration:

    gain +1  60.0429   |   gain -1  60.0449     hashes differ
    234921 pixels (22.66%) differ, by up to 14 of 255, in the green channel

which is the cleanest before/after this project has: one configuration, one scene, one frame count, and a test
whose two sides were provably identical until the representation changed.

THE 4x4 SPATIAL TABLE STAYED STRUCTURED, the step's second condition. The numbers below are the cache's
effect on the frame (gain 1 against gain 0) as a percentage of each tile's own brightness, in the same scene,
camera and configuration as the tables this file already carries:

    absolute mean difference                    as a percentage of the tile's own brightness
    -0.62  -0.12  -0.04  -0.26                  -1.50%  -0.16%  -0.03%  -0.22%
    -2.37  -1.13  -0.85  -1.64                  -7.53%  -3.64%  -0.98%  -2.22%
    -2.08  -1.27  -0.91  -0.99                  -4.92%  -3.85%  -1.45%  -0.81%
    -0.65  -0.97  -1.02  -0.52                  -4.29%  -2.94%  -3.16%  -1.06%

The span is -0.03% to -7.53% against the recorded -0.14% to -6.78%, and the ORDER is the same: the sky-facing
tiles (top right) lose a tenth to two tenths of a percent, the interior ones lose one and a half to seven and
a half. The frame itself moved by +0.0009 of mean brightness (60.0420 to 60.0429), so the directional part is
a small correction on top of a DC term that behaves as it did before.

THE FURNACE ACCEPTANCE STILL PASSES: on the convex acceptance scene (whose configs have the cache off) GI-on
and GI-off still agree to -0.0001 of mean brightness, unchanged. The captures are deterministic: two runs of
one configuration are byte-identical, before and after.

WHAT THIS DOES NOT DO, stated rather than discovered later. The directional terms are estimated from FOUR rays
per frame and blended at `ssgi_probe_rate` 0.08, so they converge over something like a hundred frames: a
single frame's first-order coefficients are mostly sampling noise, and that is the price of four rays. The
reconstruction CLAMPS AT ZERO, because four coefficients cannot describe a field that is bright on one side
and dark on the other - the energy that costs lands in exactly the directions the cache has no evidence about.
A cell still holds ONE surface offset for the visibility test, so this is not the reference implementation's
per-direction depth map. And blending neighbouring cells averages coefficients: exact for a WORLD-aligned
basis, but it rotates nothing - which is precisely why the basis is world-space.

WHERE IT LIVES: `shaders/probe_sh.glsl` (the basis, the constant, the reconstruction), `shaders/gi_probe.comp`
(the projection and the six-neighbour blend, now over four images per side), `probe_cache` in
`shaders/ssgi.comp`, the eight images and their views in `vulkan/core`, and the ping-pong, barriers and
descriptors in `vulkan/runtime.cpp`. The probe pass's push block is 40 bytes and carries no camera data at
all; the pass's own set is nine bindings (four read, four written, one geometry) and the G-buffer set gained
bindings 10..12 for the three first-order coefficients.

AND THE GATE NOW COVERS IT, which it did not before this step: the seven capture scenarios all ran with
`ssgi = false`, so the screen-space chain, the denoisers, the probe cache and every ray-traced path had NO
regression coverage at all - a break in any of them would have passed the gate. `sponza_gi` was added: same
scene and camera as `sponza`, with the traced chain, the denoisers and the probe cache switched on. Its
reference is a normal machine-local baseline (`-Update` once to seed it, as for every other scenario), and
the change that added it is the one that needed it: eight new 3D images and two re-shaped descriptor sets
would otherwise have had nothing watching them.

### L2.2a, measured: the alphaMode MASK bake, and why it is OFF by default

THE LIMITATION IT ATTACKS is documented in three places and is real: an inline ray query has no any-hit
stage, so a MASK material's `discard` cannot run during traversal and the surface is SOLID to every ray. The
raster shadow pass cuts the material's holes and a ray-traced one does not - a difference a user sees. The
reference implementation's answer, and the one the plan named, is to resolve the mask BEFORE the build.

WHAT WAS BUILT, and it is all verified plumbing: `shaders/mask_bake.comp`, a compute pass over the masked
casters that writes an EXPANDED copy of their vertices - three unique vertices per triangle, no index buffer,
32 bytes each (position, normal, UV) - with the triangles the mask covers nowhere collapsed to a single
position (a degenerate triangle traversal can never hit). Hiding a triangle that way is only safe if it owns
its vertices, which is why the copy is expanded and non-indexed; the bottom level structures and the instance
table are built from that copy instead, and the hit shading reads a zero index address as "flat vertex list".
It costs one startup dispatch per masked caster, next to nothing at 3 KB of vertices for a vase of flowers.

THE PLUMBING IS PROVEN EXACT, and the proof is the interesting part: with the rule disabled so that NOTHING
is collapsed, the traced frame is BYTE-IDENTICAL to the same frame built from the original indexed buffers
(same SHA256). So the expansion, the non-indexed build, the instance record and the flat-vertex hit path are
all exact - which matters, because it means every difference the feature makes is the RULE's doing and not
the mechanism's.

THE RULE MEASURED WORSE THAN THE RASTER PATH, so the knob defaults to off. On GlassVaseFlowers (3818 MASK
triangles), against the same frame with raster shadows as the reference (that path discards per fragment and
is therefore correct by construction):

    raster shadows (reference)          122.0241
    ray-traced, mask solid (no bake)    121.8918     error vs raster: mean 0.1323, mean|.| 0.4071
    ray-traced, bake on                 123.3137     error vs raster: mean -1.2896, mean|.| 1.6595

The bake moves the shadow 1.29 of mean brightness the WRONG WAY: 33723 pixels change, and only 1474 of them
move towards the raster reference. Reading it: the bake removes triangles that the raster path still
shadows. The rule samples every texel of a triangle's UV footprint (and KEEPS any footprint it cannot walk,
so it never removes geometry it has not looked at), so the disagreement is not sloppiness in the sampling
but the thing the mechanism cannot do - reproduce a PER-PIXEL mask with a per-triangle structure. The raster
path's own texel sampling is filtered (mip selection over a shared atlas), so it keeps fragments whose LOD 0
texels are all below the cutoff, and every such triangle is one this bake deletes.

WHAT THE THREE RULES LOOK LIKE on the same assets, evaluated on the CPU over the same texture and UVs as a
share of triangles that are cut everywhere:

    GlassVaseFlowers          vertices 1.8%      10-sample grid 1.0%      texel walk 0.1%
    DiffuseTransmissionPlant  vertices 1.5%      10-sample grid 2.3%      texel walk 0.9%

The vertex rule is the cheapest and the most destructive - a two-quad MASK plane whose pattern sits in the
middle of the quad looks 100% cut at its corners while 0% of its area is cut - and the texel walk is the only
honest one. It is also the one that shows how little there is to remove: under 1% of a plant's or a vase's
masked triangles are entirely empty, so even a perfect per-triangle rule would fix a small part of the
limitation.

WHAT A BETTER ANSWER NEEDS, which is why this is left as an instrument rather than deleted: a per-triangle
structure cannot represent a per-pixel mask, so the next attempt has to subdivide along the mask boundary
(splitting a partly-cut triangle until the pieces are clean) or use the opacity-micromap extension, which
exists for exactly this problem. Both need the plumbing this step built - the expanded, mask-aware copy of
the geometry that the structures and the hit shading already read.

THREE THINGS THIS STEP FIXED ON THE WAY, each independent of the bake and each found by a gate rather than
by inspection: `acceleration_structure::add` documented a zero index address as "a flat vertex list" but left
the caller's index type in place, which would have had the build read indices from address zero;
`runtime::make_mask_bake_pipeline` leaked its pipeline layout, which the validation layer reported as
"vkDestroyDevice(): VkDevice ... has 1 leaked objects" on every run; and a first version of the bake bound
the per-frame SCENE set and then the same command buffer rewrote that set's binding 16 later in the frame,
which invalidated the command buffer - 62 validation errors, every subsequent command reported against a
buffer "now in an invalid state". The bake now owns a descriptor set of its own, written once.

### L2.2b, the baseline measured first: an animated skinned mesh's traced shadow does not follow the pose

The limitation is documented - a skinned or morphed mesh is built from its SOURCE vertex buffer, which holds
the bind pose - and it is worth measuring before it is fixed, because the fix's acceptance IS this number.
The instrument has to cope with the object moving as well, so a plain frame difference says nothing. The four
captures below are the same animated model (Fox, three clips) at two PINNED poses (`[render] animation_time`,
which exists for this) with raster and ray-traced shadows, and the quantity that matters is the difference OF
the differences:

    d_raster = raster(t0) - raster(t1)     the object's motion PLUS its shadow following the pose
    d_rt     = rt(t0)     - rt(t1)         the object's motion plus whatever the traced shadow does
    d        = d_raster - d_rt             the part of the pose's effect only the RASTER path has,
                                           i.e. the traced shadow's MISSING pose dependence

    raster shadows, pose0 - pose1    mean +0.4845   mean|.| 2.5983
    traced shadows, pose0 - pose1    mean -0.0535   mean|.| 2.4757
    d, only the raster has it        mean +0.5380   mean|.| 0.7296
                                     1953 pixels where the raster shadow moved and the traced one did not

    d in a 4x4 table:      +0.000  +0.000  -0.000  +0.000
                           +0.000  +3.573  +3.780  +0.000
                           +0.000  +1.260  -0.006  +0.000
                           +0.000  +0.000  +0.000  +0.000

The traced shadow's pose dependence is ZERO exactly where the raster one's is up to 3.8 per tile: the model's
own pixels move in both captures (the 2.47 of d_rt) while its shadow moves in only one. THE ACCEPTANCE FOR THE
FIX is this table collapsing to the usual traced-versus-cascade difference, and it is measurable today, in one
config, with no new pass.

WHAT THE FIX NEEDS, from the plan plus the two steps before this one: a compute skinning pass writing an
expanded vertex record a structure can be built from - `shaders/mask_bake.comp` is a working example of that
exact shape, down to the flat-vertex index path in `shaders/hit_shading.glsl` - then `ALLOW_UPDATE` at build
and a per-frame `MODE_UPDATE` refit, because a skinned mesh's positions change every frame while its triangle
count does not.

### L2.2b, measured: the refit collapses the table to the noise floor, and what it costs

Built as the plan said. `shaders/compute_skin.comp` is one invocation per vertex and repeats
`shaders/pbr.vert`'s deformation term for term (four joints blended by the weights and divided by their sum,
positions as `mat4 * vec4(p, 1)`, normals through the matrices' `mat3`, the identity block at joint base 0
leaving an unskinned vertex alone), writing a 32-byte OBJECT-space record (position, normal, UV) that keeps the
primitive's own vertex order - which is what makes a refit legal: same addresses, same counts, same index
buffer, only the bytes change. `acceleration_structure::add` grew a `refittable` flag that adds `ALLOW_UPDATE`
to both the size query and the build, and `record_update` is a `MODE_UPDATE` build with source == destination
and the scratch memory it retains. The per-frame pass runs at the top of `record_top_level_structure`, the
refit right after it, and the instance record's vertex address is switched to the skinned buffer for that
frame, so traversal and hit shading read the same copy. Knob: `[render] rt_skin_bake`, default **false**.

The control came first, and it is the same one L2.2a used: with the knob off the frame must be byte-identical
to the capture the baseline was measured on, which proves the whole addition is inert when it is not asked
for. It is - `skin_rttrue_t05.png`'s SHA is `8D800A22341DCA32034571C2F909C766085806C1F33126127D69D7D6FA354C4D`
and the rebuilt `rt_skin_bake`-off frame hashes the same, with the t=0.9 pair agreeing too
(`5A2C57779547EF4B9D40D07ABA2652D543136AB3745ED7647BF669E9DD7A294D`). The knob is then a no-op by a second
route as well: on a scene with NO skinned caster the pass records nothing (no refit, nothing to refit) and
Sponza's traced-GI scenario hashes identically with the knob on and off, twice each:
`754DD38898F3E8C6EADBF9911CD5992AE07CBF34754CC7654935D61C1BF574C`.

The acceptance was the baseline table collapsing. Same instrument, same four captures, only the fix changed:

    pose pair t=0.5 -> t=0.9            REFIT OFF (the baseline)   REFIT ON
    raster shadows, pose0 - pose1        mean +0.4845 / 2.5983      (unchanged)
    traced shadows, pose0 - pose1        mean -0.0535 / 2.4757      mean +0.4845 / 2.5990
    d = only the raster has it           mean +0.5380 / 0.7296      mean -0.0001 / 0.0039
    worst 4x4 tile of d                  +3.780                     -0.008
    raster moved, traced did not         1953 px                    103 px

    d in a 4x4 table, refit ON:      +0.000  +0.000  +0.000  +0.000
                                     +0.000  +0.001  -0.008  +0.000
                                     +0.000  +0.005  +0.000  +0.000
                                     +0.000  +0.000  +0.000  +0.000

The four captures behind it, so the numbers can be re-derived rather than trusted: off t=0.5
`8D800A22341DCA32034571C2F909C766085806C1F33126127D69D7D6FA354C4D`, off t=0.9
`5A2C57779547EF4B9D40D07ABA2652D543136AB3745ED7647BF669E9DD7A294D`, on t=0.5
`3F04AE499C4EF16EFBDD28F0112C26B11E8C7DB1917508C241B1C99B921F94ED` (mean G 118.7188 against the off
arm's 117.7957), on t=0.9 `D48DCB71DA45DF16CDD0ED7B4127F2B60434957AE120FDE119C44553B14DA3DC`,
all four through `scripts/windows/run_furnace.ps1`-style runs of the release build with
`[render] animation_time` pinned and `--capture-frames 60`, raster shadows off/on as the arm dictates.

The traced shadow's pose dependence is now the raster one's to four decimal places (+0.4845 against +0.4845 of
mean brightness), the pixel count that has the raster shadow moving with nothing in the traced path falls by
95%, and what is left is 0.0039 of mean|.| - the ordinary traced-versus-cascade difference (a hard ray against
hardware PCF, thin bands on shadow edges), not a pose error. A THIRD pose pair, t=0.5 -> t=1.5, which moves the
model further (raster mean|.| 3.1355), says the same thing and is the reason the result is not a two-point
coincidence: d mean +0.0074 / mean|.| 0.0180 with the refit on, against -0.6935 / 0.8547 with it off, tiles up
to -5.573 becoming +0.077, 1673 struck pixels becoming 93.

WHAT IT COSTS, measured rather than asserted. The engine's own GPU pass timings, averaged over 60 frames and
read twice per arm (Fox: ONE skinned caster, 576 triangles, 24 joints):

    rt interval (the structures' interval, which is where the pass and the refit are recorded)
        refit off   0.03 ms / 0.03 ms       refit on   0.07 ms / 0.07 ms
    whole frame     0.41 ms / 0.41 ms                  0.46 ms / 0.46 ms

so roughly +0.04 ms of GPU on this asset, and the CPU-side phases (`scene`, `submit`) move by less than this
harness resolves. That is small because the caster is small: the pass is one invocation per vertex and the
refit touches one bottom-level structure, so the cost scales with the skinned vertex count and with how many
skinned structures exist - NEITHER OF WHICH IS MEASURED HERE, on an asset with a single 576-triangle skinned
mesh. A scene with many or heavy skinned casters is unmeasured.

WHAT IT DOES NOT DO: morph targets. They are the step before skinning in `shaders/pbr.vert` and this pass reads
only position, normal, UV, joints and weights, so a morphable mesh still traces its un-morphed shape - the same
limitation one property over. Folding them in needs the deltas and their active weights, which live at the
scene set's binding 10.

TRAPS, all paid for once. (1) `compute_skin_pipeline_layout` leaked and validation reported it exactly the way
L2.2a's `mask_bake_pipeline_layout` leak was reported - `[ERROR] vkDestroyDevice(): ... has 1 leaked objects` -
and it cost one capture, because the off arm's frame was already accepted as byte-identical before the leak was
seen; the fix is three lines in the teardown and the off arm was re-taken. (2) Binding 9 of the SCENE layout
(`SkinMatrices`) declared `VERTEX_BIT` only, and `shaders/compute_skin.comp` reads it from a compute stage: the
layout has to name `VERTEX_BIT | COMPUTE_BIT` or the binding is not usable there. (3) The first "the knob is
not a no-op" reading was a FALSE POSITIVE and the harness was at fault, not the code: the gate's Sponza
scenarios override the shared camera (`90,0,6.41,0,-18.548,0`), and the manual re-run passed the DamagedHelmet
camera, so two configs differing only in `rt_skin_bake` differed in the view as well. (4) A key appended to the
END of a config file lands in the LAST table, not in `[render]`: the first ASan smoke run had
`rt_skin_bake = true` sitting under `[lighting]`, where it was silently ignored, and the pass therefore never
ran while the run reported a clean exit. Appending is only safe if something pins the table, which is exactly
what `scripts/windows/run_furnace.ps1`'s in-table override does and what the ad-hoc command did not.

### L2.3, measured: a glossy reflection replaces the environment's specular ambient

Every indirect term in this chain was diffuse, and the specular one belonged to the lighting stage: the
split-sum lookup of the prefiltered environment along the reflection direction, which is the sky and nothing
else. So a metal panel inside a room reflected the sky, and so did a polished floor - the failure is not a
matter of degree, it is that the term has no way to know a room exists.

WHAT IS BUILT. `shaders/ssgi_spec.comp` is a pass of its own, dispatched between the tracer and the denoiser
and writing into the TRACER'S OWN IMAGE (it reads, adds to and stores the raw trace back, with a
compute-to-compute barrier between the two dispatches). It samples a half-vector from the GGX distribution the
environment chain is prefiltered with, traces along the reflection, and shades what it lands on from that
hit's own geometry - which is why it needs `ssgi_hit_shading`. Its push block carries its own ray count, which
is half of why it is a separate pass at all: the tracer's block is exactly 128 bytes (the smallest range
Vulkan guarantees) and has no lane left, and the other half matters more - a pass that is not RECORDED cannot
perturb the frame, which is a stronger statement than a branch that arithmetically cancels.
`shaders/ssgi_spatial.comp` then removes the lighting stage's specular term for those pixels, so the traced
reflection REPLACES it rather than adding to it; the two sides of that subtraction both come from a new shared
`shaders/ibl_specular.glsl`, which also replaced the two copies of the same expressions that already existed
in `shading.glsl` and `hit_shading.glsl` (verified byte-exact: 8 scenarios, 0 changed).

THE CONTROL, and it is the sharpest instrument this step has. Where a glossy ray finds no geometry, the
estimate IS the lighting stage's term - the same expression, from the same file, in the same association order
- so with the ray length short enough that nothing is reachable, the feature on and off must agree. On the
DamagedHelmet (an isolated model, so its convex-ish silhouette gives reflections somewhere to go) with the
joint-bilateral filter BYPASSED (`ssgi_spatial_sigma = 0`) and `ssgi_radius = 0.0006` (0.011 world units):

    pixels differing         856 (0.08%)      |d|>1        0        max |d|  1
    mean green              -0.0008

One 8-bit step on 0.08% of pixels and nothing beyond it: the arithmetic is exact, and what is left is the
half-float round trip through the trace image (the estimate is added, stored as RGBA16F, loaded and
subtracted again, and `(a + b) - b` is not `a` in floating point). The same test on Sponza at the same ray
length does NOT collapse (mean -0.0126, 1.12% of pixels, max 47) and the difference is the scene, not the
code: Sponza has surfaces within 0.011 units of each other (coplanar panels, double-sided walls), so its
reflection rays find geometry the helmet's cannot. Zeroing the diffuse estimate (`ssgi_intensity = 0`, which
makes the tracer write exactly 0 and leaves only the add/subtract pair) halves that residual to -0.0059 while
the outliers survive - which is what identified them as hits rather than arithmetic.

THE EFFECT, Sponza interior, the traced-hit-shading config, A/B on `ssgi_specular` alone, 120 frames
(60 frames gives -1.3763 against -1.3860, so the chain has converged and nothing drifts):

    frame mean green        56.3324 -> 54.9465      -1.3860  (-2.5%)

    4x4 tiles of mean green difference, absolute and as a percentage of that tile's own frame:
      -1.128  -2.288  -0.605  -0.343        -2.61%  -3.02%  -0.48%  -0.29%
      -1.208  -2.118  -0.481  +0.113        -4.65%  -7.33%  -0.57%  +0.16%
      -2.339  -2.329  -1.615  -0.742        -7.43%  -8.92%  -2.81%  -0.63%
      -2.085  -1.827  -1.782  -1.396       -16.78%  -9.23%  -9.14%  -3.13%

Ordered by the SCENE, in the same shape the L2.1 probe-cache tables have: the interior tiles lose 7% to 17%
while the sky-facing ones move 0.1-0.3%. That is what a correct local reflection looks like - the sky's
specular contribution to a dark interior is a large share of that interior's brightness, and the interior's
own reflection is darker than the sky. Isolating the specular channel (the diffuse estimate zeroed, so
nothing else can move) says the same thing harder: -1.1266, with the interior tiles at -20% to -33% against
-0.4% where the view is sky.

CONVERGENCE. The hit is a POINT sample of a cone whose width is the material's roughness, so one ray is the
feature's definition but not obviously enough. Measured, 1 ray against 4: +0.0028 of mean green, 0.60% of
pixels beyond 1/255 and 0.05% beyond 4 - against the feature's own -1.3860. Four times the cost buys 0.2% of
the signal after the temporal and spatial denoisers, so one ray is what the default is.

COST, the GPU timings' `gi` interval (which carries the tracer, this pass and both denoisers) at 1080x960:
1.27 ms off, 1.40 ms at one ray, 2.08 ms at four - about +0.13 ms for the first ray and +0.23 ms for each
further one. The whole-frame totals (2.80 / 2.70 / 3.69 ms) are noisier than that delta and should not be
read as the feature's cost.

THE RESIDUAL, and it is the honest half of this result. The subtraction happens after the joint-bilateral
filter has AVERAGED the estimate, while the removed value is the centre pixel's own: the added quantity is
filtered and the subtracted one is not. So the "a ray that misses changes nothing" property holds exactly
only when the filter is bypassed. On the same helmet, with the filter ON (sigma 2) and still nothing
reachable, the two arms differ by mean +0.027 with 1.0% of pixels beyond 4/255 and a worst pixel of 109 -
and zeroing the diffuse estimate leaves that unchanged (+0.0271, max 109), which is what proves it is the
filter and not a hit. This is the same class of artifact the DIFFUSE subtraction has carried since it was
written (the recorded non-mean-preserving-average note: +0.33 convex / +0.52 Sponza), and it is not fixed
here for the same reason the diffuse one was not: the correct fix is to filter the REMOVED term with the same
weights as the added one - 25 more gathers per pixel - and to do it for both terms at once, which changes
every existing capture and belongs in a step of its own with its own re-baseline.

WHAT IT DOES NOT DO: morph targets (inherited: the structures hold the bind pose), alphaMode MASK (a MASK
surface is solid to the ray), and the reflection is a point sample rather than a cone-filtered one, so a
low-roughness reflection aliases at half resolution in a way a prefiltered cube does not - the shared denoiser
takes the worst off, not all of it. Neither the ray length nor the exclusion of geometry the screen path would
reject was tuned: the length is the diffuse bounce's own radius, and a mirror that should reflect the far side
of a room needs a longer reach than the light that bounced off the floor.

THE TWO CANDIDATE FIXES FOR THAT RESIDUAL, with the trade-off already visible, recorded so the next step does
not have to re-derive it. (a) Filter the removed term with the same weights as the added one - 25 extra
gathers per pixel, three texture fetches each, so the spatial filter's cost roughly triples - and do it for
the diffuse term in the same change, which also removes the pre-existing +0.33/+0.52 note. The images upstream
stay pure radiance, which is what the multi-bounce feedback needs. (b) Have the glossy pass write the NET
correction `E - ibl_specular` instead of `E`, and drop the spatial filter's specular subtraction entirely: the
identity then cancels EXACTLY even with the filter on (`filter(A + 0) == filter(A)`, bit for bit), for zero
extra cost, and one less place for the two halves to disagree. What it costs is the property the L1 work
fought for - the trace image stops being a radiance and becomes a radiance plus a bookkeeping term, so the
bounce feedback (`[render] ssgi_bounce`, off by default) would re-emit a correction at a specular hit. (a) is
the architecturally consistent one and (b) is the cheap one; the reason this is written down rather than
decided is that the choice is a measurement away either way, and (b)'s numbers are one config change from
being taken.


