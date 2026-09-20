# UE 5.8.2 Lumen WORLD-SPACE RADIANCE CACHE — source study

Root: `C:\UnrealEngine-5.8.2-release`. All paths below relative to
`Engine/Source/Runtime/Renderer/Private/Lumen/` (C++) and `Engine/Shaders/Private/Lumen/` (shaders).

The "world-space radiance cache" (the one the GI tracer samples for off-screen/hidden hits) is
*the same class* as the irradiance field / translucency-volume cache: `LumenRadianceCache::UpdateRadianceCaches`.
Consumers: screen-probe gather GI (`LumenScreenProbeTracing.usf:756-837`), reflections
(`LumenReflectionTracing.usf:851`), translucency volume (`LumenTranslucencyVolumeLighting.usf:111-144`),
irradiance field (`LumenIrradianceFieldGather.cpp`), which only differ by the `FRadianceCacheInputs`.

⚠️ Two *different* interpolation paths exist and they matter enormously for your problem:
- `LumenRadianceCacheInterpolation.ush` (used by the screen-probe-gather world cache, reflections,
  translucency): per-probe **directional** radiance (32×32 octahedral) + per-direction depth,
  **trilinear over 8 probes, no per-probe occlusion rejection**.
- `LumenIrradianceFieldInterpolation.ush` (used by the irradiance field / screen-space irradiance cache):
  per-probe **irradiance in a 6×6 octahedral layout** + a **Chebyshev probe-occlusion** term,
  **8 probes with weight rejection**. This is the file whose weights you should copy (see §4/§5).

---

## 1. PLACEMENT — camera-anchored clipmaps, snapped to cell grid, sparsely *marked*, then optionally offset toward surfaces

**Not a fixed world grid, not on-surface placement.** It is a **camera-anchored clipmap chain**
level 0 anchored at the view origin, snapped (floored) to the cell grid of each level, and
**only actually allocated where a consumer marks it**.

Constants (screen-probe-gather world cache — the default "Lumen radiance cache"):
`LumenScreenProbeGather.cpp`
| constant | value | line |
|---|---|---|
| `r.Lumen.ScreenProbeGather.RadianceCache.NumClipmaps` | **4** | 602-608 |
| `...RadianceCache.ClipmapWorldExtent` | **2500.0** ("World space extent of the first clipmap") | 610-616 |
| `...RadianceCache.ClipmapDistributionBase` | 2.0 ("Base of the Pow() …", only used for state-reset detection now) | 618-624 |
| `...RadianceCache.GridResolution` | **48** ("Resolution of the probe placement grid within each clipmap") | 633-639 |
| `...RadianceCache.ProbeResolution` | **32** ("The number of rays traced for the probe will be ProbeResolution ^ 2") | 641-647 |
| `...RadianceCache.NumMipmaps` | 1 | 649-654 |
| `...RadianceCache.ProbeAtlasResolutionInProbes` | **128** ("Number of probes along one dimension … controls the memory usage. Overflow currently results in incorrect rendering. Aligned to the next power of two") | 656-661 |
| `...RadianceCache.ReprojectionRadiusScale` | 1.5 | 663-669 |
| `...RadianceCache.NumFramesToKeepCachedProbes` | 8 | 671-676 |
| `...RadianceCache.NumProbesToTraceBudget` | **100** | 626-631 |
| `MaxClipmaps` (C++) / `RADIANCE_PROBE_MAX_CLIPMAPS` (ush) | **6** | `LumenRadianceCacheInterpolation.h:15`, `LumenRadianceCacheInterpolation.ush:8` |

Geometry (`LumenRadianceCache.cpp:1291-1322`):
```
1299  double Level0CellSize = (ClipmapWorldExtent * 2.0f) / ClipmapResolution;
1300  FClipmapGeometry ClipmapGeometry(NewViewOrigin, NumClipmaps - 1, Level0CellSize, ClipmapResolution);
...
1308  const float ClipmapExtent = LevelGeometry.Size.X / 2.0;   // = ClipmapWorldExtent
1309  const float CellSize = LevelGeometry.CellSize.X;
1319  Clipmap.ProbeTMin = RadianceCacheInputs.CalculateIrradiance ? 0.0f
                        : FVector(CellSize, CellSize, CellSize).Size() * RadianceCacheInputs.ProbeTMinScale;
```
`FClipmapGeometry` (`:1190-1277`) — cell size **doubles per level**: `GetCellSize(Level) = Level0CellSize * Pow(2, Level)`
(`:1233-1236`); `SnapToGrid` uses `FloorToDouble` on each axis (`:1220-1231`);
`GetCenterAlignedOrigin() = SnappedOrigin - 0.5*CellSize` with the comment *"Shift the cell grid such that the
center of a cell on this level lines up with the center of a cell on every level below this one"* (`:1199-1206`).
`GetRootOrigin()` = last level's snapped origin, *"guaranteed to line up with the cell grid on every level"* (`:1268-1276`).

Derived numbers (world cache defaults, UE units = cm):
- Level cell sizes: **104.17 / 208.33 / 416.67 / 833.33 cm**; level 0 covers **50 m** across (half-extent 2500), level 3 covers **400 m**.
- Grid slots: 4 × 48³ = **442,368**; but the probe atlas holds only **128×128 = 16,384 allocated probes** at once
  (`LumenRadianceCache.cpp:1477` `MaxNumProbes = ProbeAtlasResolutionInProbes.X * .Y`).
- Grid resolution is halved in `LumenFastCameraMode` (`LumenScreenProbeGather.cpp:703-712`).

**Sparse marking.** Indirection texture = `Texture3D<uint>` sized `(GridRes*NumClipmaps, GridRes, GridRes)`, `PF_R32_UINT`
(`LumenRadianceCache.cpp:1325-1344`). It is cleared to `INVALID_PROBE_INDEX` each frame (`:1701-1716`) and then the
*consumer* marks the 8 probes around each queried position:
`LumenRadianceCacheMarkCommon.ush:102-124` (`MarkPositionUsedInIndirectionTexture` writes `USED_PROBE_INDEX` into the
8 corners of the containing cell). Screen probes call it at their reconstructed world position
(`LumenScreenProbeGather.usf:356-363`). Comment on the contract
(`LumenRadianceCacheInterpolation.ush:176-178`):
> *"Only positions that were marked during FMarkUsedRadianceCacheProbes can be queried, this version does not check if the position was marked correctly. See UnmappedDebugColor for visualizing these errors"*

**Moved/relocated — two distinct mechanisms:**

(a) **Clipmap shift → the probe follows its world position, not its grid index.**
`LumenRadianceCacheUpdate.usf:59-67` reconstructs last frame's probe world position (including the pre-view-translation
delta), then `UpdateCacheForUsedProbesCS` (`:77-165`) tests whether the *new* grid cell contains it:
```hlsl
 92  float3 ProbeTranslatedWorldPosition = GetLastFrameProbeTranslatedWorldPosition(LastFrameProbeCoord, ClipmapIndex);
 93  int3 ProbeCoord = GetRadianceProbeCoord(ProbeTranslatedWorldPosition, ClipmapIndex);
...
112  if (ProbeUsedMarker == USED_PROBE_INDEX || FrameNumber - LastUsedFrameNumber < NumFramesToKeepCachedProbes)
115      bReused = true;   // keep probe index + its cached radiance, re-bind into the new cell
...
152  if (!bReused) FreeProbeIndex(LastFrameProbe.ProbeIndex);   // push to free list
```
`NumFramesToKeepCachedProbesHeuristic = min(2, ...)` when free probes < budget (`:103-110`).
Probe index allocation: persistent free-list or linear (`AllocateUsedProbesCS`, `:288-389`, `AllocateProbeIndex` `:245-266`).

(b) **Probe inside geometry → `ProbeWorldOffset`** (`Buffer<float4>`, `PF_FloatRGBA` if typed UAV load/store else RGBA32F —
`LumenRadianceCache.cpp:1664-1665`). Computed *for the probes being traced this frame* in
`ComputeProbeWorldOffsetsCS` (`LumenRadianceCache.usf:146-282`), mode `r.Lumen.RadianceCache.IrradianceProbeOffsetMode`
(default **1**, `LumenRadianceCache.cpp:61-67`: "0: No offsetting, 1: Offset irradiance probes using GBuffer (when probes
are on screen), 2: Offset using Global SDF").
- G-buffer path (`:156-224`): 16-tap Hammersley kernel, requires the probe to project behind the depth
  `ProbeProjectedPosition.w > SampleSceneDepth - .05f * CellSize`; per-sample
  `IdealProbeOffset = (N*0.2 + V*0.8) * (0.75 * CellSize * 0.25)`, accepted only if `all(abs(SampleWorldOffset) < 0.25*CellSize)`,
  accumulated with `InterlockedAdd` at **128 quantisation steps per cell**, then averaged.
- SDF path (`:226-281`): only if `GetDistanceToNearestSurfaceGlobal(center) < 0.05f*CellSize`; samples a 4×4×4 = **64**
  offsets on the grid `(X,Y,Z)*2/3 - 1` scaled by `0.25*CellSize`, keeps the max-SDF offset, accepted if
  `BestOffset.w >= TooCloseThreshold` (0.05*CellSize).
Offsets are reset to 0 when a probe index is allocated (`LumenRadianceCacheUpdate.usf:321`) and are added to the probe
centre everywhere (trace `LumenRadianceCache.usf:839`, filter `:1032`, interpolation
`LumenRadianceCacheInterpolation.ush:126-129`).

Clipmap selection + **dithering at the seams** (`LumenRadianceCacheInterpolation.ush:136-154`): the outermost cell is
faded with `InvClipmapFadeSize` and compared against a dither random, so a position belongs to a deeper clipmap
stochastically; `GetRadianceCacheCoverage` computes
`MinTraceDistanceBeforeInterpolation = ProbeTMin + cellSize * sqrt(3)` (`:179-194`, `LumenRadianceCacheCommon.ush:46-47`).
`Coverage` is only valid if **all 8 surrounding probes are allocated**
(`GetRadianceCacheCoverageWithUncertainCoverage`, `LumenRadianceCacheCommon.ush:11-51`).

---

## 2. WHAT ONE PROBE STORES — a 32×32 octahedral radiance map + a 32×32 depth map. No SH, no single RGB.

Per probe, in the **probe atlas** (all probes tiled into one 2D texture; `ProbeAtlasResolutionModuloMask/DivideShift`
do the tiling math, `LumenRadianceCache.cpp:152-153`):

| resource | format | size (defaults) | purpose | ref |
|---|---|---|---|---|
| Radiance (trace res) | `Lumen::GetLightingDataFormat()` = `PF_FloatR11G11B10` (or `PF_FloatRGBA` / `PF_A32B32G32R32F` via `r.Lumen.LightingDataFormat`) | 128×32 = 4096² | raw traced radiance, per direction, pre-exposed | `LumenRadianceCache.cpp:1476,1585-1601`; `Lumen.cpp:94-108` |
| FinalRadianceAtlas | same, with mip chain (mips = `FinalRadianceAtlasMaxMip+1`) | 128×34 = 4352², 1 mip | post-filter + border-wrap, what interpolation samples | `LumenRadianceCache.cpp:1530-1550` |
| DepthProbeAtlas | **`PF_R16_UINT`** | 4096² | per-direction hit distance + hit flags | `:1461-1467` |
| SkyVisibility | `PF_R8` (optional, `Configuration.bSkyVisibility`) | 4096² | per-direction sky visibility | `:1604-1623` |
| FinalIrradianceAtlas | lighting format, `IrradianceProbeResolution+2` per probe | only if `CalculateIrradiance` | octahedral irradiance (for the irradiance-field consumer) | `:1479-1501` |
| ProbeOcclusionAtlas | `PF_G16R16F` (mean, mean²) | only if `CalculateIrradiance` | Chebyshev probe occlusion | `:1508-1526` |
| ProbeValid | `R8_UINT`/`R32_UINT` buffer | 1 per probe | probe-not-inside-geometry mask | `LumenRadianceCache.cpp:2657`; `LumenRadianceCache.usf:1360-1386` |
| ProbeWorldOffset | float4 buffer, 1 per probe | relocation offset (+a flag in .w) | §1b | `:1769` |

**Direction encoding**: `InverseEquiAreaSphericalMapping` / `EquiAreaSphericalMapping` (equi-area octahedral),
`LumenRadianceCacheInterpolation.ush:252`.

**Depth encoding is 16 bits total** — `LumenRadianceCacheInterpolation.ush:214-235`:
```hlsl
uint EncodeProbeDepth(FProbeDepth ProbeDepth) {
  float HitDistance = clamp(ProbeDepth.HitDistance, 0.0f, 65503.0f);
  uint Encoded = f32tof16(HitDistance) & 0x7FFE;   // f16 with the 2 low bits masked off
  Encoded |= ProbeDepth.bFrontface ? 0x8000 : 0;   // sign bit = frontface
  Encoded |= ProbeDepth.bTwoSided ? 0x1 : 0;       // 1 mantissa bit = two-sided (e.g. foliage)
  Encoded = ProbeDepth.bHit ? Encoded : 0xFFFF;    // 0xFFFF = miss
}
```
Range ~65503 cm, f16 precision ⇒ ~cm accuracy near the probe, decimetres far away.

**Where the depth is used (this is the part your unoccluded blur lacks):**
1. **Probe filter neighbour rejection** — the occlusion test *is* the depth atlas (§4). `LumenRadianceCache.usf:1029-1096`.
2. **Probe occlusion / validity** — `PrepareProbeOcclusionCS` (`:1247-1388`) turns the depth map into
   `float2(meanDepth, meanDepth²)` (Chebyshev), and `bBackface` counts + closest-frontface give `RWProbeValid`
   (HWRT: `NumBackfaces < 0.1 * Res²` **and** `ClosestFrontfaceDistance > 0.01 * CellSize`; SWRT: probe centre's
   distance to the global SDF `> SampledClipmapVoxelExtent * 0.5 * CoveredExpandSurfaceScale`).
3. **Occlusion-at-hit** for the irradiance-field consumer (`LumenIrradianceFieldInterpolation.ush:86-110`, Chebyshev).
4. **Hit distance out-parameter** of interpolation — `SampleRadianceCacheInterpolation` returns `TraceHitDistance`,
   which the screen-probe tracer passes on as `RadianceCacheHitDistance` for the spatial filter / AO / sky-leak
   (`LumenRadianceCacheInterpolation.ush:237-243, 499-513`; `LumenScreenProbeTracing.usf:825-837`).
5. **Next frame's trace pullback** (HWRT): `Ray.TMin = max(ProbeTMin, TraceHitDistance - PullbackBias)`
   (`LumenRadianceCacheHardwareRayTracing.usf:246`).

---

## 3. TRACE — 256/1024/4096 rays per probe, ~100 probes per frame, from the probe centre, into the Lumen Scene surface cache (SWRT) or TLAS (HWRT)

**Rays per probe = `RadianceProbeResolution²` = 32² = 1024** at the base level; the trace grid is
`TraceResolution = (RadianceProbeResolution / 2) << TraceTileLevel` (`LumenRadianceCache.usf:835`,
`LumenRadianceCacheHardwareRayTracing.usf:222`) so:
- level 0 = 16² = **256** rays, upsampled 2× (far / forced-downsample),
- level 1 = 32² = **1024** rays (normal),
- level 2 = 64² = **4096** rays (supersampled, downsampled to 32²; depth = min/OR).
Cost model matches exactly (`LumenRadianceCacheUpdate.ush:10-12`):
`PROBE_TRACE_COST_DOWNSAMPLED 1 / NORMAL 4 / SUPERSAMPLED 16`, budget
`GetProbeTraceCostBudget() = NumProbesToTraceBudget * PROBE_TRACE_COST_NORMAL` (`:18-21`).
Distance thresholds: `r.Lumen.RadianceCache.DownsampleDistanceFromCamera` = **4000** (`LumenRadianceCache.cpp:108-114`),
`SupersampleDistanceFromCamera` = **-1** ⇒ `max(-1,0)²=0` ⇒ supersampling by distance is **off by default**
(`:101-106`, `:1372-1376`); probes beyond 4000 cm are forced to 256 rays (`:29-38`). If a screen-probe BRDF SH is
supplied (it is, for the world cache), tiles are instead built by BRDF importance sampling
`GenerateAdaptiveProbeTraceTilesCS` with `SupersampleTileBRDFThreshold` = **0.1**, `ForcedUniformLevel` = 1
(`LumenRadianceCache.cpp:1367-1370`, `:2384-2411`, usf `:431-655`), on top of a 9-coefficient screen-probe BRDF SH
(`NUM_PDF_SH_COEFFICIENTS 9`, `LumenScreenProbeImportanceSamplingShared.ush:5`).

**Per-frame cadence = 100 probes** (`NumProbesToTraceBudget`, scaled by
`LumenFinalGatherLightingUpdateSpeed` clamped to **[0.5, 4]** and ×10 when the family is being edited:
`LumenScreenProbeGather.cpp:741-743`) ⇒ ~**102,400 rays/frame** by default. With 16,384 atlas slots a full refresh
takes ~164 frames at 100/frame. Selection is by a **16-bucket log2 priority histogram**
(`LumenRadianceCacheUpdate.usf:221-241`):
```hlsl
// [1;N]
uint FramesBetweenTracedAndUsed = LastUsedFrameIndex > LastTracedFrameIndex ? LastUsedFrameIndex - LastTracedFrameIndex : 1;
float UpdateImportance = FramesBetweenTracedAndUsed / (ClipmapIndex + 1.0f);
BucketIndex = PRIORITY_HISTOGRAM_SIZE - 1 - clamp(log2(UpdateImportance), 0, PRIORITY_HISTOGRAM_SIZE - 2);
```
Bucket 0 is reserved for **never-traced (new) probes** and they are always updated; over-budget new probes are
force-downsampled (`:466-479`). `SelectMaxPriorityBucketCS` (`:393-429`) and `AllocateProbeTracesCS` (`:431-528`) fill
the budget; `PRIORITY_HISTOGRAM_SIZE = 16` (`LumenRadianceCache.cpp:119`).

**Ray origin/direction**: `Ray.Origin = ProbeTranslatedWorldCenter (+ ProbeWorldOffset)`, direction = the octahedral
texel centre direction — **no jitter by default**, with the comment
`// No temporal accumulation, so just reads as dirty lighting` / `#define JITTER_TRACE_DIRECTION 0`
(`LumenRadianceCache.usf:843-853`). Cone half angle is set from the *uniform* solid-angle distribution, **not** from
octahedral distortion: `ConeHalfAngle = acosFast(1 - 1/(TraceResolution*TraceResolution))` (`:860`).

**Bias / ray range:**
```hlsl
// LumenRadianceCache.usf:855-856
float FinalMinTraceDistance = GetRadianceProbeTMin(TraceData.ClipmapIndex);   // = |(c,c,c)| * ProbeTMinScale(=1.0) = c*sqrt(3)
float FinalMaxTraceDistance = MaxTraceDistance;
...
TraceInput.bZeroRadianceIfRayStartsInsideGeometry = true;                     // :791
TraceInput.bExpandSurfaceUsingRayTimeInsteadOfMaxDistance = false;            // :793
```
`MaxTraceDistance = Lumen::GetMaxTraceDistance(View)` = `clamp(LumenMaxTraceDistance * r.Lumen.TraceDistanceScale, .01, 0.5*UE_OLD_WORLD_MAX)`
(`LumenDiffuseIndirect.cpp:228-231`, `Lumen.h:57`). `MinTraceDistance` for probes =
`clamp(max(r.Lumen.DiffuseIndirect.SurfaceBias(=5.0), r.Lumen.DiffuseIndirect.MinTraceDistance(=0)), .01, 1000)` with
`SurfaceBias` forced to 0 — comment: *"Probe tracing doesn't have surface bias, but should bias MinTraceDistance due to
the mesh SDF world space error"* (`LumenDiffuseIndirect.cpp:349-362`). `StepFactor = 1`, `MinSampleRadius = 10`
(`:41-55`, `:320-346`).

**Hit-shading path** — three mutually exclusive ones:
1. **SWRT default** (`TraceForProbeTexel`, `LumenRadianceCache.usf:784-803`): `ConeTraceLumenSceneVoxels` — global-SDF /
   mesh-SDF cone trace (`LumenSoftwareRayTracing.ush:862-889`) that shades the hit by **`SampleLumenMeshCards` against
   the Lumen Scene surface cache** (`EvaluateGlobalDistanceFieldHit`, `:637-710`) + `ApplySkylightToTraceResult`.
   There is no screen-space read at all. Radiance is multiplied by `CachedLightingPreExposure` (`:889`).
2. **HWRT, surface-cache pass** (`RAY_TRACING_PASS_DEFAULT`): `TraceSurfaceCacheRay` (`LumenRadianceCacheHardwareRayTracing.usf:290`),
   built with `Lumen::ESurfaceCacheSampling::AlwaysResidentPagesWithoutFeedback` (`LumenRadianceCacheHardwareRayTracing.cpp:174`).
3. **HWRT, hit-lighting pass** (`RAY_TRACING_PASS_HIT_LIGHTING`, when `LumenRadianceCache::UseHitLighting`, `.cpp:49-57`):
   `TraceAndCalculateRayTracedLighting` with real direct+skylight at the hit (`usf:277-287`), SER enabled, minimal
   payload disabled (`.cpp:135-151, 439, 612-613`).
Then a **far-field pass** fires a *second* ray **only for texels that missed** (compacted in
`RadianceCacheCompactTracesCS`, `usf:473-558`), gated by `r.Lumen.RadianceCache.HardwareRayTracing.FarField` = 1
(`.cpp:28-33`) and `Configuration.bFarField` (default true).
Each ray: `Ray.TMin = max(ProbeTMin, TraceHitDistance - PullbackBias)`,
`Ray.TMax = max(ClippedNearFieldMaxTraceDistance - PullbackBias, TMin)`, culling **disabled** —
*"Disable culling in order to minimize leaking when rays start inside walls"* (`usf:264`).

---

## 4. FILTER / LEAK AVOIDANCE — **the filter is a 6-neighbour directional gather with a bidirectional depth/occlusion test; there is no propagation blur at all.** This is your missing piece.

`FilterProbeRadianceWithGatherCS`, `LumenRadianceCache.usf:1003-1123`, enabled by
`r.Lumen.RadianceCache.SpatialFilterProbes` = **true** (`LumenRadianceCache.cpp:46-51`). It is **one pass**, reads the
*unfiltered* probe atlas of the same frame (no multi-iteration propagation), writes to a temp atlas, then
`FixupBordersAndGenerateMipsCS` copies to the final atlas.

It gathers the **6 axis neighbours only** (not 3×3×3 = 26; offsets at `:1034-1040`), at the *same* texel coordinate
(i.e. the same direction), and weights each neighbour by three independent terms:

```hlsl
// LumenRadianceCache.usf:1058-1103
float OcclusionWeight = NeighborProbeDepth.bFrontface || NeighborProbeDepth.bTwoSided ? 1.0f : 0.0f;

// Test whether probe can see neighbor probe's ray starting point and if occluded then discard the neighbor radiance to reduce leaking.
// Need to offset starting point as all probe traces start after GetRadianceProbeTMin and there's no depth information in the region where probe TMin spheres intersect.
// That offset can't be also too large due to limited probe angular resolution making it pretty inaccurate at connecting paths at larger distances.
// Also run this test in reverse by checking whether neighbor probe can see probe's ray starting point, which improves chances of finding a thin wall between two probes.
float OcclusionTestOffset = 2.0f * ProbeCellSizeForOcclusionTest;
{ // Probe -> Neighbor's ray
  float3 NeighborOcclusionTestPosition = NeighborTranslatedWorldPosition + OcclusionTestOffset * WorldConeDirection;
  float3 ToNeighborOcclusionPosition = NeighborOcclusionTestPosition - TraceData.ProbeTranslatedWorldCenter;
  uint2 ProbeTexelCoordForNeighborOcclusionPosition = InverseEquiAreaSphericalMapping(ToNeighborOcclusionPosition) * RadianceProbeResolution;
  float ProbeDepthForNeighborOcclusionPosition = DecodeProbeDepth(DepthProbeAtlasTexture[ProbeTexelCoordForNeighborOcclusionPosition + ProbeAtlasBaseCoord]).HitDistance;
  if (ProbeDepthForNeighborOcclusionPosition * ProbeDepthForNeighborOcclusionPosition < dot(ToNeighborOcclusionPosition, ToNeighborOcclusionPosition)) OcclusionWeight = 0.0f;
}
{ // Neighbor -> Probe's ray (same test, reversed) }
// Clamp neighbor's hit distance to our own.  This helps preserve contact shadows, as a long neighbor hit distance will cause a small NeighborAngle and bias toward distant lighting.
if (NeighborProbeDepth.bHit) NeighborProbeDepth.HitDistance = min(NeighborProbeDepth.HitDistance, HitDistance);
float3 NeighborHitPosition = NeighborTranslatedWorldPosition + WorldConeDirection * NeighborProbeDepth.HitDistance;
float3 ToNeighborHit = NeighborHitPosition - TraceData.ProbeTranslatedWorldCenter;
float NeighborAngle = acosFast(dot(ToNeighborHit, WorldConeDirection) / length(ToNeighborHit));
float AngleWeight = 1.0f - saturate(NeighborAngle / SpatialFilterMaxRadianceHitAngle);
float Weight = AngleWeight * OcclusionWeight;
Lighting += RadianceProbeAtlasTexture[ProbeTexelCoord + NeighborProbeAtlasBaseCoord].xyz * Weight;
TotalWeight += Weight;
...
RWRadianceProbeAtlasTexture[...] = QuantizeForFloatRenderTarget(Lighting / TotalWeight, RandomScalar);
```
Note `float TotalWeight = 1.0f;` (`:1024`) — the probe's own radiance always has weight 1, so this is a
**normalised, conservative, occlusion-gated neighbour blend**, not an unweighted blur.

Constants: `SpatialFilterMaxRadianceHitAngle` = **0.2**
(`LumenRadianceCache.cpp:69-75`, comment says "In Degrees" but the shader compares an `acosFast` result, so it is
**0.2 rad ≈ 11.5°** — the comment is wrong). `ProbeCellSizeForOcclusionTest = ProbeTMin == 0 ? cellSize : ProbeTMin`
(`usf:1029`) i.e. `cellSize*sqrt(3)` by default. `OcclusionKernelCosineExponent` = 128 is used only by the
`FILTER_PROBE_OCCLUSION 0` path (currently disabled, `:1275`).

**Six distinct leak defences, in order of importance for you:**
1. **`ProbeTMin`** = `|(cellSize,cellSize,cellSize)| * ProbeTMinScale(=1.0)` = **cellSize·√3** (`LumenRadianceCache.cpp:128, 1319`)
   — a per-clipmap *blind sphere*: no ray can report a hit closer than the probe's own cell diagonal.
   Additionally `bZeroRadianceIfRayStartsInsideGeometry` forces the radiance factor to 0 when
   `HitTime <= MinTraceDistance` (`LumenSoftwareRayTracing.ush:658-661`), and the card sample point is offset off the
   surface by one SDF voxel — *"Offset card grid cell from surface in order to minimize leaking"* (`:644-645`).
2. **The 6-neighbour bidirectional depth occlusion test** above (the core answer).
3. **`bFrontface || bTwoSided` gate on the neighbour's own depth** — a neighbour whose ray exited backwards
   (`bBackface`) contributes 0.
4. **Hit-distance clamping** to the receiving probe's depth before computing the angle weight (contact shadows).
5. **`MinTraceDistanceBeforeInterpolation = ProbeTMin + cellSize*√3`** ⇒ a caller may only consult the cache for hits
   **beyond ≈ 2·√3·cellSize** (~361 cm at level 0), enforced at every call site by clamping the trace length:
   `LumenScreenProbeTracing.usf:130, 761`; `LumenScreenProbeHardwareRayTracing.usf:134`;
   `LumenReflections.usf:410`; `LumenTranslucencyVolumeLighting.usf:119`; `LumenVisualize.usf:66`.
6. **Probe relocation** off surfaces (§1b) and **`ProbeValid`** (`LumenRadianceCache.usf:1360-1386`).

Also note the ordering guarantee: `ProbeWorldOffset` is computed **before** tracing
(`LumenRadianceCache.cpp:2216-2251`), so every stage uses the same relocated centre (`usf:839, 1032`).

**Your measured symptom** (same relative weight in dark and mid-bright thirds) is exactly what an occlusion-free
normalised blur does: normalising by total weight makes the *relative* spread independent of the light field, so the
blur acts as a low-pass with unity DC gain and no notion of whether the two probes can see each other. UE never
normalises unweighted: `Weight = AngleWeight * OcclusionWeight`, and `OcclusionWeight` becomes an all-or-nothing 0 as
soon as *either* directed depth test fails. If you cannot afford the neighbour depth maps, the cheapest faithful
subset is: keep the multi-step propagation but **do not normalise** — accumulate `Σ w·L / Σ w` where `w` includes a
"can A see B" step test derived from your screen-injection depth, and *drop* (never renormalise) contributions whose
step test fails; a dropped neighbour must lower the total weight, not be redistributed. That single change converts
your uniform spread into a light-following one.

---

## 5. INTERPOLATION AT A HIT

### 5a. World radiance cache path (`LumenRadianceCacheInterpolation.ush`) — 8 probes, parallax-corrected, **no per-probe depth rejection**

`SampleRadianceCacheInterpolated` (`:429-497`):
1. Cone → mip: `NumTexels = sqrt(1 - cos(ConeHalfAngle)) * RadianceProbeResolution;`
   `MipLevel = clamp(log2(NumTexels), 0, FinalRadianceAtlasMaxMip);` (`:443-444`).
2. `CornerProbeCoordFloat = ProbeCoordFloat - 0.5; CornerProbeCoord = floor(...); LerpAlphas = frac(...)` (`:446-448`).
3. Default path: **8 probes**, full trilinear with `LerpRadianceCacheSample` (`:475-492`). With
   `RADIANCE_CACHE_STOCHASTIC_INTERPOLATION` (a *reflections-only* permutation,
   `LumenReflectionHardwareRayTracing.cpp:130,418`) it is a single stochastically-chosen probe (`:450-472`).
4. Each probe sample goes through `SampleRadianceCacheProbeWithParallaxCorrection` (`:332-404`):
```hlsl
float ReprojectionRadius = ReprojectionRadiusScale * ProbeTMin;      // 1.5 * cellSize*sqrt(3)
float T = RayIntersectSphere(TranslatedWorldSpacePosition, WorldSpaceDirection, float4(ProbeTranslatedWorldPosition, ReprojectionRadius)).y;
ReprojectedDirection = TranslatedIntersectionPosition - ProbeTranslatedWorldPosition;
// Cancel out the attenuation effect when moving towards/away from a probe texel to mitigate the grid like pattern
CorrectionFactor = T * T / (ReprojectionRadius * dot(ReprojectedDirection, WorldSpaceDirection));
```
   i.e. the probe texel is fetched in the direction **from the probe to the ray/cone's intersection with the
   "probe sphere"** — this is the geometry-aware part of the lookup: a probe whose sphere the ray misses cannot
   contribute a wildly wrong direction.
5. **The depth test that would reject a probe belonging to another surface is DISABLED**:
   `#define RADIANCE_CACHE_DEPTH_TEST_SPHERE_PARALLAX 0` (`:19-21`) and the block at `:358-372` is additionally
   guarded by `&& 0`; `TRACE_THROUGH_PROBE_DEPTHS_REFERENCE` (`:374-399`) is likewise `0`, with the comment
   *"@note - no depth mips implemented"*.
   ⇒ **Rejection of foreign-surface probes happens at the filter stage (§4), not at interpolation.** The only
   interpolation-time guards are the 8-probe coverage check (`LumenRadianceCacheCommon.ush:26-44`) and the parallax
   sphere.
6. Sky resolve: `RadianceCacheResolveSky` lerps toward the analytic skylight by `SkyVisibility` (`:415-427`);
   the caller adds it with the remaining path throughput and then closes the path:
```hlsl
// LumenRadianceCacheInterpolation.ush:499-513
if (Transparency > 0.0f) {
  FRadianceCacheSample RadianceCacheSample = SampleRadianceCacheInterpolated(...);
  if (ShowBlackRadianceCacheLighting == 0) Lighting += RadianceCacheSample.Radiance * Transparency;
  TraceHitDistance = RadianceCacheSample.TraceHitDistance;
  Transparency = 0.0f;   // one cache lookup terminates the path
}
```

### 5b. The path that *does* reject probes by geometry+depth — `LumenIrradianceFieldInterpolation.ush` (copy this)
`InterpolateIrradianceFromProbe` (`:60-133`) builds the per-probe weight you are missing:
```hlsl
float3 SamplePositionToProbe = ProbeTranslatedWorldPosition - TranslatedSamplePosition;
float DistanceToProbe = length(SamplePositionToProbe);
float Weight = 1.0f;
// 1) wrap shading: normal-facing weight, floor of 0.2
float WrapShading = (dot(normalize(ProbeTranslatedWorldPosition - TranslatedPixelPosition), WorldSpaceNormal) + 1.0f) * .5f;
Weight *= WrapShading * WrapShading + .2f;
// 2) Chebyshev probe occlusion from the (mean, mean^2) depth atlas
float2 MeanAndMeanSq = SampleProbeOcclusion(ProbeIndex, -SamplePositionToProbe);
float ChebyshevWeight = 1.0f;
if (DistanceToProbe > MeanAndMeanSq.x) {
  float Variance = abs(Square(MeanAndMeanSq.x) - MeanAndMeanSq.y);
  float VisibilityWeight = Variance / (Variance + Square(DistanceToProbe - MeanAndMeanSq.x));
  ChebyshevWeight = max(VisibilityWeight * VisibilityWeight * VisibilityWeight, 0.0f);
}
Weight *= max(ChebyshevWeight, 0.05f);
Weight = max(Weight, MinWeight);                                // MinWeight = .0001
float WeightCrushThreshold = .2f; if (Weight < WeightCrushThreshold) Weight *= Square(Weight) / Square(WeightCrushThreshold);
// 3) validity mask
float ProbeValid = max(SampleProbeValid(ProbeIndex), .001f); Weight = max(Weight * ProbeValid, MinWeight);
Weight *= TrilinearWeight;
```
The gather (`SampleIrradianceCacheInterpolated`, `:141-241`) clamps the corner coord to
`[0, GridRes-2]`, iterates the **8** neighbours, skips `!ProbeIndirection.bValid`, and rejects the whole lookup when
`TotalWeight < MinWeight * 8 * 1.5` (`:237-241`). Normalisation is `Irradiance = Square(Σ sqrt(Sample)·w / Σw)` —
the `sqrt`/`Square` pair is the standard SH-style "bent" encoding of an irradiance probe, which is how UE gets a
usable directional falloff out of a *probe-local* octahedral layout rather than increasing SH order.

---

## 6. INFINITE BOUNCES / FEEDBACK — the cache is **not** self-recursive; the loop closes through the Lumen Scene surface cache, and is bounded by a blind radius, incremental updates, and a hard reset on global-lighting change

**The loop.** Probe traces hit the **surface cache**, whose `FinalLightingAtlas` is
`CombineFinalLighting(Albedo, Emissive, DirectLighting, IndirectLighting)` (`LumenSceneLighting.usf:503-506`) where
`IndirectLighting` comes from the previous frame's screen-probe gather (and radiosity). So:
`surface cache (frame N-1) → radiance-cache probes (frame N) → screen-probe GI (frame N) → surface cache (frame N) → …`
The radiance cache is never read by its own trace pass, and `SampleRadianceCacheAndApply` sets `Transparency = 0`
after one lookup (`LumenRadianceCacheInterpolation.ush:511`), so there is no *within-frame* recursion either.

**What stops divergence:**
1. **Self-intersection / blind radius (the actual per-frame gain limiter).**
   SWRT: `ProbeTMin` = cellSize·√3 (≈180 cm at level 0, default `ProbeTMinScale = 1.0f`,
   `LumenRadianceCache.cpp:128,1319`) plus `bZeroRadianceIfRayStartsInsideGeometry` (`LumenRadianceCache.usf:791`;
   applied at `LumenSoftwareRayTracing.ush:658-661`).
   HWRT: `r.Lumen.HardwareRayTracing.MinTraceDistanceToSampleSurfaceCache` = **10.0**, with the explicit comment
   (`LumenHardwareRayTracingCommon.cpp:160-165`):
   > *"Ray hit distance from which we can start sampling surface cache in order to fix feedback loop where surface cache texel hits itself and propagates lighting."*
   applied as zeroed radiance/direct/indirect at `LumenHardwareRayTracingCommon.ush:670-676`; plus
   `Ray.TMin = max(ProbeTMin, TraceHitDistance - PullbackBias)` (`LumenRadianceCacheHardwareRayTracing.usf:246`)
   and `PullbackBias = Lumen::GetHardwareRayTracingPullbackBias()` (`.cpp:434`).
2. **Incremental update ⇒ tiny per-frame loop gain.** Only ~100 of 16,384 probes are re-traced per frame (§3), and
   each probe holds its old value until its turn; the priority histogram plus the budget makes this a
   Jacobi-style damped iteration where the cache only "advances" one hop per ~N frames.
3. **No write-back from cache-traced probes into the surface cache feedback buffer** — the HWRT shader is compiled
   with `Lumen::ESurfaceCacheSampling::AlwaysResidentPagesWithoutFeedback`
   (`LumenRadianceCacheHardwareRayTracing.cpp:174`), and `GetLumenCardTracingParameters(..., /*bSurfaceCacheFeedback*/ false, ...)`
   (`:512`, also `LumenRadianceCache.cpp:2525`). The cache cannot ask for higher-res pages of the texels it repeatedly
   hits, which would otherwise be a positive-feedback path.
4. **Hard reset when global lighting changes materially.** `UpdateGlobalLightingState`
   (`LumenSceneRendering.cpp:2471-2540`) compares the old/new max of the directional light colour and skylight colour
   and sets `bPropagateGlobalLightingChange` when `Ratio > 4.0f || Ratio < .25f`. That flag then:
   - forces a full-size temp atlas (`LumenRadianceCache.cpp:1642-1645`),
   - sets `NumProbesToTraceBudget = UINT32_MAX` for one frame — *"Don't throttle on one-time updates"* (`:1670-1673`),
   - clears the persistent cache (`bPersistentCache = ... && !bPropagateGlobalLightingChange`, `:1691-1696`),
   - and clears the final atlases if `GRadianceCacheForceFullUpdate` (`:1573-1580`).
   (`r.LumenScene.PropagateGlobalLightingChange` can disable it, `LumenSceneRendering.cpp:222-223, 2535-2538`.)
5. **Resizes/extent/distribution-base/pre-exposure changes also reset** — `bResetState`
   (`LumenRadianceCache.cpp:1281-1284`): `ClipmapWorldExtent`, `ClipmapDistributionBase`, `CachedLightingPreExposure`,
   `ProbeOffsetMode`. Note the **pre-exposure** round-trip (`RadianceCacheOneOverCachedLightingPreExposure`,
   `LumenRadianceCacheInterpolation.ush:74,272`) — radiance is stored pre-exposed and de-exposed on read, so exposure
   changes never enter the loop.
6. **Bounded filter step** — `Σ w·L / Σ w` with `Σw ≥ 1` and neighbour hit distances clamped to the receiver's own
   (`LumenRadianceCache.usf:1024, 1092-1108`), so one filter pass is a convex-ish combination, never an amplification.
7. **Normalisation of the last step** — `Irradiance = Square(Σ sqrt(L)·w / Σ w)` and
   `bSuccess = TotalWeight >= MinWeight*8*1.5f` (`LumenIrradianceFieldInterpolation.ush:233-241`).

There is **no explicit gain/energy clamp and no convergence comment** on the cache itself; convergence is by
construction (blind radius + incremental refresh + reset-on-change), which is why "unbounded energy" is not one of
Lumen's failure modes.

---

## 7. Deltas vs. your design (actionable, derived only from the above)

| your design | UE's answer |
|---|---|
| 32³ RGBA16F grid over scene bounds | camera-anchored 4-level clipmap, 48³ cells/level, cell 1.04 m doubling; only 16,384 probes allocated, at marked positions only. Bounds-anchored is fine, but per-probe **directional** storage is what lets UE use a cheap bilinear + cone-mip lookup instead of a dense 3D DDA. |
| one RGB per probe | 32×32 octahedral radiance **+ 32×32 R16_UINT depth (f16 distance, frontface bit, two-sided bit, 0xFFFF miss)**. The depth map is the entire leak defence. If you can only add one thing: add the depth channel. |
| injected once/frame from screen GI | UE traces the **surface cache / TLAS** at 1024 rays/probe, 100 probes/frame, ~164 frames for a full refresh, jitter disabled, `TMin` = cell·√3. |
| unoccluded confidence-weighted 3³ blur, trust halved per step, tracer scales by trust | UE: **single-pass, 6-neighbour, same-direction gather**, `weight = angleWeight × occlusionWeight`, with a **bidirectional depth test against both probes' depth maps** at `2·ProbeTMin`, plus neighbour `bFrontface‖bTwoSided` gate and hit-distance clamping; normalised with the centre at weight 1. No propagation, no trust decay — trust is replaced by a binary visibility test. |
| tracer samples cache for off-screen hits | Same, but only beyond `MinTraceDistanceBeforeInterpolation = 2·cell·√3`, with sphere-parallax reprojection (`radius = 1.5·ProbeTMin`), `CorrectionFactor = T²/(R·dot(dir,rayDir))`, and 8-probe trilinear. For an SH/irradiance-style probe, use `LumenIrradianceFieldInterpolation.ush:60-133` weights (wrap shading × Chebyshev occlusion × validity × trilinear, with a `.2` crush threshold). |
| infinite bounces | Not self-recursive: cache → screen GI → surface cache → cache, bounded by the blind radius (10 cm HWRT / cell·√3 SWRT), 100 probes/frame refresh, no surface-cache write-back, and a full reset when directional/skylight colour changes by >4× or <0.25×. |

**No files were modified.** Source study only.
