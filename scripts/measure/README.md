# Measurement instruments

The scripts the GI work is read with. They are here rather than beside the captures because a capture is
4 MB of scratch that a `-Clean` may take with it, while these are the reason a recorded number can be
re-derived at all: `docs/gi_hit_shading.md` and `docs/level2_handoff.md` quote means, 4x4 tile tables and
per-material buckets, and every one of them came out of one of these.

They need Python 3 with Pillow (`python -c "import PIL"` must work) and nothing else. The captures they
read come from `scripts/windows/capture.ps1`, which lands PNGs in `<BuildDir>/gi-probe/`.

## The instruments

| Script | Question it answers |
| --- | --- |
| `mean.py <a.png> ...` | the per-channel means. Every number in the records is quoted in **mean green** - one channel, so that a colour change cannot hide a luminance change |
| `diff.py <a.png> <b.png>` | how two frames differ: signed mean per channel, how many pixels differ at all / by >1 / by >4, the worst pixel, and a 4x4 tile table of the mean difference. **The workhorse**, and the tile table is the part that matters - a single frame-level mean cannot tell a global bias from a change ordered by the scene |
| `tiletab.py <on.png> <off.png>` | the same 4x4 table as an absolute difference AND as a percentage of that tile's own mean in the off-frame. This is the form the L2.1 probe-cache result is recorded in ("interior tiles lose 4-7%, sky-facing 0.1-0.6%") and the form that shows a change is *ordered by the scene* rather than flat |
| `corr.py <a_on> <a_off> <b_on> <b_off>` | the Pearson correlation between two per-pixel difference images: do two estimators agree about *where* they moved, not merely about how much? A frame-level coincidence of means without correlation is not agreement |
| `spec_material.py <on> <off> <rough> <metal> <normal>` | an A/B delta **bucketed by material** instead of by tile - the instrument the glossy-lobe result needed, because its effect is a material response (smooth metal +10.57 against rough dielectric +0.08) and a tile table would only have shown where the spheres are. The three channel images come from `[render] gbuffer_debug` with `gbuffer_channel` 2 (roughness), 3 (metallic) and 1 (normal); the normal channel is the surface mask, because the debug view paints the cleared background black in every channel and a unit normal can never map to (0,0,0) through `normal*0.5+0.5`. **Caveat**: the debug view is tonemapped on its way to the screen, so the buckets are in display units - ordinal, not linear roughness |
| `shadow_pose.py <raster_t0> <raster_t1> <rt_t0> <rt_t1>` | does an animated skinned mesh's ray-traced shadow follow the pose? The object moves in all four captures, so the instrument is the difference OF the differences: what survives subtracting the traced motion from the raster motion is the pose dependence only the raster path has. A traced shadow stuck in the bind pose leaves it as large as the shadow's motion |
| `mask_stats.py <cascade> <rt_nobake> <rt_baked>` | is the alphaMode MASK bake moving a ray-traced shadow **towards** the raster one (which discards per pixel and is therefore the reference)? Reports the error against it before and after, and the correlation between "what the bake changed" and "how wrong the unbaked frame was" - the part that says whether it moved in the right direction rather than merely by a lot. This is the instrument whose answer was negative (docs/gi_hit_shading.md, L2.2a) |
| `mask_scan.py <ModelsDir> [asset ...]` | CPU reconnaissance over the sample assets: for every alphaMode MASK material, how many triangles would a per-triangle rule collapse at the three vertex UVs alone? The number the shader implementation is then checked against |
| `mask_rules.py <gltf> [materialIndex]` | the same question for one asset under three rules of increasing cost (3 vertex UVs / a 4x4 barycentric grid, which is what `shaders/mask_bake.comp` does / every texel of the UV triangle), plus the share the raster path effectively keeps - which is what the bake has to agree with |

## Habits these instruments encode

* **Take the control, and take it as a hash.** A knob that is supposed to do nothing must produce a
  byte-identical frame; if the claim is "this change is invisible", `scripts/windows/check_render.ps1`'s
  scenario set x2 is what verifies it. A frame that merely *looks* the same is not a control.
* **A frame-level mean is the weakest reading in this directory.** Always ask for the tile table or the
  material buckets: the whole difference between "the estimator is right" and "the frame got uniformly
  brighter" is whether the change is ordered by the scene or by the material.
* **Record the negative results with the same care.** Two of the three features in
  `docs/gi_hit_shading.md`'s later sections are off by default because a measurement said so, and the
  instrument that said it is in this directory.
* **Decompose a residual before explaining it.** The L2.3 identity test's residual looked like an
  arithmetic mismatch and was three measurements from being explained (it was rays finding geometry on a
  scene with coincident surfaces); zeroing an input to make one term vanish is what separated them.
* **Quote the arm you took the number from.** The same scene at a different ray length, frame count or
  filter setting is a different measurement - which is why the records give the config's values
  alongside every table.
