# The ZZZ-style NPR shading, and what of the reference is here

This engine's toon path began as a plain cel ramp: quantize the diffuse falloff into bands and let the
band darken the base colour. That is what `[render] toon_steps` does, and on a stylised model it reads as
grey plastic - the shadowed side is the base colour scaled down, so it loses saturation exactly where the
model wants it.

The renderer's version of the look referenced here adds a DIFFUSE WARP: the shadowed end of the ramp
lerps towards a tinted colour instead. The reference is **XIYAG's ZZZ shader** - the Blender
reproduction of *Zenless Zone Zero*'s character shading, `XIYAG-ZZZ-Shader - Base` and
`- Face` node groups, published as
[`fnoji/Blender-ZZZ-XIYAG-Shader`](https://github.com/fnoji/Blender-ZZZ-XIYAG-Shader). The credit belongs
there, and the shader source says so where the warp is implemented (`shaders/shading.glsl`,
`diffuse_warp`).

## What the reference does, and what each part is here

| the reference | where it lives here | status |
| --- | --- | --- |
| `ShadowColor1..5` - five authored shadow colours, one per band | `[render] toon_shadow_tint`, walked across the bands by `diffuse_warp` | **implemented**, one tint rather than five authored colours |
| `NTShadow Colors` / `NTSmoothstep` - the banded ramp the colours are indexed by | `toon_band` in `shaders/shading.glsl`, driven by `[render] toon_steps` / `toon_softness` | already existed; the warp rides it |
| `SpecularColor1..5` - the specular's own five bands | the cel path's hardened highlight (`toon_steps > 0`), no per-band colours | partial |
| `NTMatcap` / `Eff_MatCap` / `CombineMESphere` - the matcap, and the model's MMD sphere texture combined with it | `[render] toon_rim`: a view-space silhouette rim | **stand-in** - the rim is `pow(1 - N.V, 3)`, not a texture lookup, because the sphere maps are not in the converted model at all |
| the `- Face` group: `headOrgn` / `headFwd` / `headUp`, `Face_lightmap`, `SpecularShapeMaskDot` | - | **not implemented**: it needs a face light-map/SDF the MMD model does not carry (its extra UV sets ride along in the GLB, the map they index does not) |
| the `Outline` shader, thickness from the vertex colour | `[render] outline_color` / `outline_width` as the FALLBACK, and the material's own `mmd_edge_color` / `mmd_edge_size` from the glTF `extras` (read by the loader, carried in `material_record::npr_edge`, used by shaders/outline.vert + outline.frag) | **implemented**: per-material colour and thickness. The per-VERTEX edge scale the reference reads from the vertex colour is exported by the converter (`_EDGESCALE`) but the loader does not import that attribute yet |
| the `Glow` shader | - | **not implemented** |

## How to ask for it

```toml
[render]
toon_steps = 5                  # the ramp; 0 = plain PBR
toon_softness = 0.05            # band edge width: smaller = harder edges
toon_shadow_tint = [0.55, 0.5, 0.75]   # (1,1,1) = the warp is OFF
toon_rim = 0.8                  # 0 = no rim
outline_color = [0.06, 0.04, 0.09]     # the hull's colour, written into the G-buffer's albedo
outline_width = 0.05            # WORLD units of expansion; 0 = no hull recorded at all
```

All four defaults are the NEUTRAL values, so a config that omits them shades exactly as the renderer did
before the keys existed - which is what keeps the gate's reference frames valid. That is not a claim: the
gate is run against them.

The parameters ride two lanes appended to the light UBO (`npr_shadow`, `npr_rim`). They are appended
rather than inserted because a stage that declares the block without them still matches the buffer, which
keeps the change off every other shader that reads the light block.

## The first measurement

Two captures of the converted PMX asset (千夏 -> `scripts/pmx_to_glb.py`), one config, one camera, one
frame count, the two arms differing only in the four keys above at:

```toml
toon_steps = 5
toon_softness = 0.05
toon_shadow_tint = [0.34, 0.31, 0.44]
toon_rim = 0.5
```

`scripts/measure/diff.py` between them: 7.32% of pixels differ, 5.99% by more than 4, `max |d|` 178 - and
the 4x4 tile table puts every one of those in the model's own columns (the outer tiles read 0.000, i.e. the
sky, the background and the overlay do not move). The frame means rise by ~2.2/255 on all three channels,
which is the expected direction rather than a defect: the shadowed end of the ramp is a tinted PAINT, so it
no longer falls to zero the way the un-warped falloff did.

Three properties the same pair settled, because all three were wrong in the first version - and the third
one is the reason the warp was briefly removed from the "what does this look like" column and put back:

- **The warp changes the diffuse COLOUR, not the light factor.** The first version took the light factor
  over, arguing that the tinted colour already IS the shadowed result and an extra ndotl would darken the
  shadow side twice. This engine's sun carries a radiance of 7.5 (`shade_surface`), so that put the lit
  band at `albedo/pi * 7.5` = 2.4x the albedo and the tonemapper resolved it to white. Measured on the
  asset's face region (mean R/G/B), one camera, one frame count, and - for the two warp arms - one commit
  apart so that only the expression differs:

  | | R | G | B | R stddev |
  | --- | --- | --- | --- | --- |
  | PBR (no warp) | 151.2 | 158.1 | 168.8 | 69.1 |
  | warp, light factor dropped (wrong) | 172.9 | 178.4 | 190.1 | 65.7 |
  | warp, light factor kept | 146.8 | 154.2 | 166.7 | 68.4 |
  | pure albedo (unlit) | 181.0 | 185.2 | 190.7 | 45.8 |

  The wrong shape sits at the albedo's own level with its contrast flattened towards the unlit row's - i.e.
  it had pushed the shading into the texture. The right one bands and tints the face while leaving it where
  the PBR path has it.
- **The specular keeps the falloff, and the SAME one the diffuse uses.** Handing it the raw cosine while
  the diffuse took none - the first version's split - let the whole model, unlit side included, collect the
  sun's full specular. Both terms now carry `light_radiance * ndotl`, with the ndotl the ramp produced.
- **The tint is a paint value, not an attenuation.** (0.55, 0.5, 0.75) against this asset's near-white
  albedo still reads as a bright lavender surface rather than a shadow. That is why the shipped default is
  the neutral value: the useful range is a property of the model, and the reference's five per-material
  `ShadowColor` values are how the reference avoids the question.

What the ramp does NOT solve, and why the reference has a shader for it: on a face, a band edge crossing
the nose is a visible line, because a face is nearly flat and its shading is painted into the texture. The
reference's `- Face` group works in a `headOrgn`/`headFwd`/`headUp` frame with a face light-map and an SDF,
which is the piece listed as missing above.


## The outline, and the three things it cost

The outline is the classic inverted hull: the model is drawn a second time, expanded along its normals by
`outline_width` world units, with the FRONT faces culled, into the same five G-buffer targets the surface
writes. It is geometry, not an overlay - so the line is lit, shadowed and tonemapped with the model, and the
depth buffer keeps it attached to the silhouette. The scene pass records every non-blended leaf's hull
before the surfaces (`scene_pass::record_segment`), which is why a hull costs one draw per leaf and why the
cull mode and depth state are set once for the whole loop rather than per leaf.

Three separate defects stood between that description and a line on screen, and every one of them was
invisible to the validator (`validation clean` throughout, in fact):

1. **The vertex input layout is DERIVED from what a stage declares** (`vulkan::make_pipeline` accumulates the
   stride over the declared locations). The first outline.vert declared locations 0,1,4,5 and skipped 2 -
   the UV - so the pipeline described a 48-byte stride against a 64-byte vertex and the hull rasterized as a
   few enormous triangles across the screen. pbr.vert states the rule from its side ("shadow.vert must
   declare exactly the same inputs"); outline.vert now declares the UV and says why.
2. **`motion_base` sits between `instance_base` and `model` in the push block.** Declaring it after `model`
   put `model` four bytes early, so the hull was transformed by the wrong matrix and landed outside the
   view - drawn, valid, and nowhere. surface.glsl gets away with omitting it only because the std430
   padding happens to land `model` on the same offset.
3. **The hull pipeline had to be resynced to the frame's viewport** like the G-buffer pipeline is
   (`runtime::update_pass_geometry`). It is a runtime member rather than a pass, so no pass declaration
   resyncs it, and `begin_pipeline` re-emits whatever viewport it cached at creation.

None of the three is a subtle rendering question, and all three are the same kind of thing: a shader that
does not line up with a contract that lives somewhere else. They are written down here because the contracts
are the part of this engine a reader cannot see from the shader.

**What the outline does not do yet:** the thickness is per MATERIAL (the model's own `エッジ倍率` for hair,
face, body, ...) but not per vertex, while the reference also reads the per-vertex edge scale the PMX
converter already exports as `_EDGESCALE` and the loader does not import. And the hull's motion vector is
zero, so an animated character's line gets the camera's motion only and can crawl slightly under TAA.

### Where the per-material data comes from

The MMD inputs are application-specific, so the converter writes them into each material's `extras` -
and fastgltf does NOT keep extras: it parses them, hands the simdjson DOM object to an
`ExtrasParseCallback` and forgets it. The loader therefore installs one (`collect_mmd_extras` in
gltf_loader.cpp), which is also why that one translation unit now has the real simdjson header rather than
fastgltf's forward declaration (see the target's include dirs). An absent field leaves the NEUTRAL value
behind, and "the model authored an edge at all" is a separate flag - black is a legitimate edge colour and
0 a legitimate size, so the values cannot be asked to imply their own absence.

The GPU side is one lane APPENDED to the material record (`npr_edge`: xyz the colour, w the thickness) plus
flag bit6. Appending matters twice over: the fields above keep their offsets, and - because a storage
buffer's array stride is the struct's own size - EVERY copy of the record in the shaders has to grow with
it (seven files declare one; a copy that stopped early would index the table at the wrong pitch).

Measured on the asset, with the frame's global outline colour set to a colour nothing should use (bright
green, then magenta) and everything else equal: the two captures differ by 370 pixels of 1,036,800 (mean
0.0005/255) against a run-to-run jitter of 294 - i.e. the global colour is never reached, because every
material on this model authored its own. Its hair materials author `(0.333, 0.424, 0.302)` at size 0.5 and
the rest black at 0.5..1.0, which is what the render shows: thinner, dark-green lines on the hair and
thicker black ones on the body.



