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
| the `Outline` shader, thickness from the vertex colour | - | **not implemented** (the next slice). The PMX converter already exports the model's per-vertex edge scale, as `COLOR_0` today; nothing reads it yet |
| the `Glow` shader | - | **not implemented** |

## How to ask for it

```toml
[render]
toon_steps = 5                  # the ramp; 0 = plain PBR
toon_softness = 0.05            # band edge width: smaller = harder edges
toon_shadow_tint = [0.55, 0.5, 0.75]   # (1,1,1) = the warp is OFF
toon_rim = 0.8                  # 0 = no rim
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

Two properties the same pair settled, because both were wrong in the first version:

- **The specular keeps the UNQUANTIZED falloff.** Handing it the ramp's light factor along with the diffuse
  made the whole model - unlit side included - collect the sun's full specular, and the capture read as
  blown out. The fix is in the warp's branch: `specular * raw_ndotl`.
- **The tint is a paint value, not an attenuation.** (0.55, 0.5, 0.75) against this asset's near-white
  albedo still reads as a bright lavender surface rather than a shadow. That is why the shipped default is
  the neutral value: the useful range is a property of the model, and the reference's five per-material
  `ShadowColor` values are how the reference avoids the question.

