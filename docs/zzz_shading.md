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
| `NTMatcap` / `Eff_MatCap` / `CombineMESphere` - the matcap, and the model's MMD sphere texture combined with it | the model's own sphere map, sampled matcap-style from the view-space normal and combined per its mode (`material_record::sphere_index` + flags bits 7-8: 1 multiply, 2 add), in `gather_surface` | **implemented** for the model's sphere; mode 3 (sub-texture) is not, and the authored matcap half has no equivalent yet - `[render] toon_rim` remains a procedural stand-in for it |
| the `- Face` group: `headOrgn` / `headFwd` / `headUp`, `Face_lightmap`, `SpecularShapeMaskDot` | the same frame's cel path on the face, with the measured agreement below | **not ported, and read from the .blend rather than assumed**: `Light Vector` takes the head frame from three SCENE OBJECTS (empties placed by the artist, their position differences), `Face Factor` needs a face light map whose R is a signed distance and whose Alpha is a mask (`Map Range 0.1..0.35`) plus a mirrored-UV lookup so the shadow edge is symmetric, and `Face Shader` needs a `TData` mask texture to cut the eyes out and blend its `ShadowColor` in. A PMX carries none of the three |
| the `Outline` shader, thickness from the vertex colour | `[render] outline_color` / `outline_width` as the FALLBACK, and the material's own `mmd_edge_color` / `mmd_edge_size` from the glTF `extras` (read by the loader, carried in `material_record::npr_edge`, used by shaders/outline.vert + outline.frag) | **implemented**: per-material colour and thickness. The per-VERTEX edge scale the reference reads from the vertex colour is exported by the converter (`_EDGESCALE`) but the loader does not import that attribute yet |
| the `Glow` shader | - | **not implemented** |

## How to ask for it

```toml
[render]
toon_steps = 5                  # the ramp; 0 = plain PBR
toon_softness = 0.05            # band edge width: smaller = harder edges
toon_shadow_tint = [0.55, 0.5, 0.75]   # (1,1,1) = the warp is OFF
toon_rim = 0.8                  # 0 = no rim
toon_shadow_band = 0.0          # the reference's five-colour shadow cascade: 0 = its deepest colour
toon_shadow_band_gain = 0.25    # ... plus this much of each surface's own albedo luminance (per texel)
toon_specular = 0.8             # the mask on the reference's stepped highlight term; 0 = off
outline_color = [0.06, 0.04, 0.09]     # the hull's colour, written into the G-buffer's albedo
outline_width = 0.05            # WORLD units of expansion; 0 = no hull recorded at all
exposure = 1.0                  # the linear scale the tonemapper sees (README's "Exposure" slider)
```

Every default is the NEUTRAL value, so a config that omits them shades exactly as the renderer did before the
keys existed - which is what keeps the gate's reference frames valid. That is not a claim: the gate is run
against them. The values above are the arm measured against the in-game reference (see the tables below).

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


## The reference's own shader, read out of the .blend

The rules above were being invented from the reference's *names*. They are now read from its *graph*: the
`.blend` is dumped headless (`blender --background XIYAG_ZZZ_Shader.blend --python ...`), in four passes -
every node with its unlinked defaults, every node group's interface, every operator and interpolation type,
and every socket's exact source. That is what a port needs and what a screenshot cannot give.

Two things the dump settled that guessing had got wrong:

- **The shadow colours are MULTIPLIERS, not colours.** `ShadowColor1..5` default to pure WHITE, and white
  means "no shadow at all". The shaded albedo is therefore `albedo * mix(SC, white, light)`, not a lerp
  between two absolute colours.
- **The diffuse falloff IS the shadow multiplier.** The reference's `Light Factor` is
  `smoothstep(0, 0.25, NdotL)` - every surface past a quarter-lit is fully lit, which is where the cel edge
  comes from - and there is no separate NdotL multiply on top of it. It multiplies that by
  `smoothstep(0, 0.5, 1 - vertex_colour)`; a PMX has no vertex colours, so that term is 1 here.

The whole main path, as the graph has it:

```
light = smoothstep(0, 0.25, NdotL) * smoothstep(0, 0.5, 1 - vertex_colour)
SC    = pow( mix-chain(SC1..5, f * (0.2, 0.4, 0.6, 0.8)), 2.2 )    f = MData.x, the light-map channel
SP    = pow( mix-chain(SP1..5, f < (0.8, 0.6, 0.4, 0.2)), 2.2 )    the SAME chain, but hard thresholds
base' = Matcap(base, view_normal, light)                            per-channel, see below
out   = base' * mix(SC, white, light) + SP * (MData.z * smoothstep(0.75, 1, NdotH) * light)
```

The shadow cascade is SMOOTH (its Math nodes multiply the factor) and the specular one is HARD - which is
the opposite of what "cel shading" suggests, and the reason the reference's shadows have no visible borders
while its highlights read as shapes. The `Matcap` group is not a texture lookup alone: it combines the
matcap image with the base colour **per channel**, `mix(2*base*x, 1 - 2*(1-base)*x, base*0.5)`, which is
where the reference's extra saturation in the darks actually comes from.

**What this port substitutes, and why it has to.** The `.blend` is a TEMPLATE: its materials ship every
colour white, `Stocking` 0, `LUT` 1, and its `Eff_MatCap` image is an empty placeholder. So there is no
authored data to copy - and a PMX carries none either (no five shadow colours, no five specular colours, no
ILM light map, no vertex colours). The substitutions, each named in the code:

| reference input | here |
| --- | --- |
| `ShadowColor1..5` | derived from the material's albedo: a ramp from a vibrance+darken transform of it to WHITE, which is the reference's own lit end |
| `MData.x` (light-map band) | `[render] toon_shadow_band` as a frame-wide constant PLUS `toon_shadow_band_gain`, which adds each surface's own albedo luminance to it - the per-texel half, derived from something a PMX does have (see below) |
| `MData.z` (specular mask) | `[render] toon_specular`, one frame-wide value (0 = the highlight term off) |
| the vertex-colour shadow mask | 1 (a PMX has no vertex colours) |
| `Eff_MatCap` | the model's own MMD sphere map when it has one, otherwise no matcap |
| `Stocking` / `LUT` | not ported: the first is a per-model gradient the texture already carries here, the second needs a LUT image the template does not ship |

Measured against the in-game reference, same masks and same lit/shadow quartiles as the table above:

| | lit luma | lit sat | shadow luma | shadow sat | shadow/lit |
| --- | --- | --- | --- | --- | --- |
| reference hair | 236.9 | 0.096 | 181.1 | 0.339 | 0.76 |
| ported hair, `toon_shadow_band = 0.0`, vibrance 2.0 | 237.6 | 0.100 | 179.4 | 0.320 | 0.76 |

All five numbers within about two percent, which is what the parameters were tuned against. Two of them are
tuning artefacts rather than the reference's constants, and both are now `[render]` keys so the next
measurement can move them without a rebuild:

- **`toon_shadow_band = 0.0`** - the deepest of the five shadow colours is the right arm for THIS asset,
  whose author-painted shadow is the strong one; the compiled default stays 0.3 because a model with a real
  light map is supposed to drive the band per texel.
- **the vibrance constant is 2.0**, down from 3.0: at 3.0 the deepest band put the hair's shadow saturation
  at 0.44 against the reference's 0.34 while every other number already matched, so the boost came down
  rather than the band or the value scale moving.

`[render] toon_specular` is the other new key - the reference's `MData.z`, the mask on its stepped highlight
term. With it at 0.8 and the default band, the same hair measures lit 238.8 / shadow 185.0 (ratio 0.77), so
the highlight arm is a second, independently tunable way into the reference's numbers.

### The band factor, and why a frame-wide constant is not enough

The reference reads its band - which of the five shadow colours a texel takes - from its light map, per
texel. Sweeping a frame-wide constant on this asset shows why the reference needs that:

| band | face shadow luma/sat | hair shadow luma/sat |
| --- | --- | --- |
| reference | 196.7 / 0.157 | 180.3 / 0.246 |
| 0.15 (the hair's best) | 185.8 / 0.228 | 182.7 / 0.244 |
| 0.30 | 187.3 / 0.210 | 185.5 / 0.191 |
| gain 0.25 (band 0.0 + 0.25 x albedo luma) | 186.0 / 0.227 | 181.5 / 0.258 |

The HAIR's optimum is 0.15 - a near-exact match on both luma and saturation - and the FACE wants a value
above 0.3, so no constant serves both. `toon_shadow_band_gain` adds each surface's own albedo luminance to
the constant, which is the direction a painted light map goes (darker material, deeper shadow) and is the
one thing a PMX does carry: the hair's albedo is mid-dark (linear luma 0.42), the skin's pale (0.53). At
gain 0.25 it beats the best constant on BOTH regions at once, which is the point of a per-texel input.

Be honest about how much it buys, though: the two luminances differ by only 0.11, so the split is narrow.
The face's remaining gap (186 against the reference's 196.7) is NOT something the band can close - in the
reference the face is a different SHADER (`- Face`), not a different band, and that is the part this asset
cannot supply. Gain 0 is the compiled default, i.e. the frame-wide constant and exactly the behaviour
before this existed.


### The face, measured rather than assumed

The face groups are the one part of the reference that cannot be ported from the file, and the measurement
says it matters less than the name suggests. With a skin mask that excludes the white shirt and the mint
hair (a warm filter, `R > G > B` and `R - B > 12`) and a box on the head alone:

| | lit rgb | lit luma | lit sat | shadow luma | shadow sat | shadow/lit |
| --- | --- | --- | --- | --- | --- | --- |
| reference face | 250/233/225 | 236.1 | 0.102 | 196.7 | 0.157 | 0.83 |
| ported face, exposure 1.0 | 242/227/219 | 229.8 | 0.094 | 180.8 | 0.264 | 0.79 |

The lit half - the hue and its saturation - is within a few percent, which is what the reference's face
shader is FOR (a flat, symmetric, mask-driven face); what is left is the shadow half, where our derived
shadow is darker and more saturated than the reference's painted one.

**A measurement trap worth recording**, because it produced a wrong conclusion first: a mask of
`R >= G >= B` and `R > 150` also selects a white shirt, and a box that reaches the collar then reports the
FACE as neutral grey (247/247/246, saturation 0.005) - which reads exactly like a colour-pipeline bug and is
not one. The albedo path is warm (the unlit face measures 225/215/211), the shirt is what was neutral, and
the fix was the warmth filter in the table above rather than any shader change. For the same reason the
"shadow" of a box that reaches the chest is the red tie, not skin.

`[render] exposure` was added while chasing that: it is the exposure scale the README documents as a GUI
slider, and matching a reference render needs it reproducible. It is NOT the lever here - 0.7 pulls the lit
face to luma 220.9, further from the reference's 236.1 - but it is what let the question be answered with
numbers rather than by eye.

**The face does get its own path now, as far as this asset allows.** What the reference does with a shader,
this port does with a flag and one number: the converter marks the model's face block by material NAME
(`mmd_face`, the prefixes every PMX shares - on this asset exactly materials 0..11, the skin, mouth, teeth,
lashes, brows, eyes and eye shadow), the flag rides `material_record` bit7, and the lighting gives those
materials a shallower band, because the one thing the reference's face shader changes that is measurable
here is that its shadow is far lighter than the body's.

Bit **7** is not a free choice: the deferred path's only channel for material flags is ONE BYTE in the
G-buffer (`out_material.a`), so a flag the LIGHTING stage needs has to fit inside it. The sphere mode moved
up to bits 8-9 for the same reason and loses nothing, because the stage that applies the sphere is the
G-buffer pass, which reads the record directly.

Measured with FIXED patches inside the face (a mean over a box, so the pixel set cannot change between
arms), and the offset swept:

| band offset | cheek luma | cheek sat | forehead luma |
| --- | --- | --- | --- |
| 0.0 (no face path) | 159.6 | 0.024 | 217.6 |
| 0.2 (shipped) | 161.4 | 0.029 | 218.2 |
| 0.5 | 163.2 | 0.033 | 218.8 |
| 0.8 | 164.2 | 0.040 | 219.2 |

The direction is right and monotone, but the effect is SMALL - 4.6 luma of cheek across the whole 0..0.8
range - so 0.2 is the arm that also puts the lit quartile nearest the reference (235.4 against 236.1) and
leaves the hair untouched. The offset is a documented constant rather than a `[render]` key because the
reference has no such parameter.

**A second measurement trap, and it invalidated a whole sweep before the fixed patches replaced it.** A
quartile computed through an ABSOLUTE-threshold mask is not comparable between two shading arms: `R > 150`
admits more of a region's dark pixels as that region brightens, so the "shadow quartile" of the admitted set
can move the OPPOSITE way to the shading. It reported the face's shadow at 157/0.402 for offset 0.5, i.e. a
large effect in the wrong direction, when the fixed patches show a small effect in the right one. Quartiles
through a mask are fine for comparing two renders against a fixed benchmark; they are not fine for deciding
which way a knob moves.



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

### The sphere map, and why this model shows nothing from it

MMD's sphere map is the layer that gives a model its authored sheen: the diffuse textures are flat and the
sphere is combined on top, either multiplied or added. The converter writes the texture index beside the
name (`mmd_sphere_texture`), the loader reads both, they reach the record as `sphere_index` plus two flag
bits, and `gather_surface` looks the map up matcap-style - the view-space normal's xy remapped to [0,1],
with MMD's flipped V - and multiplies or adds it.

**On this asset none of that is visible, and the reason is the model rather than the code.** Measured, in
order:

- the only material with a sphere map is `髮+` (mode 2 = add, `spa\hair_s.bmp`), and it is a DUPLICATE of
  the hair: its bounding box matches the hair's in x and y, and 229 of 313 sampled `髮+` vertices sit
  EXACTLY (distance 0.0000) on a `髮` vertex. It is the hair's inner surface.
- so its fragments are always behind the hair's own. A probe that painted every sphere-map material
  magenta changed 39 pixels of the frame - all of them in the debug overlay, none on the model.
- and the map itself is a GREYSCALE highlight (mean 51,51,51; a white ball on black), so even where it
  showed it would add a white sheen, not a hue.

That last point is the useful one for the look: the saturation the reference art has is NOT in this model's
diffuse textures (their means are 208/187/180, 199/169/174, 165/176/162 - all pale), and not in its sphere
map either. It is authored in the REFERENCE's own shading colours - XIYAG's per-material `ShadowColor1..5`
and `SpecularColor1..5` and its matcap are hand-picked vivid colours, which is what the Diffuse Warp
interpolates towards. Our equivalent is one global `[render] toon_shadow_tint`; a per-material shadow
colour is the next thing the look needs, and the PMX does not carry one, so it has to be a RULE (derived
from each material's own albedo) rather than data.


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



