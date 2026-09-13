# Lumen's reflection denoiser: the mechanisms this renderer's glossy lobe does not have

A study of the UE 5.8.2 source (available at `C:\UnrealEngine-5.8.2-release`), read for the one question
this project's L2.3 work left open: **the glossy lobe's reflection is accumulated by the DIFFUSE chain's
temporal resolve**, which reprojects its history with the G-buffer's motion vector - the motion of the
SURFACE. That is exact for a diffuse bounce, whose value depends on the surface point and travels with it.
It is wrong for a reflection, whose value depends on where the reflection POINTED: in a mirror the reflected
image slides across the surface at its own rate, which is not the surface's.

Everything below is a mechanism, with the file and line it lives at, and a note on what it would take here.
The numbers in this project that these mechanisms would move are in `docs/gi_hit_shading.md`'s L2.3 sections.

## 1. The history is reprojected twice, from different depths, and blended

`Lumen/LumenReflectionDenoiserTemporal.usf`:

    FLightingHistory GetLightingHistory(..., float DeviceZ, float ReflectionHitDeviceZ, float4 EncodedVelocity,
                                        bool bFromReflectHit, ...)          // line 132
        if (bFromReflectHit) HistoryScreenPosition = GetHistoryScreenPosition(ScreenPosition, DeviceZ, ReflectionHitDeviceZ, EncodedVelocity);  // 139
        else                 HistoryScreenPosition = GetHistoryScreenPosition(ScreenPosition, DeviceZ, DeviceZ, EncodedVelocity);              // 143

and both are fetched, then chosen between per pixel (line 381 `/*bFromReflectHit*/ true`, line 385
`/*bFromReflectHit*/ false`). The difference is the depth used for the reprojection, in
`Lumen/LumenPosition.ush:62`:

    float4 ThisClip = float4(ScreenPosition, ReprojectDeviceZ, 1);
    float4 PrevClip = mul(ThisClip, View.ClipToPrevClip);
    ...

so the "from the hit" variant projects the HIT's depth through the previous frame's view-projection. A ray
that landed on a world point 30 units away is reprojected as a point 30 units away, not as the mirror's own
2 units. `bIsDynamicPixel` (the velocity texture's sign) then adds the pixel's own object motion on top, with
the reference being the surface depth - i.e. the object motion is attributed to the surface, and the camera
motion to the hit.

WHAT IT WOULD TAKE HERE: the GI temporal resolve would need a second reprojection per pixel (the reflection
hit's depth, which the glossy pass would have to write out - it currently writes only radiance into the
tracer's image) and a roughness-dependent choice between the two. That is a second history buffer or a
second channel in the existing one, which is why it is a step and not a patch.

## 2. A smooth surface accumulates almost no history, deliberately

`LumenReflectionDenoiserTemporal.usf:413`:

    MaxFramesAccumulated = lerp(2, MaxFramesAccumulated, saturate(Material.TopLayerRoughness / 0.05f));

A pixel with roughness 0 accumulates at most **2 frames**; only at roughness >= 0.05 does the full history
apply, and it ramps in over that range. The reasoning is the one this project measured from the other side:
a sharp reflection cannot be reconstructed from a history that has moved (the reprojection is an
approximation, the neighbourhood clamp rejects the difference, and what survives is ghosting), while a rough
reflection is low-frequency and can be averaged for a long time. The knob is essentially "how much of the
reflection is expected to be a stable, view-independent quantity".

WHAT IT WOULD TAKE HERE: the temporal resolve's blend weight is currently one function of motion
(`blend_static` / `blend_min` in `shaders/ssgi_temporal.comp`), with no roughness input at all - it does not
even declare the G-buffer's normal target, which is where this renderer stores roughness (`gbuffer_normal.w`).
Adding the input and the lerp is small; the reason to do it is that the measurement above says the
reflection loses ~40% of its own magnitude to camera motion today.

## 3. The clamp uses the history's OWN variance, not the current frame's box

The colour history carries a second moment (`FLightingHistory.SpecularSecondMoment`, line 122; written as
`Pow2(Luminance(SpecularLighting))`, line 344) and the neighbourhood is derived from it
(`Neighborhood.Extent = TemporalNeighborhoodClampScale * StdDev`, line 109) - a running mean and variance,
so the clamp tracks how noisy the pixel actually is rather than how noisy one frame looks. This project does
the opposite on purpose: `shaders/ssgi_temporal.comp` takes the clamp box from the CURRENT frame's 3x3
neighbourhood, and its own comment explains that the box is wide and rejects little, which is what lets the
accumulation average ten frames of differently-seeded rays. Both are defensible; UE's is the one that can
tighten as the estimate converges.

## 4. The ray is traced along a "specular dominant direction", not the mirror direction

`LumenReflectionDenoiserTemporal.usf:378`: `float SpecularDominantDirFactor = GetSpecularDominantDirFactor(Material.TopLayerRoughness);`
and the reflection tracing side uses the same idea. Instead of reflecting exactly and sampling from the GGX
lobe around it, the single ray is aimed at a direction biased between the mirror direction and the normal,
by an amount that grows with roughness. For a ONE-ray estimate that is the better single sample: the
dominant direction is where the GGX lobe's integral is concentrated, so the estimate is closer to the cone
average than a random lobe sample - at the cost of being biased for a mirror-like surface, which is why the
accumulation length in §2 is clamped short exactly there.

WHAT IT WOULD TAKE HERE: this project already importance-samples the GGX half-vector (`importance_sample_ggx`
in `shaders/ssgi_spec.comp`), which is unbiased; the dominant direction is a variance-reduction alternative.
The measured case for switching is weak - one ray against four rays differed by +0.0028 of mean green after
the denoisers, i.e. the noise is already handled - so this is the LAST of the four mechanisms to consider.

## What this study changes about the plan

Mechanisms 1 and 2 are the answer to the measured problem (the reflection losing ~40% of its magnitude to
camera motion, `docs/gi_hit_shading.md`'s motion section), and they are one change in shape: give the
reflection its own reprojection and its own accumulation policy inside the shared temporal resolve, keyed on
a hit depth the glossy pass exports. 3 is an alternative to a clamp this project deliberately chose, and 4 is
a variance reduction for a noise level already measured to be low. The order to try them is 1+2, then 3, and
4 only if a measurement asks.
