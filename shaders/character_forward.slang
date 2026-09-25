// shaders/character_forward.slang - THE TOON CHARACTER FORWARD STAGE.
//
// WHAT IT IS FOR, and it is the one thing this renderer was missing to draw a toon character
// correctly: a pass that runs AFTER the lighting stage and OVERWRITES the pixels it already lit, so
// the character can have its own hand-authored shading instead of the deferred one - without being lit
// TWICE, which is what happens if a character's finished colour is fed back through the G-buffer.
//
// THE ARCHITECTURE IS TAKEN FROM THE REFERENCE PORT, NOT INVENTED HERE (see _toon_ref/PORT_SPEC.md):
// DanbaidongRP's `PBRToon/Base`, `Face`, `Face_AllDirSDF`, `Hair`, `Eye` and `Stockings` each declare a
// pass named `CharacterForward` with `ZWrite Off` + `ZTest Equal`, and it is invoked from
// `Runtime/Lighting/CharacterForwardLighting.cs` at `RenderPassEvent.BeforeRenderingOpaques` - i.e.
// after `DeferredLighting` (`BeforeRenderingDeferredLights`). The `EQUAL` is the whole mechanism: the
// pass lands on exactly the surface the G-buffer pass recorded, so it REPLACES the deferred result
// rather than layering over it, and nothing needs a mask.
//
// WHY THE OUTPUT IS LINEAR AND NOT DISPLAY-REFERRED. The same reference is explicit about this and it is
// why the MME shader pack this port started from did not fit: `PBRToonBase.shader` ends at
// `PostEvaluate(...)` -> `return float4(resultColor, 1)`, and the tonemap is the post chain's alone. A
// stage that encoded sRGB here would be tonemapped a second time by post.slang's ACES composite.
//
// THIS FILE'S FIRST VERSION DRAWS THE SURFACE ALBEDO ONLY. That is deliberate and it is a PROBE, not the
// finish: it makes the pass's four claims separately visible (it records at all, it binds the
// `character_forward` pipeline, it lands on the character through depth-EQUAL, and it leaves every other
// pixel alone) before any toon maths is in the way. The toon chain replaces the last statement.
//
// THE INCLUDE ORDER IS PART OF THE CONTRACT (see gbuffer.slang): the shim first, then surface.glsl, which
// declares the push-constant block and the heap_frame_slot / heap_image_index aliases the slot macros
// expand against. Compiling a body before surface.glsl reports heap_frame_slot as an undefined
// identifier - measured.
//
// The alphaMode MASK discard is NOT written here because `gather_surface` already performs it (see
// surface.glsl): it is the material's rule, not the pass's, so no pass can forget it. The BLEND alpha rule
// IS written here, copied from unlit.slang, because it is the pipeline's coverage contract.

#include "heap_access.slang"
#include "surface.glsl"

[shader("fragment")]
float4 main(
    [[vk::location(0)]] float3 v_world_pos : TEXCOORD0,
    [[vk::location(1)]] float3 v_normal : TEXCOORD1,
    [[vk::location(2)]] float2 v_uv : TEXCOORD2,
    bool v_is_front : SV_IsFrontFace) : SV_Target
{
    // The same surface the G-buffer pass wrote, gathered by the same function: a normal-map fetch, the
    // double-sided flip and the material's factors cannot disagree with the G-buffer if there is one
    // implementation of them.
    const surface_sample s = gather_surface(v_world_pos, v_normal, v_uv);

    // THE PLACEHOLDER. Replaced by the toon chain (ramp, shadow LUT, specular ramp, rim, and the
    // screen-space shadow map this pass will sample) once that data exists; until then it is the one
    // output that proves the pass is running and where.
    const float3 colour = s.albedo;

    // the BLEND rule, copied from unlit.slang: an alphaMode BLEND material carries its own coverage, and
    // every other material is fully opaque regardless of what its base colour's alpha says
    const float out_alpha = ((s.flags & 32u) != 0u) ? s.alpha : 1.0;
    return float4(colour, out_alpha);
}
