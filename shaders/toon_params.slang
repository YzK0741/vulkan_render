// shaders/toon_params.slang - THE TOON FAMILY TABLE, shared by every stage that shades a character.
//
// WHY IT IS A FILE OF ITS OWN: two stages need it - the character-forward stage, which shades the surface,
// and the screen-space rim, which adds the contour over it - and they must agree about a family's
// parameters, or the two would draw one character with two different looks. A second copy of the table
// inside the rim stage is exactly the kind of duplicate this project refuses elsewhere: the failure is
// silent, and it shows only as a contour that does not match the shading under it.
//
// SLANG SYNTAX, deliberately: it is included by `.slang` stages only, so it uses `float3`/`uint` rather than
// the GLSL spellings. It touches no heap slot and declares nothing, so it may be included at any point in a
// stage's include order.
//
// THE FAMILY IDS MUST MATCH `gltf::toon_family` (see gltf_loader.cppm) - the same enum, written out here
// because a shader cannot include a C++ enum, and the numbers ARE the contract.
/**
 * @brief the toon stage's ramp, PER MATERIAL FAMILY
 *
 * WHY A RAMP AND NOT THE ENGINE'S `toon_band`, which is measured rather than assumed: `toon_band` is a
 * QUANTIZER - it multiplies by the step count, takes `floor`, and steps at the half-way point of each cell,
 * so a value lands on the nearest band and the output stays close to the input on average. A ramp is a
 * DIFFERENT shape: flat through the shadow side, a jump at one threshold, flat through the lit side.
 * Rendering the quantizer and the ramp and diffing them showed the quantizer moving 48624 pixels by a
 * maximum of 7/255 - present, and visually invisible, which is exactly what "the banding does not work"
 * looks like when the term is technically active.
 *
 * WHY PER FAMILY, and where the numbers come from. The references carry this pair PER MATERIAL, and the
 * differences between the families are the visible ones: hair's shadow edge is the hardest and sits highest
 * (a strand's terminator is a line), skin's is the softest and lowest (a soft terminator across a curved
 * cheek), and cloth and the eye sit between. `EF_*_SHADOW_CENTER` / `EF_*_SHADOW_SMOOTHNESS` in the MME
 * pack and `Sigmoid(blend, offset, x)` in MyZmdShaders are the same two parameters, and each family's entry
 * file gives them different values.
 *
 * THE VALUES BELOW ARE THE REFERENCES' RELATIVE ORDER, NOT THEIR FIGURES - they cannot be their figures,
 * because the reference's ramps are TEXTURES whose own contrast is part of the number and those textures are
 * assets this port cannot carry (see the note on the family textures below). What is reproduced here is the
 * STRUCTURE the order expresses: which family gets the hard edge, which gets the soft one, and which side
 * of the albedo each one's tint moves toward. They are named as the place the measured values land.
 *
 * THE FAMILY IDS MUST MATCH `gltf::toon_family` (gltf_loader.cppm) - they are the same enum, written out
 * here because a shader cannot include a C++ enum and the numbers ARE the contract.
 */
#define VR_FAMILY_BASE 1u
#define VR_FAMILY_SKIN 2u
#define VR_FAMILY_FACE 3u
#define VR_FAMILY_HAIR 4u
#define VR_FAMILY_EYE 5u
#define VR_FAMILY_CLOTH 6u

/// @brief one family's toon parameters: its diffuse ramp, its specular ramp, and its rim
struct toon_params {
    // ---- the DIFFUSE ramp ----
    float center;       // the threshold: where the dark side ends
    float softness;     // its edge width
    float3 shadow_tint; // the colour the dark side is tinted toward
    // ---- the SPECULAR ramp: the reference's SECOND ramp band ----
    // The reference samples ONE ramp atlas twice, at two different V bands: `SampleDirectShadowRamp(ramp, x,
    // 0.125)` for the diffuse and `SampleDirectSpecularRamp(ramp, x, 0.375)` for the highlight. Two bands,
    // two mechanisms: the diffuse band is indexed by the shadow-gated half-Lambert, the specular band by the
    // half-vector angle. Kept as a separate pair of numbers here for the same reason.
    float spec_center;   // the half-vector angle at which the highlight appears
    float spec_softness; // its edge width - small is the drawn highlight blob, large is a sheen
    float spec_strength; // 0 disables it, which is what the eye family needs (its highlight is a matcap)
    // ---- the RIM ----
    float rim_power;   // how tightly the rim hugs the silhouette
    float rim_strength;
    float3 rim_colour;
};

/// @brief the parameters for @p family, falling back to the family-independent ones for `none` and anything
///        this shader does not know - so an unclassified material is shaded exactly as it was before
toon_params toon_params_for(uint family) {
    // THE DEFAULT IS THE FAMILY-INDEPENDENT PATH (family 0 == none), and it is also the fallback for a family
    // this shader has no entry for. Keeping it first and returning it unchanged is what makes "an
    // unclassified material did not move" a property of the code rather than a promise - and it is why
    // EVERY field below has a defined default rather than the table only filling the fields it cares about.
    //
    // `spec_strength = 0` IS PART OF THAT CONTRACT, not a placeholder: the specular band is a term this
    // stage did not have before, so a non-zero default would grow a highlight on every material whose name
    // claimed no family - and the claim "the family table cannot change a frame with no character in it"
    // would become false the moment the table gained a term. The families OPT IN to it; `none` does not.
    // The rim fields, by contrast, default to exactly the constants they replaced (power 3.0, strength 0.9,
    // the same colour), so the rim is unchanged for `none` and only varies where a family overrides it.
    toon_params p = {0.42, 0.035, float3(0.58, 0.60, 0.72), 0.80f, 0.06f, 0.0f, 3.0f, 0.9f, float3(1.0, 0.94, 0.90)};
    if (family == VR_FAMILY_HAIR) {
        // the hardest diffuse edge and the highest threshold: a strand's terminator is a drawn line. The
        // widest, strongest highlight - the reference gives hair THREE specular models and a dedicated LUT
        // because the highlight is what identifies hair, and a broad band is the toon reduction of that.
        p.center = 0.52;
        p.softness = 0.015;
        p.shadow_tint = float3(0.42, 0.42, 0.50);
        p.spec_center = 0.72;
        p.spec_softness = 0.22;
        p.spec_strength = 0.55;
        p.rim_power = 2.2;
        p.rim_strength = 1.1;
    } else if (family == VR_FAMILY_SKIN) {
        // the softest diffuse edge and a red-shifted dark side: the MME pack's skin ships `EF_SKIN_SSS_COLOR`
        // as a warm colour, and its ramp's dark side is warm for the same reason. Skin's highlight is small
        // and weak - the reference's is a GGX lobe with `EF_SKIN_SPECULAR_STRENGTH 0.65` gated to the lit
        // side, which as a toon band is a tight, low-contrast spot.
        p.center = 0.36;
        p.softness = 0.07;
        p.shadow_tint = float3(0.80, 0.66, 0.63);
        p.spec_center = 0.86;
        p.spec_softness = 0.10;
        p.spec_strength = 0.22;
        p.rim_power = 3.4;
        p.rim_strength = 0.8;
        p.rim_colour = float3(1.0, 0.86, 0.82);
    } else if (family == VR_FAMILY_FACE) {
        // near skin's: the face is shaded by its SDF in the full reference, and this fallback should not be
        // softer than skin's before that data exists. Almost no specular: the reference's face highlight is a
        // SEPARATE lip-specular mask, not a lobe on the whole surface.
        p.center = 0.40;
        p.softness = 0.06;
        p.shadow_tint = float3(0.82, 0.70, 0.67);
        p.spec_center = 0.88;
        p.spec_softness = 0.08;
        p.spec_strength = 0.12;
        p.rim_power = 3.4;
        p.rim_strength = 0.7;
        p.rim_colour = float3(1.0, 0.88, 0.84);
    } else if (family == VR_FAMILY_CLOTH) {
        // a hard diffuse edge and a cool dark side: cloth's shadow reads as a fold, not a gradient. A BROAD,
        // dim highlight - the reference's cloth carries three lobes (direct, broad, anisotropic) and their
        // sum is what a wide low band stands in for.
        p.center = 0.46;
        p.softness = 0.025;
        p.shadow_tint = float3(0.50, 0.52, 0.64);
        p.spec_center = 0.76;
        p.spec_softness = 0.24;
        p.spec_strength = 0.30;
        p.rim_power = 2.8;
        p.rim_strength = 1.0;
    } else if (family == VR_FAMILY_EYE) {
        // the softest diffuse of all: an eye is a sphere whose highlight moves, and a hard terminator on it
        // reads as a painted dot rather than as a lit surface. NO specular - the reference's eye highlight is
        // a matcap and a separate overlay material, so a lobe here would be a second, wrong highlight.
        p.center = 0.30;
        p.softness = 0.20;
        p.shadow_tint = float3(0.74, 0.74, 0.78);
        p.spec_strength = 0.0;
        p.rim_power = 4.0;
        p.rim_strength = 0.5;
        p.rim_colour = float3(0.92, 0.94, 1.0);
    }
    return p;
}
