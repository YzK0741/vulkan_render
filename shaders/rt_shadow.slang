// shaders/rt_shadow.slang - the SLANG source of the ray-traced shadow's stages.
//
// PRODUCES `rt_shadow.rchit.spv` (and will carry the raygen, miss and any-hit entries as they are ported),
// the same names the GLSL versions produced, so the host side is untouched. This file starts with the
// CLOSEST-HIT stage, which is the smallest one and the one that carries the payload contract.
//
// THE HIT GROUP REPORTS THAT THE RAY HIT SOMETHING, and nothing else: the sun's visibility is a binary
// question and `gl_RayFlagsTerminateOnFirstHitEXT` in the raygen stops traversal here, so this stage has no
// attribute to read, no material to fetch and no shading to do. It exists because a traced ray needs a hit
// group in its shader binding table, and it is the stage a LATER step grows: an any-hit shader for alphaMode
// MASK, and an opacity micromap in front of it, belong exactly here.
//
// THE PAYLOAD IS WRITTEN HERE AND NOWHERE ELSE ON THIS PATH, and that is the contract to preserve rather than a
// detail: the raygen deliberately does NOT initialise it (an initialisation storing the same value the miss
// shader stores is what the pass header records as a bug - the compiler dropped the redundant store and every
// escaped ray came back occluded). So this 1.0 and the miss shader's 0.0 are the two halves of it, and the
// payload is one float.

struct shadow_payload
{
    float occluded;
};

/// @brief a hit along the shadow ray: the surface occludes the sun
/// @note the payload travels as an `inout` parameter, which is how Slang binds a ray payload; the GLSL side
///       declares it as `layout(location = 0) rayPayloadInEXT float`, i.e. the same single float at location 0.
[shader("closesthit")]
void chit_main(inout shadow_payload payload)
{
    payload.occluded = 1.0;
}

/// @brief the ray escaped the scene: nothing blocked the light
/// @note THIS STORE IS LOAD-BEARING AND MUST SURVIVE CODEGEN, which is the opposite of what an "empty miss
///       shader" suggests. The payload is the only channel a traversal's outcome comes back through, and
///       leaving the miss case to a value the raygen initialised is what this pass did first: measured on an
///       NVIDIA RTX 4060, a raygen storing 0.0 before the trace plus a miss shader that does not write the
///       payload made EVERY escaped ray come back occluded (a DamagedHelmet capture measured 61.22 mean over
///       the model against 76.48 for the raster shadow map, with 100% of the differing pixels DARKER - the
///       whole sunlit ground had its sun killed). Storing the SAME 0.0 the raygen stored is not enough
///       either, because then the store is redundant and the compiler drops it. What works is exactly one
///       store per path with the raygen initialising NOTHING: the raygen writes nothing, the hit group
///       writes 1.0 and this writes 0.0 - which is why the built module is checked for this store rather
///       than assumed (see docs/slang_migration.md).
[shader("miss")]
void miss_main(inout shadow_payload payload)
{
    payload.occluded = 0.0; // "nothing blocked the light": the ray escaped the scene unoccluded
}
