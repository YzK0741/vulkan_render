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
