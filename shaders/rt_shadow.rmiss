#version 460
#extension GL_EXT_ray_tracing : require

/**
 * @file shaders/rt_shadow.rmiss
 * @brief The ray-traced shadow's miss shader: deliberately empty.
 * @ingroup shaders
 *
 * The raygen sets its payload to 0.0 ("nothing blocked the light") before every trace and reads it afterwards,
 * so the miss case needs no work at all: the initial value IS the answer. The shader exists because the miss
 * region of a shader binding table must name a stage, and because the alternative - a region with no shader -
 * would make "this light is unoccluded" a fact about the binding table rather than about the estimator.
 *
 * The declaration is kept even though the variable is not written: the payload is shared state, and a stage
 * that declares a different type or location for it would be a link-time lie rather than a compile error.
 */

layout(location = 0) rayPayloadInEXT float payload_occluded;

void main() {
}
