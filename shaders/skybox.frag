#version 450

/**
 * @file shaders/skybox.frag
 * @brief Analytic sky background, evaluated per pixel from the view ray.
 * @ingroup shaders
 *
 * The sky function itself lives in shaders/sky.glsl, which the deferred lighting pass includes as
 * well: the deferred path evaluates it for the pixels the G-buffer left empty instead of drawing a
 * background pass, and both must produce the same background.
 *
 * The sky is written as linear HDR radiance like the lit geometry; the post-process pass
 * (post.frag) applies exposure + ACES + display encoding once for the whole frame, so sky and models
 * stay consistent without the skybox needing any descriptor set or push constant beyond the shared
 * scene set.
 */

#include "sky.glsl"

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec3 v_dir;

/**
 * @brief evaluate the sky for the interpolated view ray and write linear HDR radiance
 *
 * No exposure, tonemapping or encode happens here: the post pass owns all display-referred work, so
 * the sky and the lit geometry cannot drift apart.
 */
void main() {
    vec3 color = sky_color(normalize(v_dir));
    out_color = vec4(color, 1.0);
}
