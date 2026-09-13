/**
 * @file shaders/probe_sh.glsl
 * @brief The SH-2 basis the world-space probe cache stores, shared by the pass that projects into it and
 *        the tracer that reads it back.
 * @ingroup shaders
 *
 * ONE definition, because the projection and the reconstruction have to agree on the basis exactly: the
 * probe pass turns a cell's rays into coefficients with `probe_sh_basis()`, and the tracer turns them back
 * into radiance for a direction with `probe_sh_reconstruct()`. Two copies of these constants would be two
 * chances to disagree, and a disagreement here does not look like a bug - it looks like a slightly wrong
 * image.
 *
 * WHAT A CELL STORES: real SH-2, four coefficients per channel - the DC term and the three first-order
 * terms. A cell therefore answers a DIRECTION rather than a number, which is the whole of the L2.1 step: a
 * cell between a bright window and a dark wall no longer averages the two into one value.
 *
 * WHY THE PROJECTION CARRIES 4*pi: it is what makes the reconstruction EXACT for the one field whose
 * answer is known by hand. A uniform radiance L has to come back as L, and it does, because the DC
 * coefficient of a uniform field is `4*pi*L*Y_0` and `4*pi*Y_0*Y_0 = 1`. The furnace verification mode
 * checks exactly that identity, so the constant is load-bearing rather than a convention.
 */

#ifndef VULKAN_RENDER_PROBE_SH_GLSL
#define VULKAN_RENDER_PROBE_SH_GLSL

/// @brief the DC basis function: 0.5 * sqrt(1/pi)
#define PROBE_SH_DC 0.2820948
/// @brief the first-order basis functions' factor: 0.5 * sqrt(3/pi)
#define PROBE_SH_LINEAR 0.4886025
/// @brief 4*pi: the projection's normalization (see the file comment - the reconstruction depends on it)
#define PROBE_SH_FOUR_PI 12.566371

/**
 * @brief the SH-2 basis for @p dir, in the order the coefficient images are bound
 * @param dir a unit world-space direction
 * @return (DC, Y(dir.y), Y(dir.z), Y(dir.x))
 * @note the assignment of the three first-order terms to the three coefficient images is part of the
 *       storage format: swizzling them here would silently rotate the directional part of the cache, and
 *       the projection and the reconstruction would still agree with each other.
 */
vec4 probe_sh_basis(vec3 dir) {
    return vec4(PROBE_SH_DC, PROBE_SH_LINEAR * dir.y, PROBE_SH_LINEAR * dir.z, PROBE_SH_LINEAR * dir.x);
}

/**
 * @brief the radiance a cell's four coefficients describe for a direction
 * @param c0 the DC coefficient's rgb and, in .a, the cell's trust
 * @param c1 / @p c2 / @p c3 the three first-order coefficients
 * @param dir the world-space direction the radiance arrives FROM
 * @return the reconstructed radiance, clamped at zero
 * @note the clamp is not cosmetic. Four coefficients cannot describe a radiance field that has both a dark
 *       and a bright side, so a direction the cache has no evidence about reconstructs NEGATIVE. Clamping
 *       is the standard answer, and it costs a little energy in exactly those directions - which is why it
 *       is stated here rather than left to be discovered.
 */
vec3 probe_sh_reconstruct(vec4 c0, vec4 c1, vec4 c2, vec4 c3, vec3 dir) {
    const vec4 basis = probe_sh_basis(dir);
    return max(c0.rgb * basis.x + c1.rgb * basis.y + c2.rgb * basis.z + c3.rgb * basis.w, vec3(0.0));
}

#endif // VULKAN_RENDER_PROBE_SH_GLSL
