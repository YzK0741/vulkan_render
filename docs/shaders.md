/**
 * @defgroup shaders Shaders (GLSL)
 * @brief The GLSL pass sources, the descriptor/push-constant contracts they share with the C++
 *        records, and how they are compiled.
 *
 * The renderer has no shader reflection and no runtime shader compilation: `dsh`-style `.spv`
 * files sit next to their sources, are loaded by `chores::setup_pipeline()` and are turned into
 * `vk_pipeline` objects through the `runtime::make_*_pipeline()` calls (which own the render
 * state: color format, depth, MSAA, and the shared scene pipeline layout).
 *
 * @section shader_passes The pass chain
 *
 * @code
 *  light_cluster.comp            compute: sorts the punctual lights into the frame cluster grid (M5)
 *  shadow.vert + shadow.frag     depth-only, from the sun, into the per-slot shadow map array
 *                                (one layer per cascade, `[render] shadow_cascades` = 1..4)
 *        |
 *  surface.glsl                  (include) the shared material-surface gather, and
 *  shading.glsl                  (include) the shared lighting - used by the G-buffer path below and
 *                                by the transparent pass
 *        |
 *  pbr.vert + gbuffer.frag       opaque geometry -> three 1x G-buffer targets + a 1x depth image
 *                                (albedo+metallic, world normal+roughness, material id+AO+flags),
 *                                the motion-vector target, and the emissive term ADDED into the
 *                                scene color (5th attachment)
 *        |
 *  post.vert + deferred.frag     fullscreen: read the G-buffer + depth, rebuild the world position
 *                                from the depth, light the surface with shading.glsl, add the result
 *                                into the scene color - sky where no geometry wrote depth
 *        |
 *  pbr.vert + pbr.frag/unlit.frag  the TRANSPARENT pass: alphaMode BLEND geometry, shaded while it
 *                                draws and blended over the shaded frame (a G-buffer cannot carry a
 *                                blended surface)
 *        |
 *  post.vert + taa.frag          fullscreen: blend the scene color with the reprojected, clamped
 *                                history into the HDR target the post chain reads (optional; the
 *                                runtime then copies that into the history image for the next frame)
 *        |
 *  post.vert + gbuffer_debug.frag  (debug alternative, never together with the lighting stage) show
 *                                one stored channel instead of lighting it
 *        |
 *  post.vert + post.frag         mode 0 bright-pass prefilter (HDR -> bloom L0)
 *                                mode 1 downsample x3 (L0 -> L1 -> L2 -> L3)
 *                                mode 2 composite (HDR + bloom -> exposure -> ACES -> display)
 *        |
 *  post.vert + fxaa.frag         optional FXAA over the display-referred LDR image
 *        |
 *  Dear ImGui                    overlay, drawn on the final 1x swapchain image
 * @endcode
 *
 * That is the whole frame: `pbr.vert + gbuffer.frag` is the engine's only scene path (the opaque
 * surface write), and everything after it shades or composites the result. A 1x G-buffer cannot be
 * multisampled without per-sample shading, so there is no MSAA - `taa.frag` is the anti-aliasing.
 *
 * @section shader_taa Temporal anti-aliasing
 *
 * The scene renders at 1x, so `shaders/taa.frag` is the engine's anti-aliasing instead of MSAA:
 * `runtime::set_taa` jitters the projection by a Halton(2,3) sub-pixel offset every frame (the
 * G-buffer and the lighting stage both see the jittered projection - geometry and the depth
 * reconstruction agree about where each sample is), the G-buffer writes a motion vector per pixel
 * (current - previous, in UV space), and the resolve blends the current frame with the history
 * sampled at `uv - velocity`, clamped into the current 3x3 neighborhood, with a view-depth guard for
 * disocclusions. The result goes into the HDR target the post chain reads, and the runtime copies it
 * into the history image for the next frame that renders that swapchain image.
 *
 * Two invariants are worth stating because breaking either one is hard to see and expensive to find:
 * - the motion vectors come from the UNJITTERED view-projection pair (a jitter in there is read as
 *   camera motion and reprojects the history by up to a pixel every frame), and
 * - everything that is not the rendering transform itself uses the unjittered matrices: the shadow
 *   frustum fit, the BVH cull frustum, the depth-linearization terms. Letting the jitter into the
 *   shadow fit in particular re-quantizes the light-space box to whole texels every frame, so the
 *   shadow map's texel grid alternates between two alignments and a grazing-angle surface flickers
 *   between lit and shadowed - the TAA history then averages that flicker into a dark band.
 *
 * The G-buffer motion vectors are CAMERA motion at this milestone: an animated (skinned/morphed)
 * object moves without the camera and needs its own previous transform, which is a per-primitive
 * quantity (`gbuffer.frag` documents where it goes). Until then such an object can ghost slightly.
 *
 * Which pipeline a primitive draws with is decided per leaf: a default-semantics primitive asks
 * the pass for its default pipeline, so the same geometry renders through `pbr` (lit), `unlit`
 * (flat base color) or the G-buffer write without re-baking anything - see the runtime's render mode
 * combo. The G-buffer pipeline is not in the runtime's named pipeline cache: it declares four color
 * attachments, so it is only valid inside the G-buffer instance, and the pass hands it to
 * default-semantics leaves under `runtime::gbuffer_pipeline_name` ("gbuffer").
 *
 * @section shader_surface The shared material-surface gather (surface.glsl)
 *
 * `pbr.frag` and `gbuffer.frag` answer the same question - "what is this surface made of?" - and both
 * do it through `gather_surface()` in `shaders/surface.glsl`: the material table lookup, the glTF
 * alpha tests (the MASK `discard` lives in there, so no pass can forget it), the tangent-space normal
 * map with the double-sided flip, and the texture-derived factors. The include declares the descriptor
 * bindings and the push constant block it depends on (bindings 1 and 5, the shared material push
 * block), so a shader including it must not declare them again.
 *
 * @section shader_shading The shared lighting (shading.glsl)
 *
 * `pbr.frag` (forward) and `deferred.frag` (deferred) shade a surface through the SAME function,
 * `shade_surface()` in `shaders/shading.glsl`: the directional sun through the shadow test, the
 * punctual lights, the split-sum IBL ambient, the selectable BRDF/diffuse presets and the cel-shading
 * bands. The include declares the bindings a shading stage needs (0 camera UBO, 2/3/4 the IBL maps,
 * 7 the light UBO, 8 the shadow map) and takes a `shade_input` - world position, normal, albedo,
 * emissive, metallic, roughness, AO. The forward path fills that from its interpolated fragment
 * inputs, the deferred path from G-buffer texels; the lighting cannot tell the difference, which is
 * what makes the two render paths comparable instead of merely similar.
 *
 * @section shader_sky The shared sky (sky.glsl)
 *
 * `sky_color()` is a pure function of a world-space direction with no bindings at all, so both
 * backgrounds use it: the forward path's `skybox.frag` (a fullscreen triangle evaluated per pixel)
 * and the deferred path's `deferred.frag` (the pixels whose G-buffer depth is still the far plane).
 * Two backgrounds from one function means the two paths cannot disagree about the sky.
 *
 * @section shader_clusters Clustered light culling (M5)
 *
 * `shaders/light_cluster.comp` runs once per frame on the graphics queue, one invocation per
 * cluster: the screen cut into 64 px tiles and 16 EXPONENTIAL depth slices. It rebuilds the
 * cluster's view-space box (its tile's corner rays unprojected at the slice's near and far depth)
 * and appends every light whose bounding sphere intersects that box to the cluster's fixed-capacity
 * index row (`binding 12`, one atomic counter per cluster in `binding 11`). The shading stage
 * computes the same cluster from `gl_FragCoord` and its view depth - `cluster_slice_of()` in
 * `shading.glsl` and the compute shader MUST agree on the slice boundaries - then loops only that
 * row. The `cluster_grid.w` lane switches between the clustered list and the brute-force loop over
 * `light_count` lights, which is what the clustered path is verified against: the sphere test is
 * conservative, so the two produce byte-identical images. The grid dims and the slice depth range
 * ride the light UBO (`cluster_grid` / `cluster_depth`, appended AFTER the light array so the array
 * offset every other user of the block encodes stays 352).
 *
 * @section shader_ssao Screen-space ambient occlusion (M6)
 *
 * `deferred.frag` computes the AO itself, from the depth and normal it already reads: a golden-angle
 * hemisphere spiral around the pixel's view-space position, rotated per pixel by a hash of
 * `gl_FragCoord`, each sample projected back to screen and compared against the depth buffer (a
 * sample is occluded when the stored surface is closer to the camera and inside the radius, with the
 * range check fading the contact out). View-space z is negative in front of the camera, so "closer"
 * is a GREATER z - the sign that is easy to get backwards. The result multiplies the material's baked
 * AO into `shade_input.ao`, which `shade_surface()` applies to the IBL ambient (diffuse and
 * specular) and never to the direct sun. With an intensity of 0 the term returns exactly 1.0, so an
 * SSAO-off frame is bit for bit the pre-M6 frame. Parameters ride the deferred pass's push block
 * (the 4x4 inverse view-projection plus the `vec4 ssao`), so nothing new is bound. It is deliberately
 * hemisphere SSAO rather than horizon-based GTAO, and being screen-space it cannot see occluders off
 * screen - the usual set of approximations.
 *
 * @section shader_bindings The shared scene descriptor set (set 0)
 *
 * Every shader in the main pass uses the SAME descriptor set layout (created once by
 * `vulkan::core`, described in `vulkan_primitive`), so a pass binds it once and any leaf can draw:
 *
 * | binding | contents | type | written by |
 * |---------|----------|------|------------|
 * | 0 | `CameraUBO` (view, proj, camera_pos) | uniform buffer | `pace_and_acquire()` per frame slot |
 * | 1 | `textures[]` | sampled image array (bindless) | texture upload / `set_textures` |
 * | 2 | `env_sampler` prefiltered environment | cubemap | `set_ibl()` |
 * | 3 | `irradiance_sampler` | cubemap | `set_ibl()` |
 * | 4 | `brdf_lut_sampler` | 2D LUT | `set_ibl()` |
 * | 5 | `Materials` table | storage buffer | `register_material()` |
 * | 6 | `InstanceTransforms` | storage buffer | `make_instanced_primitive()` |
 * | 7 | `LightUBO` | uniform buffer | `enable_shadows()` + per-frame lanes |
 * | 8 | `shadow_map` | `sampler2DArrayShadow` | the shadow pass |
 * | 9 | `SkinMatrices` | storage buffer | `set_skin_matrices()` |
 * | 10 | `MorphData` | storage buffer | `morph_scratch()` |
 * | 11 | `ClusterCounts` (`uint counts[]`) | storage buffer | the cluster compute pass (read: fragment) |
 * | 12 | `ClusterIndices` (`uint indices[]`) | storage buffer | the cluster compute pass (read: fragment) |
 *
 * @section shader_cascades Cascaded shadows (M4)
 *
 * `shadow_map` is a 2D ARRAY of `shadow_cascades` layers rather than an array of separate samplers,
 * because the layer is chosen per FRAGMENT: indexing a sampler array dynamically needs a
 * dynamically uniform index, while a texture-array layer is just a coordinate - so the shadow test
 * can pick its cascade per pixel. `LightUBO` carries one `light_view_proj` matrix per cascade plus
 * `cascade_splits` (the view-space far distance of each cascade), `cascade_texel_world` (the world
 * size of one texel of that cascade, which drives the world-space normal offset) and
 * `cascade_count` / `cascade_blend`. `calc_shadow()` computes the receiver's view-space depth,
 * picks the cascade whose `[splits[i-1], splits[i]]` range contains it, samples that layer with
 * `calc_shadow_cascade()`, and - inside the last `cascade_blend` fraction of the range - samples
 * the next cascade too and mixes the two, so the resolution/offset step between cascades does not
 * show as a line. A receiver outside its cascade's fitted box is reported lit: each cascade's map
 * covers its own fitted box only, and `runtime::update_shadow_frustum()` keeps those boxes on the
 * part of the scene the camera can see. The shadow pass renders the same caster set once per
 * cascade, each into its own array layer with its own `push_constant` cascade index - `shadow.vert`
 * declares the material fields AND that trailing `uint cascade` in ONE block, because GLSL allows
 * only one `push_constant` block per stage, and the shared layout's single range covers both
 * (`scene_push_constant_size` + `scene_cascade_push_size`).
 *
 * Bindings 0/7/8/9/10 are per frame slot, so a frame in flight never shares a buffer with the
 * frame being written. Indexing `textures[]` with a value from the material table is what needs
 * `#extension GL_EXT_nonuniform_qualifier : enable` in pbr.frag / unlit.frag / shadow.frag.
 *
 * @section shader_push Push constants
 *
 * Material push constants carry `material_index`, the draw `flags` (bit0 = instanced), the
 * `skin_base` / `morph_*` / `instance_base` indices and the per-model `model` matrix
 * (`material_push_constants` in `vulkan_primitive`). The vertex and fragment stages of one
 * pipeline share a single push-constant range, so any stage that does not need a field still
 * declares it to keep the block layout identical - a mismatch here is silent corruption, not a
 * compile error. The post/FXAA pipelines are separate: they use `post_push_constants` (exposure,
 * bloom intensity/threshold, mode, encode_gamma, FXAA knobs) on their own layout, and the G-buffer
 * debug view uses its own `gbuffer_debug_push_constants` (channel selector + the two projection
 * terms that linearize depth) on its own layout.
 *
 * @section shader_compile Compiling
 *
 * The build owns this step: `cmake --build` regenerates every `.spv` from its `.glsl` (and from the
 * shared includes it lists as dependencies), so editing a shader is just editing a shader. The binaries
 * are **not** tracked in the repository and are not hand-synced - they are a build output, and the
 * directory is mirrored next to the executable, which is the copy the runtime loads
 * (`chores::locate_shaders_dir` prefers a `shaders/` sibling of the running binary over anything found
 * by walking up from the working directory). glslc is required, not optional: without a shader compiler
 * there would be nothing to run, and an optional step is how a stale binary gets loaded unnoticed.
 *
 * `shaders/compile_shaders.ps1` / `.sh` remain as a manual escape hatch for a machine without CMake -
 * they compile in place, which is what the build does too:
 *
 * @code
 *  powershell -ExecutionPolicy Bypass -File shaders/compile_shaders.ps1   # Windows
 *  sh shaders/compile_shaders.sh                                          # POSIX
 * @endcode
 *
 * Both scripts pass `-I shaders/`, which is what lets the fragment stages `#include "surface.glsl"`.
 *
 * @section shader_conventions Conventions and pitfalls
 *
 * - **ASCII only.** The Doxygen LaTeX manual runs the shader comments through pdflatex; a stray
 *   non-ASCII character (an em dash, a narrow no-break space) aborts the PDF build. Keep comments
 *   and docs ASCII.
 * - **`const` goes first.** GLSL wants `const float x`, not `float const x`: glslc rejects the
 *   east-const form with a bare "unexpected CONST".
 * - **Keep the interface matching.** The vertex input layout is derived from the shader's inputs
 *   (64-byte interleaved stride, locations 0,1,2,4,5), so shadow.vert declares the unused inputs
 *   and keeps them alive in a never-taken branch; a fragment output must match the pipeline's
 *   color format, which is why the composite and the FXAA pass are separate pipelines.
 * - **A multi-target pipeline must not use the forward blend state.** `make_color_blend_attachment()`
 *   blends with src alpha, which is the right convention while alpha means coverage (opaque draws
 *   have alpha 1, so the math reduces to an overwrite). In a G-buffer alpha carries DATA (metallic,
 *   roughness, flags), so the same state scales the surface by its own alpha and mixes it with the
 *   cleared target - a fully-rough fragment would erase its own albedo. G-buffer pipelines pass
 *   `color_blending = false` and use `make_color_blend_attachment_opaque()`.
 * - **Comments here are the reference.** Every non-obvious decision (bias choices, guards against
 *   NaN at grazing angles, banding, the gamma/encode split) is documented where it is implemented,
 *   and those comments are what Doxygen shows for the matching symbol.
 */
