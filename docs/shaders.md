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
 *  shadow.vert + shadow.frag     depth-only, from the sun, into the per-slot shadow map
 *        |
 *  pbr.vert + pbr.frag           forward PBR (or unlit.frag) into the MSAA HDR target,
 *  skybox.vert + skybox.frag     skybox first in the same instance
 *        |
 *  (resolve)                     MSAA HDR target -> single-sample HDR resolve target
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
 * Which pipeline a primitive draws with is decided per leaf: a default-semantics primitive asks
 * the pass for its default pipeline, so the same geometry renders through `pbr` (lit) or `unlit`
 * (flat base color) without re-baking anything - see the runtime's render mode combo.
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
 * | 8 | `shadow_map` | `sampler2DShadow` | the shadow pass |
 * | 9 | `SkinMatrices` | storage buffer | `set_skin_matrices()` |
 * | 10 | `MorphData` | storage buffer | `morph_scratch()` |
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
 * bloom intensity/threshold, mode, encode_gamma, FXAA knobs) on their own layout.
 *
 * @section shader_compile Compiling
 *
 * @code
 *  powershell -ExecutionPolicy Bypass -File shaders/compile_shaders.ps1   # Windows
 *  sh shaders/compile_shaders.sh                                          # POSIX
 * @endcode
 *
 * The compiled `.spv` files are tracked in the repository (there is no build-time shader step),
 * so a shader change must be followed by a recompile + commit of both the source and the binary.
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
 * - **Comments here are the reference.** Every non-obvious decision (bias choices, guards against
 *   NaN at grazing angles, banding, the gamma/encode split) is documented where it is implemented,
 *   and those comments are what Doxygen shows for the matching symbol.
 */
