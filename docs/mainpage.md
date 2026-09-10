# vulkan_render

A Vulkan renderer written in modern C++23 (C++20 modules / `.cppm`), built with
CMake 4.3 + Ninja on MSYS2 clang64.

Current version: **0.3.0** — single source is `project(VERSION)` in `CMakeLists.txt`
(surfaced by `--version`, the startup log banner and the Vulkan `app_info`); bump it there
and keep this line in sync.

- `vstd` - the project's STL module (modified from libc++ and trimmed to the project's
  usage; consumed as `import vstd;`, module version 0.1.0a - see `vstd/README.md`)
- `vulkan.core` - instance / device / swapchain / VMA allocator / pipeline / descriptor plumbing
- `vulkan.scene_tree` - pure-CPU scene storage (transform hierarchy of scene_node objects with abstract primitive leaves)
- `vulkan.primitive` - the GPU primitives (normal / instanced / static draws) plus the material / camera / light UBO records of the GPU scene set
- `vulkan.runtime` - the frame facade (per-frame-slot scene resources, granular frame phases:
  poll_events -> recreate_if_minimized -> pace_and_acquire -> begin_recording ->
  record_main_drawcalls -> end_recording -> submit_and_present, plus one-call render_frame()),
  drives the peer scene_tree / primitive modules, debug GUI overlay.
  Shadow + main pass commands are recorded into per-slot secondary command buffers and the
  main pass fans its leaf recording out over the shared task pool (sub_render_task batches);
  each recording worker gets its own render_environment (thread-local pipeline-bind state)
- `vulkan.render_environment` - per-recording-session render state: the session's command
  buffer, the available named pipelines (pointer to the runtime's stable name table), the
  session's default pipeline and a deduplicated binder (std::function, injected by the runtime)
  that primitives call through draw(render_environment&). Holds no Vulkan module dependency.
- `vulkan.animation` - animation::controller: glTF keyframe playback / skinning / morphs on the
  runtime scene tree (heavy animations fan per-source sampling over a small utility.thread_pool)
- `gltf_loader` - pure-CPU glTF/GLB loading: meshes, keyframe animation, skins, morph targets,
  cameras and punctual lights (KHR_lights_punctual); world-AABB + loader diagnostics
- `chores` - demo bootstrap helpers for main(): startup config analysis (config + argv merge,
  shaders/model location), pipeline setup, instancing stress grid, shader loading
- `utility` - log/panic, handle distribution, thread pool (utility.thread_pool), BVH, data blocks,
  frame_clock, pmr routing
- `app_config` - TOML startup configuration merged with argv

## Rendering

PBR (Cook-Torrance + image-based lighting), directional shadows with manual
percentage-closer filtering, skybox, keyframe animation / skinning / morph playback, and a
Dear ImGui debug overlay that is on by default (`[gui] show = false` in config disables it).

The frame is also **measured**: with `[render] gpu_timings` (on by default) each pass boundary
writes a GPU timestamp into a per-frame-slot query pool, and the values are read back after the
slot completed and averaged over a 60-frame window (logged and shown in the overlay). Every
rendering feature below is steered by those numbers rather than by guesswork.

The renderer is being evolved toward a deferred pipeline, one milestone at a time. **M1 (done)** is
the G-buffer: the opaque pass stores the surface (albedo/metallic, world normal/roughness, material
id/AO/flags) in three 1x targets, with a channel debug view. **M2 (done)** is the deferred lighting
stage: `shaders/deferred.frag` shades every pixel from those targets and adds the result into the HDR
target (sky where no geometry wrote depth), through the *same* function the forward path calls
(`shaders/shading.glsl`), with emissive added by the base pass and the material-surface gather shared
in `shaders/surface.glsl`. The forward path stays as the A/B reference - measured on Sponza the two
agree to 0.32/255 mean absolute luminance difference. **M3 (done)** is temporal anti-aliasing on that
path: a Halton(2,3) projection jitter, per-pixel camera motion vectors written by the G-buffer, and
`shaders/taa.frag` resolving the jitter against a reprojected, neighborhood-clamped history (with a
view-depth guard for disocclusions) - measured against the same frame without TAA, 1.05/255 mean
difference with 9% less high-frequency energy, and the forward path byte-identical. TAA is the
deferred path's anti-aliasing because a 1x G-buffer has no MSAA to fall back on, which is why
`[render] msaa` now distinguishes auto (0) from off (1). **M4 (done)** is cascaded shadow maps: the
sun's shadow pass fills a layered 2D-array depth map (1..4 cascades, three by default), each cascade
fitting its own light-space box to its own slice of the view range (practical split scheme, lambda
0.75), with the fragment shader selecting its cascade per pixel from the view depth and blending
across the boundary - one cascade is byte-identical to the pre-M4 single-map path. **M5 (done)** is
clustered light culling: `shaders/light_cluster.comp` (the engine's first compute pipeline, on the
graphics queue) sorts up to 128 punctual lights into a 64 px-tile x 16-exponential-depth-slice grid
once per frame with one atomic counter per cluster, and the shading stage loops only its own cluster's
list - measured with 64 lights, the forward shading pass drops 1.28 -> 0.44 ms and the deferred
lighting pass 0.25 -> 0.08 ms with a byte-identical image, since the cluster test is conservative.
**M6 (done)** is screen-space ambient occlusion in the deferred lighting stage: a golden-angle
hemisphere spiral traced against the G-buffer depth and normals, folded into the `shade_input` ao so
it scales the IBL ambient exactly like a baked AO map, with no extra pass or render target and an
off path that is byte-identical to the pre-M6 frame. **M7 (done)** is the collation pass: the shadow
map size became a config knob, the documented example config is parsed and pinned by a unit test,
and the configuration / GUI / reference documentation cover every milestone. **Shadow-map reuse** (the
first optimization that reads the pass graph instead of shrinking it) followed: a slot's cascade maps
are re-rendered only when the fitted matrices (a refit counter) or the caster geometry (an XXH3-64
fingerprint of every caster's world matrix, of the uploaded skin matrices and of a morph-scratch
revision) changed
since that slot last rendered them, so the skip is byte-identical by construction - a skinned caster
keeps a constant world matrix, which is why the skin upload has to be part of the signal or an
animated model silently keeps a frozen map. Still ahead on this
path: alpha-blended geometry in the deferred path and per-object motion vectors.

## Modular composition

Most modules are independent building blocks that meet only through narrow
interfaces, so you are free to recombine or rewire them:

- `gltf_loader`, `app_config` and `utility` are **pure CPU with no Vulkan
  dependency** — standalone libraries that embed into any host application;
- `vulkan.animation` is **format-neutral and runtime-agnostic**: it drives
  whatever scene storage a caller injects through the `backend` surface and
  initializes from any loader whose data satisfies the structural `source`
  concept (it imports no loader and no `vulkan.runtime`);
- `vulkan.core` / `vulkan.runtime` are a configurable facade
  (`core_create_info`, granular per-frame phase calls) — the demo entry point
  (`main.cpp` + `chores`) is a thin glue layer on top and can be replaced
  wholesale.

Use the modules as-is to extend this renderer (new pass / primitive strategy /
loader) or link only the ones you need into your own project.

Module reference is grouped under the `vulkan_core`, `vulkan_runtime`, `vulkan_runtime_scene_tree`,
`vulkan_render_environment`, `vulkan_animation`, `vulkan_gui`, `vulkan_math`, `gltf_loader`, `chores`,
`utility` and `app_config` groups; the GLSL passes are collected under the `shaders` group (see the
shader reference page for the pass chain, the shared scene set and the conventions). See the README at the
repository root for the controls and config reference.
