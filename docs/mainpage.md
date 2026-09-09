# vulkan_render

A Vulkan renderer written in modern C++23 (C++20 modules / `.cppm`), built with
CMake 4.3 + Ninja on MSYS2 clang64.

Current version: **0.1.0** — single source is `project(VERSION)` in `CMakeLists.txt`
(surfaced by `--version`, the startup log banner and the Vulkan `app_info`); bump it there
and keep this line in sync.

- `vstd` - the project's STL module (modified from libc++ and trimmed to the project's
  usage; consumed as `import vstd;`, module version 0.1.0 - see `vstd/README.md`)
- `vulkan.core` - instance / device / swapchain / VMA allocator / pipeline / descriptor plumbing
- `vulkan.runtime` - the frame facade (per-frame-slot scene resources, granular frame phases:
  poll_events -> recreate_if_minimized -> pace_and_acquire -> begin_recording ->
  record_main_drawcalls -> end_recording -> submit_and_present, plus one-call render_frame()),
  scene tree, GPU primitives (normal / instanced / static draws), debug GUI overlay.
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
`utility` and `app_config` groups. See the README at the repository root for the controls and config reference.
