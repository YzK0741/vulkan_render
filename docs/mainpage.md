# vulkan_render

A Vulkan renderer written in modern C++23 (C++20 modules / `.cppm`), built with
CMake 4.3 + Ninja on MSYS2 clang64.

- `vulkan.core` - instance / device / swapchain / VMA allocator / pipeline / descriptor plumbing
- `vulkan.runtime` - the frame facade (per-frame-slot scene resources, split begin/end frame),
  scene tree, GPU primitives, debug GUI overlay
- `vulkan.animation` - animation_controller: glTF keyframe playback / skinning / morphs on the
  runtime scene tree
- `gltf_loader` - pure-CPU glTF/GLB loading: meshes, keyframe animation, skins, morph targets,
  cameras and punctual lights (KHR_lights_punctual)
- `utility` - log/panic, handle distribution, thread pool, BVH, data blocks, frame_clock,
  pmr routing
- `app_config` - TOML startup configuration merged with argv

Rendering: PBR (Cook-Torrance + image-based lighting), directional shadows with manual
percentage-closer filtering, skybox, keyframe animation / skinning / morph playback, and a
Dear ImGui debug overlay that is on by default (`[gui] show = false` in config disables it).

Module reference is grouped under the `vulkan_core`, `vulkan_runtime`, `vulkan_runtime_scene_tree`,
`vulkan_animation`, `vulkan_gui`, `vulkan_math`, `gltf_loader`, `utility` and `app_config`
groups. See the README at the repository root for the demo controls and config reference.
