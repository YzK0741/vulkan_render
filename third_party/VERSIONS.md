# Vendored Third-Party Versions

Pinned versions + local patches for the vendored code under `third_party/`
(see each subtree's LICENSE). These are the checked-in copies the build
uses; system-package equivalents (e.g. MSYS2 packages) are not vendored and
are not listed here.

## imgui

- version: 1.92.9b (`IMGUI_VERSION` in `third_party/imgui/imgui.h`)
- upstream: https://github.com/ocornut/imgui
- local changes: none recorded - vendored as-is in commit 3e34dc6, which
  states "vendored imgui 1.92.9b core + GLFW/Vulkan backends"
- backends: `imgui_impl_glfw.*` and `imgui_impl_vulkan.*` ship with the
  source under `third_party/imgui/backends/` (used by the `vulkan.gui`
  overlay)
- build: compiled as a standalone plain-C++ `imgui` static target (see
  CMakeLists.txt); imgui is not a module, is exempt from the repo's
  `-Werror` set (upstream code triggers warnings we do not own), and is
  excluded from the clang-format glob (which only covers
  main/gltf_loader/utility/vulkan, never `third_party`)

## xxhash

- version: 0.8.3 (`XXH_VERSION_MAJOR` / `XXH_VERSION_MINOR` /
  `XXH_VERSION_RELEASE` in `third_party/xxhash/xxhash.h`)
- upstream: https://github.com/Cyan4973/xxHash
- local changes: no local patches

## spirv-reflect

- version: none pinned - the vendored tree carries no version marker (no
  SPIRV_REFLECT version macro in `spirv_reflect.h`; its VERSION HISTORY
  comment only records the 2018-03-27 1.0 initial release, so it cannot pin
  this snapshot)
- upstream: https://github.com/KhronosGroup/SPIRV-Reflect
- local changes: vendored with local patches; upstream baseline unknown -
  check git history (the subtree entered the repo at the initial commit
  under the repo root and was moved into `third_party/` by commit 2c4c4c2).
  The one tracked local change is commit 44f4f9e ("fix(spirv-reflect):
  disable crt debug memory mapping macro"): `#define _CRTDBG_MAP_ALLOC` in
  `spirv_reflect.c` is commented out.
