# vulkan_render

A Vulkan renderer written in modern C++23 (C++20 modules / `.cppm`), implementing a glTF 2.0 PBR (metallic-roughness) pipeline with CPU-precomputed split-sum IBL lighting.

![Screenshot](snapshot/YboodwVAwF.png)

## Features

- **Vulkan wrapper (`vulkan` module)**: C++ modules wrapping the full initialization flow — instance / device / swapchain / render pass / pipeline / descriptor set / command buffer. The `vulkan::runtime` facade sets up everything from window to swapchain in one line and caches named pipelines.
- **PBR rendering**: standard metallic-roughness workflow with five texture slots — albedo (sRGB), metallic-roughness, normal, occlusion, emissive — falling back to a 1×1 white texture when missing; material parameters are passed via push constants.
- **IBL lighting**: CPU-precomputed environment cubemap → GGX importance-sampled prefiltered environment (mip chain), irradiance map, and BRDF LUT (split-sum), uploaded as `R16G16B16A16_SFLOAT` cubemaps / `R16G16_SFLOAT` LUT.
- **glTF loading (`gltf_loader` module)**: standalone CPU module built on [fastgltf](https://github.com/spnda/fastgltf), supporting `.gltf` / `.glb` / data URIs and exporting a world-space flattened node hierarchy with raw de-interleaved vertex/index data.
- **Orbit camera**: left-drag to rotate, wheel to zoom, per-frame camera UBO updates, with `MAX_FRAMES_IN_FLIGHT` frames in flight.
- **Utility library (`utility` module)**: handle distribution, stack-style destructor mixin, `pmr` manager, thread pool, BVH, data block, and more.
- **Engineering practices**: automatic `clang-format` before every build, `-Wall -Wextra -Werror`, and **exceptions disabled in all build configurations** (`-fno-exceptions`; the vendored `std` module makes this work), plus `-flto -march=native -fno-rtti` in Release builds.

## Documentation

The project uses **Doxygen** to generate API documentation; every module, class, and interface is annotated in-source with `@defgroup` / `@brief`. The generated docs live in `docs/html/` (tracked in the repo) — open `docs/html/index.html` to browse:

| Page | Link |
|---|---|
| Doxygen main page | [docs/html/index.html](docs/html/index.html) |
| Modules overview | [docs/html/modules.html](docs/html/modules.html) |
| Class index | [docs/html/annotated.html](docs/html/annotated.html) |
| Class hierarchy | [docs/html/hierarchy.html](docs/html/hierarchy.html) |
| Function index | [docs/html/functions.html](docs/html/functions.html) |
| File list | [docs/html/files.html](docs/html/files.html) |

Per-module Doxygen groups:

- [vulkan group](docs/html/group__vulkan.html) ([core](docs/html/group__vulkan__core.html) / [handles](docs/html/group__vulkan__handles.html) / [init_utils](docs/html/group__vulkan__init__utils.html) / [pipeline](docs/html/group__vulkan__pipeline.html) / [spirv_parser](docs/html/group__vulkan__spirv__parser.html) / [model](docs/html/group__vulkan__model.html) / [runtime](docs/html/group__vulkan__runtime.html) / [vma](docs/html/group__vulkan__vma.html))
- [utility group](docs/html/group__utility.html) ([better_pmr](docs/html/group__better__pmr.html) / [bvh](docs/html/group__bvh.html) / [data_block](docs/html/group__data__block.html) / [thread_pool](docs/html/group__thread__pool.html))
- [gltf_loader group](docs/html/group__gltf__loader.html)

Related docs:

- [gltf_loader usage guide](docs/gltf_loader_usage.md) (API semantics, data formats, Vulkan integration examples)
- [docs/official-shaders/](docs/official-shaders/): reference shaders (IBL / PBR / primitive)

To regenerate the Doxygen docs (outputs to `docs/html` and `docs/latex`):

```bash
doxygen Doxyfile
```

## Layout

```
├── main.cpp                 # Demo: pipeline loading + glTF model + PBR/IBL render loop
├── CMakeLists.txt           # CMake 4.3, C++23 modules build
├── Doxyfile                 # Doxygen config (PROJECT_NAME: "vulkan render")
├── vulkan/                  # vulkan module (core / vma / handles / init_utils / pipeline / spirv_parser / model / runtime)
├── utility/                 # utility module (better_pmr / BVH / thread_pool / data_block)
├── gltf_loader/             # gltf_loader module (CPU-side glTF/GLB loading)
├── std/                     # std / std.compat modules (vendored libc++ module; required by the -fno-exceptions builds,
│                            #   avoids configuring CMake's experimental C++ modules flags)
├── shaders/                 # GLSL sources + precompiled SPIR-V (recompile via compile_shaders.ps1)
├── gltf_model/              # Sample model (DamagedHelmet)
├── snapshot/                # Screenshots
├── docs/                    # Doxygen output (html/ is tracked because this README links to it) + usage guides
└── third_party/             # Vendored dependencies (spirv-reflect)
```

## Dependencies & Build

### Requirements

- CMake ≥ 4.3 and a compiler with C++23 / C++20 modules support (this project uses MSYS2 clang64's clang)
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) (includes `glslc`)
- System packages: `glfw3`, `glm`, `OpenSSL`, `boost` (for `boost/stacktrace` in the debug callback), `fastgltf` (MSYS2 package; `libfastgltf.dll` + `libsimdjson.dll` must be on PATH), Vulkan Memory Allocator (VMA, `vma/vk_mem_alloc.h`)
- `spirv-reflect` is vendored under `third_party/`

### Build

Build directories are not committed to the repo, so pick any name — e.g. `build`:

```bash
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

> `clang-format` runs automatically before compilation; a `clang-format-check` target is also provided for CI (check-only, no modifications).

### Run

Run from the project root or any build directory (the program walks upward to locate `shaders/` and `gltf_model/`):

```bash
./build/vulkan_render
# or load a different model:
./build/vulkan_render path/to/model.glb
```

By default it loads `gltf_model/DamagedHelmet.gltf` and renders it with PBR + IBL. Controls: **left-drag** to orbit, **wheel** to zoom, **ESC** to quit.

> Release builds are Windows GUI-subsystem executables: no console window appears when running, and the log output goes to `debug.log` in the working directory (the previous session's content is rotated to `debug.log.old` with a session timestamp on startup). Debug builds keep the terminal.

### Recompile shaders

```powershell
powershell -ExecutionPolicy Bypass -File shaders/compile_shaders.ps1
```

## License

[MIT](LICENSE) © 2026 YzK0741
