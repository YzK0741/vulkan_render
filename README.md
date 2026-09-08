# vulkan_render

A Vulkan renderer written in modern C++23 (C++20 modules / `.cppm`), implementing a glTF 2.0 PBR (metallic-roughness) pipeline with CPU-precomputed split-sum IBL lighting, a scene tree with BVH frustum culling, directional shadows, and a Dear ImGui debug overlay.

<p align="center">
  <img src="snapshot/DamagedHelmet.png" width="49%" alt="DamagedHelmet with PBR + IBL + shadows" />
  <img src="snapshot/FlightHelmet.png" width="49%" alt="FlightHelmet with PBR + IBL + shadows" />
</p>

## Features

- **Vulkan wrapper (`vulkan` module)**: C++ modules wrapping the full initialization flow — instance / device / swapchain / pipeline / command buffer. The `vulkan::runtime` facade is created with a `core_create_info` (window size / title / vsync / MSAA / validation layers) and drives each frame through **granular phases** — `poll_events()` (window events / minimized skip) → `recreate_if_minimized()` → `pace_and_acquire()` (wait the frame slot's timeline, acquire the next image, write the camera UBO) → `begin_recording()` (command buffer + world accumulation + frustum culling) → `record_main_drawcalls()` (shadow + main passes) → `end_recording()` → `submit_and_present()`. Callers that write per-frame data (animation poses, skin matrices, morph weights) do so between `pace_and_acquire()` and `begin_recording()`; `render_frame()` runs all the phases in one call for callers without interleaved updates. Each phase returns a `frame_status` (`proceed` / `skipped` / `closed` / per-stage `*_failed`). All scene-wide GPU resources (shared camera UBO, texture array, IBL, material table, shadow maps, skin-matrix and morph buffers) are owned by the runtime. Frame synchronization uses **timeline semaphores**: one timeline per frame slot counts completed submissions and serves as the host pacing wait (`wait_frame_slot`, no fences); the image-available and present-ready signals stay **binary** (required by acquire/present — the present-ready gate is one per swapchain image so a separate present queue cannot race a re-signal).
- **Scene tree (`vulkan.runtime.scene_tree`)**: scene storage is a tree of nodes (name + local transform + children + a primitive leaf), walked once per frame to accumulate world matrices. Every GPU drawable is a `primitive` subclass (`normal_draw_primitive` / `instanced_draw_primitive`...) with a polymorphic `draw()` — new draw strategies only add a subclass. The tree is **caller-owned**: the app declares the `scene_tree::scene` and binds it with `runtime::set_scene()`; the runtime renders it but never owns it. The tree must be destroyed before the runtime (declaration order in the app gives that) so the leaves' GPU buffers — `vk_buffer`/`vk_image` RAII owners from `vulkan.core.vma.handles` — release through the still-alive allocator.
- **BVH frustum culling**: per-frame, a BVH is built over every leaf's world AABB and tested against the camera frustum; the tree is rebuilt only when the scene changed and the culled result is reused while the camera is static. Toggle with `set_frustum_culling()`.
- **Dynamic rendering**: frames are rendered through `vkCmdBeginRendering` (Vulkan 1.3 dynamic rendering) when the device supports it — no render pass / framebuffer objects exist on that path; devices without dynamic rendering automatically fall back to a classic render pass + framebuffers. MSAA resolve works on both paths. The shadow pass is depth-only dynamic rendering.
- **Single descriptor set (descriptor indexing)**: every pipeline shares the flat scene layout — camera UBO (binding 0), `sampler2D textures[]` runtime array (binding 1, partially bound + non-uniform indexing), shared IBL images (bindings 2–4), the GPU material table (binding 5, a storage buffer), per-instance transforms (binding 6), the light UBO (binding 7), the shadow map (binding 8), the per-joint skin matrices (binding 9, identity block for unskinned draws) and the per-primitive morph data (binding 10, deltas + active weights). The runtime keeps **one descriptor set per frame slot**: a slot's bindings 0/8/9/10 always point at that slot's own camera / shadow map / skin / morph buffers and are written once at scene-set creation, so an in-flight frame can never observe the next frame's descriptors (no update-after-bind). The layout is an agreed fixed convention (not parsed from SPIR-V), so no per-pipeline layout objects exist; each primitive only pushes a 96-byte block (`material_index` + `skin_base` + morph base/targets/vertex count + model matrix).
- **GPU material table**: each material is one `material_record` in the storage buffer (5 texture-array indices + factors + flags). A primitive references a material by index, the shader reads `materials[push.material_index]` and samples `textures[<index>]` — material data is stored once on the GPU and shareable between primitives.
- **PBR rendering**: standard metallic-roughness workflow with five texture slots — albedo (sRGB), metallic-roughness, normal, occlusion, emissive — falling back to a 1×1 white texture when missing; all material parameters come from the material table.
- **IBL lighting**: CPU-precomputed environment cubemap → GGX importance-sampled prefiltered environment (mip chain), irradiance map, and BRDF LUT (split-sum), uploaded as `R16G16B16A16_SFLOAT` cubemaps / `R16G16_SFLOAT` LUT. Precompute resolutions are configurable.
- **Directional shadows**: orthographic shadow pass rendering the scene's depth from the sun (2048² dynamic-rendering depth pass, manual PCF in the shader); enabled over the imported scene's bounds. The pass's depth bias is dynamic state, tunable live from the debug GUI (slope factor for acne on angled surfaces, constant factor for a fixed push).
- **Debug GUI (`vulkan.gui`)**: a Dear ImGui overlay driven inside the runtime's frame steps. `vulkan::gui::widget` subclasses (`label_widget` / `checkbox_widget` / `slider_widget` / `vec3_widget` / `combo_widget`...) stack into `debug_panel`s that register with `gui_content` via `runtime::debug_gui()`. The demo panel shows fps and toggles frustum culling, the skybox pass, the shadow pass, adjusts the shadow depth bias (slope / constant sliders), and edits the camera orbit target; animated models additionally get play/pause, a time scrubber and an animation dropdown. Window layout (position / size) persists to `imgui_layout.ini`.
- **glTF loading (`gltf_loader` module)**: standalone CPU module built on [fastgltf](https://github.com/spnda/fastgltf), supporting `.gltf` / `.glb` / data URIs and exporting both a flattened drawable stream and the retained node hierarchy (name + local transform + children, plus per-node TRS base pose, asset-node index, skin and morph references) with raw de-interleaved vertex/index data (JOINTS_0 / WEIGHTS_0 and morph-target deltas included) — and the file's keyframe animations (channels/samplers incl. morph `weights`, times + values decoded to floats, with pure-CPU LINEAR/STEP/CUBICSPLINE sampling), skins (joints + inverse bind matrices) and morph targets (POSITION/NORMAL deltas + default weights).
- **Keyframe animation + skinning + morph targets (`vulkan.animation`)**: `vulkan::animation::controller` (a module of the engine core library) plays channel-bearing animations automatically on a loop — it samples the animation (slerped rotations, morph weights) into the scene tree's node locals (animated roots keep the import offset), rebuilds the per-frame skin matrices (`inv(W_mesh) · W_joint · IBM`) and updates the active morph weights into the runtime's per-frame-slot buffers, so skinned and morphable meshes deform live. It is **format-neutral**: animation data is value-copied into its own `vulkan::animation` structures (samplers/channels/clips/skins/base poses, pure CPU) at init, and `init()` is a template over the `source` concept — any loader exposing the required member shapes (glTF's `scenes` does; a future format just implements them) can drive it, so the module never imports a loader. Heavy animations (many channels, e.g. the recursive-skeleton stress sample with 840 channels over 924 nodes) fan the per-source sampling out over a small `utility.thread_pool` (2–6 threads, sized to the machine) — each source only touches its own runtime nodes, so slices run concurrently; light animations skip the pool and sample on the frame thread. The demo drives it with `frame_clock` (cheap per-frame stamped time) and the `gui` overlay adds play/pause, a time scrubber and an animation dropdown for multi-animation files. Non-indexed glTF meshes (e.g. Fox) are handled by synthesizing indices. Authored glTF cameras are exported and picked as orbit-camera viewpoint seeds (gui "camera" dropdown); punctual lights (KHR_lights_punctual) are exported too, but the demo still shades with the fixed analytic sun, so they do not drive lighting yet.
- **Orbit camera**: left-drag to rotate, wheel to zoom; one shared camera UBO is updated once per frame, with `MAX_FRAMES_IN_FLIGHT` frames in flight.
- **Utility library (`utility` module)**: handle distribution, stack-style destructor mixin, thread pool (`utility.thread_pool`: RAII pool of `jthread` workers with priority queue + `wait_until_free()`, used by the animation sampling fan-out), BVH (used by frustum culling), data block, a per-frame stamped clock (`utility.frame_clock`: single-writer stamps, atomic-load readers), and more. A mimalloc-backed `pmr` manager (`utility::init_pmr()` via `better_pmr`) routes all `std::pmr` allocations — including the runtime's per-frame cull/visible vectors — through mimalloc (vendored under `third_party/mimalloc`); it is idempotent and initialized before `main` from every TU that uses it. Content hashing for GPU-resource dedup is [xxHash](https://github.com/Cyan4973/xxHash) `XXH3_64bits` (`utility::xxh3_64bits`, vendored under `third_party/xxhash`).
- **Startup configuration (`app_config` module)**: TOML config (`config.toml`, `--config <path>` override) merged with argv, covering model/demo/grid, resource paths, window/render settings (size, title, vsync, MSAA, clear color, skybox/shadow toggles) and IBL resolutions. See `config.example.toml`.
- **Engineering practices**: automatic `clang-format` before every build, `-Wall -Wextra -Werror`, and **exceptions disabled in all build configurations** (`-fno-exceptions`; the vendored `std` module makes this work), plus `-flto -march=native -fno-rtti` in Release builds.

## Documentation

The project uses **Doxygen** for API documentation; every module, class, and interface is annotated in-source with `@defgroup` / `@brief`. The generated HTML and the LaTeX manual are **not** committed to the repo (they would drown the source tree in generated files) — build them whenever you need them with the platform wrapper script: `scripts/windows/build_docs.ps1` (PowerShell / Windows) or `scripts/posix/build_docs.sh` (POSIX sh — WSL / Linux / macOS / an MSYS2 shell). They run `doxygen Doxyfile`, then compile the LaTeX manual into `docs/latex/refman.pdf`:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts/windows/build_docs.ps1

# POSIX (Linux / WSL / macOS)
sh scripts/posix/build_docs.sh
```

Raw equivalent: `doxygen Doxyfile` (HTML only).

Output: `docs/html/` (open `docs/html/index.html`) and `docs/latex/` + `docs/latex/refman.pdf` (all gitignored). The LaTeX step needs a TeX distribution (`pdflatex`/`makeindex`; `make`, `latexmk`, or bare `pdflatex` all work — MiKTeX's per-user install under `%LOCALAPPDATA%` is found automatically).

Related source docs (tracked in the repo):

- [gltf_loader usage guide](docs/gltf_loader_usage.md) (API semantics, data formats, Vulkan integration examples)
- [docs/official-shaders/](docs/official-shaders/): reference shaders (IBL / PBR / primitive)

## Layout

```
├── main.cpp                 # Demo entry point: start async loads -> runtime init -> scene import
│                            #   -> animation/camera/gui setup -> granular frame-phase render loop
├── chores.cppm / chores.cpp # chores module (root-level demo bootstrap): analyse_config (config +
│                            #   argv merge, shaders/model location), setup_pipeline,
│                            #   add_instancing_grid, shader loading
├── CMakeLists.txt           # CMake 4.3, C++23 modules build
├── config.example.toml      # Annotated startup-config reference (copy to config.toml)
├── Doxyfile                 # Doxygen config (PROJECT_NAME: "vulkan render")
├── app_config/              # app_config module (TOML startup config + argv merge)
├── vulkan/                  # vulkan modules (core / vma / handles / init_utils / pipeline / spirv_parser / math /
│                            #   runtime / runtime.scene_tree / gui / animation)
├── utility/                 # utility module (data_block / better_pmr / BVH / thread_pool / frame_clock)
├── gltf_loader/             # gltf_loader module (CPU-side glTF/GLB loading)
├── std/                     # std / std.compat modules (vendored libc++ module; required by the -fno-exceptions builds,
│                            #   avoids configuring CMake's experimental C++ modules flags)
├── shaders/                 # GLSL sources + precompiled SPIR-V (recompile via compile_shaders.ps1 / .sh)
├── gltf_model/              # Sample model (DamagedHelmet)
├── snapshot/                # Screenshots
├── docs/                    # Usage guides + reference shaders; Doxygen HTML is generated on demand (gitignored)
├── scripts/                 # Platform-split helpers: windows/ (PowerShell build/run/docs; only setup.sh
│                            #   is sh — it must run inside MSYS2) and posix/ (sh), plus config
│                            #   generators; build_docs = windows/build_docs.ps1 + posix/build_docs.sh
└── third_party/             # Vendored dependencies (spirv-reflect, imgui, xxhash, fastgltf, simdjson, stb_image, mimalloc)
```

## Dependencies & Build

### Requirements

- CMake ≥ 4.3 and a compiler with C++23 / C++20 modules support (this project uses MSYS2 clang64's clang)
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) (includes `glslc`; also provides VMA, `vma/vk_mem_alloc.h`, under its `Include/`)
- System packages: `glfw3`, `glm`, `tomlplusplus` (header-only; MSYS2 `mingw-w64-clang-x86_64-{glfw,glm,tomlplusplus}`)
- Everything else is vendored under `third_party/`: `spirv-reflect`, Dear ImGui (GLFW/Vulkan backends), xxHash, **fastgltf + simdjson** (the glTF parser and its JSON backend, compiled from source into a `fastgltf_vendored` target), **stb_image** (texture decode) and **mimalloc** (allocator behind `utility.better_pmr`, compiled into a `mimalloc_vendored` static target). No system fastgltf/simdjson/mimalloc package and no network fetch is needed — the build is self-contained on both Windows/MSYS2 and Linux. The **Windows Release** executable links fully static (`-static`: libc++ / libc++abi / libunwind, glfw3, mimalloc are all pulled in statically), so `build-release-clang64/vulkan_render.exe` is a single portable file — only the OS's own DLLs (kernel32, the UCRT, `vulkan-1.dll`) remain dynamic. Debug builds stay dynamic for faster iteration.

### Scripts

The repo ships setup / build / run scripts under `scripts/` for the two
main platforms — **Windows** (MSYS2 clang64; the environment check + package
install is a POSIX `sh` script run inside MSYS2, configure/build/run are
PowerShell) and **POSIX** (Linux / WSL / macOS; everything is `sh`).
Python is used for the config generators (cross-platform, no shell needed):

```bash
# config.toml helpers (any platform, Python 3.8+):
python scripts/make_default_config.py     # copy config.example.toml as-is (asks where to put it)
python scripts/make_config.py             # interactive: asks every setting (types + defaults shown)
python scripts/make_config.py path/to/dir # write config.toml into an explicit directory
```

**Windows (MSYS2)**

```bash
# 1. in an MSYS2 shell: check clang64, install missing pacman packages,
#    find the Vulkan SDK, create config.toml from the example
sh scripts/windows/setup.sh

# 2. build Debug + Release (PowerShell; clang64 bin must be reachable)
powershell -ExecutionPolicy Bypass -File scripts/windows/build.ps1
#    or one config / clean:
powershell -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Type Release -Clean

# 3. run (PowerShell; forwards extra args to the executable)
powershell -ExecutionPolicy Bypass -File scripts/windows/run.ps1
powershell -ExecutionPolicy Bypass -File scripts/windows/run.ps1 -Model path/to/model.glb -Demo gui
```

Builds land in `build-debug-clang64/` and `build-release-clang64/`.

**POSIX (Linux / WSL / macOS)**

```sh
sh scripts/posix/setup.sh     # detect distro, install glfw/glm/toml++/Vulkan via apt/dnf/pacman/brew
sh scripts/posix/build.sh     # Debug + Release
sh scripts/posix/run.sh       # run; extra args (model / demo) are forwarded
sh scripts/posix/run.sh path/to/model.glb gui
```

Builds land in `build-debug/` and `build-release/`.

The Doxygen HTML + LaTeX manual build with `scripts/windows/build_docs.ps1`
(Windows / PowerShell) or `scripts/posix/build_docs.sh` (POSIX sh) — see the
[Documentation](#documentation) section.

### Manual build

The scripts above are thin wrappers over the same two commands — configure
once with Ninja + a C++23-module compiler, then build:

```bash
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

> `clang-format` runs automatically before compilation; a `clang-format-check` target is also provided for CI (check-only, no modifications).

### Run

Run from the project root or any build directory (the program walks upward to locate `shaders/` and `gltf_model/` when they are not configured):

```bash
./build-release/vulkan_render                # or build-release-clang64/vulkan_render.exe on Windows
# or load a different model:
./build-release/vulkan_render path/to/model.glb
# or a demo / debug mode (see below):
./build-release/vulkan_render path/to/model.glb gui
```

By default it loads `gltf_model/DamagedHelmet.gltf` and renders it with PBR + IBL. Controls: **left-drag** to orbit, **wheel** to zoom, **drag the window border** to resize (the swapchain is recreated on the fly), **ESC** to quit. Loaded models that carry keyframe animations (e.g. glTF-Sample-Assets `AnimatedCube` / `BoxAnimated`) play automatically on a loop.

#### Startup configuration

Startup is driven by a TOML config file — copy `config.example.toml` to
`config.toml` (working directory) or point at one explicitly:

```bash
./build-release/vulkan_render --config my_config.toml
```

Two helpers generate `config.toml` for you: `make_default_config.py`
(writes the example file as-is; only asks where to put it) and
`make_config.py` (asks every setting with type hints and defaults) — see
the [Scripts](#scripts) section.

Positional argv overrides the file: `argv[1]` = model path, `argv[2]` = grid side (a number) or demo, `argv[3]` = demo. Configurable: model / demo / instancing grid, `shaders_dir` / `model_dir` paths, window size / title / vsync / MSAA / clear color, skybox & shadow stage toggles, IBL precompute resolutions, and the debug-panel default size.

#### Demo / debug modes

The third positional argument (or `demo = "..."` in the config) selects a mode:

| demo | effect |
|---|---|
| *(none)* | static view — drag to orbit, wheel to zoom |
| `spin` | rotate the whole scene around its own center (BVH rebuilt every frame) |
| `spin-subtree` | rotate one geometry-carrying scene-tree node in place |
| `nocull` | disable frustum culling (compare fps to verify culling) |
| `closeup` | pull the camera into a partial close-up (expect partial culling) |
| `gui` | force the Dear ImGui debug overlay on (fps, frustum-culling / skybox / shadow toggles, camera-target drag; animated models additionally get play/pause, a time scrubber and an animation dropdown). The overlay is also on **by default** — disable it with `[gui] show = false` in the config |

> Release builds are Windows GUI-subsystem executables: no console window appears when running, and the log output goes to `debug.log` in the working directory (the previous session's content is rotated to `debug.log.old` with a session timestamp on startup). Debug builds keep the terminal.

### Recompile shaders

PowerShell (Windows):

```powershell
powershell -ExecutionPolicy Bypass -File shaders/compile_shaders.ps1
```

POSIX sh (git-bash / MSYS2 / WSL / Linux):

```bash
sh shaders/compile_shaders.sh
```

## License

[MIT](LICENSE) © 2026 YzK0741
