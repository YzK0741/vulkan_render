# vulkan_render

A Vulkan renderer written in modern C++23 (C++20 modules / `.cppm`), implementing a glTF 2.0 PBR (metallic-roughness) pipeline with CPU-precomputed split-sum IBL lighting, a scene tree with BVH frustum culling, directional shadows, and a Dear ImGui debug overlay.

<p align="center">
  <img src="snapshot/DamagedHelmet.png" width="49%" alt="DamagedHelmet with PBR + IBL + shadows" />
  <img src="snapshot/FlightHelmet.png" width="49%" alt="FlightHelmet with PBR + IBL + shadows" />
</p>

## Version

**0.1.0** — single source of truth is `project(VERSION)` in `CMakeLists.txt`; CMake injects
`VULKAN_RENDER_VERSION_{MAJOR,MINOR,PATCH}` into the code. To release a new version, bump it
there and update this line (plus `docs/mainpage.md`). The version is surfaced by `--version`,
the startup log banner (`vulkan_render x.y.z`), and the Vulkan instance's `app_info`
(`applicationVersion` / `engineVersion`).

Each **independently reusable module set** also carries its own `module version` annotation
in a comment block at the top of its main interface unit — `utility`, `gltf_loader`,
`app_config`, `vulkan.core`, `vulkan.math`, `vulkan.runtime` (incl. its `scene_tree` /
`render_environment` submodules), `vulkan.animation`, `vulkan.gui` and `vulkan.constant_init`
(the compile-time Vulkan info-struct builders that `vulkan.core` and `vulkan.runtime`
embed). All started at 0.1.0; `vulkan.core` and `vulkan.animation` are at 0.1.1,
`vulkan.runtime` at 0.1.2, `vulkan.constant_init` at 0.1.3.
They evolve on their own cadence (bump MAJOR on breaking interface changes, MINOR on additive
features, PATCH on fixes), independent of the app version and of each other. An appended `a`
suffix (e.g. `vulkan.core` 0.1.1a) marks an **internal revision**: source-compatible style or
implementation changes that do not consume a semver slot.

## Features

> **Modular by design — free to combine.** Most modules here are independent,
> dependency-light units that talk to each other through narrow interfaces, so
> you can pick the pieces you need and assemble them however fits your project:
>
> - `gltf_loader`, `app_config` and `utility` are **pure CPU with no Vulkan
>   dependency** — usable as standalone libraries in any host application;
> - `vulkan.animation` is **format-neutral and runtime-agnostic**: it is driven
>   by an injected `backend` surface and a structural `source` concept, so it
>   neither imports a loader nor depends on `vulkan::runtime` — swap in another
>   format or drive it from your own scene storage;
> - `vulkan.core` / `vulkan.runtime` expose a configured facade
>   (`core_create_info`, per-frame phase calls), and the top-level
>   `main.cpp` + `chores` are just a thin glue layer — replace them with your own
>   entry point, or wire only the stages you want.
>
> In short: extend this renderer (add a pass, a primitive strategy, a loader) or
> embed its modules into your own project — the build is per-module CMake
> targets, so you link only what you use.

- **Vulkan wrapper (`vulkan` module)**: C++ modules wrapping the full initialization flow — instance / device / swapchain / pipeline / command buffer. The `vulkan::runtime` facade is created with a `core_create_info` (window size / title / vsync / MSAA / validation layers, or an optional caller-provided `GLFWwindow` via `core_create_info::window` — the core then binds to that window instead of creating/destroying its own) and drives each frame through **granular phases** — `poll_events()` (window events / minimized skip) → `recreate_if_minimized()` → `pace_and_acquire()` (wait the frame slot's timeline, acquire the next image, write the camera UBO) → `begin_recording()` (command buffer + world accumulation + frustum culling) → `record_main_drawcalls()` (shadow + main passes) → `end_recording()` → `submit_and_present()`. Callers that write per-frame data (animation poses, skin matrices, morph weights) do so between `pace_and_acquire()` and `begin_recording()`; `render_frame()` runs all the phases in one call for callers without interleaved updates. Each phase returns a `frame_status` (`proceed` / `skipped` / `closed` / per-stage `*_failed`). All scene-wide GPU resources (shared camera UBO, texture array, IBL, material table, shadow maps, skin-matrix and morph buffers) are owned by the runtime. Frame synchronization uses **timeline semaphores**: one timeline per frame slot counts completed submissions and serves as the host pacing wait (`wait_frame_slot`, no fences); the image-available and present-ready signals stay **binary** (required by acquire/present — the present-ready gate is one per swapchain image so a separate present queue cannot race a re-signal). Pass recording is **multi-threaded** (Vulkan 1.3 dynamic rendering): the shadow pass and the main pass are recorded into per-slot **secondary command buffers** (one `{command pool, command buffer}` pair per task-pool worker, vma-style) and executed from the primary inside `vkCmdBeginRendering` — the main pass splits its visible leaves into per-worker `sub_render_task` segments recorded in parallel on the shared `utility.thread_pool` (a dedicated `recording` priority group), so command generation scales past one thread; the Dear ImGui overlay stays on the primary thread. When the scene is small the main pass falls back to a single segment.
- **Scene tree / GPU primitives (`vulkan.scene_tree` + `vulkan.primitive`)**: pure-CPU scene storage (vulkan.scene_tree) is a tree of nodes (name + local transform + children + an abstract primitive leaf), walked once per frame to accumulate world matrices; the GPU drawables live in the peer `vulkan.primitive` — every GPU drawable is a `primitive` subclass (`normal_draw_primitive` / `instanced_draw_primitive` / `static_draw_primitive`) with a polymorphic `draw()` — new draw strategies only add a subclass. `scene_node` / `scene` expose mounting helpers (`add_root()` / `add_child()` / `attach()` / `find_node()`) so programmatic scenes build without hand-rolled node packing. **Pipeline binding is decoupled from the scene tree**: a leaf carries either *default semantics* (empty `pipeline_name` — it draws with whatever pipeline the recording pass binds as its default) or an explicit pipeline name, and `draw(render_environment&)` requests the pipeline through the per-recording-worker `vulkan.render_environment` — the environment also carries the session's command buffer, so draw has no separate buffer parameter (`bind_default()` / `bind_pipeline(name)`, deduplicated — no re-bind while the requested pipeline is already bound). Custom draw strategies select their pipeline by name; default leaves are pipeline-agnostic, so the same geometry renders under any default (main pass, shadow depth, or a swapped debug pipeline) without re-baking. The tree is **caller-owned**: the app declares the `scene_tree::scene` and binds it with `runtime::set_scene()`; the runtime renders it but never owns it. The tree must be destroyed before the runtime (declaration order in the app gives that) so the leaves' GPU buffers — `vk_buffer`/`vk_image` RAII owners from `vulkan.core.vma.handles` (copy = share, last owner frees) — release through the still-alive allocator.
- **Multi-pipeline scenes (`vulkan.render_environment`)**: pipelines are named and cached in the runtime (`make_pipeline(name, vs, frag)`); the **first created pipeline becomes the implicit default**, `set_default_pipeline(name)` overrides. Every pipeline shares the single flat scene descriptor layout, so any number can coexist in one scene — leaves choose per draw. Each parallel recording worker builds its own `render_environment` (thread-local bind state, never shared), which describes the session's available pipelines and hands `draw()` a deduplicated binder: default leaves request the session default, custom strategies request a name. The shadow pass binds its depth-only pipeline through the same mechanism (its environment ignores the requested name), so custom leaves still cast their geometry into the shadow map.
- **BVH frustum culling**: per-frame, a BVH is built over every leaf's world AABB and tested against the camera frustum; the tree is rebuilt only when the scene changed and the culled result is reused while the camera is static. Toggle with `set_frustum_culling()`.
- **Dynamic rendering**: the engine requires Vulkan 1.3 (device selection enforces `apiVersion >= 1.3`), so every frame records through `vkCmdBeginRendering` — no render pass / framebuffer objects exist. MSAA resolve is inline (a resolve attachment in `VkRenderingInfo`), and the shadow pass is depth-only dynamic rendering.
- **Static/merged batching (`static_draw_primitive`)**: one primitive OWNS a merged vertex/index buffer and draws a chunk table over it — one buffer bind, then one offset draw per chunk with each chunk's own material. N static sub-meshes cost 1 bind + N draws instead of N binds + N draws; the same buffer can carry many different materials. Built via `runtime::make_static_draw(static_draw_create_info)` from already-merged CPU geometry (the packing itself stays at the call site).
- **Single descriptor set (descriptor indexing)**: every pipeline shares the flat scene layout — camera UBO (binding 0), `sampler2D textures[]` runtime array (binding 1, partially bound + non-uniform indexing), shared IBL images (bindings 2–4), the GPU material table (binding 5, a storage buffer), per-instance transforms (binding 6, one shared region sliced per instanced primitive via `instance_base`), the light UBO (binding 7), the shadow map (binding 8), the per-joint skin matrices (binding 9, identity block for unskinned draws) and the per-primitive morph data (binding 10, deltas + active weights). The runtime keeps **one descriptor set per frame slot**: a slot's bindings 0/8/9/10 always point at that slot's own camera / shadow map / skin / morph buffers and are written once at scene-set creation, so an in-flight frame can never observe the next frame's descriptors (no update-after-bind). The layout is an agreed fixed convention (not parsed from SPIR-V), so no per-pipeline layout objects exist; each primitive only pushes a 96-byte block (`material_index` + `skin_base` + morph base/targets/vertex count + `instance_base` + model matrix).
- **GPU material table**: each material is one `material_record` in the storage buffer (5 texture-array indices + factors + flags). A primitive references a material by index, the shader reads `materials[push.material_index]` and samples `textures[<index>]` — material data is stored once on the GPU and shareable between primitives.
- **Screenshots**: press **F12** - the runtime copies the presented swapchain image back to host memory (`runtime::acquire_current_frame_image()`, 8-bit RGBA) and the demo writes it as `screenshot_<timestamp>.png` via `utility::write_png()` (a dependency-free PNG encoder in the utility module). The base directory comes from `[paths] screenshot_dir` (empty = current working directory, created on demand).
- **Exposure**: a linear exposure scale applied before the ACES tonemapper in both the model and skybox passes (gui "exposure" slider -> `runtime::set_exposure`).
- **Post-processing**: the forward passes (pbr / unlit / skybox) render into an **HDR offscreen target** (RGBA16F MSAA, resolved per swapchain image); a fullscreen post pass then applies the linear **exposure** scale, ACES tonemapping and gamma into the swapchain, and the debug overlay draws on top of that result. All display-referred processing lives in one place (`shaders/post.frag`), which is where the bloom chain will be composited next.
- **PBR rendering**: standard metallic-roughness workflow with five texture slots — albedo (sRGB), metallic-roughness, normal, occlusion, emissive — falling back to a 1×1 white texture when missing; all material parameters come from the material table. glTF alpha modes are honored: `OPAQUE`, `MASK` (fragment discard at `alphaCutoff`) and `BLEND` — blended materials draw **alpha-transparent** in a separate pass after the opaque geometry (depth-write off, sorted back-to-front per frame), so overlapping glass / leaves / decals compose correctly. Blending is always enabled on the pipeline but reduces to the source color for opaque draws (alpha 1), and depth-write is a per-draw dynamic state, so no second pipeline is needed. The BRDF theory model is switchable live from the debug GUI ("brdf model" / "diffuse model" combos): GGX + joint Smith (default), GGX + height-correlated Smith, Beckmann or Blinn-Phong NDFs (all with Schlick Fresnel), and Lambert or Oren-Nayar diffuse — each preset differs by exactly one piece, so the models compare A/B on the same scene. Note the presets drive the **direct lights only**: the IBL ambient (prefiltered GGX environment + GGX BRDF LUT with Fdez-Aguera multiscatter compensation, Lambert irradiance) always uses the fixed GGX model, so the GUI is an honest direct-light A/B rather than a whole-scene model swap. On top of the fixed directional sun, punctual lights can be configured at runtime via `runtime::set_point_lights(std::span<vulkan::punctual_light>)` (up to `vulkan::max_punctual_lights` = 2): each light has a world `position`, a linear `color` × `intensity`, an optional smooth `range` cutoff, and - when `spot` is set - a spot axis + outer half-angle cosine (soft cone, inner = `mix(outer, 1, 0.6)`; glTF `innerConeAngle` is not surfaced yet, so spot support is stub-level). Intensity/range are **artistic units**, not physical: the falloff is `1/(1+d²)` (well-behaved at zero) with a `(1-(d/r)²)²` range fade instead of the physical/Khronos `1/d²` and `(1-(d/r)⁴)²`. Punctual lights shade through the same BRDF path as the sun and do not cast shadows in this version. The demo debug GUI currently exposes two point lights (enable + position / color / intensity / range).
- **IBL lighting**: CPU-precomputed environment cubemap → GGX importance-sampled prefiltered environment (mip chain), irradiance map, and BRDF LUT (split-sum), uploaded as `R16G16B16A16_SFLOAT` cubemaps / `R16G16_SFLOAT` LUT. Precompute resolutions are configurable.
- **Directional shadows**: orthographic shadow pass rendering the scene's depth from the sun (2048² dynamic-rendering depth pass, sampled with a LINEAR depth-compare sampler = hardware percentage-closer filtering); enabled over the imported scene's bounds. The pass's depth bias is dynamic state, tunable live from the debug GUI (slope factor for acne on angled surfaces, constant factor for a fixed push). Shadow casters are selected per scene size: scenes up to 1500 leaves render **every leaf** into the shadow map (a wall arbitrarily far up-light still throws its parallel shadow column into the view — a small camera-margin would leak sunlight through interiors like Sponza); very heavy scenes keep a camera + up-light-margin caster subset so the depth pass stays cheap (the runtime logs once when the heuristic kicks in, since crossing the threshold silently could regress the interior-leak fix).
- **Debug GUI (`vulkan.gui`)**: a Dear ImGui overlay driven inside the runtime's frame steps (**F1** shows/hides it at runtime without losing its panels; the key even initializes it on demand when `[gui] show` started false). `vulkan::gui::widget` subclasses (`label_widget` / `checkbox_widget` / `slider_widget` / `vec3_widget` / `combo_widget`...) stack into `debug_panel`s that register with `gui_content` via `runtime::debug_gui()`. The demo panel shows fps and toggles frustum culling, the skybox pass, the shadow pass, switches the **render mode** (`pbr` lit ↔ `unlit` flat base color — the runtime default pipeline, so every default-semantics leaf re-shades live, useful as a shading-free reference or for normal-less test assets), adjusts the shadow depth bias (slope / constant sliders), edits the camera orbit target, and drives two demo **point lights** (enable + position / color / intensity / range, pushed to the runtime once per frame via `chores::apply_point_lights`); animated models additionally get play/pause, a time scrubber and an animation dropdown. Window layout (position / size) persists to `imgui_layout.ini`.
- **glTF loading (`gltf_loader` module)**: standalone CPU module built on [fastgltf](https://github.com/spnda/fastgltf), supporting `.gltf` / `.glb` / data URIs and exporting both a flattened drawable stream and the retained node hierarchy (name + local transform + children, plus per-node TRS base pose, asset-node index, skin and morph references) with raw de-interleaved vertex/index data (JOINTS_0 / WEIGHTS_0 and morph-target deltas included) — and the file's keyframe animations (channels/samplers incl. morph `weights`, times + values decoded to floats, with pure-CPU LINEAR/STEP/CUBICSPLINE sampling), skins (joints + inverse bind matrices) and morph targets (POSITION/NORMAL deltas + default weights). Meshes without a `NORMAL` attribute get per-vertex normals **generated from the triangle connectivity** (smooth, area-weighted) instead of shading with a hard-coded direction.
- **Keyframe animation + skinning + morph targets (`vulkan.animation`)**: `vulkan::animation::controller` (a module of the engine core library) plays channel-bearing animations automatically on a loop — it samples the animation (slerped rotations, morph weights) into the scene tree's node locals (animated roots keep the import offset), rebuilds the per-frame skin matrices (`inv(W_mesh) · W_joint · IBM`) and updates the active morph weights into the runtime's per-frame-slot buffers, so skinned and morphable meshes deform live. It is **format-neutral**: animation data is value-copied into its own `vulkan::animation` structures (samplers/channels/clips/skins/base poses, pure CPU) at init, and `init()` is a template over the `source` concept — any loader exposing the required member shapes (glTF's `scenes` does; a future format just implements them) can drive it, so the module never imports a loader. Heavy animations (many channels, e.g. the recursive-skeleton stress sample with 840 channels over 924 nodes) fan the per-source sampling out over a small `utility.thread_pool` (2–6 threads, sized to the machine) — each source only touches its own runtime nodes, so slices run concurrently; light animations skip the pool and sample on the frame thread. The demo drives it with `frame_clock` (cheap per-frame stamped time) and the `gui` overlay adds play/pause, a time scrubber and an animation dropdown for multi-animation files. Non-indexed glTF meshes (e.g. Fox) are handled by synthesizing indices. Authored glTF cameras are exported and picked as orbit-camera viewpoint seeds (gui "camera" dropdown); punctual lights (KHR_lights_punctual) are exported too, but the demo's lights are the two GUI-configured point lights (`runtime::set_point_lights`) — glTF-authored lights are not auto-imported into them yet.
- **Orbit camera**: left-drag to rotate, wheel to zoom; one shared camera UBO is updated once per frame, with `MAX_FRAMES_IN_FLIGHT` frames in flight.
- **Utility library (`utility` module)**: handle distribution, stack-style destructor mixin, thread pool (`utility.thread_pool`: RAII pool of `jthread` workers with priority queue + `wait_until_free()`, used by the animation sampling fan-out), BVH (used by frustum culling), data block, a per-frame stamped clock (`utility.frame_clock`: single-writer stamps, atomic-load readers), and more. A mimalloc-backed `pmr` manager (`utility::init_pmr()` via `better_pmr`) routes all `std::pmr` allocations — including the runtime's per-frame cull/visible vectors — through mimalloc (vendored under `third_party/mimalloc`); it is idempotent and initialized before `main` from every TU that uses it. Content hashing for GPU-resource dedup is [xxHash](https://github.com/Cyan4973/xxHash) `XXH3_128bits` (`utility::xxh3_128bits`, vendored under `third_party/xxhash`).
- **Startup configuration (`app_config` module)**: TOML config (`config.toml`, `--config <path>` override) merged with argv, covering model / instancing grid, resource paths, window/render settings (size, title, vsync, MSAA, clear color, skybox/shadow toggles) and IBL resolutions. See `config.example.toml`.
- **Engineering practices**: automatic `clang-format` before every build, `-Wall -Wextra -Werror`, and **exceptions disabled in all build configurations** (`-fno-exceptions`; the vendored `std` module makes this work), plus `-flto -march=native -fno-rtti` in Release builds (LTO also covers the vendored libs). Release additionally runs dead-code elimination (`-ffunction-sections -fdata-sections` on every target + link-time `--gc-sections`) and strips symbols at link (`-s`), keeping the single-file exe at ~2.9 MB with no runtime cost.

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
- [scene tree design notes](docs/scene_tree_design.md) (design history of the scene-tree / import rework)
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
│                            #   runtime / scene_tree / render_environment / gui / animation)
├── utility/                 # utility module (data_block / better_pmr / BVH / thread_pool / frame_clock)
├── gltf_loader/             # gltf_loader module (CPU-side glTF/GLB loading)
├── vstd/                     # vstd module — modified from libc++ (LLVM), trimmed to the project's
│                            #   STL usage (import vstd; see vstd/README.md)
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
sh scripts/posix/run.sh       # run; extra args (model / grid side) are forwarded
sh scripts/posix/run.sh path/to/model.glb
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
# or load a different model / lay it out as an instancing grid:
./build-release/vulkan_render path/to/model.glb
./build-release/vulkan_render path/to/model.glb 8   # grid_side 8: one instanced draw call
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

Positional argv overrides the file: `argv[1]` = model path, `argv[2]` = grid side (a number). Configurable: model / instancing grid, `shaders_dir` / `model_dir` paths, window size / title / vsync / MSAA / clear color, skybox & shadow stage toggles, IBL precompute resolutions, and the debug-panel default size.

The Dear ImGui debug overlay is on **by default** — disable it with `[gui] show = false` in the config.

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
