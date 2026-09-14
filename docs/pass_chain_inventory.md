# PRE-GI PBR CHAIN INVENTORY

Repo `C:\Users\23530\Desktop\yzk\vulkan_render`, branch `pass-chain` at 281ab06. Read-only inventory: no source file modified, nothing committed. Paths are repo-relative. HEAD is the tracked state; note that `docs/html/` and `docs/latex/` in the working tree are **gitignored doxygen output from a later state** (`.gitignore:11`) and contain `vulkan_8pass` / `runtime__gi` / `render_resource` pages that do **not** correspond to this commit.

## 0. Findings that matter for the pass work

**1. Named pipelines' colour format vs the transparent instance's attachment format — OPEN QUESTION.**
- `runtime::make_pipeline()` (vulkan/runtime.cpp:3182-3206) builds every named scene pipeline through `core::make_pipeline` (vulkan/core/core.cpp:1424-1453), which passes `this->swap_chain_image_format` as the rendering colour format (core.cpp:1431).
- `core::choose_swap_surface_format` prefers `VK_FORMAT_B8G8R8A8_SRGB` (vulkan/core/init_utils/init_util.cpp:523-525), so on a normal surface the named pipelines declare **B8G8R8A8_SRGB**.
- The transparent pass records those same named pipelines into a rendering instance whose single colour attachment is declared as `vulkan::hdr_format` = **R16G16B16A16_SFLOAT** (vulkan/runtime.cpp:2418, `std::array<VkFormat, 1> const color_formats = {vulkan::hdr_format};`), and the leaves bind the named default pipeline through `record_main_segment(..., gbuffer_pass=false)` (runtime.cpp:2424 → 2088-2110).
- **Status: unresolved.** The capture gate reported no validation finding on this state (no format-mismatch VUID was observed), so either the mismatch is tolerated by the driver/validation configuration in use, or the two formats are in fact compatible in this path. This inventory did not run the renderer or validation layers itself; the next session should confirm it (or record it as a known-tolerated deviation) before moving the transparent pass into a pass declaration.

**2. Stale / dead references in this state** (they read as live features but are not):
- "skybox" pipeline: there is **no** skybox pipeline member and no `make_skybox_pipeline` in this state. The sky is the `sky()` function called by the lighting stage (`shaders/deferred.frag:43` includes `sky.glsl`; `shaders/shading.glsl:19`). `draw_skybox` survives only in documentation (runtime.cppm:1245, runtime.cpp:2044, runtime.cpp:2128) with no matching parameter in any signature.
- `feature_available("skybox")` is documented as a valid name (runtime.cppm:1852) but the implementation only handles `gbuffer-debug`, `taa`, `fxaa`, `shadow`, `clustered` (runtime.cpp:3356-3375).
- Overlay-record comment: `begin_recording()` claims the overlay's draw is "recorded at the end of record_main_drawcalls() while the main rendering instance is still open" (runtime.cpp:1387). The overlay is in fact recorded from `record_fullscreen_triangle()` with `overlay_after` (runtime.cpp:2904), i.e. from the composite (2992) or the FXAA pass (3012). The ImGui frame is opened in `begin_recording` (runtime.cpp:1388-1390).
- Shadow-sampler doc contradiction: core.cppm:515-521 claims `make_shadow_sampler()` creates a NEAREST sampler with manual PCF, while the actual sampler is LINEAR min/mag with `compareEnable = VK_TRUE` (hardware PCF) per `make_shadow_sampler_info` (vulkan/constant_init/constant_init.cppm:97-116) and per the layout comment (core.cppm:874). The constant_init.cppm:97-116 version is the true behaviour.

**3. What this state genuinely lacks** (verified with `git ls-tree -r HEAD` and `git grep` over tracked sources):
- **No `vulkan/pass/`** — no `pass.cppm`, `scene.cppm`, `taa.cppm`, `transparent.cppm`, `gi_probe.cppm`.
- **No `vulkan/render_resource/`** — no `render_resource.cppm`, no `shared.cppm`.
- Neither path appears in CMakeLists.txt (module list at CMakeLists.txt:113-149).
- **No acceleration structures**: no `vulkan/acceleration_structure/`, no `vkCmdTraceRays*` call site; the single mention of ray tracing in tracked sources is the feature-policy comment that it is excluded (vulkan/core/init_utils/init_util.cpp:79).
- No GI/SSGI/probe/ray-traced-shadow shaders or C++ (`git grep -i -E "ssgi|gi_probe|probe_sh|ray.?trac|acceleration_structure"` over tracked `*.cpp`/`*.cppm` matches only init_util.cpp:79).
- **`docs/html/` and `docs/latex/` in the working tree are gitignored doxygen output generated from a LATER commit, not from this one** (`.gitignore:11` covers both). They contain pages for `vulkan.pass`, `runtime_gi.cpp`, `render_resource` and `gi_probe` that do not exist at 281ab06. HEAD tracks only 7 docs files (`git ls-tree -r HEAD docs/`). Do not treat those generated pages as a description of this branch.

# 1. THE RESOURCE SET

**Device-wide / persistent** (created once; `core` ctor order core.cpp:18-46)
- `instance/device/physical_device/surface/queues` core.cppm:192-218; `swap_chain`, `swap_chain_images`, `swap_chain_image_format`, `swap_chain_extent` core.cppm:220-227 (format prefers `B8G8R8A8_SRGB`, init_util.cpp:523-525); `swap_chain_image_views` core.cppm:235.
- `depth_format = find_depth_format(...)` core.cpp:448 (preference list init_util.cpp:579-585, `D32_SFLOAT_S8_UINT` first); main `depth_images/memories/views` core.cppm:327-329 — used only by the no-G-buffer fallback instance (runtime.cpp:999), never sampled.
- `descriptor_pool` core.cppm:339 / core.cpp:828-856; `vma` core.cppm:348; `command_pool` core.cppm:335; timestamp query pool core.cppm:382 + core.cpp:998-1036.
- **Shared layouts**: `scene_descriptor_set_layout` + `scene_pipeline_layout` core.cppm:344-345, built by `core::init_scene_layouts()` core.cpp:858-952 (§4).
- Runtime-owned persistent: texture array `texture_array_views`/`owned_texture_views`/`owned_textures`/`white_texture_index` runtime.cppm:156-160 (`scene_texture_capacity=128` core.cppm:67); IBL `ibl_views`/`ibl_images` runtime.cppm:162-163; `material_buffer` runtime.cppm:169 (`material_capacity=16384`, 80 B records, primitive.cppm:385/397); `instance_buffer` runtime.cppm:193 (`instance_capacity=8192` mat4, primitive.cppm:403); `readback_staging` runtime.cppm:591 (`vk_buffer staging` readback.cppm:60); named pipeline cache `pipelines` runtime.cppm:671; per-image set families `gbuffer_family`/`taa_family`/`post_family` runtime.cppm:328/390/501.
- **Samplers**: `texture_sampler` runtime.cpp:320 (REPEAT, maxLod 12), `env_sampler` runtime.cpp:758 (CLAMP), `shadow_sampler` runtime.cpp:469 (LINEAR + compareEnable ⇒ hardware PCF, constant_init.cppm:97-116 — the "NEAREST + manual PCF" claim at core.cppm:515-521 is stale/wrong), `gbuffer_sampler` runtime.cpp:2285 (NEAREST), `taa_sampler` runtime.cpp:2516 (mag LINEAR / min NEAREST), `post_sampler` runtime.cpp:2154 (CLAMP).
- **Acceleration structures: NONE.** No acceleration-structure module exists in HEAD; the only mention is the feature-policy comment that ray tracing is excluded (init_util.cpp:79).

**Per-swapchain-image** (all sized `swap_chain_image_views.size()`; created/recreated by `core::create_render_targets` core.cpp:485-766, teardown registered once 661-765)

| resource | decl | format | usage (core.cpp) |
|---|---|---|---|
| `hdr_images/memories/views` | core.cppm:242-244 | `hdr_format` = R16G16B16A16_SFLOAT (core.cppm:87) | COLOR\|SAMPLED\|TRANSFER_SRC :500 |
| `ldr_images/...` | core.cppm:263-265 | hdr_format (display-encoded) | COLOR\|SAMPLED :525 |
| `bloom_images/...` (array<4> of vectors) | core.cppm:254-256 | hdr_format, extent>>(level+1) | COLOR\|SAMPLED :648 |
| `gbuffer_images[3]` | core.cppm:271-273 | RGBA8_UNORM / RGBA16F / RGBA8_UNORM (core.cppm:108-112) | COLOR\|SAMPLED :556 |
| `gbuffer_depth_images/...` | core.cppm:279-281 | `depth_format`, DEPTH-aspect view :627 | DEPTH_STENCIL_ATTACHMENT\|SAMPLED :620 |
| `velocity_images/...` | core.cppm:284-286 | `gbuffer_velocity_format` = R16G16_SFLOAT (core.cppm:121) | COLOR\|SAMPLED :589 |
| `scene_color_images/...` (TAA input) | core.cppm:293-295 | hdr_format | COLOR\|SAMPLED :590 |
| `taa_history_images/...` | core.cppm:302-304 | hdr_format | TRANSFER_DST\|SAMPLED :603 |
| `present_ready_semaphores` (binary, per image) | core.cppm:361 | — | present gate |

Per-image flags owned by the runtime: `gbuffer_depth_written` (`std::vector<bool>`, runtime.cppm:453; set runtime.cpp:1814-1816, consumed+cleared 2679/2689), `taa_history_valid` (`std::vector<bool>`, runtime.cppm:400; read 2569, written 2670-2672, reset 1088/2498), `image_view_proj` (`std::vector<glm::mat4>`, runtime.cppm:399; read 1253, written 2667-2669).

**Per-frame-slot** (`MAX_FRAMES_IN_FLIGHT = 2`, core.cppm:370)
- core: `image_available_semaphores` :360, `frame_done_semaphores`/`frame_done_values` :362-363, `gpu_timing_marks` :389, `gpu_timing_read_value` :392.
- runtime: `command_buffers` runtime.cppm:740 (ctor 154-156); `secondary_command_buffers` runtime.cppm:768 — exactly 1 per slot (`secondary_pass::transparent`, enum 766-767); `shadow_recording` runtime.cppm:613 — one {pool, secondary} per **cascade** per slot (ctor 177-183); `main_segments` runtime.cppm:777 — one {pool, secondary} per task-pool worker per slot (ctor 184-190); `scene_sets` runtime.cppm:226 (one set per slot, bindings.cppm:92); `shadow_rendered_version`/`shadow_rendered_models` runtime.cppm:626-627.
- runtime per-slot buffers: `camera_buffers`+`camera_mapped` (147-148, `sizeof(camera_ubo)`=320, primitive.cppm:66), `light_buffers`+`light_mapped` (549-550, `sizeof(light_ubo)`=8576, primitive.cppm:182), `skin_buffers` (212-213, 2048 mat4), `morph_buffers` (218-219, 8 MiB floats), `motion_buffers` (202-203, 8192 mat4), `shadow_images`+`shadow_array_views`+`shadow_layer_views` (529-531, layered `depth_format`, `shadow_map_size=2048` :514, one layer per cascade), `cluster_count_buffers` (643, `max_cluster_count`=12288 uints) + `cluster_index_buffers` (645, 12288×32 uints; primitive.cppm:115).

**The per-slot vs per-image line (the bug-prone distinction):** the shadow maps and the entire scene descriptor set are per **frame slot**; every render target (HDR/LDR/bloom/G-buffer/velocity/scene_color/TAA history/main depth) is per **swapchain image**. Only three pieces of state are per image: `gbuffer_depth_written`, `taa_history_valid`, `image_view_proj`. `on_swapchain_recreated()` runtime.cpp:1060-1093 retires the three set families and re-sizes the per-image flags.

# 2. THE FRAME'S RECORDING SPINE (vulkan/runtime.cpp)

`gpu_mark_id` order runtime.cppm:238-253; labels runtime.cppm:268-278; order is validated in `gpu_mark` 1095-1111.

1. `begin_recording()` **1362-1580** — `vkBeginCommandBuffer` 1376, `begin_gpu_timing` 1383, `gpu_mark(frame_begin, TOP_OF_PIPE)` **1384**, ImGui `new_frame()` 1389, `update_world` 1396-1398, `advance_motion_transforms()` 1401 (per-slot motion buffers), collect leaves 1404-1407, BVH rebuild + camera cull + shadow-caster set 1424-1541, opaque/transparent split + far→near sort 1549-1578.
2. `record_main_drawcalls()` **1601-1768** — `record_cluster_pass` 1614; `active_features()` 1632 + `shadow_geometry_signature()` 1635 + per-slot reuse test 1639; shadow: one task per cascade recording `record_shadow_content` into a per-cascade secondary (1665-1682), one depth-only instance per cascade + barrier + execute (1686-1712), array-wide hand-back barrier (1725-1729); shadow-off layout fixup 1737-1759; `gpu_mark(shadow_end)` **1763**; `record_scene()` 1767.
3. `record_scene()` **1770-1774** — `record_scene_attachments`, `update_pass_geometry`, `record_opaque_scene`.
4. `record_scene_attachments()` **1778-1821** — 5 attachment barriers + depth barrier 1798-1817; sets `gbuffer_depth_written[image]=true` 1814-1816.
5. `update_pass_geometry()` **1824-1888** — resync cached viewport/scissor of named pipelines (unique lock 1842), post/post_hdr/fxaa, gbuffer, gbuffer_debug, deferred, taa (shadow excluded — fixed map size).
6. `record_opaque_scene()` **1892-1977** — inheritance for 5 color formats 1911-1916; `segment_count = min(main_segments.size(), max(1, leaf_count))` 1919; single-segment path 1921-1940; parallel tasks 1949-1967; `run_tasks(..., recording)` 1967; `begin_rendering(..., SECONDARY_COMMAND_BUFFERS)` 1969 + execute in order 1970-1976.
7. `sub_render_task::operator()` **2131-2147** → `record_main_segment()` **2051-2124**.
8. `end_recording()` **3048-3085** — `record_post_process` 3056, optional `record_screenshot_copy` 3062, present barrier 3073-3077, `gpu_mark(frame_end)` **3080**, `vkEndCommandBuffer` 3081.
9. `record_post_process()` **3022-3047** (only caller is `end_recording`) — `record_scene_tail()` 3026; `ensure_post_descriptors()` 3028; bail-out 3029-3031; HDR→SHADER_READ 3036; bloom weight forced 0 for the debug view 3042-3043; `record_bloom_chain` 3045; `record_composite` 3046.
10. `record_scene_tail()` **2832-2860** — `vkCmdEndRendering` 2833, `gpu_mark(scene_end)` **2836**, lighting+transparent only if `deferred_lit_active()` 2842-2845, `gpu_mark(lighting_end)` **2846**, TAA 2851, `gpu_mark(taa_end)` **2852**, debug view 2856-2858, `gpu_mark(main_end)` **2859**.
11. `record_bloom_chain()` **2909-2952** — 4 passes or layout fixups 2931-2947, `gpu_mark(bloom_end)` **2951**.
12. `record_composite()` **2954-3020** — composite triangle 2992 (+overlay when `!fxaa`), `gpu_mark(composite_end)` **2995**, FXAA 2997-3013, `gpu_mark(fxaa_end)` **3016**.
13. `submit_and_present()` **3087-3111**; `render_frame()` **3117-3140** runs the phases in order.

# 3. THE PBR/SCENE CHAIN

**Opaque / G-buffer pass** — `record_opaque_scene` 1892-1977 → `begin_rendering` 953-1003 → `record_main_segment` 2051-2124 (+`sub_render_task::operator()` 2131-2147).
- Sets: scene set (set 0) bound once per segment against `core::scene_pipeline_layout` (2056-2066).
- Pipeline: `gbuffer_pipeline` as the pass default (`gbuffer_pipeline_name` = "gbuffer" runtime.cppm:1686; chosen 2088-2092); a leaf requesting any other name is logged and skipped (2099-2103).
- Render targets: 5 color attachments — gbuffer 0/1/2, velocity, `scene_target_view(image)` (HDR, or `scene_color` under TAA), all CLEARed (963-975) — plus `gbuffer_depth_image_views[image]` cleared with `STORE_OP_STORE` (981); extent = swapchain (982).
- Push constants: per-leaf `material_push_constants`, 96 B at offset 0, VERTEX|FRAGMENT (primitive.cpp:26-31).
- Barriers: `record_scene_attachments` 1790-1820 (5 attachment transitions + depth); per-image flag write 1814-1816.
- Fallback instance with no G-buffer pipeline 987-1002: `hdr_image_views[image]` CLEARed with `clear_color` + main depth DONT_CARE.

**Skybox — NO pass exists.** No skybox pipeline member and no `make_skybox_pipeline`; the sky is the `sky()` function inside the lighting stage (`shaders/deferred.frag:43` includes `sky.glsl`; shading.glsl:19). `draw_skybox` survives only in stale docs (runtime.cppm:1245, runtime.cpp:2044, 2128) with no matching parameter, and `feature_available("skybox")` is documented (runtime.cppm:1852) but unimplemented (runtime.cpp:3356-3375).

**Deferred lighting stage** — `record_lighting_pass` 2296-2383.
- Barriers before the instance: 3 G-buffer targets → `hdr_sampling_transition` + `color_attachment_dependency` on `scene_target_image` (2322-2331) + `ensure_gbuffer_depth_sampled()` 2332 (2675-2691, `shadow_map_sampling_transition`, clears `gbuffer_depth_written[image]` 2689).
- Sets: **set 0 = scene set (frame slot), set 1 = `gbuffer_family.set(image,0)`** (2367-2368) against `deferred_pipeline_layout` (scene layout + G-buffer layout, pipelines.cppm:244).
- Pipeline `deferred_pipeline`; target `scene_target_view(image)` LOADed, no depth (2357-2358); missing-set path clears `scene_color_images[index]` 2341-2350.
- Push: `deferred_push_constants` (mat4 `inv_view_proj`, vec4 `ssao`, float `unlit`; runtime.cppm:344-354), offset 0, ≈96 B, FRAGMENT (2371-2380). SSAO rides this push (intensity 0 ⇒ shader returns exactly 1.0).

**Transparent pass** — `record_transparent_pass` 2392-2442 (skipped when `frame_transparent` empty).
- Barriers: `sampling_to_depth_attachment_transition` on `gbuffer_depth_images[image]` + `color_attachment_dependency` on the scene target (2407-2413).
- Recorded into the per-slot `secondary_pass::transparent` secondary, inheritance color `{vulkan::hdr_format}` (2418-2429), via `record_main_segment(..., gbuffer_pass=false)` 2424 ⇒ leaves bind the **named** default pipeline ("pbr"), which `core::make_pipeline` built with `swap_chain_image_format` (core.cpp:1431). See §0.1: suspected format mismatch, unresolved.
- Target: scene target LOAD + G-buffer depth LOAD (2434-2436); leaves sorted far→near (1574-1577), depth writes off (primitive.cpp:42).

**Shadow passes** — inline in `record_main_drawcalls` 1640-1759 + `record_shadow_content` 1984-2042.
- Gate `active_features().shadow` (3285) + per-slot reuse fingerprint (`shadow_geometry_signature` 1589-1599, test 1639).
- Sets: scene set (1987-1997); pipeline `shadow_pipeline` forced for every leaf regardless of name (2005-2010); `env.two_sided = true` (2024); depth-write forced TRUE (2016-2018); live `vkCmdSetDepthBias` 2000.
- Push: 1 uint cascade index at offset `scene_cascade_push_offset` = 96 (`scene_cascade_push_size` = 4, core.cppm:79-80), pushed per cascade secondary (1676).
- Targets: one depth-only instance per cascade into `shadow_layer_views[slot][cascade]`, LOAD CLEAR / STORE (1691-1711); per-layer barrier 1691-1695; array-wide hand-back 1725-1729; shadow-off path 1737-1759 (`undefined_to_depth_sampling_transition`). `transparent` leaves are skipped (2037-2039).

**TAA resolve** — `record_taa_pass` 2563-2673; active only if `taa_active()` (2464-2468).
- Sets: `taa_family.set(image,0)` (1 set/image, 4 bindings: scene_color / history / velocity / G-buffer depth; layout pipelines.cppm:200-207; writes 2540-2556) against `taa_pipeline_layout` (2616-2617).
- Barriers: scene_color + velocity → `hdr_sampling_transition`, history `undefined_to_sampling_transition` only when not valid (2584-2596), `ensure_gbuffer_depth_sampled` 2599, HDR → `color_attachment_transition` 2601-2604; copy HDR→history (2637-2652); hand-back 2656-2662.
- Push: `taa_push_constants` (8 floats: history_valid, blend_static, blend_min, texel_size_x/y, depth_scale, depth_offset, unused; runtime.cppm:401-410), offset 0, ≈32 B, FRAGMENT (2618-2627).
- Flags: reads `taa_history_valid[image]` 2569; writes `image_view_proj[image]` and `taa_history_valid[image]` 2667-2672.

**G-buffer debug view** — `record_gbuffer_debug_pass` 2763-2830; runs INSTEAD of the lighting stage (2856-2858).
- Barriers: HDR→color attachment, 3 targets + velocity → sampling, then `ensure_gbuffer_depth_sampled` (2788-2799).
- Sets: `gbuffer_family.set(image,0)` against `gbuffer_pipeline_layout` (2816); layout = 5 COMBINED_IMAGE_SAMPLER bindings (pipelines.cppm:152-159); writes 2736-2756.
- Pipeline `gbuffer_debug_pipeline`; target `hdr_image_views[image]` CLEARed (2804-2805).
- Push: `gbuffer_debug_push_constants` (channel, proj_22, proj_32, motion_gain; runtime.cppm:334-343), offset 0, ≈16 B, FRAGMENT (2817-2824); channel clamped 0..8 (2709-2711).

**Post chain** — `record_post_process` 3022-3047 → `ensure_post_descriptors` 2183-2236 → `record_bloom_chain` 2909-2952 → `record_composite` 2954-3020 → `record_fullscreen_triangle` 2887-2907 (binds `post_pipeline_layout`, pushes `post_push_constants`, viewport/scissor/cull NONE, draws 3 vertices; `overlay_after` draws ImGui inside the same instance 2903-2905).
- Post sets: `post_family`, 5 sets/image (0 prefilter, 1-3 downsample, 4 composite), 6 bindings each, fingerprints {hdr views, bloom level-0 views, ldr views} (2196-2233).
- Bloom: level barriers + `undefined_to_sampling_transition` fixups when disabled (2931-2947); all bloom levels and the FXAA input use `post_hdr_pipeline`.
- Composite: target = LDR when FXAA is on, else swapchain (2966-2968); pipeline variant chosen 2972; `encode_gamma` from `is_srgb_format(swap_chain_image_format)` 2977.
- FXAA: LDR→sampling, swapchain→color attachment, `post_fxaa_pipeline` with `post_family.set(image,4)` (3010-3012); overlay rides this instance.
- Push: `post_push_constants` (exposure, bloom_intensity, bloom_threshold, mode, encode_gamma, fxaa_subpixel, fxaa_edge_threshold; runtime.cppm:462-477), offset 0, ≈28 B, FRAGMENT (2899).
- Overlay is recorded only from `record_fullscreen_triangle` (2904); the comment at runtime.cpp:1387 claiming it is drawn at the end of `record_main_drawcalls()` is **stale** (see §0.2). The ImGui frame opens in `begin_recording` (1388-1390).

**Cluster compute (pre-pass)** — `record_cluster_pass` 3446-3502: binds the scene set at `VK_PIPELINE_BIND_POINT_COMPUTE` with `scene_pipeline_layout` (3468), `cluster_pipeline` (3469), dispatch `cluster_count/64` (3470-3471), 2 COMPUTE→FRAGMENT buffer barriers (3477-3501). Counts are memset by the host in `pace_and_acquire` before upload (1322-1325).

# 4. THE SHARED DESCRIPTOR SETS

**Layout** — `core::init_scene_layouts()` core.cpp:858-952, 14 bindings (documented identically in core.cppm:45-63):
0 UNIFORM_BUFFER VERTEX|FRAGMENT|COMPUTE (camera UBO, per slot) :861 · 1 COMBINED_IMAGE_SAMPLER ×128 FRAGMENT (texture array; PARTIALLY_BOUND) :862/:896 · 2/3/4 COMBINED_IMAGE_SAMPLER (prefiltered env / irradiance / BRDF LUT) :863-865 · 5 STORAGE_BUFFER FRAGMENT (material table) :867 · 6 STORAGE_BUFFER VERTEX (instance transforms) :869 · 7 UNIFORM_BUFFER VERTEX|FRAGMENT|COMPUTE (light UBO) :872 · 8 COMBINED_IMAGE_SAMPLER FRAGMENT (shadow map array) :876 · 9 STORAGE_BUFFER VERTEX (skin) :879 · 10 STORAGE_BUFFER VERTEX (morph) :882 · 11/12 STORAGE_BUFFER FRAGMENT|COMPUTE (cluster counts/indices) :886-887 · 13 STORAGE_BUFFER VERTEX (previous world matrices) :891.
Pipeline layout: 1 set + ONE push range, VERTEX|FRAGMENT, offset 0, size `scene_push_constant_size + scene_cascade_push_size` = **96+4 = 100** (core.cpp:916-940, static_assert ≤128 :930).

**Writers** (runtime.cpp): `ensure_scene_set()` 531-624 creates one set per slot (bindings.cppm:173-181) and writes 0, 5, 6, 9, 10, 11, 12, 13 per slot (560-618); `write_light_and_shadow_bindings()` 632-672 (7, 8 per slot); `write_ibl_bindings()` 674-699 (2/3/4 → `scene_bindings::write_ibl` bindings.cppm:207-233, white placeholder when IBL unloaded). `register_material()` 765-951 writes binding 1 array elements to every slot via `update_all_scene_sets()` 626-630 → `scene_bindings::update_all` (bindings.cppm:194-205, dstSet substituted per slot). `init_scene_resources()` 274-423, `ensure_shadow_resources()` 425-490, `ensure_cluster_buffers()` 492-529 create the buffers; the last two are called lazily from `ensure_scene_set` (538-542).

**Per-image families** (`bindings::image_set_family`, bindings.cppm:96-171): `post_family` = 5 sets/image × 6 bindings (`ensure_post_descriptors` 2183-2236); `gbuffer_family` = 1 set/image × 5 bindings (`ensure_gbuffer_descriptors` 2713-2761); `taa_family` = 1 set/image × 4 bindings (`ensure_taa_descriptors` 2527-2561). The family owns the pool, the "views unchanged ⇒ no rebind" rule and pool retirement (bindings.cppm:275-342, 368-378).

# 5. THE PIPELINE REGISTRY AND BUILDERS

`std::map<std::string, vk_pipeline, std::less<>> pipelines` runtime.cppm:671; `make_pipeline()` runtime.cpp:3182-3206 → `core::make_pipeline` core.cpp:1424-1453 (scene layout, `swap_chain_image_format`, `depth_format`, 1x, cached viewport); the first created becomes `default_pipeline_name` (680, 3201-3203). Named pipelines are **not usable in the G-buffer instance** (runtime.cppm:1677-1686; such leaves are skipped 2099-2103) — here they serve the transparent pass and hypothetical custom leaves.

| member (decl) | builder | shaders | set layouts | push range |
|---|---|---|---|---|
| named `pipelines` (671) | runtime.cpp:3182 → core.cpp:1424 | any vertex/fragment | scene | scene layout (100 B) |
| `gbuffer_pipeline` (307) | `core::make_gbuffer_pipeline` core.cpp:1455-1503 | pbr.vert + gbuffer.frag (chores.cpp:219-221) | scene | 100 B (scene layout) |
| `gbuffer_debug_pipeline` (310) | `pipelines::build_gbuffer_debug` pipelines.cppm:144-192 | post.vert + gbuffer_debug.frag (chores.cpp:228-230) | own 5-binding layout | FRAGMENT, `sizeof(gbuffer_debug_push_constants)` |
| `deferred_pipeline` (313) | `pipelines::build_deferred` pipelines.cppm:241-266 | post.vert + deferred.frag (chores.cpp:236-237) | scene + gbuffer (2 sets) | FRAGMENT, `sizeof(deferred_push_constants)` |
| `taa_pipeline` (384) | `pipelines::build_taa` pipelines.cppm:196-240 | post.vert + taa.frag (chores.cpp:243-244) | own 4-binding layout | FRAGMENT, `sizeof(taa_push_constants)` |
| `post_pipeline` (478) / `post_hdr_pipeline` (484) | `pipelines::build_post` pipelines.cppm:77-140 | post.vert + post.frag | own 6-binding layout | FRAGMENT, `sizeof(post_push_constants)` |
| `post_fxaa_pipeline` (490) | `pipelines::build_fxaa` pipelines.cppm:268-276 | post.vert + fxaa.frag | reuses post layout | reuses post range |
| `shadow_pipeline` (592) | `core::make_depth_pipeline` core.cpp:1505-1528 (wrapped runtime.cpp:3215-3238, sets shadow-map viewport/scissor) | shadow.vert + shadow.frag | scene | 100 B (scene layout; cascade idx at 96) |
| `cluster_pipeline` (607) | `core::make_cluster_pipeline` core.cpp:1530-1554 (runtime.cpp:3240-3249) | light_cluster.comp | scene | scene layout (unused) |

Blend states: G-buffer = 4 opaque + 1 additive on the scene-color attachment (core.cpp:1469-1475); deferred = additive (runtime.cpp:2263).

**Created in `chores::setup_pipeline` (chores.cpp:141-254), in this order** (sole caller main.cpp:175): `pbr` (144) → `unlit` (149) → post (159, panic on failure) → fxaa (173, optional) → shadow (189, optional) → cluster (204, optional) → nested: gbuffer (221) → gbuffer debug (230) → deferred (237) → taa (244). The nesting exists because fxaa needs post's layout, deferred needs the G-buffer debug layout, and taa needs deferred's stage.

# 6. THE SCENE DRAW PATH

- `primitive` (primitive.cppm:499-598) owns vertex/index `vk_buffer`s, `index_type/count`, `pipeline_name` (521), `push` (:523), `motion_slot_index` (:529), `double_sided`/`transparent` (:535-539), local AABB (:545-547); `set_world()` writes `push.model` (primitive.cpp:9-13).
- Per-leaf push struct: **`material_push_constants`** primitive.cppm:452-481, 96 B (`static_assert(sizeof == scene_push_constant_size)` :481), field order: `material_index`(u32@0) `flags`(4) `skin_base`(8) `morph_base`(12) `morph_targets`(16) `morph_vertices`(20) `instance_base`(24) `motion_base`(28) `alignas(16) model`(mat4@32).
- Draw methods: `normal_draw_primitive::draw` primitive.cpp:40-47 (`bind_default` → depth-write(!transparent) → cull(double_sided) → bind geometry + push 96 B → `vkCmdDrawIndexed`); `instanced_draw_primitive::draw` 66-85 (source geometry, `instance_count`, push flag bit0); `static_draw_primitive::draw` 95-129 (one bind, one offset draw per chunk with its own `material_index` push). Shared `bind_geometry_and_push` 20-32 pushes at offset 0, VERTEX|FRAGMENT, against `env.layout`.
- Environment `render_environment` (render_environment.cppm:54-131): fields `command_buffer`, `default_name`, `bind`, `set_depth_write_fn`, `layout`, `bound`, `set_cull_mode_fn`, `two_sided`, plus dedup state; methods `in_default_pipeline` 82-84, `bind_default` 87-92, `bind_pipeline` 95-100, `set_depth_write` 107-114, `set_cull_mode` 124-131.
- Pipeline choice: default-semantics leaves call `env.bind_default()` and draw with the pass default; a custom leaf calls `env.bind_pipeline(this->pipeline_name)`. `create_primitive()` checks the name exists but stores nothing on `normal_draw_primitive` (runtime.cpp:3940-3950); `make_primitive` copies the name into the scene node name (4020-4021).
- Parallel per-segment recording: `main_segments` = one {own `VkCommandPool`, secondary} per worker per slot (ctor runtime.cpp:184-190; `make_command_pool` core.cppm:449-456); `segment_count = min(worker_pairs, max(1, leaf_count))` (1919); `leaf_count < 4` or 1 segment ⇒ single-threaded record+execute (1921-1940); otherwise contiguous spans (1952-1954) become `sub_render_task` values (struct runtime.cppm:1358-1378, built 1955-1965) posted via `run_tasks(tasks, task_priority::recording)` 1967; `run_tasks` (127-133) = `task_pool.post_batch` + `wait_until_priority_done`; `task_priority` = {animation=1, recording=0} (runtime.cppm:108-112) and higher numeric runs first (thread_pool.cppm:42-48; per-priority pending counts 53-58); pool sized `hardware_concurrency()/4`, floor 1 (runtime.cpp:113-121), declared before the pipeline state for destruction order (runtime.cppm:650-656).

# 7. WHAT IS NOT THERE YET

Verified with `git ls-tree -r HEAD` and `git grep` over tracked sources:
- **No `vulkan/pass/`** (no `pass.cppm`, `scene.cppm`, `taa.cppm`, `transparent.cppm`, `gi_probe.cppm`) and **no `vulkan/render_resource/`**. Neither appears in CMakeLists.txt (module list 113-149).
- **No GI/SSGI/probe/ray-traced-shadow code**: `git grep -i -E "ssgi|gi_probe|probe_sh|ray.?trac|acceleration_structure"` over tracked `*.cpp`/`*.cppm` matches exactly one comment (init_util.cpp:79, "take most features except ray tracing"); there is no acceleration-structure module and no `vkCmdTraceRays*` call site.
- **Working-tree caution**: `docs/html/` and `docs/latex/` are gitignored doxygen output from a later state (`.gitignore:11`) containing `module__vulkan_8pass`, `runtime__gi_8cpp`, `render_resource` pages. HEAD tracks only 7 docs files (`git ls-tree -r HEAD docs/`).
- Features master has that HEAD lacks (from `Compare-Object` of `git ls-tree -r`): `vulkan/pass/*`, `vulkan/render_resource/{render_resource,shared}.cppm`, `vulkan/acceleration_structure/*`; shaders `gi_probe.comp`, `ssgi.comp`, `ssgi_spatial.comp`, `ssgi_spec.comp`, `ssgi_temporal.comp`, `rt_shadow.comp`, `hit_shading.glsl`, `probe_sh.glsl`, `ibl_specular.glsl`, `mask_bake.comp`, `compute_skin.comp`; tests `test_pass.cpp`, `test_render_resources.cpp`; docs `gi_hit_shading.md`, `pass_io_design.md`, `runtime_split.md`, `level2_handoff.md`, `reference/lumen_*.md`; `scripts/measure/*`, `scripts/windows/capture.ps1`, `snapshot/README.md`.
- Stale/dead references to things this state no longer has: "skybox" pipeline (none; sky is `shaders/deferred.frag:43`), the forward/MSAA path (removed; core.cppm:231-234), `draw_skybox` doc with no parameter (runtime.cppm:1245 / runtime.cpp:2044), `feature_available("skybox")` documented but unimplemented (runtime.cppm:1852 vs runtime.cpp:3356), the overlay-record comment at runtime.cpp:1387, and the shadow-sampler doc at core.cppm:515-521 contradicting constant_init.cppm:97-116.

# 8. A PROPOSED PASS DECOMPOSITION (only what the code supports)

| pass | owns (targets / instance / state) | must be handed per frame | blockers visible in this code |
|---|---|---|---|
| Shadow (per cascade) | `shadow_images[slot]`, array/layer views, `shadow_pipeline`, `shadow_recording[slot][cascade]`, depth bias, `shadow_content_version` + `shadow_rendered_*[slot]` | scene set (slot), `shadow_casters`, per-cascade light matrices, `shadow_map_size` | per-SLOT resources + per-cascade pools; caster set built in `begin_recording` (1489-1541); reuse fingerprint spans caller-driven skin/morph writes (1589-1599) |
| Cluster compute | `cluster_pipeline`, count/index buffers[slot], `cluster_tiles_x/y` | scene set (slot), grid + depth range in the light UBO, host memset of counts | needs the set bound at COMPUTE (the scene push range excludes compute, core.cpp:924-927) and a barrier to FRAGMENT (3477-3501) |
| G-buffer (opaque) | the scene instance 953-1003, `gbuffer_pipeline`, `main_segments[slot]`, `frame_visible`, `gbuffer_depth_written[image]`, `update_pass_geometry` viewports | scene set (slot), visible leaf spans, `scene_target_view(image)` choice, inheritance formats | per-IMAGE flag produced here and consumed by three later stages (1814-1816 / 2679); the instance spans two functions (`record_scene_attachments` + `record_opaque_scene`); segments need the `owner` pointer + per-worker pools |
| Deferred lighting | `deferred_pipeline` + layout, `gbuffer_family` sets, `deferred_push_constants`, SSAO state, `current_inv_view_proj` | scene set (slot) + G-buffer set (image), image index, `unlit`, ssao knobs, scene-target choice | two sets from two owners; needs the depth-layout handoff (`ensure_gbuffer_depth_sampled`) and a dependency barrier on the LOADed scene color (2322-2331) |
| Transparent | `secondary_pass::transparent[slot]`, `frame_transparent`, named pipelines | scene set (slot), sorted transparent leaves, G-buffer depth + scene target views | named-pipeline format vs instance format discrepancy (§0.1; core.cpp:1431 vs runtime.cpp:2418); depends on the lighting stage having consumed `gbuffer_depth_written`; transparent leaves carry no motion vectors (runtime.cppm:1193-1195) |
| TAA resolve | `taa_pipeline`+layout+`taa_family`, `taa_sampler`, `taa_history_valid[image]`, `image_view_proj[image]`, jitter index, `scene_color_images` / `taa_history_images` | image index, jitter/blend params, `history_valid`, projection terms | per-IMAGE history validity + per-image previous view-proj (the documented smear trap, runtime.cppm:395-399); resolves into the HDR target the post chain binds; `taa_active()` depends on the deferred stage (2464-2468) |
| G-buffer debug | `gbuffer_debug_pipeline` + set layout + pipeline layout + `gbuffer_sampler`, `gbuffer_family`, `gbuffer_channel_index` | image index, channel, projection terms, motion gain, scene-target view | mutually exclusive with the lighting stage (2703-2707, 2856); needs the same depth handoff; forces bloom weight 0 (3042-3043) |
| Bloom | `post_hdr_pipeline`, `post_family` sets 0-3, `bloom_images[4]` per image, `post_sampler` | image index, exposure/threshold, swapchain extent | skipped-but-transitioned when intensity 0 (2931-2947); its sets come from `ensure_post_descriptors`, which also needs the HDR + LDR view lists (2188-2196) |
| Composite + FXAA + overlay | `post_pipeline`/`post_fxaa_pipeline`, `post_family` set 4, `ldr_images[image]`, ImGui overlay | image index, exposure, bloom sum weight, `encode_gamma`, fxaa params, swapchain format | one instance may carry the overlay (2887-2905) and must be the last writer; target/pipeline chosen per frame (2966-2977) |
| Post-frame tail (screenshot + present) | `readback_staging`, `screenshot_*`, present barrier | swapchain image index, whether post wrote the swapchain | ordering constraint: the copy and present transition are only legal in `end_recording` (3048-3080) |
| Frame orchestration (not a pass) | `active_slot`, `current_image_index`, `current_ubo`, `frame_leaves`/`frame_visible`/`frame_transparent`/`shadow_casters`, cull BVH + camera key, `gpu_mark` order | the frame's derived state | `active_features()` is the single derivation every pass reads (3272-3297); `gpu_mark` requires strict `gpu_mark_id` order (1104-1108) |
