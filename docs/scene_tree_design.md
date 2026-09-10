# Scene Graph Storage — Design Document

Status: migration complete — storage, import and the render loop are tree-driven.
The scene concept now spans **two peer modules**: `vulkan.scene_tree` (pure-CPU
scene storage + the abstract leaf interface, absorbed the former `vulkan.model`)
and `vulkan.primitive` (the GPU drawables — `vulkan::primitive` and its
normal/instanced/static strategies — plus the material / camera / light UBO
records of the GPU scene set; a later rename split them out of
`vulkan.runtime.scene_tree`). Per-node transforms work through
`runtime::get_scene()` and the per-frame animation writes (the former `spin` /
`spin-subtree` demo modes were removed), and keyframe TRS animation,
skinning and morph targets play through the same per-node locals (see §8 —
loader sampling in `gltf_loader`, per-frame playback / skin matrices / morph
weights in `main`). The scene tree is **caller-owned**: `main` declares the
`scene_tree::scene` and binds it with `runtime::set_scene(scene&)`; the runtime
renders it but never owns or destroys it (the tree must be destroyed before the
runtime so the leaves' GPU buffers release through the still-alive vma
allocator — declaration order in `main` guarantees this). Programmatic scenes
build through the scene-tree mounting helpers (`add_root` / `add_child` /
`attach` / `find_node`, §4.2); merged/static geometry renders as one owned
`static_draw_primitive` (one buffer bind + N offset draws), and the frame's pass
recording is multi-threaded over secondary command buffers (see the README's
runtime bullet — per-worker `{command pool, secondary}` pairs, `sub_render_task`
segments on the shared task pool). **Pipeline binding is decoupled from the tree**:
leaves carry default semantics (empty `pipeline_name`) or an explicit name, and
`draw()` receives a per-recording-worker `vulkan.render_environment` (thread-local,
deduplicated `bind_default()` / `bind_pipeline(name)`; see §4.1/§4.3). Remaining:
mesh sharing / GPU dedup (future pass, §8).

## 1. Motivation

*Why this pass exists* — as originally written, the renderer stored the scene as
*two flattened levels* and the original glTF hierarchy was lost between them:

1. **Loader side** (`gltf_loader.cpp` `load_scene`): the glTF node tree is expanded
   by `fastgltf::iterateSceneNodes` into a flat `gltf::scene { nodes: [...] }`
   where every `node` carries only `meshes` + a **world** `transform_matrix`.
   Parent/child edges, per-node local transforms and mesh-sharing semantics are
   gone the moment the file is loaded.
   *(Fixed — the loader now keeps the tree, see §3.1 and step 1 in §6.)*

2. **Runtime side** (`vulkan/runtime.cppm`): models live in
   `std::map<std::string, std::vector<std::unique_ptr<model>>>` — a flat list per
   pipeline. Each `model` is a self-contained leaf (geometry + material + baked
   world matrix in `push.model`). There is no notion of a parent transform, so you
   cannot rotate/translate a *group*, re-parent parts programmatically, or express
   TRS animation later.
   *(Fixed — runtime storage is now a scene tree, see §3.2 and step 2 in §6.)*

Goals (agreed with maintainer):

- Support **whole-group / programmatic transforms** (move, rotate, scale a subtree).
- **Lay the foundation for animation / skinning** (per-node local TRS is the
  prerequisite; actual animation/skinning is out of scope for this pass).
- **Remove the design debt**: stop destroying the loaded hierarchy.

Non-goals for this pass (explicitly deferred, see §8):

- Mesh sharing / GPU geometry dedup across nodes (`instanced_draw_model` already
  covers "one geometry, drawn many times"; glTF mesh *sharing* is a separate step).
- Animation samplers, skinning, IK.
- Per-node material overrides (glTF primitives reference materials by index; a
  later step may allow overriding per instance).

## 2. Starting shape, pre-migration (historical reference)

Before this work the shapes were flat, world-space at every level:

```cpp
// gltf_loader: flat, world-space
struct node { std::vector<mesh> meshes; glm::mat4 transform_matrix; };
struct scene { std::string name; std::vector<node> nodes; };

// runtime: flat per-pipeline
std::map<std::string, std::vector<std::unique_ptr<model>>> models;
//   model: geometry buffers + pipeline ptr + material_push_constants push (baked world)
//   normal_draw_model / instanced_draw_model (polymorphic draw())
```

Consumers of the flat lists (all re-pointed at the tree now):

- `render_frame()`: per pipeline -> `begin_pipeline()` -> `model->draw()` (main pass
  and shadow pass both iterate the same model lists).
- `import_scene()`: runtime template drives a structural `drawable_iterator`
  (pure CPU, no Vulkan in gltf_loader), calls `make_model("pbr", info)` per
  drawable, bakes `offset * node_world` into `push.model`.
- `main.cpp`: bounding scan over `gltf::scenes` (flat iterate), grid stress via
  `make_instanced_model(source, transforms)` appended to `"pbr"`.

## 3. Overall design

Keep two separable concerns distinct:

- **Storage / scene semantics = a tree** (what the file says, and what users
  manipulate: local transforms, parents, whole-group transforms).
- **Rendering = a flat per-pipeline draw list** (what the GPU wants: bind a
  pipeline once, issue many draws). The tree is *walked* to produce the flat
  render list; the render loop itself barely changes.

The flat "before" shapes are replaced by:

```
gltf (loader, pure CPU)               runtime (renderer)
----------------------------          ----------------------------
scene (node pool)                     scene_tree::scene (caller-owned, set_scene)
└── roots: indices into nodes          └── roots: scene_node { name, local,
    ├── node { name, local_matrix,          children, drawable_leaf }
    │     meshes, children: indices }       ├── ... (transform-only ok)
    │                                       └── leaf: model* (drawable)
    └── ...                                    push.model = accumulated world
```

### 3.1 Loader: keep the tree (DONE — `87ef7e8`)

`load_scene()` now walks `asset.scenes[i].nodeIndices` recursively instead of
`fastgltf::iterateSceneNodes`, and stores the result as a **node pool** (DFS
pre-order of every reachable node, roots first) with parent/child edges as
indices — the glTF hierarchy survives the load:

```cpp
struct node {
    std::string name = {};                   // debugging / future animation lookup
    glm::mat4 local_transform = mat4(1);     // relative to the parent (TRS-composed or raw matrix)
    std::vector<mesh> meshes = {};           // geometry attached at THIS node (copied per
                                             // referencing node for now; sharing is deferred)
    std::vector<std::size_t> children = {};  // indices into the owning scene.nodes
    glm::mat4 transform_matrix = mat4(1);    // world (parent * local) — kept for back-compat
};
struct scene {
    std::string name;
    std::vector<node> nodes;                 // node pool, DFS pre-order
    std::vector<std::size_t> root_indices;   // which pool entries are roots
};
```

Notes (implemented):

- Nodes keep a full **matrix** local transform (TRS-composed at load). Decomposing
  matrix -> TRS is deferred to the animation pass (§8); the tree work only needs
  the accumulated world = parent * local.
- The **flat view stays available** for existing consumers (bounds scan,
  single-pass `drawable_iterator`): `scenes::begin()` iterates the node pool
  (skipping mesh-less nodes) and still yields
  `drawable_ref { primitive*, world transform }` — same world matrices, same
  order as before. Verified: scene bounds / fps unchanged.
- Mesh-less (transform-only) nodes are kept in the pool so the hierarchy is
  complete; the flattening iterators skip them.

### 3.2 Runtime: scene tree storage (DONE — `ca6769a`, `7a445d5`)

New module `vulkan.scene_tree` (pure CPU: glm + std only, no Vulkan
types) owns the storage; the old private `models` map is gone:

```cpp
// vulkan/scene_tree/scene_tree.cppm
namespace vulkan::scene_tree {
    class primitive {                      // abstract leaf; GPU primitives implement it
        virtual ~primitive() = default;
        virtual void set_world(glm::mat4 const& world) = 0;  // push.model = world
    };
    struct scene_node {
        std::string name = {};             // glTF node name after import
        glm::mat4 local = mat4(1);         // programmatic transforms land here
        std::vector<scene_node> children;  // value semantics; move-only (clone() for copies)
        std::unique_ptr<primitive> primitive_leaf = {};  // null for transform-only nodes
    };
    struct scene { std::string name; std::vector<scene_node> roots; };
    void update_world(scene_node& n, glm::mat4 const& parent_world);   // DFS accumulate
    template <class F> void visit_primitives(scene_node const&, glm::mat4 const&, F&&); // DFS leaves
}
```

- `vulkan::primitive` (the GPU side of this same module) is
  `public vulkan::scene_tree::primitive`; `primitive::set_world` writes
  `push.model = world`, so the existing draw path (`primitive->draw()`) is
  untouched. (Both live in `vulkan.scene_tree` — the former separate
  `vulkan.model` module was merged into it; see step 4c in §6.)
- The tree is **caller-owned**: the runtime binds it with `set_scene(scene&)`
  and never owns/destroys it. Leaves release their GPU buffers automatically
  when the tree is destroyed (`vk_buffer`/`vk_image` RAII members), so the
  caller must destroy the tree BEFORE the runtime — `main` declares the scene
  after the runtime, so C++ reverse declaration order provides exactly that.
- `make_primitive` / `make_instanced_primitive` / `make_static_draw` attach a **root** leaf
  whose `name` records the requested pipeline (the leaf itself carries default semantics —
  empty `pipeline_name` — or an explicit name, and `draw()` binds through the recording
  `render_environment`; `instanced_draw_primitive` / `static_draw_primitive` get a tree slot
  like any primitive).
  `make_static_draw` uploads ONE owned merged vertex/index buffer plus a chunk table
  (`static_draw_chunk`: index range + own material), so a batch of static sub-meshes renders
  with one buffer bind + N offset draws — the primitive-level form of a static scene.
- `scene_node::attach(unique_ptr<primitive>)` (with `runtime::create_primitive`, which builds
  WITHOUT attaching) places a primitive under any node; `scene::add_root()` /
  `scene_node::add_child()` / `find_node(name)` round out programmatic scene building.
- **Scene offset / whole-scene transform**: `runtime::set_scene_transform(mat4)`
  applies one extra world matrix on top of every root before `update_world`
  (identity default → rendering identical to pre-tree). The `import_scene`
  `offset` is applied to each root node's local (step 2b).

### 3.3 World accumulation (DONE — `ca6769a`)

Every frame `render_frame()` runs the DFS before recording either pass:

```cpp
// scene_tree.cppm (as implemented)
void update_world(scene_node& n, glm::mat4 const& parent_world) {
    glm::mat4 const world = parent_world * n.local;
    if (n.primitive_leaf) n.primitive_leaf->set_world(world);  // primitive: push.model = world
    for (auto& c : n.children) update_world(c, world);
}
// render_frame(): scene_transform_ * root.local for each root, then per-leaf
//   set_world pushes the accumulated matrix into push.model via the vtable
```

- Writes only the leaf's `push.model` through `primitive::set_world`; the
  push-constant block, vertex layout and draw path are untouched.
- Cost is O(leaves) per frame with a tiny constant — negligible at current scene
  sizes; a dirty-flag skip can come later with animation.

## 4. Render loop changes

### 4.1 Flat draw list derived from the tree (DONE — `7a445d5`, `896d5b3`)

As of the render_environment refactor (`fed35fe`) the recording walk does NOT group
by pipeline anymore: each `draw(render_environment&)` binds the pipeline it needs
itself, through the per-worker environment (deduplicated, so consecutive leaves of
one pipeline still share a single bind). The environment also carries the session's
command buffer, so draw has no separate buffer parameter. The flat leaf list is
still derived from the tree **once per frame** — no persistent `render_lists_`
map, no explicit cache to invalidate:

```cpp
// begin_recording(), after update_world:
std::vector<primitive const*> frame_leaves;              // DFS collect (once)
for (scene_node const& root : scene_.roots) collect_leaf_primitives(root, frame_leaves);
//   main pass (record_main_segment): one render_environment per recording worker
//     (command buffer, available names, default pipeline, injected binder +
//     shared scene layout); leaves draw via m->draw(env) - default leaves request
//     env.bind_default(), custom leaves env.bind_pipeline(name)
//   shadow pass (record_shadow_content): its own environment always binds the shadow
//     pipeline (the binder ignores the requested name), then m->draw(env)
```

(The pre-refactor text below records the earlier design, kept for history.)

- `collect_leaf_primitives` / `get_primitives` are thin wrappers over the scene_tree
  module's own `visit_primitives()` DFS (`896d5b3`) — no hand-rolled traversal in
  the runtime.
- Alternative considered (original design): draw inline while walking the tree —
  rejected, it would re-bind pipelines per leaf (breaks batching). The environment's
  dedup gives the same single-bind-per-pipeline batching while letting each leaf
  choose its pipeline.
- Alternative considered (original design): group `frame_leaves` by `p->pipeline`
  and `begin_pipeline()` once per group in the recorder. Replaced by the
  environment-based binding: pipeline selection moved into `draw()`, so the record
  layer no longer needs to know every leaf's pipeline up front (multi-pipeline
  scenes, custom strategies) and each worker's bind state stays thread-local.

### 4.2 Public runtime API (current)

Current surface (post-`4ee1b82` / `fed35fe`; per-pipeline names kept — see §9 Q2):

```cpp
void set_scene(scene_tree::scene& scene);   // bind the caller-owned tree the runtime renders
std::expected<void, std::string> make_pipeline(std::string_view name, vs, fs); // named pipeline;
//   the FIRST created pipeline becomes the implicit runtime default (default-semantics leaves draw it)
void set_default_pipeline(std::string_view name);   // override the implicit default
primitive* make_primitive(std::string_view pipeline_name, primitive_create_info const& info); // build + attach root leaf
primitive* make_instanced_primitive(primitive const& source, std::span<glm::mat4 const> transforms);
primitive* make_static_draw(static_draw_create_info const& info); // one OWNED merged buffer + chunk table (static batch)
std::vector<primitive const*> get_primitives(std::string_view pipeline_name) const; // DFS by effective pipeline name
void clear_primitives(std::string_view pipeline_name);   // DFS: strips matching leaves anywhere in the tree
scene_import_result import_scene(NI nfirst, NI nlast, DI dfirst, DI dlast, glm::vec3 const& offset);
//   node stream (scene_node_iterator) + aligned drawable stream (scene_drawable_iterator);
//   rebuilds the real hierarchy (one scene_node per loader node), offset lands on each root
void set_scene_transform(glm::mat4 const& transform);  // extra world on top of every root
void scene_changed();              // caller edited node locals / structure via get_scene() -> culling BVH rebuilds
void enable_shadows(glm::vec3 const& scene_center, float scene_radius);
void log_scene_tree() const; // diagnostic: prints the runtime tree (names + [primitive] leaves)
```

`get_primitives` / `clear_primitives` match by the leaf's EFFECTIVE pipeline: an
explicit `pipeline_name` when the leaf carries one, otherwise the runtime default —
so default-semantics leaves (the normal/instanced/static draws, which leave
`pipeline_name` empty) are matched when the requested name equals the default.

(Step-2b is done — see §6. Programmatic scenes build through the scene-tree mounting
helpers — `scene::add_root()` / `scene_node::add_child()` / `scene_node::attach(unique_ptr<primitive>)`
(used with `runtime::create_primitive`, which builds without attaching) / `find_node(name)` —
instead of hand-rolling `scene_node` packing; structural edits must be followed by
`runtime::scene_changed()`. Legacy helpers (bounds scan / instancing grid in main.cpp)
keep working through `get_primitives` / `scenes::begin()`.)

### 4.3 Shadow pass

Unchanged structurally (and now caster-culled, see §4.4): the shadow pass iterates
the collected casters (a subset of `frame_leaves` — see the runtime's
`shadow_casters`, built from the frustum-visible leaves plus the leaves inside the
shadow frustum itself). Because the shadow pipeline shares the vertex layout / push block /
scene layout, leaves drawn into the shadow map still work via `draw()` — through
the shadow pass's own `render_environment`, whose injected binder always binds the
shadow pipeline (it ignores the requested pipeline name), so custom leaves that
would draw with a named pipeline in the main pass still cast their geometry here.
`alphaMode MASK` leaves are drawn too: `shadow.frag` runs the same alpha-cutoff
discard as `pbr.frag` off the material record, so a cut-out caster (foliage, a
curtain) throws a cut-out shadow rather than none or a solid one. `BLEND` leaves
are still skipped — a depth-only pass cannot blend a transparent shadow.
The pass also rasterizes **two-sided** (`render_environment::two_sided`): a caster
must never be dropped for facing away from the light, or a single-sided wall plane
that faces into the room casts no shadow at all and the sun pours through it.

### 4.4 Shadow caster culling (`f094c31`, revised)

The shadow pass used to re-draw every scene leaf each frame ("the whole scene casts
shadows"), so on large scenes (NodePerformanceTest: 10000 rocks) the camera view
direction barely moved the fps — only the main pass was culled. Small scenes
(<= 1500 leaves) still render **every** leaf: caster culling is only an
approximation (a caster arbitrarily far up-light still throws its parallel shadow
column into the view, so a finite camera-frustum margin visibly leaked the sun
through Sponza's walls) and at that size the full depth render is cheap. Heavier
scenes draw `cull_visible` unioned with a BVH `frustum_cull` against the
**shadow frustum** (`light_state.light_view_proj`), which `update_shadow_frustum`
refits to the camera each frame and widens by every leaf whose shadow column can
reach the view — so off-screen casters such as the wall behind the camera are
included, while the set stays far smaller than the whole scene. Rebuilt only when
the camera moved or the scene changed (same reuse rule as the main-pass cull);
`record_shadow_content` walks that set.

## 5. Loader <-> runtime bridge (the key design decision)

The runtime must not depend on `gltf_loader` types (existing invariant: loader is
pure CPU, runtime converts values via structural concepts). Options:

- **A. Structural tree iterator (recommended, consistent with current design).**
  Loader exposes a node-level DFS iterator; runtime's `import_scene` becomes a
  template over a new structural concept `scene_node_source` with member shapes
  like `get_local_matrix()`, `children_count()`, child access, and per-node
  drawable iteration (reusing the existing `scene_drawable_iterator` getters for
  geometry/material). Runtime rebuilds the tree while converting values.
- **B. Callback-driven import.** Loader walks itself and invokes runtime callbacks
  (`on_node_enter(local)`, `on_leaf(vertex, index, material...)`). Simpler
  iterator-free recursion, but inverts the current "runtime drives traversal"
  philosophy and scatters glTF traversal knowledge into the callback wiring.
- **C. Export a plain scene-graph data blob** (nodes with local matrices +
  primitive indices, no iteration) and let runtime/loader-shared headers walk it.
  Least structural coupling but duplicates traversal logic.

Recommendation: **A**, because it extends the existing decoupled
`scene_drawable_iterator` pattern (one structural concept, runtime converts,
loader stays Vulkan-free). The existing flat `drawable_iterator` remains for
single-pass consumers (bounds scan) and as the per-node leaf iterator inside the
tree concept.

## 6. Migration plan (stepwise, each step builds + renders + commits)

Status, kept in sync with git history:

- ✅ **1 — Loader keeps the tree** (`87ef7e8`). `gltf::node` gains `name`,
  `local_transform`, `children` (indices into a node pool); `gltf::scene` gains
  `root_indices`. `load_scene` builds real roots/children; `scenes::begin()` keeps
  DFS-flatten semantics (same world matrices, same order) so nothing downstream
  breaks. Verified: scene bounds / fps unchanged.
- ✅ **2a — Runtime scene tree storage** (`ca6769a`, `7a445d5`, `896d5b3`). New
  `vulkan.scene_tree` module (scene/scene_node/primitive + update_world +
  visit_primitives); `vulkan::primitive` (renamed from `vulkan::model`) implements
  `scene_tree::primitive`; the `models` map is replaced by a `scene_tree::scene`;
  `make_primitive` / `make_instanced_primitive` attach leaves; `render_frame`
  accumulates world transforms (`update_world` → `primitive::set_world` →
  `push.model`) and walks the tree for both the main pass (leaves grouped by
  `primitive->pipeline`) and the shadow pass; destructor / `clear_primitives` /
  `get_primitives` traverse the tree; runtime traversal reuses the module's
  `visit_primitives` (`896d5b3`). Verified: default scene + spin demo +
  instancing grid all render, fps unchanged.
- ✅ **2b — import_scene builds the real hierarchy** (`cba9b3d`, `dde3498`).
  Loader exposes `gltf::scene_node_iterator` (DFS pre-order over the
  retained tree, transform-only nodes included: name / local_transform / depth /
  drawable_count); `import_scene` becomes a template over that structural node
  stream PLUS the aligned drawable stream, and rebuilds the scene tree 1:1 with
  the loader tree (a gltf node -> a named `scene_node` with its local transform;
  a node's drawables become leaf primitives attached at that node — extra
  primitives of one node become identity-local child leaves). Scene `offset` moves
  onto each root node's local. `make_primitive` splits into `create_primitive`
  (build) + attach-as-root-leaf. Verified: default + hierarchy-style assets show the
  runtime tree mirroring the loader tree (`node_group_root -> 2 helmet leaves`),
  fps unchanged, instancing grid + spin demo still render.
- ✅ **3 — Whole-scene + per-node transform API** (`4ee1b82`, `74b18bc`, `9cc791a`).
  `runtime::set_scene_transform` applies one world matrix on top of every root
  (identity default = unchanged rendering); the then-current `argv == "spin"`
  demo spun the whole scene around its sink (demo modes were removed later).
  `runtime::scene()` exposes the tree so callers edit per-node `local` in place
  (structure is fixed after import); the `"spin-subtree"` demo rotated one
  primitive-leaf node about its own position — on a hierarchy-style asset a single
  helmet spun while its sibling stayed put (per-node transform over the 2b
  hierarchy).
- ✅ **4 — Remove the flat `models` map.** No flat storage remains.
- ✅ **4b — Rename model → primitive** (`386c772`). The GPU classes `vulkan::model`
  / `normal_draw_model` / `instanced_draw_model` are now `vulkan::primitive` /
  `normal_draw_primitive` / `instanced_draw_primitive`, the scene-tree leaf
  interface is `scene_tree::primitive` (was `drawable`), and the runtime API is
  `make_primitive` / `make_instanced_primitive` / `get_primitives` /
  `clear_primitives` / `create_primitive`, with `primitive_create_info`. The
  shader-facing `push.model` / `model_matrix` terms (model matrix) are kept.
  Verified: full regression matrix unchanged.
- ✅ **4c — Merge vulkan.model into scene_tree** (next commit). The separate
  `vulkan.model` module (and its files) is gone: its exports now live in
  `vulkan.scene_tree` (scene storage + abstract leaf primitive + the GPU
  primitives + material/UBO records + structural iterator concepts in one module,
  which imports vulkan.core for the GPU types). `runtime` / `main` import
  `vulkan.scene_tree` instead of `vulkan.model`; CMake module list
  updated. Verified: Release builds (no ICE), full regression matrix unchanged.
- ✅ **5 — Morph targets landed on the tree substrate** (export `5bea625`, GPU blend
  `f530889`, weights `60c7c1f`; see §8): per-primitive morph deltas + default weights bake
  into the scene morph buffer (binding 10) and animated weights are rewritten per frame.
  Remaining future work: glTF mesh sharing / GPU dedup and per-instance material overrides (§8).
- ✅ **6 — Scene tree ownership moves to the caller** (vma RAII first — `5abc5db`; then
  `set_scene`). The runtime's private `scene_tree::scene` member becomes a non-owning
  `bound_scene` pointer set by `runtime::set_scene(scene&)`: `main` declares the scene
  (AFTER the runtime, so C++ reverse declaration order destroys it BEFORE the runtime)
  and binds it before import. The runtime renders the bound tree every frame
  (`begin_recording` collects leaves from it) but never owns or destroys it; the old
  `~runtime` leaf-teardown walk and `destroy_leaf_primitives` are gone — leaves release
  their GPU buffers automatically when the caller destroys the tree (`vk_buffer` /
  `vk_image` RAII members, allocator still alive because the runtime outlives the tree).
  The animation backend now also receives the caller's `scene*` directly (the opaque
  tree-callback decoupling was reverted — with the tree externalized, depending on the
  pure-CPU scene types is the natural contract). Verified: RecursiveSkeletons render +
  ESC exit cleanly, fps unchanged.
  Remaining future work: glTF mesh sharing / GPU dedup and per-instance material
  overrides (§8).

(Detailed step list below is folded into the status above; this file is the single
source of truth for what each commit changed.)

## 7. Risks / trade-offs (as realized)

- **glTF mesh sharing stays unmodeled**: two nodes referencing the same glTF mesh
  still produce two independent leaf primitives (no regression — same as before —
  but the tree now makes the sharing opportunity visible; addressed later, §8).
- **Matrix vs TRS**: the loader stores the composed local *matrix* (TRS or raw
  node matrix). Re-decomposing to TRS for animation is deferred (§8); glTF forbids
  shear in matrix nodes, so the later decomposition is lossless in practice.
- **Loader children are indices, runtime children are values**: loader keeps a
  node pool with `children` as indices (stable, reorderable without copying);
  runtime `scene_node` owns children inline as values and is move-only
  (`clone()` for deep copies). Runtime edits go through `scene_node&` handles
  whose stability must be documented (vector may reallocate — use index access or
  `clone` before structural edits).
- **Render list rebuild cost**: `render_frame` recollects `frame_leaves` every
  frame (O(nodes)); trivial for current scenes. A structure-dirty skip can come
  later with animation if it ever matters.
- **main.cpp churn**: bounds scan / grid stress re-pointed at `get_primitives` /
  `scenes::begin()`; no compatibility helper was needed.

## 8. Deferred (design hooks left open)

- glTF mesh sharing / GPU dedup: loader keeps `mesh` copies per node for now; a
  future `scenes.meshes[]` pool + node->mesh index enables dedup without changing
  the runtime tree shape (leaves then reference shared geometry).
- ✅ **Animation (keyframe TRS)** — landed on the tree substrate: the loader exports the
  file's keyframe animations and per-node TRS base pose (`gltf::animation` /
  `gltf::node`; matrix nodes are not animatable per the spec, so no matrix→TRS
  decomposition was needed) and samples them pure-CPU (`gltf::sample_channel` /
  `sample_node`, LINEAR / STEP / CUBICSPLINE, slerped rotations). `main.cpp` plays the
  first channel-bearing animation on a loop: `scene_node::source_index` (recorded by
  `import_scene`) maps channel targets onto the live tree, and each frame the evaluated
  T·R·S locals are written back through `runtime::scene_changed()`. The `gui` overlay
  adds play/pause, a time scrubber and an animation dropdown. See
  `docs/gltf_loader_usage.md` §8.
- ✅ **Skinning** — landed on the same substrate: the loader exports skins (joint asset-node
  indices + inverse bind matrices) and JOINTS_0 / WEIGHTS_0 live in the shared 64-byte vertex
  layout; `main.cpp` resolves skin joints onto the live tree, rebuilds the per-frame skin
  matrices (`inv(W_mesh) · W_joint · IBM`, joints following the keyframe animation above) into
  a scene skin buffer (binding 9, identity block for unskinned draws) and points each skinned
  primitive at its block via `material_push_constants::skin_base`. Verified with
  `RiggedSimple` / `BrainStem`. See `docs/gltf_loader_usage.md` §9.
- ✅ **Morph targets** — landed on the same shared vertex layout: the loader exports morph
  targets (POSITION/NORMAL deltas), mesh/node default weights and `weights` animation channels
  (sampler `per_key`); the demo bakes per-primitive deltas + default weights into the scene
  morph buffer (binding 10) and rewrites the active weights each frame from the animation.
  `pbr.vert`/`shadow.vert` blend morph deltas before skinning. Verified with
  `AnimatedMorphCube` / `SimpleMorph` / `MorphStressTest`. See `docs/gltf_loader_usage.md` §10.
- Mesh sharing / GPU dedup and per-instance material overrides remain open: two nodes
  referencing the same glTF mesh still get independent leaf primitives, and material
  identity lives in the leaf primitive's material_index.
- Authored glTF cameras are consumed as orbit-camera viewpoint seeds (main + the gui
  "camera" selector); punctual point/spot lights (KHR_lights_punctual) are loaded by
  main into the runtime's editable gui light slots (up to the 4-light cap, base-pose world
  transform), adjustable in the overlay. KHR directional lights are not mapped - the sun stays the analytic
  shadow-casting light from `enable_shadows()`.

## 9. Open questions for the maintainer (answers)

1. **Storage module** — answered: `vulkan.scene_tree` is the single
   scene-tree module. The former separate `vulkan.model` module was merged into
   it (step 4c), so scene storage, the abstract leaf `primitive`, the GPU
   primitives and the material/UBO records all live in one module — no cycle, no
   module-name juggling left.
2. **API names** — renamed: the GPU classes and runtime API now use `primitive`
   (`make_primitive` / `make_instanced_primitive` / `get_primitives` /
   `clear_primitives`, `primitive_create_info`), matching the scene-tree leaf
   concept; the shader-facing `push.model` / `model_matrix` (model matrix)
   terminology is kept.
3. **Whole-group transform demo** — answered: temporary auto-spin was accepted and
   shipped (`argv == "spin"` whole-scene rotation around the sink, `spin-subtree`
   per-node rotation — `9cc791a`); both demo modes were removed later, but the
   `set_scene_transform` / per-node-`local` APIs they exercised remain.
