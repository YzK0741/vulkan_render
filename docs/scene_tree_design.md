# Scene Graph Storage — Design Document

Status: migration in progress — storage + render loop are tree-driven
(`vulkan.runtime.scene_tree`, commits below); the loader keeps the glTF
hierarchy; remaining work: import_scene builds the real hierarchy (§5 option A /
step 2b) and `vulkan.model` becomes a scene_tree primitive.

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
scene (node pool)                     scene_tree::scene (owned by runtime)
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

New module `vulkan.runtime.scene_tree` (pure CPU: glm + std only, no Vulkan
types) owns the storage; the old private `models` map is gone:

```cpp
// vulkan/runtime/scene_tree/scene_tree.cppm
namespace vulkan::scene_tree {
    class drawable {                       // abstract leaf; the future "primitive"
        virtual ~drawable() = default;
        virtual void set_world(glm::mat4 const& world) = 0;  // push.model = world
    };
    struct scene_node {
        std::string name = {};             // pipeline name today; glTF node name later
        glm::mat4 local = mat4(1);         // programmatic transforms land here
        std::vector<scene_node> children;  // value semantics; move-only (clone() for copies)
        std::unique_ptr<drawable> drawable_leaf = {};  // null for transform-only nodes
    };
    struct scene { std::string name; std::vector<scene_node> roots; };
    void update_world(scene_node& n, glm::mat4 const& parent_world);   // DFS accumulate
    template <class F> void visit_drawables(scene_node const&, glm::mat4 const&, F&&); // DFS leaves
}
```

- `vulkan::model` (model.cppm) now `public vulkan::scene_tree::drawable`;
  `model::set_world` writes `push.model = world`, so the existing draw path
  (`model->draw()`) is untouched.
- Runtime owns models inside the tree: `scene_` is a `scene_tree::scene`;
  `~runtime` walks the tree and calls `model->destroy(vma)` per leaf before the
  VkDevice goes away.
- `make_model` / `make_instanced_model` attach a **root** leaf whose `name`
  records the pipeline (models record their pipeline in `model->pipeline`;
  `instanced_draw_model` gets a tree slot like any model).
- **Scene offset / whole-scene transform**: `runtime::set_scene_transform(mat4)`
  applies one extra world matrix on top of every root before `update_world`
  (identity default → rendering identical to pre-tree). The `import_scene`
  `offset` argument is still baked into each leaf's `local` today for back-compat;
  step 2b will move it onto the root (see §4.2).

### 3.3 World accumulation (DONE — `ca6769a`)

Every frame `render_frame()` runs the DFS before recording either pass:

```cpp
// scene_tree.cppm (as implemented)
void update_world(scene_node& n, glm::mat4 const& parent_world) {
    glm::mat4 const world = parent_world * n.local;
    if (n.drawable_leaf) n.drawable_leaf->set_world(world);  // model: push.model = world
    for (auto& c : n.children) update_world(c, world);
}
// render_frame(): scene_transform_ * root.local for each root, then per-leaf
//   set_world pushes the accumulated matrix into push.model via the vtable
```

- Writes only the leaf's `push.model` through `drawable::set_world`; the
  push-constant block, vertex layout and draw path are untouched.
- Cost is O(leaves) per frame with a tiny constant — negligible at current scene
  sizes; a dirty-flag skip can come later with animation.

## 4. Render loop changes

### 4.1 Flat draw list derived from the tree (DONE — `7a445d5`, `896d5b3`)

To keep "bind pipeline once, draw batch" (main and shadow passes) exactly as
before, `render_frame()` derives the flat draw set from the tree **once per
frame** — no persistent `render_lists_` map, no explicit cache to invalidate:

```cpp
// render_frame(), after update_world:
std::vector<model const*> frame_leaves;              // DFS collect (once)
for (scene_node const& root : scene_.roots) collect_leaf_models(root, frame_leaves);
//   main pass: group frame_leaves by m->pipeline -> begin_pipeline() once, m->draw() per leaf
//   shadow pass: bind shadow pipeline, m->draw() over the same frame_leaves
```

- `collect_leaf_models` / `get_models` are thin wrappers over the scene_tree
  module's own `visit_drawables()` DFS (`896d5b3`) — no hand-rolled traversal in
  the runtime.
- Alternative considered: draw inline while walking the tree — rejected, it would
  re-bind pipelines per leaf (breaks batching).

### 4.2 Public runtime API (current, and the step-2b delta)

Current surface (post-`4ee1b82`; per-pipeline names kept — see §9 Q2):

```cpp
model* make_model(std::string_view pipeline_name, model_create_info const& info); // root leaf
model* make_instanced_model(model const& source, std::span<glm::mat4 const> transforms);
std::vector<model const*> get_models(std::string_view pipeline_name) const; // DFS by pipeline
void clear_models(std::string_view pipeline_name);
scene_import_result import_scene(I first, S last, glm::vec3 const& offset);   // per-drawable today
void set_scene_transform(glm::mat4 const& transform);  // extra world on top of every root
void enable_shadows(glm::vec3 const& scene_center, float scene_radius);
```

Step-2b delta (builds on §5 option A, keeps `import_scene`'s signature):

- `import_scene` walks loader *nodes* (name + `local_transform` + children +
  per-node drawables) and rebuilds the hierarchy: one `scene_node` per glTF node
  (root = loader scene root), leaf models attached at their node, scene `offset`
  moved to the scene root's `local` (or folded into `scene_transform_`).
- Add `scene_node&`-based handles once real subtrees exist:
  `set_local_transform(scene_node&, glm::mat4 const&)`, subtree demo in main.
- Legacy helpers (bounds scan / instancing grid in main.cpp) keep working through
  `get_models` / `scenes::begin()` — no `leaf_models()` helper was needed.

### 4.3 Shadow pass

Unchanged structurally: the shadow pass iterates the same `frame_leaves` collected
from the tree (step 2a). Because the shadow pipeline shares the vertex layout /
push block / scene layout, leaves drawn into the shadow map still work via
`model->draw()`.

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
  `vulkan.runtime.scene_tree` module (scene/scene_node/drawable + update_world +
  visit_drawables); `vulkan::model` implements `scene_tree::drawable`; the `models`
  map is replaced by a `scene_tree::scene`; `make_model` / `make_instanced_model`
  attach leaves; `render_frame` accumulates world transforms (`update_world` →
  `drawable::set_world` → `push.model`) and walks the tree for both the main pass
  (leaves grouped by `model->pipeline`) and the shadow pass; destructor /
  `clear_models` / `get_models` traverse the tree; runtime traversal reuses the
  module's `visit_drawables` (`896d5b3`). Verified: default scene + spin demo +
  instancing grid all render, fps unchanged.
- ⏳ **2b — import_scene builds the real hierarchy** (design §5 option A). Today
  every drawable imports as a separate root leaf with its full world matrix as
  `local`, so the loader tree's transform-only nodes / shared parents are not
  reflected and rotating an arbitrary subtree is not possible. Next: node-level
  structural iteration from the loader + recursive tree building in `import_scene`
  (scene offset becomes the root transform). Verify visually + instancing grid
  still identical.
- ✅ **3 — Whole-scene transform API** (`4ee1b82`, `74b18bc`).
  `runtime::set_scene_transform` applies one world matrix on top of every root
  (identity default = unchanged rendering); main's `argv[3] == "spin"` demo spins
  the whole scene around its sink. A per-node `set_local_transform` for arbitrary
  subtrees waits for 2b.
- ✅ **4 — Remove the flat `models` map.** No flat model storage remains.
- ⏳ **5 — Future:** animation / skinning / mesh sharing (separate pass, §8).

(Detailed step list below is folded into the status above; this file is the single
source of truth for what each commit changed.)

## 7. Risks / trade-offs (as realized)

- **glTF mesh sharing stays unmodeled**: two nodes referencing the same glTF mesh
  still produce two independent leaf models (no regression — same as before — but
  the tree now makes the sharing opportunity visible; addressed later, §8).
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
- **main.cpp churn**: bounds scan / grid stress re-pointed at `get_models` /
  `scenes::begin()`; no compatibility helper was needed.

## 8. Deferred (design hooks left open)

- glTF mesh sharing / GPU dedup: loader keeps `mesh` copies per node for now; a
  future `scenes.meshes[]` pool + node->mesh index enables dedup without changing
  the runtime tree shape (leaves then reference shared geometry).
- Animation: needs matrix -> TRS decomposition on loader nodes (name + local
  transform are the substrate); samplers (fastgltf `animations`) + per-frame
  `set_local_transform` land on top.
- Skinning: needs joint node naming/indices + per-vertex joint data + bone UBO —
  independent of the tree storage change, tree only provides the skeleton
  hierarchy.
- Per-instance material overrides: later; today material identity lives in the
  leaf model's material_index.

## 9. Open questions for the maintainer (answers)

1. **Storage module** — answered: a dedicated `vulkan.runtime.scene_tree` module
   (pure CPU, no Vulkan types) was created; `vulkan.model` imports and implements
   its `drawable`, so no cycle and no new ICE-prone imports.
2. **API names** — kept for now: `make_model` / `get_models` / `clear_models`
   still name the tree-leaf operations (per-pipeline semantics unchanged);
   scene-oriented renames can come with 2b's handle-based API if wanted.
3. **Whole-group transform demo** — answered: temporary auto-spin accepted and
   shipped behind `argv[3] == "spin"` (whole-scene rotation around the sink,
   `4ee1b82`); a per-subtree demo waits for 2b.
