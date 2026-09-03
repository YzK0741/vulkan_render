# Scene Graph Storage — Design Document

Status: proposal (no code changed yet)

## 1. Motivation

The renderer currently stores the scene as *two flattened levels* and the original
glTF hierarchy is lost between them:

1. **Loader side** (`gltf_loader.cpp` `load_scene`): the glTF node tree is expanded
   by `fastgltf::iterateSceneNodes` into a flat `gltf::scene { nodes: [...] }`
   where every `node` carries only `meshes` + a **world** `transform_matrix`.
   Parent/child edges, per-node local transforms and mesh-sharing semantics are
   gone the moment the file is loaded.

2. **Runtime side** (`vulkan/runtime.cppm`): models live in
   `std::map<std::string, std::vector<std::unique_ptr<model>>>` — a flat list per
   pipeline. Each `model` is a self-contained leaf (geometry + material + baked
   world matrix in `push.model`). There is no notion of a parent transform, so you
   cannot rotate/translate a *group*, re-parent parts programmatically, or express
   TRS animation later.

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

## 2. Current shape (reference)

```cpp
// gltf_loader: flat, world-space
struct node { std::vector<mesh> meshes; glm::mat4 transform_matrix; };
struct scene { std::string name; std::vector<node> nodes; };

// runtime: flat per-pipeline
std::map<std::string, std::vector<std::unique_ptr<model>>> models;
//   model: geometry buffers + pipeline ptr + material_push_constants push (baked world)
//   normal_draw_model / instanced_draw_model (polymorphic draw())
```

Consumers of the flat lists:

- `render_frame()`: per pipeline -> `begin_pipeline()` -> `model->draw()` (main pass
  and shadow pass both iterate the same model lists; see runtime.cpp:729, 844).
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

The two current "flat" structures are replaced by:

```
gltf (loader, pure CPU)               runtime (renderer)
----------------------------          ----------------------------
scene                                scene graph (owned by runtime)
└── roots: node                       └── scene_node { local, children }
    ├── node { local TRS, meshes }        ├── ... (transform-only ok)
    │                                     └── leaf: model* (drawable)
    └── ...                                   push.model = accumulated world
```

### 3.1 Loader: keep the tree

New `gltf::node` shape (replaces the flat world-space node):

```cpp
struct node {
    std::string name;                 // for debugging / future animation lookup
    glm::vec3 translation = {0,0,0};  // local TRS
    glm::quat rotation = glm::identity<glm::quat>();
    glm::vec3 scale    = {1,1,1};
    std::vector<mesh> meshes;         // geometry attached at THIS node (copied per
                                      // referencing node for now; sharing is deferred)
    std::vector<node> children;       // value semantics: simple ownership, no cycles
};
struct scene { std::string name; std::vector<node> nodes; }; // nodes = roots
```

Notes:

- fastgltf nodes can carry either TRS or a matrix. Decompose to TRS at load time
  (matrix -> TRS) so animation can later target `rotation/translation/scale`
  directly; keep a helper `local_matrix() = T * R * S`.
- The **flat view stays available** for existing consumers (bounds scan,
  single-pass `drawable_iterator`): `scenes::begin()` is re-implemented as a
  DFS flattening of the tree (document order = glTF order), still yielding
  `drawable_ref { primitive*, world transform }`. `scene_iterator`'s cursor
  becomes a small DFS stack instead of flat indices.
- `load_scene()` walks `asset.scenes[i].nodeIndices` recursively instead of
  `iterateSceneNodes`, preserving edges + local transforms.

### 3.2 Runtime: scene tree storage

```cpp
// runtime.cppm (private storage, replacing `models`)
struct scene_node {
    glm::mat4 local = glm::mat4(1.0f);   // programmatic transforms land here
    std::vector<scene_node> children;
    std::unique_ptr<model> drawable;     // leaf payload; null for transform-only nodes
};
std::vector<scene_node> scene_roots_;    // one entry per imported scene / call site
```

- Runtime owns models inside the tree (same lifetime rules as today: destroyed in
  `~runtime` before the VkDevice goes away, `model->destroy(vma)` per leaf).
- `model` keeps its current shape (geometry + pipeline ptr + push) — **no geometry
  sharing in this pass**, so leaves stay self-contained.
- `instanced_draw_model` is orthogonal to the tree: an instancing stress helper
  referencing a `source` model. It does not need a scene-tree slot (see §5.3).
- **Root offset**: the `offset` argument of today's `import_scene` becomes an
  extra transform on the scene root (or a runtime-level scene transform applied
  when accumulating worlds), instead of being multiplied into every leaf.

### 3.3 World accumulation

Every frame (or on demand when the tree is marked dirty):

```cpp
// DFS: node_world = parent_world * node.local
void update_world(scene_node& n, glm::mat4 const& parent_world) {
    glm::mat4 const world = parent_world * n.local;
    if (n.drawable) n.drawable->push.model = world; // existing field, reused
    for (auto& c : n.children) update_world(c, world);
}
```

- Writes only the leaf's `push.model`; the push-constant block, vertex layout and
  draw path are untouched.
- Cost is O(leaves) per frame with a tiny constant — negligible at current scene
  sizes; a dirty-flag skip can come later with animation.

## 4. Render loop changes

### 4.1 Flat draw list derived from the tree

To keep "bind pipeline once, draw batch" (main and shadow passes) exactly as it is
today, the runtime maintains a derived flat structure that mirrors the current
`models` map, rebuilt when the tree changes (import / clear / programmatic edit):

```cpp
// derived (kept in sync with the tree, or rebuilt lazily per frame)
std::map<std::string, std::vector<model const*>, std::less<>> render_lists_;
//   pipeline name -> leaves in tree order
```

- `render_frame()` loops over `render_lists_` exactly like today (begin_pipeline,
  then `model->draw()` per leaf) — the polymorphic draw and the shadow pass
  (which iterates the same lists) keep working unchanged.
- `update_world()` runs once per frame before recording; it walks the tree and
  can collect leaves into per-pipeline lists in the same pass (cheap, no extra
  map lookups).
- Alternative if we want zero derived state: DFS inline and draw as we go, but
  that would re-bind pipelines per leaf — rejected (keeps current batching).

### 4.2 Public runtime API

```cpp
// — scene tree management (replaces per-pipeline make_model/import_scene) —
scene_handle add_scene(gltf-view or structural tree source);   // tbd, see §5.1
scene_node* scene_root(scene_handle);
void        set_local_transform(scene_node&, glm::mat4 const& local);
void        clear_scenes();

// convenience: import a glTF through the existing structural iterator
// (now building a tree of transform nodes + leaf models instead of flat list)
scene_handle import_scene(I first, S last, glm::vec3 const& offset); // repurposed

// model access for legacy helpers (bounds scan / instancing) is re-pointed at
// tree leaves; exact shape TBD in implementation.
```

### 4.3 Shadow pass

Unchanged structurally: shadow pass iterates `render_lists_` too. Because the
shadow pipeline shares the vertex layout / push block / scene layout, leaves drawn
into the shadow map still work via `model->draw()`.

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

1. **Loader keeps the tree.** Add `children` + local TRS to `gltf::node`;
   `load_scene` builds real roots/children; keep `scenes::begin()` semantics
   (DFS flatten, same world matrices, same iteration order) so nothing downstream
   breaks. Verify: same frame as before (bounds, IBL, render).
2. **Runtime scene tree storage.** Introduce `scene_node` + roots; re-point
   `import_scene` to build the tree (transform-only nodes included), keep the
   offset as root transform; build `render_lists_` derived map; `render_frame`
   loops the derived lists. All `models` map reads move to tree walks.
   API: keep `get_models("pbr")`-style access working via the derived list until
   callers migrate. Verify visually + instancing grid still identical.
3. **Whole-group transform API.** Add `set_local_transform(root_of_group, ...)`
   and a demo in main (e.g. slowly spin the imported scene's root / a subtree) to
   prove group transforms. Verify visually + fps.
4. **Remove the flat `models` map** entirely once all callers (main bounds? no —
   loader-side; grid stress) go through the tree/derived lists.
5. (future, separate pass) animation samplers, skinning, mesh sharing.

## 7. Risks / trade-offs

- **glTF mesh sharing stays unmodeled**: two nodes referencing the same glTF mesh
  still produce two independent leaf models (same as today — no regression, but
  the tree makes the sharing opportunity visible; addressed later).
- **Matrix vs TRS**: nodes authored as matrices decompose losslessly only if no
  shear is present; glTF forbids shear in matrix nodes in practice. Acceptable.
- **Value-semantics children** (`std::vector<node>`): deep copies are fine at load
  time; runtime edits go through `scene_node&` handles whose stability must be
  documented (like today's "vector may reallocate, use index access" caveat).
- **Render list rebuild cost**: trivial for current scenes; keep the derived list
  incremental (rebuild on structure change, not per frame) to avoid regressions.
- **main.cpp churn**: grid stress + bounds scan touch points change; keep a thin
  compatibility helper (`runtime::leaf_models()`) until step 4.

## 8. Deferred (design hooks left open)

- glTF mesh sharing / GPU dedup: loader keeps `mesh` copies per node for now; a
  future `scenes.meshes[]` pool + node->mesh index enables dedup without changing
  the runtime tree shape (leaves then reference shared geometry).
- Animation: local TRS + `name` on nodes is the required substrate; samplers
  (fastgltf `animations`) + per-frame `set_local_transform` land on top.
- Skinning: needs joint node naming/indices + per-vertex joint data + bone UBO —
  independent of the tree storage change, tree only provides the skeleton
  hierarchy.
- Per-instance material overrides: later; today material identity lives in the
  leaf model's material_index.

## 9. Open questions for the maintainer

1. Storage module: tree lives in `vulkan.runtime` (like today's `models`) vs a new
   `vulkan.model`-level type? (Lean: runtime, to avoid touching model.cppm's
   stable public surface and module count.)
2. API: keep `make_model`/`get_models` names (now over tree leaves) or rename to
   scene-oriented names?
3. Whole-group transform demo: acceptable to add a temporary auto-spin in main
   (or a keyboard toggle) to verify, or keep main static and test via a unit/API
   check only?
