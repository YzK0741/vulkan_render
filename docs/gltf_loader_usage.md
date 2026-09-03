# gltf_loader Usage Guide

> CPU-side module for loading glTF / GLB files, with no Vulkan dependency.
> Backend: [fastgltf](https://github.com/spnda/fastgltf) (installed as an MSYS2 clang64 package on this machine — `libfastgltf.dll` + `libsimdjson.dll`), textures decoded with stb_image.

- Language standard: C++23 (C++20 module)
- Public interface: `gltf_loader/gltf_loader.cppm` (`import gltf_loader;`)
- Entry point: `gltf::load_model(path)` → `std::expected<gltf::scenes, gltf::error_code>`

---

## 1. API Overview

```cpp
namespace gltf {
    enum class component_type : int {          // glTF OpenGL constants
        byte_t = 5120, unsigned_byte_t = 5121, short_t = 5122, unsigned_short_t = 5123,
        int_t = 5124, unsigned_int_t = 5125, float_t = 5126, double_t = 5130, unknown = 0,
    };
    enum class element_type { scale, vec2, vec3, vec4, mat2, mat3, mat4, unknown };

    constexpr component_type to_component_type(int gltf_constant);
    constexpr uint8_t        get_component_size(component_type);
    constexpr int            to_gltf_macro_type(component_type);
    constexpr element_type   to_element_type(int gltf_constant);
    constexpr uint8_t        get_element_size(element_type);

    enum class error_code { file_not_found, file_type_error, file_load_failed };

    struct texture_data  { std::vector<unsigned char> data; uint32_t width, height; uint8_t component; };
    struct vertex_portion{ std::vector<unsigned char> data; component_type component; };
    struct material_factors { glm::vec4 base_color_factor; glm::vec3 emissive_factor;
                              float metallic_factor; float roughness_factor; float normal_scale; };
    struct material      { material_factors factors; std::map<std::string, uint16_t> texture_indices; };
    struct primitive     { std::map<std::string, vertex_portion> vertex;
                           std::vector<unsigned char> index;
                           component_type index_component_type;
                           uint32_t material_index; };   // into scenes::materials, UINT32_MAX = none
    struct mesh          { std::vector<primitive> primitives; };
    struct node          { std::string name; glm::mat4 local_transform;
                           std::vector<mesh> meshes; std::vector<std::size_t> children;
                           glm::mat4 transform_matrix; };   // see semantics below
    struct scene         { std::string name; std::vector<node> nodes;
                           std::vector<std::size_t> root_indices; };
    struct scenes        { std::vector<texture_data> textures; std::vector<material> materials;
                           std::vector<scene> scene; };

    std::expected<scenes, error_code> load_model(std::string_view file_name);
}
```

**Semantics**

- The node hierarchy is **kept**: `scene.nodes` holds every node reachable from the scene
  roots in **DFS pre-order** (roots first, then each subtree), and `root_indices` lists the
  indices of the scene's roots inside `nodes`. Each node records its direct children as
  indices into the same `nodes` list (`node.children`, always greater than the node's own
  index), so the full parent→child structure is recoverable. Transform-only (intermediate)
  nodes appear too, with empty `meshes`.
- `local_transform` is the node's **own** transform relative to its parent (TRS-composed
  matrix, or the raw matrix when the asset stored one). `transform_matrix` keeps the
  accumulated **world-space** matrix (`parent_world * local_transform`) — same value the
  loader exposed before the hierarchy was retained, so existing world-space consumers keep
  working unchanged.
- Keys of `primitive.vertex` are glTF attribute names (`POSITION` / `NORMAL` / `TEXCOORD_0` / `COLOR_0` …).
- Vertex and index data are **de-interleaved raw bytes**; their type is described by `component` / `index_component_type` (byteStride and sparse accessors are already handled by fastgltf).
- `scenes.materials` holds one entry per glTF material (in material order); each `material.texture_indices` uses the fixed roles `albedo` / `metallic_roughness` / `normal` / `occlusion` / `emissive`, and the values index into `scenes::textures` (in glTF texture order). `primitive.material_index` selects the primitive's material (`UINT32_MAX` when the glTF primitive has none).

---

## 2. Quick Start: Loading and Error Handling

```cpp
import gltf_loader;
import std;

int main() {
    auto result = gltf::load_model("assets/box.glb");   // auto-detects GLB / JSON
    if (!result) {
        switch (result.error()) {
            case gltf::error_code::file_not_found:  std::println("file not found"); break;
            case gltf::error_code::file_type_error:  std::println("not a valid glTF/GLB"); break;
            case gltf::error_code::file_load_failed: std::println("parse failed (details on stderr)"); break;
        }
        return 1;
    }
    const auto& model = *result;   // gltf::scenes
    std::println("scenes={} textures={}", model.scene.size(), model.textures.size());
}
```

> Three input forms are supported: ASCII `.gltf` (with external `.bin` / images), binary `.glb`, and embedded files using data URIs. Loading requires
> `LoadExternalBuffers | LoadExternalImages` (already configured inside the module).

---

## 3. Traversing the Scene Hierarchy

Two ways to visit the scene:

1. **Flat (drawable-oriented)** — iterate `scene.nodes` in order; every node that has meshes
   is a drawable with a ready world matrix. This is what the renderer's bounds scan and
   `drawable_iterator` use.

```cpp
for (const auto& scene : model.scene) {
    std::println("scene: {}", scene.name);
    for (const auto& node : scene.nodes) {
        const glm::mat4& world = node.transform_matrix;   // world matrix, ready to use
        for (const auto& mesh : node.meshes) {
            for (const auto& prim : mesh.primitives) {
                // each primitive is an independently drawable unit
            }
        }
    }
}
```

2. **Tree (hierarchy-oriented)** — start at `scene.root_indices` and recurse through
   `node.children`, composing world matrices from `local_transform` when you need to
   re-evaluate them (e.g. for animation / programmatic whole-group transforms):

```cpp
for (const auto& scene : model.scene) {
    // node is an index into scene.nodes; parent_world is identity at the roots
    auto walk = [&](auto&& self, std::size_t node, const glm::mat4& parent_world) -> void {
        const auto& n = scene.nodes[node];
        const glm::mat4 world = parent_world * n.local_transform;
        for (std::size_t child : n.children) self(self, child, world);
    };
    for (std::size_t root : scene.root_indices) walk(walk, root, glm::mat4(1.0f));
}
```

---

## 4. Reading Vertices and Indices

```cpp
const auto& prim = /* see above */;

// --- Vertex attributes ---
const auto& pos = prim.vertex.at("POSITION");
if (pos.component == gltf::component_type::float_t) {
    auto* p = reinterpret_cast<const float*>(pos.data.data());
    const std::size_t vertex_count = pos.data.size() / (3 * sizeof(float));
    // p[i*3 + 0..2] = position of the i-th vertex
}

// --- Indices (uint16 / uint32 are common in glTF) ---
if (prim.index_component_type == gltf::component_type::unsigned_short_t) {
    auto* idx = reinterpret_cast<const uint16_t*>(prim.index.data());
    const std::size_t index_count = prim.index.size() / sizeof(uint16_t);
}
```

Per-attribute byte sizes can be cross-checked with the helper functions:

```cpp
// one element = get_element_size(vec3) * get_component_size(float) = 3 * 4 = 12 bytes
if (pos.data.size() != gltf::get_element_size(gltf::element_type::vec3)
                       * gltf::get_component_size(gltf::component_type::float_t)
                       * vertex_count) { /* inconsistent data */ }
```

---

## 5. Using Textures

```cpp
// material lookup (default factors when the primitive has no material)
if (prim.material_index < model.materials.size()) {
    const gltf::material& mat = model.materials[prim.material_index];
    // factors: mat.factors.base_color_factor / metallic_factor / roughness_factor / ...
    if (const auto it = mat.texture_indices.find("albedo"); it != mat.texture_indices.end()) {
        const gltf::texture_data& tex = model.textures[it->second];
        // tex.data: width * height * component 8-bit pixels (already decoded, not raw PNG/JPEG bytes)
        std::println("texture {}x{} channels={} bytes={}",
                     tex.width, tex.height, tex.component, tex.data.size());
    }
}
```

| role                 | source                                          |
|----------------------|-------------------------------------------------|
| `albedo`             | `pbrMetallicRoughness.baseColorTexture`         |
| `metallic_roughness` | `pbrMetallicRoughness.metallicRoughnessTexture` |
| `normal`             | `normalTexture`                                 |
| `occlusion`          | `occlusionTexture`                              |
| `emissive`           | `emissiveTexture`                               |

---

## 6. Integration with Vulkan Rendering

`vulkan/model`'s `draw()` currently hardcodes `VK_INDEX_TYPE_UINT32`, while the loader preserves the original index type, so a conversion is needed before upload:

```cpp
std::vector<uint32_t> indices32;
if (prim.index_component_type == gltf::component_type::unsigned_short_t) {
    auto* src = reinterpret_cast<const uint16_t*>(prim.index.data());
    indices32.assign(src, src + prim.index.size() / 2);
} else if (prim.index_component_type == gltf::component_type::unsigned_int_t) {
    auto* src = reinterpret_cast<const uint32_t*>(prim.index.data());
    indices32.assign(src, src + prim.index.size() / 4);
}
```

Vertex buffers can be uploaded by memcpy'ing the raw bytes directly, provided the pipeline's vertex input layout matches the data layout (e.g. POSITION/NORMAL as `VK_FORMAT_R32G32B32_SFLOAT`, stored contiguously); otherwise re-arrange into an interleaved struct before upload.

---

## 7. Notes and Limitations

- The module is compiled with `-fno-exceptions`: fastgltf reports errors via `Expected` and never throws.
- Runtime dependencies: `libfastgltf.dll`, `libsimdjson.dll` (both in `C:\msys64\clang64\bin`; must be on PATH).
- Only static render data is exported: meshes, vertices/indices, textures, and material texture references. Animations, skins, cameras, lights, and morph targets are **not** currently exported into the public interface.
- All `asset.scenes` are loaded; `asset.defaultScene` is not separately marked yet.
- Verified samples (in `main.cpp`): the glTF/GLB variants of `glTF-Sample-Assets/Models/{Box, BoxInterleaved, BoxTextured}`, covering ASCII + external bin, binary containers, interleaved byteStride, embedded textures, and the missing-file error code.
