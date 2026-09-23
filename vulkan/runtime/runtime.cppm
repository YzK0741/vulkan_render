// ============================================================================
// module: vulkan.runtime
// module version: 0.73.1  (independent of the app version in CMakeLists project(VERSION))
//
// The renderer core: per-frame-slot frame facade (pace/record/submit phases,
// scene resources, parallel secondary-CB recording). It re-exports its peer
// modules vulkan.scene_tree (scene storage + GPU primitives) and
// vulkan.render_environment (per-worker draw state) - the frame draws through
// both, so they are versioned as ONE unit because they share the scene / draw
// interface and evolve together.
// Depends on vulkan.core (GPU), vulkan.math (IBL), vulkan.shadow_fit (the
// cascade fit it gathers for and caches), vulkan.readback (the screenshot's
// staging buffer and host read) and utility, with the frame struct
// fills coming from vulkan.constant_init.
//
// evolve: bump MAJOR on breaking interface changes, MINOR on additive features,
//         PATCH on internal fixes - independently of the rest of the project.
// ============================================================================

// THE PRIMARY INTERFACE IS DELIBERATELY THIS SMALL, for the reason recorded in vulkan.core's:
// under -Werror, clang 22 rejects a primary that imports its own implementation partitions, and it does
// not need to - CMake compiles every partition in the module's file set, so their definitions are
// archived and the linker finds them.

export module vulkan.runtime;
export import :declarations;