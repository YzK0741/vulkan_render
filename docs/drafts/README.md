# DRAFTS - not built, not wired, kept on purpose

`ssgi_temporal.cppm.txt` and `ssgi_temporal.cpp.txt` are the DRAFT of the module that would take step 1a in
`../pass_chain_plan.md`'s handoff: the GI chain's diffuse temporal resolve, with the pass owning the frame's
RECORDING (the barriers, the dispatch, the two push lanes that describe its own state, the history copy and the
hand-backs) while its set layout, its pipeline and its per-image families stay the renderer's - supplied through
`resolved_io::own_set` and `resolved_io::pipelines`, which is inside the framework's contract.

WHY THEY ARE `.txt`: they must not be compiled, format-checked or doxygen-scanned. They are here because the
shape they encode was arrived at expensively - two failed attempts, three validation errors and the per-image
view channel that came out of them (all recorded in `../pass_chain_plan.md`) - and a draft in a temporary
directory is a draft nobody can find.

WHAT IS STILL MISSING before it can be installed, and it is only the host side:

1. `vulkan.pass.ssgi_temporal` added to `vulkan/pass/` and to `CMakeLists.txt`;
2. `runtime.cppm`: the import, the `pass::ssgi_temporal_pass ssgi_temporal` member with its stage array, and the
   declarations of `make_ssgi_denoise_frame()` and `resolve_ssgi_temporal(pass::resolved_io&)`;
3. `runtime.cpp`: the resolver (the seven `own` handles, the two `barrier_images`, `own_set` from
   `ssgi_temporal_family.set(index, 0)`, `pipelines[0]` from the runtime's pipeline, the push values, and the
   extent from `pass_extent()`), a branch in `resolve_pass`, the driver's `record_stage` call around the existing
   `record_ssgi_denoise_pass` body, and the stage in `create_passes` and in `on_swapchain_recreated`;
4. `resolved_` reported back so the reflection's resolve and `gi_spec_resolved` keep their current gating.

Acceptance is the same as every other step: Release + Debug + ASan+UBSan clean, `ctest` all green, doxygen exit 0
with an empty warning stream, and the gate 12 x 2 with 0 changed / 0 flaky against
`%LOCALAPPDATA%\vulkan_render\baseline`.
