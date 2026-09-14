# The pass chain, rebuilt on the pre-GI state

This branch (`pass-chain`, forked at `281ab06` - the commit immediately before the first GI commit) exists to do
one thing the later work could not do in place: let the PBR rendering be **built** as passes rather than
migrated into them, and then let GI be **added** to that architecture pass by pass. The later state on `master`
holds the GI implementation, its measurements and its documentation; this branch starts without them and
earns them back one pass at a time.

## THE ONE RULE THAT PROTECTS THE ACCEPTANCE ANCHOR

The capture gate's references live outside the repository, in
`%LOCALAPPDATA%\vulkan_render\baseline`, and on this machine they were captured from `master` WITH GI. Those
hashes are the acceptance test for restoring GI later (`default_gi` `BF180E98ADB29E7E`, `sponza_gi`
`58EC848DFABE654A`, `metal_rough_glossy` `46F9851B7BC89872`, `glossy_motion` `98B06F2190B49519`).

**Therefore every gate run on this branch must use its own baseline directory:**

```powershell
$env:VR_RENDER_BASELINE_DIR = "$env:LOCALAPPDATA\vulkan_render\baseline-pass-chain"
pwsh -NoProfile -File scripts/windows/check_render.ps1            # compare (never -Update on the default dir)
```

`-Update` may only ever be used with that variable set. Running it without is how the anchor would be
destroyed, and it would be destroyed silently: the next gate run would compare against the wrong references and
report "0 changed" for a branch that changed everything.

This branch's gate has **7 scenarios**; the five GI-era scenarios and their configs live on `master` with the
GI work. `sponza` is the useful overlap: its hash on this branch (`0EBA5300E8F84E6F`) is IDENTICAL to
`master`'s, which is the evidence that the frames this branch renders are the frames `master` renders whenever
GI is not involved.

## WHAT IS ON THIS BRANCH SO FAR

| commit | what |
|---|---|
| `bfed4cd` | the description layer (24 schema families - this state's resources, not master's 37), the shared sampler module, and `vulkan.pass` (the framework). Nothing consumes it: gate 7 x 2, 0 changed |
| `de23673` | `/build/` is ignored (this branch predates that rule, which is why a bare `build/` was committed once and had to be amended out) |
| `2593e7e` | `test_render_resources` ported and adapted (89 checks): the schema, the validators, the three declarations this state can carry |

`docs/pass_chain_inventory.md` (192 lines) is the read-only map of this state: its resource set, the frame's
recording spine and mark intervals, every function of the PBR/scene chain with its attachments, sets,
pipelines, pushes and barriers, the shared sets binding by binding, the pipeline registry, the leaf draw path,
and what this state does NOT have.

## THE ORDER, AND WHY IT IS THIS ORDER

1. **the generators** (`make_set_layout`, `write_set`, and the `image_set_family` API) must take a
   `VkDevice` rather than a whole `core`, because a pass's create step is handed a device and nothing else.
2. **`scene`** - the PBR surface write: the rendering instance over its declared targets, the segmented draw of
   the visible leaves, and the instance's close. This is the pass that fixes the coupling the inventory found:
   the instance is opened by one function and closed by another, three subsystems later.
3. **`transparent`**, then **`taa`** - both are small once `scene` exists, and both are on this branch already
   (TAA predates GI).
4. **`lighting`** and **`post`** - the rest of what "PBR as a pass chain" means.
5. **then GI**: `ssgi` (trace) -> `temporal` -> `spatial` -> `spec` -> `probe cache`, each added as a pass and
   each verified against `master`'s references for the GI scenarios.

Acceptance is the same for every step and is not negotiable: Release + Debug + ASan+UBSan build clean, `ctest`
all green, doxygen exit 0 with an empty warning stream, and the gate `0 changed / 0 flaky` against the
branch's own baseline directory (and against `master`'s references once GI starts coming back).

## WHAT THE INVENTORY ALREADY FOUND (recorded, not yet fixed)

* The named scene pipelines are built with `swap_chain_image_format` (`B8G8R8A8_SRGB`) while the transparent
  pass's instance declares `vulkan::hdr_format` (`R16G16B16A16_SFLOAT`) for its single colour attachment. The
  gate reports no validation finding on this state, so it is either tolerated or the formats are in fact
  compatible - but a declaration that names its target's format would settle it, which is one of the reasons
  the transparent pass gets a declaration.
* `sampler_hint::probe_grid` is part of the hint vocabulary on this branch, and no sampler backs it here (the
  six-hint set is `master`'s). A hint nothing can resolve is a lie in the vocabulary.
* Stale references in comments and docs: a "skybox" pipeline that does not exist (the sky is evaluated inside
  the lighting stage's shader), a `draw_skybox` parameter that no signature has, an implemented-but-undocumented
  `feature_available("skybox")`, an overlay-record comment pointing at the wrong function, and a shadow-sampler
  doc that contradicts `constant_init`.
