# vstd — the project's STL module

> **What this is**: `vstd` is **modified from libc++** (LLVM's C++ standard
> library, the C++23 library shipped with clang). It is *not* a rewrite of
> the STL, *not* derived from any other std implementation (e.g. GNU
> libstdc++), and *not* an independent implementation: it is libc++'s own
> generated `std` module, renamed to `vstd` and **trimmed to the headers this
> project actually uses**. The exported entities are the libc++ entities,
> re-exported with `using`; the definitions come from the toolchain's libc++
> headers at compile time.
>
> The whole module is therefore **byte-bound to the matching libc++** of the
> MSYS2 clang64 toolchain — keep it in sync on toolchain upgrades (see below).

## Layout

```
vstd/
  vstd.cppm      module interface: global-fragment #include <X> lines
                 (pull libc++ definitions) + export module vstd; +
                 one #include "std/X.inc" per used header (editable whitelist)
  std/           the .inc partitions referenced by vstd.cppm (71 today),
                 copied from libc++'s module output - do not hand-edit
  LICENSE        LLVM project license (Apache-2.0 with LLVM exceptions),
                 covering the .inc content
  README.md      this file
```

Each `.inc` partition is a pure re-export block derived from the libc++
original: `export namespace std { using std::vector; ... }`. The full
upstream `std.cppm` / `std.compat.cppm` are **not** vendored here; only the
used partitions are kept.

## Portability notes (inspected)

The partitions are mechanically generated from libc++ and were audited:

- every line is a `using`-declaration inside `export namespace std` (plus
  comments/blank lines); no pragmas, no platform branches
  (`_WIN32`/`_MSC_VER`/`__linux__`/`__APPLE__`: zero hits), no `#elif`/`#else`;
  `#if`/`#endif` pairs match 1:1.
- the only conditionals are libc++ capability macros (`_LIBCPP_STD_VER`,
  `_LIBCPP_HAS_*`) and `__has_builtin`, and the only vendor tokens are
  `_LIBCPP_USING_IF_EXISTS` (a conditional-using suffix defined in libc++
  `<__config>`, which `vstd.cppm` includes first) and six
  `using std::__cpo::...` exports (the libc++ definition sites of the
  standard comparison CPOs).

So the content is **toolchain-bound, not platform-bound**: it compiles
against the matching libc++ headers and must be regenerated with them, but it
has no Windows/Linux/compiler-specific forks of its own.

## Versioning

`vstd` follows the same independent version scheme as every other module in the
repo (banner at the top of `vstd.cppm`, independent of the app version in
`project(VERSION)`):

- current: **0.1.0a**
- MAJOR: breaking interface changes
- MINOR: additive features — including project-local **extensions** (`vstd`
  is the natural home for STL additions beyond libc++; the module is not just a
  re-export shim, so bump MINOR when one lands)
- PATCH: internal fixes (trim changes, partition refreshes on a libc++ patch
  upgrade)

## Editing / trimming

`vstd.cppm` keeps one `#include <X>` + `#include "std/X.inc"` pair per used
header, in the upstream order.

- **Adding** an STL feature: the compiler reports the missing entity; add that
  header's pair to `vstd.cppm` and copy its partition into `std/` (from the
  toolchain's upstream libc++ module output).
- **Removing**: drop both lines and the partition file.

## Toolchain upgrade

The snapshot must match the installed libc++ (byte-compatible headers), as
always. On a clang64 upgrade:

1. From the new toolchain's libc++ include tree, obtain the fresh upstream
   module output (the generated `std.cppm` and its `std/` partition set - the
   same source this repo originally modified).
2. For every header currently listed in `vstd.cppm`, copy the fresh `X.inc`
   over `std/X.inc` (contents may change between libc++ versions) and confirm
   the `#include <X>` lines still exist upstream; re-verify the trim with a
   full build.
