# vstd STL module

The project's STL module for `import vstd;` builds (C++23 modules on clang +
libc++, exceptions disabled).

## Layout

- `vstd.cppm` - the module interface: a global-module-fragment block of
  `#include <X>` lines (which pull in the definitions from the toolchain's
  libc++ headers) followed by `export module vstd;` and one
  `#include "std/X.inc"` per header.
- `std/` - the `.inc` partitions **actually referenced by `vstd.cppm`** (71
  today). Each partition is a pure re-export block:
  `export namespace std { using std::vector; ... }`. The unreferenced
  upstream partitions were deleted; the full upstream `std.cppm` /
  `std.compat.cppm` are **not** kept in-tree.
- `LICENSE` - the LLVM project license (Apache-2.0 with LLVM exceptions),
  covering the `.inc` content (each file carries the upstream header).

The `.inc` files are copies of the output of libc++'s module generator
(`utils/generate_libcxx_cppm_in.py`, lives upstream in the LLVM tree, not
vendored). Do not hand-edit their contents; treat `vstd.cppm` as the editable
whitelist.

## Portability notes (inspected)

The partitions are mechanically generated and were audited:

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

## Editing / trimming

`vstd.cppm` keeps one `#include <X>` + `#include "std/X.inc"` pair per used
header, in the upstream order.

- **Adding** an STL feature: the compiler reports the missing entity; add that
  header's pair to `vstd.cppm` and copy its partition into `std/` (from the
  toolchain's upstream module output).
- **Removing**: drop both lines and the partition file.

## Toolchain upgrade

The snapshot must match the installed libc++ (byte-compatible headers), as
always. On a clang64 upgrade:

1. From the new toolchain's libc++ include tree, obtain the fresh upstream
   module output (the generated `std.cppm` and its `std/` partition set - the
   same source this repo originally snapshotted).
2. For every header currently listed in `vstd.cppm`, copy the fresh
   `X.inc` over `std/X.inc` (contents may change between libc++ versions) and
   confirm the `#include <X>` lines still exist upstream; re-verify the trim
   with a full build.
