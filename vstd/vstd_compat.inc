// -*- C++ -*-
//===----------------------------------------------------------------------===//
// vstd portability layer
//
// WHY THIS EXISTS. The partitions in std/ are libc++'s generated re-export blocks, and the only
// implementation-specific things they contain are CONDITIONALS: `_LIBCPP_STD_VER`, the
// `_LIBCPP_HAS_*` capability flags, the `_LIBCPP_ENABLE_*` opt-ins, and `_LIBCPP_USING_IF_EXISTS`.
// Every entity they export is a STANDARD name (libc++ generates them from the standard's synopsis),
// so the partitions themselves do not need to change per toolchain - the macros do.
//
// This file gives each of those one VSTD_* spelling, and it is included after <__config> and before
// any partition, so a partition names VSTD_* while the value still comes from the toolchain:
//
//   * under libc++ every VSTD_* is defined AS its _LIBCPP_* counterpart whenever that is defined, so
//     the module this project has been building is the same module (the gate's nine scenarios are the
//     proof);
//   * under another standard library the same names come from the standard's own feature-test macros
//     and from __has_include, which is what makes the toolchain swap possible at all.
//
// THE ONE RULE THAT MATTERS, and it is a rule about DEFINEDNESS rather than about value: a VSTD_*
// macro is defined ONLY WHEN THE CAPABILITY IS THERE, never as 0. The partitions test these two ways -
// `#if VSTD_HAS_THREADS` and `#ifdef VSTD_ENABLE_EXPERIMENTAL` - and only "defined iff capable" makes
// both forms mean exactly what the libc++ original meant. Defining them as 0 breaks the `#ifdef`
// form silently: the first version of this file did that, and it compiled libc++'s EXPERIMENTAL time
// zone database into the module because `#ifdef VSTD_ENABLE_EXPERIMENTAL` was true while
// `_LIBCPP_ENABLE_EXPERIMENTAL` was not defined at all.
//
// WHAT THIS CANNOT FIX, stated so nobody expects otherwise: a name that the *other* library has not
// implemented still fails to compile where a partition says `using std::that_name;`. That residue is
// found by compiling against that library and guarded there, one line at a time - no macro can
// conjure it, and `_LIBCPP_USING_IF_EXISTS` softens it only on a compiler with
// __attribute__((using_if_exists)) (clang; GCC and MSVC do not have it, so there the using is hard).
//
// MAINTENANCE: when a std/X.inc is refreshed from a newer libc++, a new _LIBCPP_* macro in it shows
// up as a compile error here - the intended failure mode. Add its VSTD_* spelling below.
//===----------------------------------------------------------------------===//

#ifndef VSTD_COMPAT_INC
#define VSTD_COMPAT_INC

// ---- the conditional-using attribute, made portable the same way libc++ itself makes it portable ----
// clang has it; GCC and MSVC do not, where it expands to nothing and a using of a name the library
// lacks becomes a hard error (the residue described above).
#ifndef VSTD_USING_IF_EXISTS
#  if defined(__has_attribute)
#    if __has_attribute(using_if_exists)
#      define VSTD_USING_IF_EXISTS __attribute__((using_if_exists))
#    endif
#  endif
#  ifndef VSTD_USING_IF_EXISTS
#    define VSTD_USING_IF_EXISTS
#  endif
#endif

#if defined(_LIBCPP_VERSION)

// ============================ libc++ ============================
// Every name IS the libc++ name, each one defined exactly when libc++ defines it, so this layer is
// invisible on the toolchain the project builds with today.
#  define VSTD_IMPLEMENTATION_LIBCXX 1

#  ifdef _LIBCPP_STD_VER
#    define VSTD_STD_VER _LIBCPP_STD_VER
#  endif
#  ifdef _LIBCPP_HAS_WIDE_CHARACTERS
#    define VSTD_HAS_WIDE_CHARACTERS _LIBCPP_HAS_WIDE_CHARACTERS
#  endif
#  ifdef _LIBCPP_HAS_LOCALIZATION
#    define VSTD_HAS_LOCALIZATION _LIBCPP_HAS_LOCALIZATION
#  endif
#  ifdef _LIBCPP_HAS_THREADS
#    define VSTD_HAS_THREADS _LIBCPP_HAS_THREADS
#  endif
#  ifdef _LIBCPP_HAS_FILESYSTEM
#    define VSTD_HAS_FILESYSTEM _LIBCPP_HAS_FILESYSTEM
#  endif
#  ifdef _LIBCPP_HAS_TIME_ZONE_DATABASE
#    define VSTD_HAS_TIME_ZONE_DATABASE _LIBCPP_HAS_TIME_ZONE_DATABASE
#  endif
#  ifdef _LIBCPP_HAS_CHAR8_T
#    define VSTD_HAS_CHAR8_T _LIBCPP_HAS_CHAR8_T
#  endif
#  ifdef _LIBCPP_HAS_UNICODE
#    define VSTD_HAS_UNICODE _LIBCPP_HAS_UNICODE
#  endif
#  ifdef _LIBCPP_HAS_RTTI
#    define VSTD_HAS_RTTI _LIBCPP_HAS_RTTI
#  endif
#  ifdef _LIBCPP_HAS_MONOTONIC_CLOCK
#    define VSTD_HAS_MONOTONIC_CLOCK _LIBCPP_HAS_MONOTONIC_CLOCK
#  endif
#  ifdef _LIBCPP_HAS_ATOMIC_HEADER
#    define VSTD_HAS_ATOMIC_HEADER _LIBCPP_HAS_ATOMIC_HEADER
#  endif
#  ifdef _LIBCPP_NO_LOCK_FREE_TYPES
#    define VSTD_NO_LOCK_FREE_TYPES _LIBCPP_NO_LOCK_FREE_TYPES
#  endif
#  ifdef _LIBCPP_HAS_EXPERIMENTAL_SYNCSTREAM
#    define VSTD_HAS_EXPERIMENTAL_SYNCSTREAM _LIBCPP_HAS_EXPERIMENTAL_SYNCSTREAM
#  endif
#  ifdef _LIBCPP_HAS_EXPERIMENTAL_OPTIONAL_ITERATOR
#    define VSTD_HAS_EXPERIMENTAL_OPTIONAL_ITERATOR _LIBCPP_HAS_EXPERIMENTAL_OPTIONAL_ITERATOR
#  endif
// The three opt-ins are DEFINED-NESS tests in the partitions (#ifdef), so they must stay undefined
// unless the toolchain defines them - which is what makes the module's experimental surface follow
// the toolchain's, exactly as the libc++ original did.
#  ifdef _LIBCPP_ENABLE_EXPERIMENTAL
#    define VSTD_ENABLE_EXPERIMENTAL _LIBCPP_ENABLE_EXPERIMENTAL
#  endif
#  ifdef _LIBCPP_ENABLE_CXX26_REMOVED_WSTRING_CONVERT
#    define VSTD_ENABLE_CXX26_REMOVED_WSTRING_CONVERT _LIBCPP_ENABLE_CXX26_REMOVED_WSTRING_CONVERT
#  endif
#  ifdef _LIBCPP_ENABLE_CXX20_REMOVED_UNCAUGHT_EXCEPTION
#    define VSTD_ENABLE_CXX20_REMOVED_UNCAUGHT_EXCEPTION _LIBCPP_ENABLE_CXX20_REMOVED_UNCAUGHT_EXCEPTION
#  endif

#else // !_LIBCPP_VERSION

// ============================ another standard library ============================
// Same rule: define a VSTD_* only when the capability is really there. The libc++-only opt-ins and
// libc++-experimental capabilities are therefore simply NOT defined, so the partitions that name them
// export nothing - which is correct, because nothing else has those entities.
#  define VSTD_IMPLEMENTATION_LIBCXX 0

// The standard language level in libc++'s numbering (20/23/26), so `VSTD_STD_VER >= 23` keeps meaning
// "the library is in C++23 mode or later".
#  if __cplusplus >= 202302L
#    define VSTD_STD_VER 23
#  elif __cplusplus >= 202002L
#    define VSTD_STD_VER 20
#  elif __cplusplus >= 201703L
#    define VSTD_STD_VER 17
#  endif

#  if defined(__cpp_char8_t)
#    define VSTD_HAS_CHAR8_T 1
#  endif
#  if __has_include(<cwchar>)
#    define VSTD_HAS_WIDE_CHARACTERS 1
#  endif
#  if __has_include(<locale>)
#    define VSTD_HAS_LOCALIZATION 1
#  endif
#  if __has_include(<thread>)
#    define VSTD_HAS_THREADS 1
#  endif
#  if __has_include(<filesystem>)
#    define VSTD_HAS_FILESYSTEM 1
#  endif
#  if defined(__cpp_lib_chrono) && __cpp_lib_chrono >= 201907L
#    define VSTD_HAS_TIME_ZONE_DATABASE 1
#  endif
#  if defined(__cpp_lib_print)
#    define VSTD_HAS_UNICODE 1
#  endif
#  if defined(__cpp_exceptions)
#    define VSTD_HAS_RTTI 1
#  endif
#  if __has_include(<chrono>)
#    define VSTD_HAS_MONOTONIC_CLOCK 1
#  endif
#  if __has_include(<atomic>)
#    define VSTD_HAS_ATOMIC_HEADER 1
#  endif

#endif // _LIBCPP_VERSION

#endif // VSTD_COMPAT_INC
