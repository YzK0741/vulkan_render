// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export namespace std {
  using std::div_t VSTD_USING_IF_EXISTS;
  using std::ldiv_t VSTD_USING_IF_EXISTS;
  using std::lldiv_t VSTD_USING_IF_EXISTS;
  using std::size_t VSTD_USING_IF_EXISTS;

  // [support.start.term], start and termination
  using std::_Exit VSTD_USING_IF_EXISTS;
  using std::abort VSTD_USING_IF_EXISTS;
  using std::at_quick_exit VSTD_USING_IF_EXISTS;
  using std::atexit VSTD_USING_IF_EXISTS;
  using std::exit VSTD_USING_IF_EXISTS;
  using std::quick_exit VSTD_USING_IF_EXISTS;

  using std::getenv VSTD_USING_IF_EXISTS;
  using std::system VSTD_USING_IF_EXISTS;

  // [c.malloc], C library memory allocation
  using std::aligned_alloc VSTD_USING_IF_EXISTS;
  using std::calloc VSTD_USING_IF_EXISTS;
  using std::free VSTD_USING_IF_EXISTS;
  using std::malloc VSTD_USING_IF_EXISTS;
  using std::realloc VSTD_USING_IF_EXISTS;

  using std::atof VSTD_USING_IF_EXISTS;
  using std::atoi VSTD_USING_IF_EXISTS;
  using std::atol VSTD_USING_IF_EXISTS;
  using std::atoll VSTD_USING_IF_EXISTS;
  using std::strtod VSTD_USING_IF_EXISTS;
  using std::strtof VSTD_USING_IF_EXISTS;
  using std::strtol VSTD_USING_IF_EXISTS;
  using std::strtold VSTD_USING_IF_EXISTS;
  using std::strtoll VSTD_USING_IF_EXISTS;
  using std::strtoul VSTD_USING_IF_EXISTS;
  using std::strtoull VSTD_USING_IF_EXISTS;

  // [c.mb.wcs], multibyte / wide string and character conversion functions
  using std::mblen VSTD_USING_IF_EXISTS;
#if VSTD_HAS_WIDE_CHARACTERS
  using std::mbstowcs VSTD_USING_IF_EXISTS;
  using std::mbtowc VSTD_USING_IF_EXISTS;
  using std::wcstombs VSTD_USING_IF_EXISTS;
  using std::wctomb VSTD_USING_IF_EXISTS;
#endif
  // [alg.c.library], C standard library algorithms
  using std::bsearch VSTD_USING_IF_EXISTS;
  using std::qsort VSTD_USING_IF_EXISTS;

  // [c.math.rand], low-quality random number generation
  using std::rand VSTD_USING_IF_EXISTS;
  using std::srand VSTD_USING_IF_EXISTS;

  // [c.math.abs], absolute values
  using std::abs VSTD_USING_IF_EXISTS;

  using std::labs VSTD_USING_IF_EXISTS;
  using std::llabs VSTD_USING_IF_EXISTS;

  using std::div VSTD_USING_IF_EXISTS;
  using std::ldiv VSTD_USING_IF_EXISTS;
  using std::lldiv VSTD_USING_IF_EXISTS;
} // namespace std
