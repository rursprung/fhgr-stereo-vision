#ifndef GCC_BUG_117560_WORKAROUND_HPP
#define GCC_BUG_117560_WORKAROUND_HPP

#if defined(__GLIBCXX__) && __GLIBCXX__ <= 20240908
  // see: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=117560
  // patch taken from here: https://gcc.gnu.org/git/?p=gcc.git;a=blobdiff;f=libstdc%2B%2B-v3/include/bits/fs_dir.h;h=79dc6764b7b27bb7b6d2ed6dda93b5f5e164ea29;hp=d669f2a74681509d769c62271d10f50beca89b0b;hb=eec6e8923586b9a54e37f32cef112d26d86e8f01;hpb=9ede072ffafcde27d0e9fe76bb7ffacb4f48a2d6
  // once GCC 14.3.0 has been released this should be removed and GCC users should upgrade.

  // Copyright (C) 2014-2024 Free Software Foundation, Inc.
  // SPDX-License-Identifier: GPL-3.0-or-later

  namespace std {
    // _GLIBCXX_RESOLVE_LIB_DEFECTS
    // 3480. directory_iterator and recursive_directory_iterator are not ranges
    namespace ranges
    {
      template<>
        inline constexpr bool
        enable_borrowed_range<filesystem::directory_iterator> = true;
      template<>
        inline constexpr bool
        enable_borrowed_range<filesystem::recursive_directory_iterator> = true;

      template<>
        inline constexpr bool
        enable_view<filesystem::directory_iterator> = true;
      template<>
        inline constexpr bool
        enable_view<filesystem::recursive_directory_iterator> = true;
    } // namespace ranges
  }
#endif

#endif //GCC_BUG_117560_WORKAROUND_HPP
