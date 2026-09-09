#pragma once
// Tiny CHECK harness for the headless unit tests - deliberately no external test
// framework (the repo stays dependency-free). Every failed CHECK prints file:line
// and bumps the failure counter; the test's main returns 1 when anything failed so
// CTest sees the test as failed. One test executable = one translation unit.

#include <cstdio>

namespace vk_test {
    [[nodiscard]] inline int& failures() {
        static int count = 0;
        return count;
    }
    [[nodiscard]] inline int& checks() {
        static int count = 0;
        return count;
    }

    inline void report(char const* expression, char const* file, int const line, char const* message) {
        ++failures();
        if (message != nullptr) {
            std::printf("FAIL %s:%d: %s  (%s)\n", file, line, expression, message);
        } else {
            std::printf("FAIL %s:%d: %s\n", file, line, expression);
        }
    }

    /** @brief print the summary and return the process exit code (0 = all checks passed) */
    inline int finish(char const* test_name) {
        int const failed = failures();
        std::printf("[%s] %d checks, %d failed -> %s\n", test_name, checks(), failed, failed == 0 ? "PASS" : "FAIL");
        return failed == 0 ? 0 : 1;
    }
} // namespace vk_test

#define CHECK(cond)                                                                                                  \
    do {                                                                                                             \
        ++::vk_test::checks();                                                                                       \
        if (!(cond)) {                                                                                               \
            ::vk_test::report(#cond, __FILE__, __LINE__, nullptr);                                                   \
        }                                                                                                            \
    } while (false)

#define CHECK_MSG(cond, message)                                                                                     \
    do {                                                                                                             \
        ++::vk_test::checks();                                                                                       \
        if (!(cond)) {                                                                                               \
            ::vk_test::report(#cond, __FILE__, __LINE__, (message));                                                 \
        }                                                                                                            \
    } while (false)
