/**
 * @file utility/platform_sleep.cpp
 * @brief The platform half of utility::sleep_until - a plain (non-module) translation unit.
 * @ingroup utility
 *
 * Deliberately NOT a module implementation unit: <windows.h> pulls in (via crtdbg) placement forms of
 * operator new, and a module's global module fragment feeds the global module - including it there made
 * libc++'s operator new ambiguous in every TU that imported `utility`. Keeping the system headers in a
 * plain TU contains them to this file, and the module side only sees the C entry point below.
 */

#include <cstdint>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <time.h>
#endif

extern "C" void utility_platform_sleep_ns(std::int64_t const nanoseconds) {
    if (nanoseconds <= 0) {
        return;
    }
#if defined(_WIN32)
    // A high-resolution waitable timer (Windows 10 1803+) waits to within microseconds without calling
    // timeBeginPeriod, which would change the timer resolution for every process on the machine. One
    // timer per process, created on first use.
    static HANDLE const timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (timer == nullptr) {
        Sleep(static_cast<DWORD>(nanoseconds / 1000000));
        return;
    }
    LARGE_INTEGER due = {};
    due.QuadPart = -nanoseconds / 100; // 100 ns units, negative = relative
    if (due.QuadPart == 0) {
        due.QuadPart = -1;
    }
    if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE) == 0) {
        Sleep(static_cast<DWORD>(nanoseconds / 1000000));
        return;
    }
    WaitForSingleObject(timer, INFINITE);
#else
    // POSIX: a relative nanosleep, restarted on EINTR. The caller re-computes the remaining time every
    // frame, so the relative form does not drift.
    timespec request = {};
    request.tv_sec = static_cast<time_t>(nanoseconds / 1000000000);
    request.tv_nsec = static_cast<long>(nanoseconds % 1000000000);
    while (nanosleep(&request, &request) == -1 && errno == EINTR) {
    }
#endif
}