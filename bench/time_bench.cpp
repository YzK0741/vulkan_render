// Micro-benchmark: cheap time reads.
//
// Compares three ways to obtain a "current time" value:
//   1. std::chrono::steady_clock::now()               (what the demo does per frame)
//   2. reading an atomic refreshed by a single writer (the "cached clock" shape)
//   3. reading an atomic refreshed by a dedicated 1 ms sleeper thread (the originally
//      proposed time_xxx design), including its staleness
//
// Batch timing: each batch times `iters` calls with one clock read around the whole loop,
// so the per-call cost of now() itself never distorts the measured op. Results are per-op
// nanoseconds (median + best of several batches).
//
// Build/run (Release matters: -O3):
//   cmake --build cmake-build-release-clang64 --target time_bench
//   ./cmake-build-release-clang64/time_bench.exe
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

namespace {
    uint64_t ns_between(clk::time_point const& a, clk::time_point const& b) {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
    }

    // One batch: run `op` `iters` times, time the whole batch, fold results into `sink`
    // (volatile, so the optimizer cannot elide the measured work).
    template <typename Op>
    uint64_t run_batch(Op&& op, int const iters, volatile uint64_t& sink) {
        clk::time_point const t0 = clk::now();
        uint64_t acc = 0;
        for (int i = 0; i < iters; ++i) {
            acc += op();
        }
        clk::time_point const t1 = clk::now();
        sink += acc;
        return ns_between(t0, t1);
    }
} // namespace

int main() {
    constexpr int iters = 1'000'000;
    constexpr int batches = 7;
    volatile uint64_t sink = 0;

    std::atomic<uint64_t> stamped{0};

    auto const bench = [&](char const* name, auto&& op) {
        std::vector<uint64_t> per(batches);
        for (int b = 0; b < batches; ++b) {
            per[static_cast<std::size_t>(b)] = run_batch(op, iters, sink);
        }
        std::sort(per.begin(), per.end());
        std::printf("%-38s median %7.2f ns/op   best %7.2f ns/op\n",
                    name,
                    static_cast<double>(per[batches / 2]) / static_cast<double>(iters),
                    static_cast<double>(per[0]) / static_cast<double>(iters));
    };

    auto const now_ticks = [] { return static_cast<uint64_t>(clk::now().time_since_epoch().count()); };

    bench("steady_clock::now()", now_ticks);
    bench("atomic load (relaxed)", [&] { return stamped.load(std::memory_order_relaxed); });
    bench("atomic load (seq_cst)", [&] { return stamped.load(); });
    bench("now() + atomic store", [&] {
        stamped.store(now_ticks(), std::memory_order_relaxed);
        return stamped.load(std::memory_order_relaxed);
    });

    // The originally proposed shape: a dedicated 1 ms sleeper thread refreshing the atomic.
    {
        std::atomic<bool> stop{false};
        std::thread writer([&] {
            while (!stop.load(std::memory_order_relaxed)) {
                stamped.store(now_ticks(), std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // let the writer warm up

        bench("read of 1ms-thread-stamped atomic", [&] { return stamped.load(std::memory_order_relaxed); });

        // Staleness: how far the cached value lags the true clock over a 200 ms window.
        uint64_t max_lag = 0;
        clk::time_point const window_end = clk::now() + std::chrono::milliseconds(200);
        while (clk::now() < window_end) {
            uint64_t const cached = stamped.load(std::memory_order_relaxed);
            uint64_t const real = now_ticks();
            max_lag = std::max(max_lag, real - cached);
        }
        std::printf("1ms-thread stamp: max observed staleness ~%.3f ms\n", static_cast<double>(max_lag) / 1.0e6);

        stop.store(true);
        writer.join();
    }

    std::printf("(sink=%llu)\n", static_cast<unsigned long long>(sink));
    return 0;
}
