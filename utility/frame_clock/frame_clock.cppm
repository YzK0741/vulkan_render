module;

#include <cstdint> // ::uint64_t (used unqualified below)

export module utility.frame_clock;

import std;

/**
 * @ingroup utility
 * @defgroup frame_clock Frame Clock
 * @brief per-frame cheap time: one writer thread stamps a couple of atomics each frame, any
 *        number of reader threads get the time as an atomic load (see bench/time_bench.cpp for
 *        the numbers: now() is ~23 ns, an atomic read ~1 ns; a dedicated 1 ms sleeper thread
 *        would lag up to the ~15.6 ms Windows timer granularity, which is why the stamping
 *        writer is the frame/update thread itself - readers are at most one frame behind).
 *
 * Typical use: the render (or game/update) thread calls stamp() once per frame, parallel
 * workers / animation code read last_ns() or delta_ns() anywhere else. The clock is
 * monotonic with an implementation-defined origin (steady_clock since its epoch).
 */
namespace utility {
    /**
     * @ingroup frame_clock
     * @brief monotonic per-frame clock with single-writer stamps and cheap multi-reader reads
     * @note
     *     - stamp() must only be called from ONE thread (the frame owner); every other method
     *       is safe to call from any thread at any time (relaxed atomic loads)
     *     - last_ns()/delta_ns() are raw nanoseconds since the steady clock's origin; convert
     *       with delta_seconds()/last_seconds() or read now_ns() for a true 'right now'
     *     - readers may observe last_ns() ahead of delta_ns()'s matching value by one stamp:
     *       values are individually consistent, not a lock-free snapshot pair
     */
    export class frame_clock {
    public:
        /** @brief record the current steady time as the new frame stamp (frame owner thread) */
        void stamp() noexcept {
            uint64_t const now = now_ns();
            uint64_t const previous = this->last_ns_.load(std::memory_order_relaxed);
            this->delta_ns_.store(now - previous, std::memory_order_relaxed);
            this->last_ns_.store(now, std::memory_order_relaxed);
        }

        /** @brief nanoseconds (steady origin) of the most recent stamp */
        [[nodiscard]] uint64_t last_ns() const noexcept {
            return this->last_ns_.load(std::memory_order_relaxed);
        }

        /** @brief nanoseconds elapsed between the two most recent stamps (0 until the second one) */
        [[nodiscard]] uint64_t delta_ns() const noexcept {
            return this->delta_ns_.load(std::memory_order_relaxed);
        }

        /** @brief seconds of the most recent stamp (float; convenience) */
        [[nodiscard]] double last_seconds() const noexcept {
            return static_cast<double>(this->last_ns()) * 1e-9;
        }

        /** @brief seconds elapsed between the two most recent stamps (float; convenience) */
        [[nodiscard]] double delta_seconds() const noexcept {
            return static_cast<double>(this->delta_ns()) * 1e-9;
        }

        /** @brief true 'right now' in the same steady-clock scale (baseline / calibration) */
        [[nodiscard]] static uint64_t now_ns() noexcept {
            return static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        }

    private:
        std::atomic<uint64_t> last_ns_{0};
        std::atomic<uint64_t> delta_ns_{0};
    };
} // namespace utility
