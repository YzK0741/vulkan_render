module;

#include <cstdint>

export module utility.frame_stats;

import vstd;

/**
 * @ingroup utility
 * @defgroup frame_stats Frame Statistics
 * @file frame_stats.cppm
 * @brief per-frame FPS statistics with a rolling report window (utility::frame_stats)
 *
 * The render loop calls tick() once per actually-presented frame; the class accumulates a
 * one-second window of frame-gap times and exposes both the running smoothed value (for a
 * per-frame overlay label) and the completed window's value (for the once-per-second log
 * line). Minimized / swapchain-recreate iterations call on_skipped() instead of tick(), so
 * pauses never distort the window - the frame-gap baseline is refreshed without counting.
 *
 * Typical use (the demo's render loop):
 * @code {.cpp}
 * utility::frame_stats stats;
 * while (...) {
 *     if (frame skipped) { stats.on_skipped(); continue; }
 *     // ... frame phases ...
 *     if (frame presented) {
 *         stats.tick();
 *         if (stats.window_rolled()) {
 *             utility::log("fps: {:.1f} ({:.2f} ms/frame)",
 *                          stats.window_fps(), stats.window_frame_ms());
 *         }
 *     }
 * }
 * @endcode
 */
namespace utility {
    /**
     * @ingroup frame_stats
     * @brief rolling FPS statistics over a report window (one second by default)
     * @note single-threaded by design: tick()/on_skipped() run on the frame owner thread only
     */
    export class frame_stats {
    public:
        /**
         * @brief configure the report window
         * @param window_seconds window length; tick() reports when the window fills
         */
        explicit frame_stats(double window_seconds = 1.0) noexcept
            : window_seconds_{window_seconds} {
        }

        /**
         * @brief record one presented frame: add its frame gap, count it, and roll the window
         *        over when the accumulated time reaches the window length
         * @note call ONLY for real frames; skipped iterations must call on_skipped() instead
         */
        void tick() noexcept {
            auto const now = std::chrono::steady_clock::now();
            if (this->last_frame_.time_since_epoch().count() == 0) {
                this->last_frame_ = now; // first tick: no gap yet, start the baseline
            } else {
                this->window_elapsed_ += std::chrono::duration<double>(now - this->last_frame_).count();
                this->last_frame_ = now;
                this->window_frames_ += 1;
            }
            if (this->window_elapsed_ >= this->window_seconds_ && this->window_frames_ > 0) {
                // window filled: publish its final numbers (same formula the overlay showed
                // live), then start a fresh window
                this->window_fps_ = static_cast<double>(this->window_frames_) / this->window_elapsed_;
                this->window_frame_ms_ = 1000.0 * this->window_elapsed_ / static_cast<double>(this->window_frames_);
                this->window_elapsed_ = 0.0;
                this->window_frames_ = 0;
                this->rolled_ = true;
            } else {
                this->rolled_ = false;
            }
        }

        /**
         * @brief a frame was skipped (minimized / swapchain recreation): refresh the frame-gap
         *        baseline without counting a frame, so the pause never inflates the window
         */
        void on_skipped() noexcept {
            this->last_frame_ = std::chrono::steady_clock::now();
            this->rolled_ = false;
        }

        /**
         * @brief whether the report window just filled on the last tick(); the caller runs its
         *        once-per-second reporting (log line) when this is true
         */
        [[nodiscard]] bool window_rolled() const noexcept {
            return this->rolled_;
        }

        /**
         * @brief running smoothed fps of the current window (live value for a per-frame
         *        overlay label); equals the completed window's fps right after a rollover
         */
        [[nodiscard]] double smoothed_fps() const noexcept {
            return this->window_elapsed_ > 0.0 ? static_cast<double>(this->window_frames_) / this->window_elapsed_ : 0.0;
        }

        /** @brief fps of the last completed window (0.0 before the first rollover) */
        [[nodiscard]] double window_fps() const noexcept {
            return this->window_fps_;
        }

        /** @brief milliseconds per frame of the last completed window (0.0 before the first rollover) */
        [[nodiscard]] double window_frame_ms() const noexcept {
            return this->window_frame_ms_;
        }

    private:
        double window_seconds_ = 1.0;                        // rolling-window length
        std::chrono::steady_clock::time_point last_frame_{}; // zero = no frame ticked yet
        double window_elapsed_ = 0.0;                        // summed frame gaps in this window
        uint32_t window_frames_ = 0;                         // frames counted in this window
        bool rolled_ = false;                                // set when the last tick() filled a window
        double window_fps_ = 0.0;                            // fps of the last completed window
        double window_frame_ms_ = 0.0;                       // ms/frame of the last completed window
    };
} // namespace utility
