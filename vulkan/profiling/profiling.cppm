// module version: 0.1.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/profiling/profiling.cppm
 * @brief Frame timing instrumentation: the CPU frame phases, measured per frame and reported as a
 *        60-frame window (see cpu_phases), with the RAII scope timer the pass recorders use.
 * @ingroup vulkan_profiling
 *
 * Extracted from vulkan.runtime, whose implementation had grown past 4900 lines with the timing, the
 * pass recorders, the pipelines and the resources all in one file. It is a module rather than an
 * implementation partition because the timing state is legitimately useful on its own: an overlay, a
 * test or another layer can read cpu_phases without going through the runtime facade. The GPU half
 * (the per-pass timestamps) follows in a second step.
 */

module;

#include <array>
#include <chrono>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>

export module vulkan.profiling;

namespace vulkan::profiling {
    /**
     * @brief the CPU frame phases that are measured per frame
     * @ingroup vulkan_profiling
     *
     * A phase whose name ends in '*' is a SUB-phase: it is measured inside another phase, so it is
     * reported but deliberately left out of the reported total (adding it would count that time twice).
     */
    export enum class cpu_phase : uint32_t {
        pace,
        begin,
        scene,
        cluster, // sub-phase of scene
        shadow,  // sub-phase of scene
        post,
        submit,
        submit_queue, // sub-phase of submit
        present,      // sub-phase of submit
        count,
    };

    /// phase names in cpu_phase order; the trailing '*' marks a sub-phase (see cpu_phase)
    export constexpr std::array<std::string_view, static_cast<std::size_t>(cpu_phase::count)> cpu_phase_names = {
        "pace", "begin", "scene", "cluster*", "shadow*", "post", "submit", "submit-queue*", "present*"};

    /**
     * @brief the rolling window of measured CPU frame phases
     * @ingroup vulkan_profiling
     *
     * add()/end() accumulate the frame being measured; fold() closes the window every @c window_length
     * frames and publishes it as a label, because a per-frame running mean makes an overlay re-wrap its
     * text every frame (the label is only as wide as its widest value). The label is what both the log
     * and the overlay read, so it holds the last COMPLETED window rather than a partially filled one.
     */
    export class cpu_phases {
    public:
        static constexpr uint32_t window_length = 60;

        void add(cpu_phase const phase, std::chrono::steady_clock::duration const elapsed) noexcept {
            this->frame[static_cast<std::size_t>(phase)] += std::chrono::duration<double, std::milli>(elapsed).count();
        }

        void end(cpu_phase const phase, std::chrono::steady_clock::time_point const start) noexcept {
            this->add(phase, std::chrono::steady_clock::now() - start);
        }

        /// close the current window when it is full; call once per frame
        void fold() noexcept {
            if (++this->window_frames < window_length) {
                return;
            }
            std::string label = std::format("cpu frame phases (avg of {} frames):", window_length);
            double total = 0.0;
            for (std::size_t i = 0; i < cpu_phase_names.size(); ++i) {
                double const mean = this->sum[i] / static_cast<double>(window_length);
                label += std::format(" {} {:.2f} ms |", cpu_phase_names[i], mean);
                if (!cpu_phase_names[i].ends_with('*')) {
                    total += mean;
                }
            }
            label += std::format(" total {:.2f} ms", total);
            this->report_label = std::move(label);
            this->last_window = this->sum;
            this->sum = {};
            this->window_frames = 0;
        }

        [[nodiscard]] std::string const& summary() const noexcept {
            return this->report_label;
        }

        [[nodiscard]] std::array<double, static_cast<std::size_t>(cpu_phase::count)> means() const noexcept {
            std::array<double, static_cast<std::size_t>(cpu_phase::count)> result = {};
            for (std::size_t i = 0; i < result.size(); ++i) {
                result[i] = this->last_window[i] / static_cast<double>(window_length);
            }
            return result;
        }

    private:
        std::array<double, static_cast<std::size_t>(cpu_phase::count)> frame = {};       // the frame being measured
        std::array<double, static_cast<std::size_t>(cpu_phase::count)> sum = {};         // the window being accumulated
        std::array<double, static_cast<std::size_t>(cpu_phase::count)> last_window = {}; // the last completed window
        uint32_t window_frames = 0;
        std::string report_label = {};
    };

    /**
     * @brief scope timer: measures one CPU phase from construction to destruction
     * @ingroup vulkan_profiling
     */
    export class cpu_phase_timer {
    public:
        cpu_phase_timer(cpu_phases& target, cpu_phase const which) noexcept
            : phases{&target}
            , phase{which}
            , start{std::chrono::steady_clock::now()} {
        }

        ~cpu_phase_timer() {
            this->phases->end(this->phase, this->start);
        }

        cpu_phase_timer(cpu_phase_timer const&) = delete;
        cpu_phase_timer& operator=(cpu_phase_timer const&) = delete;
        cpu_phase_timer(cpu_phase_timer&&) = delete;
        cpu_phase_timer& operator=(cpu_phase_timer&&) = delete;

    private:
        cpu_phases* phases = nullptr;
        cpu_phase phase = cpu_phase::count;
        std::chrono::steady_clock::time_point start = {};
    };
} // namespace vulkan::profiling