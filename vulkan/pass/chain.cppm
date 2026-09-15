// module version: 0.1.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/chain.cppm
 * @brief A chain of passes: the list, its order, and the two runner calls over it.
 * @defgroup vulkan_pass_chain Pass Chain
 *
 * WHY THIS EXISTS. The runner already had both halves of "run a group of passes" (`create_stage` and
 * `record_stage`), and the renderer still re-stated the LIST at every call site: one `std::array<frame_pass*, 1>`
 * member per pass, one `stage{.name = ..., .passes = ..., .marks = false}` construction per call, and the order
 * written in the frame loop's statement sequence. A chain makes that a VALUE instead of a convention: the passes
 * are added once, in order, and the two calls take the chain.
 *
 * WHAT IT IS NOT: not a scheduler and not an ownership container. It holds NON-OWNING pointers, because a pass
 * in this renderer is a member of whatever built it (the runtime) and the chain must not extend or shorten
 * anyone's lifetime - the framework's own rule for everything a pass is handed (see `pass_context`). Nor does it
 * decide ANYTHING per pass: each pass's `feature()` gate, its resolver and its behaviour still belong to the
 * runner, so a chain of four passes runs exactly like the four separate stages it replaced, down to the
 * per-pass `resolved_io` the runner builds for each of them.
 *
 * THE ORDER IS THE DATA, and that is the point of the container: `trace -> lobe -> denoise -> filter` is what
 * makes the GI chain work (each stage reads what the one before it wrote), and in this class that order is a
 * sequence of `add` calls rather than the line order of a frame loop that also has to interleave barriers,
 * marks and the renderer's own work between the stages.
 *
 * WHAT A CHAIN DELIBERATELY DOES NOT EXPRESS: a pass that must run only when an EARLIER pass in the same chain
 * recorded (the spatial filter, which must not filter a stale accumulation). That stays where it belongs - the
 * feature registry, whose `feature_active` is asked per pass, in order, by the runner - because a chain that
 * could skip its own tail would be a scheduler, and this renderer's feature gates are already the one place
 * where "does this pass run this frame" is answered.
 */

module;

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

export module vulkan.pass.chain;

import vulkan.pass;

export namespace vulkan::pass {

    /**
     * @brief an ordered list of passes, with the runner's create and record steps over it
     *
     * The two calls are the framework's own (`create_stage` / `record_stage`), which is what keeps a chain from
     * growing into a second runner: everything a pass is given, everything the runner does around it and every
     * report it produces are unchanged.
     */
    class pass_chain {
        /// the name the runner is handed for the whole chain (used by marks, which every chain here disables)
        std::string_view name_ = {};
        /// the passes, in the order they were added; fixed once the frame starts, so nothing allocates per frame
        std::vector<frame_pass*> passes_ = {};
        /// whether the runner writes a mark pair around the chain (false for every chain in this renderer: the
        /// frame loop owns the marks and their positions are the timing report's contract)
        bool marks_ = false;

    public:
        pass_chain() = default;
        explicit pass_chain(std::string_view name, bool marks = false) noexcept
            : name_(name)
            , marks_(marks) {
        }

        /// @brief append a pass to the end of the chain; the order of the calls IS the order of the stages
        void add(frame_pass& pass) {
            this->passes_.push_back(&pass);
        }

        [[nodiscard]] std::string_view name() const noexcept {
            return this->name_;
        }
        [[nodiscard]] std::size_t size() const noexcept {
            return this->passes_.size();
        }
        [[nodiscard]] bool empty() const noexcept {
            return this->passes_.empty();
        }

        /// @brief the pass whose DECLARATION is called @p name, or nullptr (a lookup by the declaration's own
        ///        vocabulary, so a caller never has to know the chain's order to find one pass)
        [[nodiscard]] frame_pass* find(std::string_view const name) const noexcept {
            for (frame_pass* const pass : this->passes_) {
                if (pass != nullptr && pass->io().name == name) {
                    return pass;
                }
            }
            return nullptr;
        }

        /// @brief the stage the runner is handed: this chain's name, list and mark policy
        /// @note non-const because `stage::passes` carries MUTABLE pointers (a pass writes its own state in
        ///       `record`); the chain itself is fixed once the frame starts
        [[nodiscard]] stage as_stage() noexcept {
            return stage{.name = this->name_, .passes = std::span<frame_pass*>(this->passes_.data(), this->passes_.size()), .marks = this->marks_};
        }

        /// @brief the runner's create step over every pass in the chain, in order
        [[nodiscard]] run_report init(pass_context const& context) {
            return create_stage(this->as_stage(), context);
        }
        /// @brief the runner's record step over every pass in the chain, in order
        [[nodiscard]] run_report record(pass_host const& host) {
            return record_stage(this->as_stage(), host);
        }
    };

} // namespace vulkan::pass
