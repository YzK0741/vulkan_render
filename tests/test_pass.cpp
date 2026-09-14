// Headless unit tests: vulkan.pass (the framework, no device) =================
// The framework's whole value is a CONTRACT: create once per generation, resolve per frame, skip an inactive
// pass without resolving it, apply the behaviour before recording, mark once per stage, and tell every pass
// when the swapchain was rebuilt. Each of those is something this renderer does by hand today, and does
// inconsistently - the manual reset list misses two of six descriptor families, and the viewport resync is a
// hand-kept pipeline list. So this test asserts the contract with a FAKE host: no device, no Vulkan call, and
// every assertion is about what the runner DID, in what order, and what it deliberately did not do.
#include "vk_test.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>

import vulkan.pass;
import vulkan.render_resource;

namespace {
    namespace vp = vulkan::pass;
    namespace rr = vulkan::render_resource;

    /// the declaration every fake pass carries, renamed to the pass it belongs to
    rr::pass_io named_io(std::string_view const name) {
        return rr::pass_io{
            .name = name,
            .own_set = 1,
            .bindings = rr::gi_probe_bindings, // a declaration the schema accepts, reused rather than invented
            .push = std::nullopt,
        };
    }

    /// the host's own state: the log every callback writes, plus what the fake host answers
    struct host_state {
        std::vector<std::string> log;
        vp::frame_identity frame = {.image_index = 3, .slot = 1, .extent = {640, 480}};
        std::vector<std::string_view> active_features = {"gi"};
        std::string_view failing_pass = {}; // resolve returns false for this pass's name
        std::array<vp::resolved_binding, 9> own = {};
        std::array<VkPipeline, 2> pipelines = {reinterpret_cast<VkPipeline>(0x1), reinterpret_cast<VkPipeline>(0x2)};
    };

    vp::frame_identity host_frame(void* ctx) {
        return static_cast<host_state*>(ctx)->frame;
    }

    bool host_feature_active(void* ctx, std::string_view const feature) {
        auto const& active = static_cast<host_state*>(ctx)->active_features;
        return std::find(active.begin(), active.end(), feature) != active.end();
    }

    bool host_resolve(void* ctx, vp::frame_pass const& pass, vp::resolved_io& out) {
        auto& state = *static_cast<host_state*>(ctx);
        state.log.emplace_back(std::string("resolve:") + std::string(pass.io().name));
        if (pass.io().name == state.failing_pass) {
            return false;
        }
        out.frame = state.frame;
        out.own = state.own;
        out.pipelines = state.pipelines;
        return true;
    }

    void host_apply_behaviour(void* ctx, vp::frame_pass const& pass, vp::resolved_io const&) {
        auto& state = *static_cast<host_state*>(ctx);
        state.log.emplace_back(std::string("behaviour:") + std::string(pass.io().name) +
                               (pass.behaviour().resync_viewport ? ":resync" : ":plain"));
    }

    void host_mark_begin(void* ctx, std::string_view const name) {
        static_cast<host_state*>(ctx)->log.emplace_back(std::string("mark_begin:") + std::string(name));
    }

    void host_mark_end(void* ctx, std::string_view const name) {
        static_cast<host_state*>(ctx)->log.emplace_back(std::string("mark_end:") + std::string(name));
    }

    vp::pass_host make_host(host_state& state) {
        return vp::pass_host{
            .context = &state,
            .frame = host_frame,
            .feature_active = host_feature_active,
            .resolve = host_resolve,
            .apply_behaviour = host_apply_behaviour,
            .mark_begin = host_mark_begin,
            .mark_end = host_mark_end,
        };
    }

    /// position of an entry in the log, or a large number when it is absent (so ordering asserts read clearly)
    std::size_t at(std::vector<std::string> const& log, std::string_view const entry) {
        auto const it = std::find(log.begin(), log.end(), entry);
        return it == log.end() ? std::size_t{9999} : static_cast<std::size_t>(it - log.begin());
    }

    bool has(std::vector<std::string> const& log, std::string_view const entry) {
        return at(log, entry) != 9999;
    }

    /// a declaration the schema must refuse: a binding whose resource was never set
    rr::pass_io const& bad_io() {
        static constexpr std::array<rr::pass_binding, 1> bindings = {{
            {.set = 1, .binding = 0, .owner = rr::set_owner::own, .kind = rr::binding_kind::sampled_image, .resource = rr::resource_id::none, .sampler = rr::sampler_hint::post},
        }};
        static constexpr rr::pass_io io = {.name = "bad", .own_set = 1, .bindings = bindings, .push = std::nullopt};
        return io;
    }

    /// a pass that records what the runner did to it, in the order it did it
    class fake_pass final : public vp::frame_pass {
    public:
        fake_pass(rr::pass_io io, vp::behaviour const behaviour, std::string_view const feature, host_state& state)
            : io_(io)
            , behaviour_(behaviour)
            , feature_(feature)
            , state_(&state) {
        }

        [[nodiscard]] rr::pass_io const& io() const noexcept override {
            return io_;
        }
        [[nodiscard]] vp::behaviour const& behaviour() const noexcept override {
            return behaviour_;
        }
        [[nodiscard]] std::string_view feature() const noexcept override {
            return feature_;
        }
        void create(vp::pass_host const&) override {
            state_->log.emplace_back(std::string("create:") + std::string(io_.name));
        }
        void on_swapchain_recreated(vp::pass_host const&) override {
            state_->log.emplace_back(std::string("recreate:") + std::string(io_.name));
        }
        void record(vp::resolved_io const& io) const override {
            state_->log.emplace_back(std::string("record:") + std::string(io_.name));
            last_image_index = io.frame.image_index;
            last_slot = io.frame.slot;
            last_pipeline_count = io.pipelines.size();
        }

        mutable uint32_t last_image_index = 0;
        mutable uint32_t last_slot = 0;
        mutable std::size_t last_pipeline_count = 0;

    private:
        rr::pass_io io_;
        vp::behaviour behaviour_;
        std::string_view feature_;
        host_state* state_;
    };

    constexpr vp::behaviour compute_behaviour = {.kind = vp::behaviour_kind::compute, .group_size_x = 4, .group_size_y = 4, .group_size_z = 4, .extent = vp::extent_rule::resource};
    constexpr vp::behaviour fullscreen_behaviour = {.kind = vp::behaviour_kind::fullscreen, .pipelines = 2, .resync_viewport = true};
} // namespace

int main() {
    using namespace vulkan::pass;

    host_state state;
    pass_host const host = make_host(state);

    fake_pass probe{named_io("probe"), compute_behaviour, "gi", state};
    fake_pass tail{named_io("tail"), fullscreen_behaviour, "gi", state};
    fake_pass gated{named_io("gated"), compute_behaviour, "off", state};
    fake_pass bad{bad_io(), compute_behaviour, "gi", state};

    // ---- build: every declaration is validated first, and a bad one refuses the whole stage ----
    {
        std::array<frame_pass*, 2> passes = {&probe, &tail};
        stage const st = {.name = "scene", .passes = passes};
        state.log.clear();
        run_report const report = create_stage(st, host);
        CHECK(report.created == 2);
        CHECK(report.rejected.empty());
        CHECK(at(state.log, "create:probe") < at(state.log, "create:tail")); // declaration order, not container order
    }
    {
        std::array<frame_pass*, 2> passes = {&probe, &bad};
        stage const st = {.name = "scene", .passes = passes};
        state.log.clear();
        run_report const report = create_stage(st, host);
        CHECK(report.rejected == "bad"); // the pass's own name, for a startup message that says which one
        CHECK(report.created == 1);      // and the stage stops there rather than running a bad declaration
        CHECK(!has(state.log, "create:bad"));
    }
    {
        std::array<frame_pass*, 3> passes = {&probe, nullptr, &tail};
        stage const st = {.name = "scene", .passes = passes};
        state.log.clear();
        CHECK(create_stage(st, host).created == 2); // a null slot is skipped, not counted
    }

    // ---- record: the order of resolve, behaviour and record, and the stage's mark around all of it ----
    {
        std::array<frame_pass*, 2> passes = {&probe, &tail};
        stage const st = {.name = "scene", .passes = passes};
        state.log.clear();
        run_report const report = record_stage(st, host);
        CHECK(report.recorded == 2);
        CHECK(report.skipped_inactive == 0);
        CHECK(report.marked == 1);
        CHECK(state.log.front() == "mark_begin:scene"); // the stage owns the mark: first and last
        CHECK(state.log.back() == "mark_end:scene");
        // per pass: resolve, then the behaviour's mechanical part, then record
        CHECK(at(state.log, "resolve:probe") < at(state.log, "behaviour:probe:plain"));
        CHECK(at(state.log, "behaviour:probe:plain") < at(state.log, "record:probe"));
        CHECK(at(state.log, "record:probe") < at(state.log, "resolve:tail"));
        // ... and the behaviour is passed through, not interpreted: the fullscreen pass asked for a resync
        CHECK(has(state.log, "behaviour:tail:resync"));
        // the pass saw the frame the host declared, and its declared pipeline count
        CHECK(probe.last_image_index == 3);
        CHECK(probe.last_slot == 1);
        CHECK(tail.last_pipeline_count == 2);
    }

    // ---- an inactive feature is skipped WITHOUT being resolved: what makes an off feature byte-exact ----
    {
        std::array<frame_pass*, 2> passes = {&probe, &gated};
        stage const st = {.name = "scene", .passes = passes};
        state.log.clear();
        run_report const report = record_stage(st, host);
        CHECK(report.recorded == 1);
        CHECK(report.skipped_inactive == 1);
        CHECK(!has(state.log, "resolve:gated")); // not resolved...
        CHECK(!has(state.log, "record:gated"));  // ... and not recorded
    }

    // ---- a frame that cannot resolve a pass skips it, and does not apply its behaviour either ----
    {
        std::array<frame_pass*, 2> passes = {&probe, &tail};
        stage const st = {.name = "scene", .passes = passes};
        state.log.clear();
        state.failing_pass = "tail";
        run_report const report = record_stage(st, host);
        state.failing_pass = {};
        CHECK(report.recorded == 1);
        CHECK(report.skipped_unresolved == 1);
        CHECK(!has(state.log, "behaviour:tail:resync")); // a pass recorded with unresolved handles is worse
        CHECK(!has(state.log, "record:tail"));
    }

    // ---- a stage that nests inside another's instance can decline its own mark pair ----
    {
        std::array<frame_pass*, 1> passes = {&probe};
        stage const st = {.name = "nested", .passes = passes, .marks = false};
        state.log.clear();
        run_report const report = record_stage(st, host);
        CHECK(report.marked == 0);
        CHECK(!has(state.log, "mark_begin:nested"));
        CHECK(has(state.log, "record:probe"));
    }

    // ---- a rebuilt swapchain tells EVERY pass: the hazard the manual reset list keeps missing ----
    {
        std::array<frame_pass*, 3> passes = {&probe, &tail, &gated};
        stage const st = {.name = "scene", .passes = passes};
        state.log.clear();
        run_report const report = recreate_stage(st, host);
        CHECK(report.recreated == 3);
        CHECK(has(state.log, "recreate:probe"));
        CHECK(has(state.log, "recreate:tail"));
        CHECK(has(state.log, "recreate:gated")); // even the pass this frame skipped
    }

    return vk_test::finish("test_pass");
}
