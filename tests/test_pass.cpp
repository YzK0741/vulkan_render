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
import vulkan.bindings;

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

    /// fake handles, so a resolved pass can be told apart from an unresolved one without a device
    /// (not `constexpr`: a handle comes from `reinterpret_cast`, which is not a constant expression)
    VkCommandBuffer const fake_cmd = reinterpret_cast<VkCommandBuffer>(0xC0);
    VkDescriptorSet const fake_own_set = reinterpret_cast<VkDescriptorSet>(0x0F);
    VkDescriptorSet const fake_scene = reinterpret_cast<VkDescriptorSet>(0x5E);
    std::array<VkPipeline, 4> const fake_pipelines = {
        reinterpret_cast<VkPipeline>(0x1), reinterpret_cast<VkPipeline>(0x2), reinterpret_cast<VkPipeline>(0x3), reinterpret_cast<VkPipeline>(0x4)};

    /// the host's own state: the log every callback writes, plus what the fake host answers
    struct host_state {
        std::vector<std::string> log;
        vp::frame_identity frame = {.image_index = 3, .slot = 1, .extent = {640, 480}};
        std::vector<std::string_view> active_features = {"gi"};
        std::string_view failing_pass = {}; // resolve returns false for this pass's name
        std::array<vp::resolved_binding, 9> own = {};
    };

    vp::frame_identity host_frame(void* ctx) {
        return static_cast<host_state*>(ctx)->frame;
    }

    bool host_feature_active(void* ctx, std::string_view const feature) {
        auto const& active = static_cast<host_state*>(ctx)->active_features;
        return std::find(active.begin(), active.end(), feature) != active.end();
    }

    /// the resolver a real runtime would write: the extent rule is applied HERE, from the declaration
    bool host_resolve(void* ctx, vp::frame_pass const& pass, vp::resolved_io& out) {
        auto& state = *static_cast<host_state*>(ctx);
        state.log.emplace_back(std::string("resolve:") + std::string(pass.io().name));
        if (pass.io().name == state.failing_pass) {
            return false;
        }
        vp::behaviour const& behaviour = pass.behaviour();
        out.frame = state.frame;
        out.cmd = fake_cmd;
        out.own = state.own;
        out.own_set = fake_own_set;
        out.shared.scene = fake_scene;
        out.pipelines = std::span<VkPipeline const>(fake_pipelines.data(), behaviour.pipelines.size());
        out.extent = behaviour.extent == vp::extent_rule::half ? VkExtent2D{state.frame.extent.width / 2u, state.frame.extent.height / 2u} : state.frame.extent;
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
            last_cmd = io.cmd;
            last_own_set = io.own_set;
            last_scene_set = io.shared.scene;
            last_pipelines = io.pipelines.size();
            last_extent = io.extent;
            last_image_index = io.frame.image_index;
            last_slot = io.frame.slot;
        }

        mutable VkCommandBuffer last_cmd = VK_NULL_HANDLE;
        mutable VkDescriptorSet last_own_set = VK_NULL_HANDLE;
        mutable VkDescriptorSet last_scene_set = VK_NULL_HANDLE;
        mutable std::size_t last_pipelines = 0;
        mutable VkExtent2D last_extent = {0, 0};
        mutable uint32_t last_image_index = 0;
        mutable uint32_t last_slot = 0;

    private:
        rr::pass_io io_;
        vp::behaviour behaviour_;
        std::string_view feature_;
        host_state* state_;
    };

    // the names are `vulkan.runtime`'s own pipeline keys, which is what makes this cost nothing new
    constexpr std::array<std::string_view, 1> compute_pipeline_names = {"gi_probe"};
    constexpr std::array<std::string_view, 2> fullscreen_pipeline_names = {"post_composite", "fxaa"};
    constexpr vp::behaviour compute_behaviour = {.kind = vp::behaviour_kind::compute, .group_size_x = 4, .group_size_y = 4, .group_size_z = 4, .extent = vp::extent_rule::resource, .pipelines = compute_pipeline_names};
    constexpr vp::behaviour fullscreen_behaviour = {.kind = vp::behaviour_kind::fullscreen, .extent = vp::extent_rule::half, .pipelines = fullscreen_pipeline_names, .resync_viewport = true};
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
    }

    // ---- what a pass is GIVEN, which is the reason it needs no device state of its own ----
    {
        std::array<frame_pass*, 2> passes = {&probe, &tail};
        stage const st = {.name = "scene", .passes = passes};
        state.log.clear();
        run_report const given = record_stage(st, host);
        CHECK(given.recorded == 2);
        CHECK(probe.last_cmd == fake_cmd);         // the command buffer is handed out per frame...
        CHECK(probe.last_own_set == fake_own_set); // ... with the pass's own resolved descriptor set...
        CHECK(probe.last_scene_set == fake_scene); // ... and the shared sets it declared usage of
        CHECK(probe.last_image_index == 3);        // ... and both frame counters, kept apart
        CHECK(probe.last_slot == 1);
        CHECK(probe.last_pipelines == 1);      // one declared name -> one resolved pipeline
        CHECK(tail.last_pipelines == 2);       // two names -> two, in the declared order
        CHECK(probe.last_extent.width == 640); // extent_rule::resource, with a fake host that hands the frame's
        CHECK(tail.last_extent.width == 320);  // extent_rule::half is half of it, applied by the resolver
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

    // ---- the declaration -> Vulkan mapping, which is what the generator builds a layout from ----
    // These live in this test rather than in test_render_resources because they are Vulkan-typed: the
    // description layer itself stays pure CPU, and everything that has to name a VkDescriptorType lives on the
    // bindings side. No device is created - an enum mapping needs none - so this still runs in CI.
    {
        using namespace vulkan::bindings;
        CHECK(descriptor_type_of(rr::binding_kind::sampled_image) == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        CHECK(descriptor_type_of(rr::binding_kind::storage_image) == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        CHECK(descriptor_type_of(rr::binding_kind::sampler) == VK_DESCRIPTOR_TYPE_SAMPLER);
        CHECK(descriptor_type_of(rr::binding_kind::uniform_buffer) == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        CHECK(descriptor_type_of(rr::binding_kind::storage_buffer) == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        CHECK(descriptor_type_of(rr::binding_kind::input_attachment) == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        CHECK(descriptor_type_of(rr::binding_kind::acceleration_structure) == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
        CHECK(stage_flags_of(rr::stage_flag::compute) == VK_SHADER_STAGE_COMPUTE_BIT);
        CHECK(stage_flags_of(rr::stage_flag::fragment | rr::stage_flag::compute) == (VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT));
        CHECK(stage_flags_of(rr::stage_flag::none) == 0u);
        // the layout mapping, and the sampler CHOICE a declaration makes instead of a handle
        CHECK(image_layout_of(rr::image_layout::sampled) == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        CHECK(image_layout_of(rr::image_layout::general) == VK_IMAGE_LAYOUT_GENERAL);
        sampler_set const samplers = {.gbuffer = reinterpret_cast<VkSampler>(0x11), .probe_grid = reinterpret_cast<VkSampler>(0x22)};
        CHECK(samplers.of(rr::sampler_hint::probe_grid) == reinterpret_cast<VkSampler>(0x22));
        CHECK(samplers.of(rr::sampler_hint::gbuffer) == reinterpret_cast<VkSampler>(0x11));
        CHECK(samplers.of(rr::sampler_hint::none) == VK_NULL_HANDLE); // "no sampler", which the validator enforces
        // the probe declaration's nine own bindings all declare GENERAL: both ping-pong sides and the geometry
        // stay in that layout for the whole update, which is what makes the propagation's barriers same-layout
        for (rr::pass_binding const& b : rr::gi_probe_io.bindings) {
            if (b.set == rr::gi_probe_io.own_set) {
                CHECK(b.layout == rr::image_layout::general);
            }
        }
        // the probe declaration's own set is exactly the nine bindings its shader declares, in order, and the
        // generated layout is what `pipelines::build_gi_probe` now builds from them
        uint32_t own = 0;
        for (rr::pass_binding const& b : rr::gi_probe_io.bindings) {
            if (b.set == rr::gi_probe_io.own_set) {
                CHECK(b.binding == own); // contiguous from zero: the index IS the binding number
                ++own;
            }
        }
        CHECK(own == 9);
    }

    return vk_test::finish("test_pass");
}
