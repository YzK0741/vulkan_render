// Headless unit tests: vulkan.render_resource (pure CPU) ======================
// The description layer exists so that a pass's resources are stated ONCE and the layout, the descriptor
// writes and the pool counts are generated from that statement instead of being kept in agreement by hand.
// These tests pin the invariants that make that worth doing, and they can do it without a device - which is
// the point of keeping the layer Vulkan-free: the capture gate cannot run in CI at all (its references are
// tied to one machine's driver), and this can.
//
// THE FAILURE CASES ARE THE TEST. A validator that only ever says "ok" would pass every check here, so each
// malformed declaration below is asserted to be REJECTED: an unset resource, a binding kind that does not
// fit the resource, an access the kind cannot perform, an element outside the family, a descriptor count of
// zero, a sampled image with no sampler (and a uniform buffer with one), a duplicated (set, binding), an own
// binding outside the pass's own set, a gap in the own bindings, and a push block that does not fit.
#include "vk_test.h"

#include <array>
#include <expected>
#include <span>
#include <string>

import vulkan.render_resource;

namespace {
    namespace rr = vulkan::render_resource;

    /// validate a declaration that holds exactly one binding, so a wrong one can be built in one line
    std::expected<void, std::string> validate_one(rr::pass_binding const& binding) {
        rr::pass_io const io = {
            .name = "probe",
            .own_set = 1,
            .bindings = std::span<rr::pass_binding const>(&binding, 1),
            .push = std::nullopt,
        };
        return rr::validate(io);
    }

    /// a well-formed own binding, the starting point of every malformed one below
    constexpr rr::pass_binding good_binding = {
        .set = 1,
        .binding = 0,
        .owner = rr::set_owner::own,
        .kind = rr::binding_kind::sampled_image,
        .resource = rr::resource_id::gi_history,
        .element = 0,
        .descriptor_count = 1,
        .access = rr::binding_access::read,
        .sampler = rr::sampler_hint::post,
        .stages = rr::stage_flag::compute,
    };
} // namespace

int main() {
    // ---- the schema describes itself, completely ----
    auto const schema = rr::validate_schema();
    CHECK_MSG(schema.has_value(), schema.has_value() ? "" : schema.error().c_str());
    CHECK(rr::resource_schema.size() == 37);
    CHECK(static_cast<uint32_t>(rr::resource_id::count_) == 38); // 37 families plus `none`
    CHECK(rr::find(rr::resource_id::none) == nullptr);
    CHECK(rr::find(rr::resource_id::probe_grid) != nullptr);
    CHECK(rr::find(rr::resource_id::probe_grid)->count == 8);      // side*4+coefficient, as core indexes it
    CHECK(rr::find(rr::resource_id::gbuffer_targets)->count == 3); // albedo, normal+roughness, material+AO
    CHECK(rr::find(rr::resource_id::top_level_structure)->kind == rr::resource_kind::accel_struct);
    CHECK(rr::find(rr::resource_id::swapchain_image)->lifetime == rr::resource_lifetime::imported);
    // the two scopes this project has been bitten by, asserted where they live
    CHECK(rr::find(rr::resource_id::shadow_map)->scope == rr::resource_scope::per_frame_slot);
    CHECK(rr::find(rr::resource_id::gi_history)->scope == rr::resource_scope::per_swapchain_image);
    CHECK(rr::find(rr::resource_id::probe_grid)->scope == rr::resource_scope::device_wide);

    // ---- the probe cache's declaration, read off shaders/gi_probe.comp ----
    auto const probe = rr::validate(rr::gi_probe_io);
    CHECK_MSG(probe.has_value(), probe.has_value() ? "" : probe.error().c_str());
    CHECK(rr::gi_probe_io.bindings.size() == 16); // 9 own (set 1) + 7 shared (set 0)
    // the own set is 4 sampled read + 5 storage write = the nine bindings the shader declares
    rr::descriptor_counts const own = rr::descriptor_counts_for(rr::gi_probe_io, 1);
    CHECK(own.sampled_image == 4);
    CHECK(own.storage_image == 5);
    CHECK(own.total() == 9);
    // ... and the shared scene set is what the shader uses of it, not what it owns
    rr::descriptor_counts const scene = rr::descriptor_counts_for(rr::gi_probe_io, 0);
    CHECK(scene.sampled_image == 4);
    CHECK(scene.storage_buffer == 1);
    CHECK(scene.uniform_buffer == 1);
    CHECK(scene.acceleration_structure == 1);
    CHECK(scene.total() == 7);
    // the ping-pong halves are ELEMENTS of one resource, which is how core indexes the family
    CHECK(rr::gi_probe_io.bindings[0].element == 0);
    CHECK(rr::gi_probe_io.bindings[4].element == 4);
    CHECK(rr::gi_probe_io.bindings[8].resource == rr::resource_id::probe_surface);

    // ---- the well-formed binding, so the failures below mean something ----
    CHECK(validate_one(good_binding).has_value());
    CHECK(rr::compatible(rr::binding_kind::storage_buffer, rr::resource_kind::buffer));
    CHECK(!rr::compatible(rr::binding_kind::storage_image, rr::resource_kind::buffer));
    CHECK(rr::has_stage(rr::stage_flag::compute | rr::stage_flag::fragment, rr::stage_flag::fragment));
    CHECK(!rr::has_stage(rr::stage_flag::compute, rr::stage_flag::fragment));
    CHECK(rr::name_of(rr::binding_kind::acceleration_structure) == "acceleration_structure");

    // ---- every malformed declaration is REJECTED (a validator that cannot fail is not one) ----
    {
        rr::pass_binding b = good_binding;
        b.resource = rr::resource_id::none;
        CHECK(!validate_one(b).has_value()); // an unset resource
    }
    {
        rr::pass_binding b = good_binding;
        b.kind = rr::binding_kind::sampled_image;
        b.resource = rr::resource_id::light_ubo; // a buffer bound as an image
        CHECK(!validate_one(b).has_value());
    }
    {
        rr::pass_binding b = good_binding;
        b.kind = rr::binding_kind::uniform_buffer;
        b.resource = rr::resource_id::light_ubo;
        b.access = rr::binding_access::write; // a uniform buffer cannot be written
        b.sampler = rr::sampler_hint::none;
        CHECK(!validate_one(b).has_value());
    }
    {
        rr::pass_binding b = good_binding;
        b.resource = rr::resource_id::probe_grid;
        b.element = 8; // the family holds 0..7
        CHECK(!validate_one(b).has_value());
    }
    {
        rr::pass_binding b = good_binding;
        b.sampler = rr::sampler_hint::none; // a sampled image must name one
        CHECK(!validate_one(b).has_value());
    }
    {
        rr::pass_binding b = good_binding;
        b.kind = rr::binding_kind::storage_image;
        b.resource = rr::resource_id::gi_history;
        b.access = rr::binding_access::write;
        CHECK(!validate_one(b).has_value()); // ... and a storage image must not
    }
    {
        rr::pass_binding b = good_binding;
        b.descriptor_count = 0;
        CHECK(!validate_one(b).has_value());
    }
    {
        rr::pass_binding b = good_binding;
        b.set = 2; // own bindings live in the pass's own set
        CHECK(!validate_one(b).has_value());
    }
    {
        rr::pass_binding b = good_binding;
        b.binding = 3; // ... numbered contiguously from zero
        CHECK(!validate_one(b).has_value());
    }
    {
        std::array<rr::pass_binding, 2> const dup = {good_binding, good_binding};
        rr::pass_io const io = {.name = "probe", .own_set = 1, .bindings = dup, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value()); // the same (set, binding) twice
    }
    {
        rr::pass_io const io = {.name = {}, .own_set = 1, .bindings = {}, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value()); // an unnamed pass
    }
    {
        rr::pass_io const io = {.name = "probe", .own_set = 1, .bindings = {}, .push = rr::push_block{.offset = 0, .size = 132, .stages = rr::stage_flag::compute}};
        CHECK(!rr::validate(io).has_value()); // past the 128-byte guaranteed minimum
    }
    {
        rr::pass_io const io = {.name = "probe", .own_set = 1, .bindings = {}, .push = rr::push_block{.offset = 0, .size = 6, .stages = rr::stage_flag::compute}};
        CHECK(!rr::validate(io).has_value()); // not a whole number of 4-byte lanes
    }

    return vk_test::finish("test_render_resources");
}
