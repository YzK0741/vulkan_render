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
        b.kind = rr::binding_kind::storage_image;
        b.resource = rr::resource_id::gi_history;
        b.access = rr::binding_access::write;
        b.sampler = rr::sampler_hint::none;
        // ... and the LAYOUT is not free either: a storage image's descriptor must declare GENERAL, and a
        // declaration that leaves it at the default `sampled` describes an image the pass may not write
        CHECK(!validate_one(b).has_value());
    }
    {
        rr::pass_binding b = good_binding;
        b.layout = rr::image_layout::general; // a SAMPLED image may declare GENERAL (the probe grid does)
        CHECK(validate_one(b).has_value());
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

    // ---- the render TARGETS: an attachment is a use that cannot be a descriptor, so it is declared here ----
    rr::render_target const hdr_target = {.resource = rr::resource_id::hdr, .element = 0};
    rr::render_target const no_target = {.resource = rr::resource_id::none, .element = 0};
    rr::render_target const past_the_family = {.resource = rr::resource_id::hdr, .element = 4};
    rr::render_target const not_an_image = {.resource = rr::resource_id::light_ubo, .element = 0};
    {
        std::array<rr::render_target, 1> const target = {hdr_target};
        rr::pass_io const io = {.name = "taa", .own_set = 1, .bindings = {}, .targets = target, .push = std::nullopt};
        CHECK(rr::validate(io).has_value()); // the frame's HDR target, written by a fullscreen resolve
    }
    {
        std::array<rr::render_target, 1> const target = {no_target};
        rr::pass_io const io = {.name = "taa", .own_set = 1, .bindings = {}, .targets = target, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value()); // an unset resource, exactly as for a binding
    }
    {
        std::array<rr::render_target, 1> const target = {past_the_family};
        rr::pass_io const io = {.name = "taa", .own_set = 1, .bindings = {}, .targets = target, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value()); // element 4 of a per-swapchain-image family that holds one
    }
    {
        std::array<rr::render_target, 1> const target = {not_an_image};
        rr::pass_io const io = {.name = "taa", .own_set = 1, .bindings = {}, .targets = target, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value()); // a buffer is not an image a pass can render into
    }
    {
        std::array<rr::render_target, 2> const target = {hdr_target, hdr_target};
        rr::pass_io const io = {.name = "taa", .own_set = 1, .bindings = {}, .targets = target, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value()); // the same image twice
    }
    {
        // a DEPTH target: the scene pass declares one, and an instance has exactly one
        rr::render_target const depth_target = {.resource = rr::resource_id::gbuffer_depth, .element = 0, .kind = rr::target_kind::depth};
        std::array<rr::render_target, 2> const one_depth = {hdr_target, depth_target};
        rr::pass_io const io = {.name = "scene", .own_set = 1, .bindings = {}, .targets = one_depth, .push = std::nullopt};
        CHECK(rr::validate(io).has_value()); // one colour plus one depth is what a scene instance is
    }
    {
        rr::render_target const depth_a = {.resource = rr::resource_id::gbuffer_depth, .element = 0, .kind = rr::target_kind::depth};
        rr::render_target const depth_b = {.resource = rr::resource_id::shadow_map, .element = 0, .kind = rr::target_kind::depth};
        std::array<rr::render_target, 2> const two_depths = {depth_a, depth_b};
        rr::pass_io const io = {.name = "scene", .own_set = 1, .bindings = {}, .targets = two_depths, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value()); // two depth attachments cannot be recorded
    }

    // ---- the SECOND declaration: the TAA resolve, whose bindings are FRAGMENT and whose own set is 0 with no
    //      shared set beside it, and which is the first one to declare a render TARGET ----
    {
        CHECK(rr::validate(rr::taa_io).has_value());
        CHECK(rr::taa_io.own_set == 0);
        CHECK(rr::taa_io.bindings.size() == 4);
        for (rr::pass_binding const& b : rr::taa_io.bindings) {
            CHECK(b.set == rr::taa_io.own_set);
            CHECK(b.owner == rr::set_owner::own);
            CHECK(b.kind == rr::binding_kind::sampled_image);
            CHECK(b.sampler == rr::sampler_hint::taa);
            // THE STAGE FLAGS COME FROM THE DECLARATION, so a fragment binding declared compute would build a
            // layout the fragment stage cannot see: every declaration before this one was a compute pass and
            // took the default.
            CHECK(rr::has_stage(b.stages, rr::stage_flag::fragment));
            CHECK(!rr::has_stage(b.stages, rr::stage_flag::compute));
        }
        CHECK(rr::taa_io.targets.size() == 1);
        CHECK(rr::taa_io.targets[0].resource == rr::resource_id::hdr);
        CHECK(rr::taa_io.push->size == 32); // eight floats: the history flag, two weights, texel size, two depth terms
        CHECK(rr::taa_io.push->stages == rr::stage_flag::fragment);
        rr::descriptor_counts const own = rr::descriptor_counts_for(rr::taa_io, rr::taa_io.own_set);
        CHECK(own.sampled_image == 4);
        CHECK(own.total() == 4);
        // nothing is declared in a shared set: the resolve's four inputs are all its own
        CHECK(rr::descriptor_counts_for(rr::taa_io, 1).total() == 0);
    }

    // ---- the THIRD declaration: the scene pass, the first one with NO own bindings (it binds a whole shared
    //      set and writes the frame's surface) and the first with six targets including the DEPTH slot ----
    {
        CHECK(rr::validate(rr::scene_io).has_value());
        CHECK(rr::scene_io.bindings.empty()); // it owns no set: everything arrives through set 0
        CHECK(rr::scene_io.shared_sets.size() == 1);
        CHECK(rr::scene_io.shared_sets[0] == 0); // the shared scene set
        CHECK(rr::scene_io.targets.size() == 6);
        CHECK(rr::scene_io.targets[0].resource == rr::resource_id::gbuffer_targets);
        CHECK(rr::scene_io.targets[2].element == 2); // all three stored surface targets
        CHECK(rr::scene_io.targets[3].resource == rr::resource_id::velocity);
        CHECK(rr::scene_io.targets[4].resource == rr::resource_id::scene_color); // the frame decision (see the doc)
        CHECK(rr::scene_io.targets[5].resource == rr::resource_id::gbuffer_depth);
        CHECK(rr::scene_io.targets[5].kind == rr::target_kind::depth); // exactly one depth, as an instance wants
        std::size_t depth_targets = 0;
        for (rr::render_target const& t : rr::scene_io.targets) {
            if (t.kind == rr::target_kind::depth) {
                ++depth_targets;
            }
        }
        CHECK(depth_targets == 1);
        CHECK(!rr::scene_io.push.has_value()); // the per-leaf pushes belong to the leaves
        CHECK(rr::descriptor_counts_for(rr::scene_io, rr::scene_io.own_set).total() == 0);
    }
    {
        // a set cannot be both the pass's own and one it only binds
        std::array<uint32_t, 1> const conflicting = {1};
        rr::pass_io const io = {.name = "scene", .own_set = 1, .bindings = rr::gi_probe_bindings, .shared_sets = conflicting, .targets = {}, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value());
    }
    {
        std::array<uint32_t, 2> const twice = {0, 0};
        rr::pass_io const io = {.name = "scene", .own_set = 1, .bindings = {}, .shared_sets = twice, .targets = {}, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value()); // the same shared set twice
    }

    // ---- the FOURTH declaration: the SSGI tracer, which binds TWO shared sets and no own binding, and the
    //      first one whose IMAGES have to be named without being descriptors (see pass_io::barrier_images) ----
    {
        CHECK(rr::validate(rr::ssgi_trace_io).has_value());
        CHECK(rr::ssgi_trace_io.own_set != 0 && rr::ssgi_trace_io.own_set != 1); // it owns nothing anywhere
        CHECK(rr::ssgi_trace_io.bindings.empty());
        CHECK(rr::descriptor_counts_for(rr::ssgi_trace_io, rr::ssgi_trace_io.own_set).total() == 0); // nothing generated
        CHECK(rr::ssgi_trace_io.shared_sets.size() == 2);
        CHECK(rr::ssgi_trace_io.shared_sets[0] == 0); // the scene set
        CHECK(rr::ssgi_trace_io.shared_sets[1] == 1); // the G-buffer set: the second shared set any pass declares
        CHECK(rr::ssgi_trace_io.targets.empty());     // it dispatches; it renders into nothing
        CHECK(rr::ssgi_trace_io.push.has_value());
        CHECK(rr::ssgi_trace_io.push->size == 128); // the full guaranteed range, five vectors
        CHECK(rr::ssgi_trace_io.push->stages == rr::stage_flag::compute);
        // THE BARRIER IMAGES, in the order the pass indexes them (its record() documents what each slot is for)
        CHECK(rr::ssgi_trace_io.barrier_images.size() == 12);
        CHECK(rr::ssgi_trace_io.barrier_images[0].resource == rr::resource_id::gi_trace);
        CHECK(rr::ssgi_trace_io.barrier_images[1].resource == rr::resource_id::gi_spec_resolve);
        CHECK(rr::ssgi_trace_io.barrier_images[2].resource == rr::resource_id::gi_resolve);
        for (std::size_t i = 3; i < 11; ++i) {
            CHECK(rr::ssgi_trace_io.barrier_images[i].resource == rr::resource_id::probe_grid);
            CHECK(rr::ssgi_trace_io.barrier_images[i].element == i - 3);
        }
        CHECK(rr::ssgi_trace_io.barrier_images[11].resource == rr::resource_id::probe_surface);
    }
    {
        // the barrier-image rule the validator enforces: they must be images the schema declares, and a pass
        // indexes them by POSITION, so the same one twice is a declaration that cannot be read
        std::array<rr::barrier_image, 1> const not_an_image = {rr::barrier_image{.resource = rr::resource_id::camera_ubo, .element = 0}};
        rr::pass_io const io = {.name = "ssgi_trace", .own_set = 2, .bindings = {}, .shared_sets = {}, .targets = {}, .barrier_images = not_an_image, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value());
        std::array<rr::barrier_image, 2> const twice = {rr::barrier_image{.resource = rr::resource_id::gi_trace, .element = 0},
                                                        rr::barrier_image{.resource = rr::resource_id::gi_trace, .element = 0}};
        rr::pass_io const dup = {.name = "ssgi_trace", .own_set = 2, .bindings = {}, .shared_sets = {}, .targets = {}, .barrier_images = twice, .push = std::nullopt};
        CHECK(!rr::validate(dup).has_value());
    }

    return vk_test::finish("test_render_resources");
}
