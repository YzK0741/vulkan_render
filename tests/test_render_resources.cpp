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
        .resource = rr::resource_id::ml_history,
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
    CHECK(rr::resource_schema.size() == 30);
    CHECK(static_cast<uint32_t>(rr::resource_id::count_) == 31); // 30 families plus `none`
    CHECK(rr::find(rr::resource_id::none) == nullptr);
    CHECK(rr::find(rr::resource_id::furnace_cube) != nullptr);
    CHECK(rr::find(rr::resource_id::shadow_map)->count == 4);      // one layer per cascade, as core indexes it
    CHECK(rr::find(rr::resource_id::gbuffer_targets)->count == 3); // albedo, normal+roughness, material+AO
    // the bloom chain's FOUR levels are elements of one family (core::bloom_images is an array of four vectors),
    // and the count is what makes `element = 3` legal and `element = 4` refused - the contract the post chain's
    // declarations are written against, asserted where it lives rather than assumed by them
    CHECK(rr::find(rr::resource_id::bloom)->count == 4);
    {
        std::array<rr::render_target, 1> const last_level = {rr::render_target{.resource = rr::resource_id::bloom, .element = 3, .kind = rr::target_kind::color}};
        rr::pass_io const io = {.name = "bloom", .own_set = 0, .bindings = {}, .shared_sets = {}, .targets = last_level, .push = std::nullopt};
        CHECK(rr::validate(io).has_value()); // the deepest level a pass may render into
        std::array<rr::render_target, 1> const past_the_end = {rr::render_target{.resource = rr::resource_id::bloom, .element = 4, .kind = rr::target_kind::color}};
        rr::pass_io const bad = {.name = "bloom", .own_set = 0, .bindings = {}, .shared_sets = {}, .targets = past_the_end, .push = std::nullopt};
        CHECK(!rr::validate(bad).has_value()); // ... and the one past it, which names no image at all
    }
    CHECK(rr::find(rr::resource_id::top_level_structure)->kind == rr::resource_kind::accel_struct);
    CHECK(rr::find(rr::resource_id::swapchain_image)->lifetime == rr::resource_lifetime::imported);
    // the two scopes this project has been bitten by, asserted where they live
    CHECK(rr::find(rr::resource_id::shadow_map)->scope == rr::resource_scope::per_frame_slot);
    CHECK(rr::find(rr::resource_id::ml_history)->scope == rr::resource_scope::per_swapchain_image);
    CHECK(rr::find(rr::resource_id::furnace_cube)->scope == rr::resource_scope::device_wide);

    // The shared-set rule the validator enforces is checked below against a declaration built for the purpose.
    {
        // One of the SCENE set's bindings: something a pass reaches through a shared set rather than owning, which
        // is the shape the rule is about. (static: the declaration holds a span, so the array must outlive it.)
        static constexpr std::array<rr::pass_binding, 1> scene_binding = {{{.set = 0,
                                                                            .binding = 1,
                                                                            .owner = rr::set_owner::scene,
                                                                            .kind = rr::binding_kind::sampled_image,
                                                                            .resource = rr::resource_id::scene_textures,
                                                                            .access = rr::binding_access::read,
                                                                            .sampler = rr::sampler_hint::post}}};
        static constexpr std::array<rr::shared_set, 1> scene_set_list = {{{.family = 0}}};
        // a declaration that BINDS a shared set and NAMES none is refused ...
        rr::pass_io const unnamed = {.name = "set-like",
                                     .own_set = 1,
                                     .bindings = scene_binding,
                                     .shared_sets = {},
                                     .targets = {},
                                     .push = rr::push_block{.offset = 0, .size = 56, .stages = rr::stage_flag::compute}};
        auto const refused = rr::validate(unnamed);
        CHECK(!refused.has_value());
        // ... while the same declaration WITH the set is accepted
        rr::pass_io const named = {.name = "set-like",
                                   .own_set = 1,
                                   .bindings = scene_binding,
                                   .shared_sets = scene_set_list,
                                   .targets = {},
                                   .push = rr::push_block{.offset = 0, .size = 56, .stages = rr::stage_flag::compute}};
        CHECK(rr::validate(named).has_value());
    }

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
        b.resource = rr::resource_id::shadow_map;
        b.element = 4; // the family holds 0..3
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
        b.resource = rr::resource_id::ml_history;
        b.access = rr::binding_access::write;
        CHECK(!validate_one(b).has_value()); // ... and a storage image must not
    }
    {
        rr::pass_binding b = good_binding;
        b.kind = rr::binding_kind::storage_image;
        b.resource = rr::resource_id::ml_history;
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
        CHECK(rr::scene_io.shared_sets[0].family == 0); // the shared scene set
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
        std::array<rr::shared_set, 1> const conflicting = {{{.family = 1}}};
        std::array<rr::pass_binding, 1> const own_at_one = {{{.set = 1,
                                                              .binding = 0,
                                                              .owner = rr::set_owner::own,
                                                              .kind = rr::binding_kind::sampled_image,
                                                              .resource = rr::resource_id::ml_trace,
                                                              .access = rr::binding_access::read,
                                                              .sampler = rr::sampler_hint::gbuffer}}};
        rr::pass_io const io = {.name = "scene", .own_set = 1, .bindings = own_at_one, .shared_sets = conflicting, .targets = {}, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value());
    }
    {
        std::array<rr::shared_set, 2> const twice = {rr::shared_set{.family = 0}, rr::shared_set{.family = 0}};
        rr::pass_io const io = {.name = "scene", .own_set = 1, .bindings = {}, .shared_sets = twice, .targets = {}, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value()); // the same shared set twice
    }

    // The GI tracer's declaration was asserted here (two shared sets, no own binding, twelve barrier images).
    // The pass is gone, so what remains is the validator rule it exercised.
    {
        // the barrier-image rule the validator enforces: they must be images the schema declares, and a pass
        // indexes them by POSITION, so the same one twice is a declaration that cannot be read
        std::array<rr::barrier_image, 1> const not_an_image = {rr::barrier_image{.resource = rr::resource_id::camera_ubo, .element = 0}};
        rr::pass_io const io = {.name = "barrier-like", .own_set = 2, .bindings = {}, .shared_sets = {}, .targets = {}, .barrier_images = not_an_image, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value());
        std::array<rr::barrier_image, 2> const twice = {rr::barrier_image{.resource = rr::resource_id::ml_trace, .element = 0},
                                                        rr::barrier_image{.resource = rr::resource_id::ml_trace, .element = 0}};
        rr::pass_io const dup = {.name = "barrier-like", .own_set = 2, .bindings = {}, .shared_sets = {}, .targets = {}, .barrier_images = twice, .push = std::nullopt};
        CHECK(!rr::validate(dup).has_value());
    }

    // ---- the SIXTH declaration: the stochastic punctual lighting chain's temporal resolve ----
    // (the GI denoiser's declaration stood here; it went with that chain) - the stochastic punctual
    // lighting chain's resolve - is a declaration of its own (megalights_temporal_io) with the same shape, and
    // it is covered by the lighting chain's own tests rather than by a GI scenario.

    // The spatial filter's declaration was asserted here - the third pass on the shared-sets shape. It went with
    // the chain; the shape itself is still covered by the ray-traced shadow's declaration below.

    // ---- the EIGHTH declaration, and the first one OUTSIDE the chain: the ray-traced shadow, the same
    //      shared-sets shape as the tracer and the spatial filter but at the FRAME's resolution, and its one
    //      barrier image is a per-frame-slot resource rather than a per-swapchain-image family ----
    {
        CHECK(rr::validate(rr::rt_shadow_io).has_value());
        CHECK(rr::rt_shadow_io.bindings.empty()); // the camera, the light UBO and the TLAS are the scene set's
        CHECK(rr::rt_shadow_io.targets.empty());  // a compute pass
        CHECK(rr::rt_shadow_io.shared_sets.size() == 2);
        CHECK(rr::rt_shadow_io.shared_sets[0].family == 0); // the scene set: the top level structure lives at binding 16
        CHECK(rr::rt_shadow_io.shared_sets[1].family == 1); // the G-buffer set: the surface each ray starts from
        CHECK(rr::rt_shadow_io.barrier_images.size() == 1);
        CHECK(rr::rt_shadow_io.barrier_images[0].resource == rr::resource_id::rt_shadow_visibility);
        CHECK(rr::find(rr::resource_id::rt_shadow_visibility)->scope == rr::resource_scope::per_frame_slot);
        CHECK(rr::rt_shadow_io.push.has_value());
        CHECK(rr::rt_shadow_io.push->size == 80); // inv_view_proj (64) + the four ray-offset terms (16)
        CHECK(rr::rt_shadow_io.push->stages == rr::stage_flag::compute);
        CHECK(rr::descriptor_counts_for(rr::rt_shadow_io, rr::rt_shadow_io.own_set).total() == 0);
    }

    // ---- the NINTH declaration: the clustered-light sort, the first pass whose resources are BUFFERS it
    //      orders without binding (they are the shared scene set's bindings 11 and 12) - which is what the
    //      barrier_buffers channel was added for ----
    {
        CHECK(rr::validate(rr::cluster_io).has_value());
        CHECK(rr::cluster_io.bindings.empty());       // it binds the whole shared scene set, so it enumerates nothing
        CHECK(rr::cluster_io.targets.empty());        // a compute pass
        CHECK(!rr::cluster_io.push.has_value());      // light_cluster.comp declares no push_constant block at all
        CHECK(rr::cluster_io.barrier_images.empty()); // it moves no image
        CHECK(rr::cluster_io.shared_sets.size() == 1);
        CHECK(rr::cluster_io.shared_sets[0].family == 0); // the scene set: the camera, the light UBO, the two buffers
        CHECK(rr::cluster_io.barrier_buffers.size() == 2);
        CHECK(rr::cluster_io.barrier_buffers[0].resource == rr::resource_id::cluster_counts);
        CHECK(rr::cluster_io.barrier_buffers[1].resource == rr::resource_id::cluster_indices);
        CHECK(rr::find(rr::resource_id::cluster_counts)->kind == rr::resource_kind::buffer);
        CHECK(rr::find(rr::resource_id::cluster_indices)->kind == rr::resource_kind::buffer);
        CHECK(rr::find(rr::resource_id::cluster_counts)->scope == rr::resource_scope::per_frame_slot);
        CHECK(rr::descriptor_counts_for(rr::cluster_io, rr::cluster_io.own_set).total() == 0);
    }
    {
        // the barrier-BUFFER rule, the same three checks one resource class over: it must be a buffer the
        // schema declares, and a pass indexes these by position so the same one twice cannot be read
        std::array<rr::barrier_buffer, 1> const not_a_buffer = {rr::barrier_buffer{.resource = rr::resource_id::ml_trace, .element = 0}};
        rr::pass_io const io = {.name = "cluster", .own_set = 1, .bindings = {}, .shared_sets = {}, .targets = {}, .barrier_buffers = not_a_buffer, .push = std::nullopt};
        CHECK(!rr::validate(io).has_value());
        std::array<rr::barrier_buffer, 2> const twice = {rr::barrier_buffer{.resource = rr::resource_id::cluster_counts, .element = 0},
                                                         rr::barrier_buffer{.resource = rr::resource_id::cluster_counts, .element = 0}};
        rr::pass_io const dup = {.name = "cluster", .own_set = 1, .bindings = {}, .shared_sets = {}, .targets = {}, .barrier_buffers = twice, .push = std::nullopt};
        CHECK(!rr::validate(dup).has_value());
        // ... and an element the family does not hold is rejected too (the schema's count is the contract)
        std::array<rr::barrier_buffer, 1> const out_of_range = {rr::barrier_buffer{.resource = rr::resource_id::cluster_counts, .element = 9}};
        rr::pass_io const bad_element = {.name = "cluster", .own_set = 1, .bindings = {}, .shared_sets = {}, .targets = {}, .barrier_buffers = out_of_range, .push = std::nullopt};
        CHECK(!rr::validate(bad_element).has_value());
    }

    // ---- the TENTH declaration: the deferred lighting stage, a fullscreen pass over the same two shared sets the
    //      traced compute passes bind, whose only resource of its own is the target it renders into - and whose target is a
    //      recorded deviation (it names scene_color; the host hands over hdr on the frames TAA is off) ----
    {
        CHECK(rr::validate(rr::deferred_io).has_value());
        CHECK(rr::deferred_io.bindings.empty()); // every binding it uses is in one of the two shared sets
        CHECK(rr::deferred_io.shared_sets.size() == 2);
        CHECK(rr::deferred_io.shared_sets[0].family == 0); // the scene set: camera, IBL, light UBO, shadow map
        CHECK(rr::deferred_io.shared_sets[1].family == 1); // the G-buffer set: the surface it shades
        CHECK(rr::deferred_io.targets.size() == 1);        // the only resource of its own: what it renders into
        CHECK(rr::deferred_io.targets[0].resource == rr::resource_id::scene_color);
        CHECK(rr::deferred_io.targets[0].kind == rr::target_kind::color);
        CHECK(rr::deferred_io.barrier_images.empty()); // it moves no image of its own
        CHECK(rr::deferred_io.push.has_value());
        CHECK(rr::deferred_io.push->size == 88);                         // mat4 + vec4 + one float
        CHECK(rr::deferred_io.push->stages == rr::stage_flag::fragment); // the vertex stage pushes nothing
        CHECK(rr::descriptor_counts_for(rr::deferred_io, rr::deferred_io.own_set).total() == 0);
    }

    // ---- the post chain's FIVE declarations: the bloom chain's four levels and the composite, all on the post
    //      shared set, with the level IS the pass boundary and the chain's edges declared where they are not the
    //      frame loop's ----
    {
        CHECK(rr::post_bloom_io.size() == 4);
        for (std::size_t level = 0; level < rr::post_bloom_io.size(); ++level) {
            rr::pass_io const& io = rr::post_bloom_io[level];
            auto const valid = rr::validate(io);
            CHECK_MSG(valid.has_value(), valid.has_value() ? "" : valid.error().c_str());
            CHECK(io.bindings.empty()); // everything it reads belongs to the post set
            CHECK(io.shared_sets.size() == 1);
            // EACH LEVEL'S OWN SET of the post family: family 2, and the element IS the level (the set that reads
            // the level before it) - which is the fact six declarations could not express before `shared_set`
            CHECK(io.shared_sets[0].family == 2 && io.shared_sets[0].element == level);
            CHECK(io.targets.size() == 1);
            CHECK(io.targets[0].resource == rr::resource_id::bloom); // the level it writes...
            CHECK(io.targets[0].element == level);                   // ... which IS the pass boundary
            CHECK(io.targets[0].kind == rr::target_kind::color);
            // the level it READS is declared exactly when it has to move it: level 0's input is the HDR target,
            // whose transition is the frame loop's (the composite reads it too, and on a bloom-off frame nobody
            // else does), and levels 1..3 own the transition of the level before them
            CHECK((level == 0 ? io.barrier_images.empty() : io.barrier_images.size() == 1));
            if (level > 0) {
                CHECK(io.barrier_images[0].resource == rr::resource_id::bloom);
                CHECK(io.barrier_images[0].element == level - 1);
            }
            CHECK(io.push.has_value());
            CHECK(io.push->size == rr::post_push_bytes); // the chain's ONE block, all 52 bytes of it
            CHECK(io.push->stages == rr::stage_flag::fragment);
        }

        auto const composite = rr::validate(rr::post_composite_io);
        CHECK_MSG(composite.has_value(), composite.has_value() ? "" : composite.error().c_str());
        CHECK(rr::post_composite_io.shared_sets.size() == 1);
        CHECK(rr::post_composite_io.shared_sets[0].family == 2 && rr::post_composite_io.shared_sets[0].element == 4);
        CHECK(rr::post_composite_io.targets.size() == 1);
        // the RECORDED DEVIATION: the declaration names the swapchain, and the host hands over the LDR image (and
        // the R16F pipeline that goes with it) on the frames FXAA runs
        CHECK(rr::post_composite_io.targets[0].resource == rr::resource_id::swapchain_image);
        CHECK(rr::post_composite_io.barrier_images.empty()); // the HDR transition is the frame loop's, on every frame
        CHECK(rr::post_composite_io.push.has_value());
        CHECK(rr::post_composite_io.push->size == rr::post_push_bytes);
    }

    // ---- the FOURTEENTH declaration: the FXAA pass, the frame's LAST writer whenever it runs ----
    {
        auto const fxaa = rr::validate(rr::fxaa_io);
        CHECK_MSG(fxaa.has_value(), fxaa.has_value() ? "" : fxaa.error().c_str());
        CHECK(rr::fxaa_io.bindings.empty()); // everything it reads belongs to the post set
        CHECK(rr::fxaa_io.shared_sets.size() == 1);
        CHECK(rr::fxaa_io.shared_sets[0].family == 2 && rr::fxaa_io.shared_sets[0].element == 4); // the SAME post set the composite binds
        CHECK(rr::fxaa_io.targets.size() == 1);
        CHECK(rr::fxaa_io.targets[0].resource == rr::resource_id::swapchain_image); // it finishes the frame
        CHECK(rr::fxaa_io.targets[0].kind == rr::target_kind::color);
        // ITS INPUT IS DECLARED, unlike the composite's, and the difference is the frame: the LDR image is written
        // by the composite and read here, and on a frame without FXAA nobody touches it at all - so this pass is
        // the one that moves it, and there is no "nobody ran" case to hand over to the frame loop
        CHECK(rr::fxaa_io.barrier_images.size() == 1);
        CHECK(rr::fxaa_io.barrier_images[0].resource == rr::resource_id::ldr);
        CHECK(rr::fxaa_io.push.has_value());
        CHECK(rr::fxaa_io.push->size == rr::post_push_bytes); // FXAA is mode 3 of the chain's one shader block
        CHECK(rr::fxaa_io.push->stages == rr::stage_flag::fragment);
    }

    // ---- the FIFTEENTH declaration: the G-buffer debug view - the stored surface, one channel at a time ----
    {
        auto const debug = rr::validate(rr::gbuffer_debug_io);
        CHECK_MSG(debug.has_value(), debug.has_value() ? "" : debug.error().c_str());
        CHECK(rr::gbuffer_debug_io.bindings.empty()); // everything it reads is in the shared G-buffer set
        CHECK(rr::gbuffer_debug_io.shared_sets.size() == 1);
        CHECK(rr::gbuffer_debug_io.shared_sets[0].family == 1); // the G-buffer set (1), NOT the scene's or the post's
        // THE HDR TARGET, declared as itself: this is the one graphics declaration in the chain with no deviation -
        // the image it writes is the image it names
        CHECK(rr::gbuffer_debug_io.targets.size() == 1);
        CHECK(rr::gbuffer_debug_io.targets[0].resource == rr::resource_id::hdr);
        CHECK(rr::gbuffer_debug_io.targets[0].kind == rr::target_kind::color);
        // the four images it moves to a sampled layout: the three stored targets and the motion vectors. The DEPTH
        // is deliberately NOT here - its old layout depends on whether the G-buffer instance rendered this frame,
        // which is the host's per-image bookkeeping (the accessor the two other sampling stages share)
        CHECK(rr::gbuffer_debug_io.barrier_images.size() == 4);
        CHECK(rr::gbuffer_debug_io.barrier_images[0].resource == rr::resource_id::gbuffer_targets);
        CHECK(rr::gbuffer_debug_io.barrier_images[0].element == 0);
        CHECK(rr::gbuffer_debug_io.barrier_images[1].element == 1);
        CHECK(rr::gbuffer_debug_io.barrier_images[2].element == 2);
        CHECK(rr::gbuffer_debug_io.barrier_images[3].resource == rr::resource_id::velocity);
        CHECK(rr::gbuffer_debug_io.push.has_value());
        CHECK(rr::gbuffer_debug_io.push->size == 16); // the channel, two projection terms, the motion gain
        CHECK(rr::gbuffer_debug_io.push->stages == rr::stage_flag::fragment);
    }

    // ---- the SIXTEENTH declaration: the shadow pass, one cascade at a time into a layer of the map array ----
    {
        auto const shadow = rr::validate(rr::shadow_io);
        std::string const shadow_error = shadow.has_value() ? std::string{} : shadow.error(); // a temporary`s c_str() would dangle
        CHECK_MSG(shadow.has_value(), shadow_error.c_str());
        CHECK(rr::shadow_io.bindings.empty()); // the scene set carries everything the depth-only draw reads
        CHECK(rr::shadow_io.shared_sets.size() == 1);
        CHECK(rr::shadow_io.shared_sets[0].family == 0); // the SCENE set (the light UBO, the material table, the textures)
        // ONE TARGET ENTRY, N ELEMENTS - the RUN of cascade layers, and this is what replaced "one target by
        // declaration, N by frame": the validator refuses a second DEPTH target (a rendering instance has exactly
        // one depth attachment), so the host used to hand the extra layers over itself. `render_target::count` is
        // the vocabulary that says what that was trying to say - ONE family, rendered one INSTANCE per element -
        // and the frame still caps it, because the map has exactly the layers the cascade knob asked for.
        CHECK(rr::shadow_io.targets.size() == 1);
        CHECK(rr::shadow_io.targets[0].resource == rr::resource_id::shadow_map);
        CHECK(rr::shadow_io.targets[0].element == 0); // the first cascade; the run is consecutive from there
        CHECK(rr::shadow_io.targets[0].kind == rr::target_kind::depth);
        CHECK(rr::shadow_io.targets[0].count == 4); // every layer max_shadow_cascades allows
        // ... and the map family holds exactly that many layers: FOUR, like the image ensure_shadow_resources
        // creates (the count was 1 until this declaration, the same correction the bloom family needed)
        CHECK(rr::find(rr::resource_id::shadow_map)->count == 4);
        CHECK(rr::shadow_io.barrier_images.empty()); // each layer is moved through its own target
        CHECK(rr::shadow_io.push.has_value());
        CHECK(rr::shadow_io.push->size == 4);    // the cascade index
        CHECK(rr::shadow_io.push->offset == 96); // where the scene's own push block ends (scene_push_constant_size)
        CHECK(rr::shadow_io.push->stages == (rr::stage_flag::vertex | rr::stage_flag::fragment));

        // THE RUN'S OWN CHECKS: a run that reaches past its family is a declaration error ...
        std::array<rr::render_target, 1> const past_the_family = {rr::render_target{.resource = rr::resource_id::shadow_map, .element = 2, .kind = rr::target_kind::depth, .count = 4}};
        rr::pass_io const bad_run = {.name = "shadow", .own_set = 0, .bindings = {}, .shared_sets = {}, .targets = past_the_family, .push = std::nullopt};
        CHECK(!rr::validate(bad_run).has_value());
        // ... and so is claiming NO element, which would be a target that renders nothing ...
        std::array<rr::render_target, 1> const empty_run = {rr::render_target{.resource = rr::resource_id::shadow_map, .element = 0, .kind = rr::target_kind::depth, .count = 0}};
        rr::pass_io const no_run = {.name = "shadow", .own_set = 0, .bindings = {}, .shared_sets = {}, .targets = empty_run, .push = std::nullopt};
        CHECK(!rr::validate(no_run).has_value());
        // ... and two runs of ONE resource may not overlap, which is "rendering into one image twice" one layer
        // out. (Two depth runs would be refused by the one-depth rule first, so this case is a colour family's.)
        std::array<rr::render_target, 2> const overlapping = {rr::render_target{.resource = rr::resource_id::bloom, .element = 0, .count = 3},
                                                              rr::render_target{.resource = rr::resource_id::bloom, .element = 2, .count = 2}};
        rr::pass_io const overlap = {.name = "post_hdr", .own_set = 0, .bindings = {}, .shared_sets = {}, .targets = overlapping, .push = std::nullopt};
        CHECK(!rr::validate(overlap).has_value());
        // ... while two runs that TOUCH but do not overlap are legal, which keeps the rule about the IMAGES a
        // declaration claims rather than about adjacency
        std::array<rr::render_target, 2> const adjacent = {rr::render_target{.resource = rr::resource_id::bloom, .element = 0, .count = 2},
                                                           rr::render_target{.resource = rr::resource_id::bloom, .element = 2, .count = 2}};
        rr::pass_io const touching = {.name = "post_hdr", .own_set = 0, .bindings = {}, .shared_sets = {}, .targets = adjacent, .push = std::nullopt};
        CHECK(rr::validate(touching).has_value());
    }
    return vk_test::finish("test_render_resources");
}
