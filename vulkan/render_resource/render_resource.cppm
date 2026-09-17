// module version: 0.12.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/render_resource/render_resource.cppm
 * @brief A pass's resources, described as DATA: what exists, and what each pass does with it.
 * @defgroup vulkan_render_resource Render Resource Descriptions
 *
 * WHY THIS MODULE EXISTS. In this renderer a pass's inputs and outputs already ARE descriptor sets - the
 * G-buffer set is the interface between the G-buffer pass, the lighting stage, the GI chain, the probe cache
 * and the ray-traced shadow pass; the scene set is the substrate; each pass's private family is its own I/O.
 * What exists today is that interface written TWICE BY HAND and kept in agreement by discipline: the layout
 * in `vulkan/pipelines`, the descriptor writes in `vulkan/runtime`'s `ensure_*_descriptors()`. Both drifts
 * that pair can have are already in this project's history, and both were found by the validation layer
 * rather than by review: a pool sized for four descriptors per set while the layout asked for five, and a
 * binding whose type changed without its writer noticing. One declaration, from which both are generated,
 * is what removes the pair.
 *
 * WHAT IT IS NOT, deliberately rather than unfinished:
 *
 *  * NOT a capability surface. Nothing here can reach a resource: no `core&`, no `runtime&`, no handles. It
 *    is copyable `constexpr` data with predicates over it - the opposite of a context object handed to code
 *    that must BIND things, which is the kind of object that grows as needs appear. A declaration cannot
 *    grow a capability.
 *  * NOT an allocator. Every image, buffer, view and sampler stays where it is: `vulkan.core` creates them,
 *    `vulkan.runtime` owns the pipelines. The schema says a resource EXISTS and what its scope and lifetime
 *    are; it does not say who makes it or how.
 *  * NOT Vulkan. The vocabulary is this module's own enums, so the description layer is pure CPU like
 *    `vulkan.math` and `gltf_loader` - which is what lets its invariants be tested in `ctest`, on a machine
 *    with no GPU. (The capture gate cannot run in CI at all: its references are tied to one machine's
 *    driver.) Mapping these enums onto `VkDescriptorType`/`VkShaderStageFlags` belongs to the generator,
 *    which imports Vulkan the way the pipeline builders already do.
 *
 * THE RULE THAT KEEPS THE TWO HALVES FROM DRIFTING: the SCHEMA owns what a resource IS (kind, scope,
 * lifetime, how many images the family holds); a PASS owns what it DOES with one (which set and binding,
 * read or write, which sampler, which stage). The only token they share is `resource_id`. A pass that
 * restates a resource's kind or scope would be a second copy of a fact - exactly the drift this module
 * exists to remove.
 *
 * OWN SETS VERSUS SHARED ONES, which the first real declaration made unavoidable: a pass with a private
 * family still binds the shared scene set (a probe cell's ray needs the top level structure, the material
 * table and the texture array to resolve and shade what it finds). Such a binding is declared here as USAGE -
 * which resource, which access - and `set_owner` says whose set it is; the set's LAYOUT and its descriptor
 * counts stay with that set's owner (`vulkan.bindings`' `scene_bindings`, and the G-buffer family in
 * `vulkan.runtime`), because a pass does not own them and must not be able to change them.
 */

module;

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>

export module vulkan.render_resource;

export namespace vulkan::render_resource {

    // =============================================================================================
    // 1. WHAT EXISTS - the schema. Declared here, implemented by vulkan.core.
    // =============================================================================================

    /**
     * @brief every resource family this renderer declares
     *
     * `none` is first on purpose: it is the value a binding holds before it is filled in, and the validator
     * rejects it, so a binding that was never given a resource cannot pass silently.
     *
     * THE ORDER OF THIS ENUM IS NOT A CONTRACT - unlike `gpu_mark_id`, whose order IS the timing contract and
     * whose 15 call sites must agree with it. Nothing indexes by this; it is an identity and nothing more.
     * That is why this list can live in one place without becoming the positional coupling that enum is.
     */
    enum class resource_id : uint32_t {
        none = 0,
        // ---- the swapchain image itself, owned outside ----
        swapchain_image,
        // ---- the HDR chain and the G-buffer (one per swapchain image) ----
        hdr,
        bloom,
        ldr,
        gbuffer_targets, // 3: albedo (RGBA8), normal + roughness (RGBA16F), material + baked AO (RGBA8)
        gbuffer_depth,
        velocity,
        scene_color,
        taa_history,
        // ---- the GI chain (one per swapchain image) ----
        gi_trace,
        gi_resolve,
        gi_history,
        gi_spatial,
        gi_spec_trace,
        gi_spec_reproject,
        gi_spec_resolve,
        gi_spec_history,
        // ---- the stochastic punctual lighting chain (one per swapchain image) ----
        ml_trace,
        ml_resolve,
        ml_history,
        // ---- one per frame slot ----
        shadow_map,
        rt_shadow_visibility,
        camera_ubo,
        light_ubo,
        material_table,
        instance_table,
        motion_vectors,
        skin_matrices,
        morph_targets,
        cluster_counts,
        cluster_indices,
        // ---- device wide ----
        probe_grid, // 8: four SH-2 coefficients per channel, two ping-pong sides, side*4+coefficient
        probe_surface,
        furnace_cube,
        scene_textures,
        white_texture,
        ibl_env,
        ibl_irradiance,
        brdf_lut,
        // ---- what every traced pass needs ----
        top_level_structure,
        /// the number of enumerators, so the schema's completeness can be checked by ITERATING rather than
        /// against a hand-kept list: `validate_schema()` requires exactly one entry per id in [1, count_)
        count_,
    };

    /** @brief what a resource IS, which decides the descriptor types it can be bound as */
    enum class resource_kind : uint8_t {
        image2d,
        image3d,
        image_cube,
        buffer,
        accel_struct,
    };

    /**
     * @brief which frame counter a resource is indexed by
     *
     * THIS IS THE FIELD THIS MODULE EXISTS FOR, and it is not a classification exercise: this project's
     * documented per-image-lifetime trap is the difference between the two, and getting it wrong has already
     * produced a real bug here (one `bool` guarding a resource that is one-per-swapchain-image, so every slot
     * after the first never received its first-use transition). A pass that uses a `per_frame_slot` resource
     * is handed the frame slot; one that uses a `per_swapchain_image` resource is handed the image index.
     */
    enum class resource_scope : uint8_t {
        per_frame_slot,      // shadow maps, the light/camera/material/instance buffers, the scene sets
        per_swapchain_image, // every GI image, the TAA history, the G-buffer, the HDR chain
        device_wide,         // the probe grid and its geometry, the furnace cube, the IBL cubes
    };

    /** @brief how long a resource lives: what decides whether a first-use transition or a clear exists */
    enum class resource_lifetime : uint8_t {
        per_frame,  // rewritten every frame by its owner
        persistent, // holds state ACROSS frames (histories, the accumulated grid) - see the reset sites
        imported,   // owned outside this renderer (the swapchain image)
    };

    /** @brief one family of resources: what it is, how it is indexed, and how many images it holds
     *  @ingroup vulkan_render_resource */
    struct resource_info {
        resource_id id = resource_id::none;
        std::string_view name = {};
        resource_kind kind = resource_kind::image2d;
        resource_scope scope = resource_scope::per_swapchain_image;
        resource_lifetime lifetime = resource_lifetime::per_frame;
        /// how many images the family holds: 3 G-buffer targets, 8 probe coefficients (side*4+coefficient,
        /// exactly how `core` indexes them), 1 for an ordinary image. A binding names one ELEMENT of this.
        uint16_t count = 1;
    };

    /**
     * @brief the schema itself
     *
     * EACH ENTRY'S PROVENANCE, so the table can be re-derived rather than trusted: the HDR chain, the
     * G-buffer, the velocity and scene-colour targets and the TAA history are created in
     * `core::create_render_targets` (`core.cpp:506-693`); the GI chain's images at `636-749`; the probe grid
     * and its geometry at `798-811`; the furnace cube at `814-825`; the ray-traced shadow visibility at
     * `831-844`, one per FRAME SLOT rather than per image; the shadow map is the one family `vulkan.runtime`
     * creates itself (`runtime.cpp:495-560`, layered, one image per slot); the scene buffers and the IBL
     * textures are `vulkan.runtime`'s (`runtime.cpp:148-231`, `163-166`); the top level structure belongs to
     * `vulkan.acceleration_structure` and reaches shaders through the scene set.
     *
     * WHAT AN ENTRY DOES NOT SAY YET, on purpose: its format and its extent. Those are needed by the CREATION
     * step, not by the invariants checkable today, and a format copied here before it is verified against
     * `core.cpp` would be a second copy of a fact - the thing this file's header warns about.
     * @ingroup vulkan_render_resource
     */
    inline constexpr std::array<resource_info, 40> resource_schema = {{
        {.id = resource_id::swapchain_image, .name = "swapchain_image", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::imported},
        {.id = resource_id::hdr, .name = "hdr", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        // FOUR LEVELS, not one: `core::bloom_images` is `std::array<std::vector<VkImage>, bloom_level_count>`
        // with `bloom_level_count == 4`, and the post chain's passes each own ONE of them. The count was 1 (the
        // default) until the bloom chain was extracted, which made `element = level` illegal for every pass but
        // the first - and the validator's `element >= info->count` check is what says so, so this field IS the
        // contract a declaration is written against.
        {.id = resource_id::bloom, .name = "bloom", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame, .count = 4},
        {.id = resource_id::ldr, .name = "ldr", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::gbuffer_targets, .name = "gbuffer_targets", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame, .count = 3},
        {.id = resource_id::gbuffer_depth, .name = "gbuffer_depth", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::velocity, .name = "velocity", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::scene_color, .name = "scene_color", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::taa_history, .name = "taa_history", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::persistent},
        {.id = resource_id::gi_trace, .name = "gi_trace", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::gi_resolve, .name = "gi_resolve", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::gi_history, .name = "gi_history", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::persistent},
        {.id = resource_id::gi_spatial, .name = "gi_spatial", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::gi_spec_trace, .name = "gi_spec_trace", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::gi_spec_reproject, .name = "gi_spec_reproject", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::gi_spec_resolve, .name = "gi_spec_resolve", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::gi_spec_history, .name = "gi_spec_history", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::persistent},
        // The stochastic punctual lighting chain's raw estimate: written as a storage image by the trace,
        // read as a sampled image by the lighting stage that adds it (and, from the next stage, by the
        // temporal resolve). HALF resolution, like the GI chain's images, because the signal is a
        // light-level quantity rather than a shaded surface.
        {.id = resource_id::ml_trace, .name = "ml_trace", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        // The temporal resolve's output - what the lighting stage actually samples - and the accumulation it
        // reads back next frame. The history holds the radiance in rgb and the FRAME COUNT in alpha (an
        // integer up to 12 fits exactly in a float16), which is the per-pixel "how many frames of samples is
        // this": UE keeps it in an R8_UINT image of its own, and the alpha lane is the free one here.
        {.id = resource_id::ml_resolve, .name = "ml_resolve", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::ml_history, .name = "ml_history", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::persistent},
        // FOUR LAYERS, not one: the runtime creates the shadow map as ONE layered image (`max_shadow_cascades` layers, one per cascade, plus the spare layers a shrank count leaves behind - see ensure_shadow_resources), and the shadow pass renders into them one at a time. The count was 1 (the default) until that pass was declared, which made `element = cascade` illegal for every cascade but the first - the same correction the bloom family needed, and the validator`s element check is what says so.
        {.id = resource_id::shadow_map, .name = "shadow_map", .kind = resource_kind::image2d, .scope = resource_scope::per_frame_slot, .lifetime = resource_lifetime::per_frame, .count = 4},
        {.id = resource_id::rt_shadow_visibility, .name = "rt_shadow_visibility", .kind = resource_kind::image2d, .scope = resource_scope::per_frame_slot, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::camera_ubo, .name = "camera_ubo", .kind = resource_kind::buffer, .scope = resource_scope::per_frame_slot, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::light_ubo, .name = "light_ubo", .kind = resource_kind::buffer, .scope = resource_scope::per_frame_slot, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::material_table, .name = "material_table", .kind = resource_kind::buffer, .scope = resource_scope::per_frame_slot, .lifetime = resource_lifetime::persistent},
        {.id = resource_id::instance_table, .name = "instance_table", .kind = resource_kind::buffer, .scope = resource_scope::per_frame_slot, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::motion_vectors, .name = "motion_vectors", .kind = resource_kind::buffer, .scope = resource_scope::per_frame_slot, .lifetime = resource_lifetime::persistent},
        {.id = resource_id::skin_matrices, .name = "skin_matrices", .kind = resource_kind::buffer, .scope = resource_scope::per_frame_slot, .lifetime = resource_lifetime::persistent},
        {.id = resource_id::morph_targets, .name = "morph_targets", .kind = resource_kind::buffer, .scope = resource_scope::per_frame_slot, .lifetime = resource_lifetime::persistent},
        {.id = resource_id::cluster_counts, .name = "cluster_counts", .kind = resource_kind::buffer, .scope = resource_scope::per_frame_slot, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::cluster_indices, .name = "cluster_indices", .kind = resource_kind::buffer, .scope = resource_scope::per_frame_slot, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::probe_grid, .name = "probe_grid", .kind = resource_kind::image3d, .scope = resource_scope::device_wide, .lifetime = resource_lifetime::persistent, .count = 8},
        {.id = resource_id::probe_surface, .name = "probe_surface", .kind = resource_kind::image3d, .scope = resource_scope::device_wide, .lifetime = resource_lifetime::persistent},
        {.id = resource_id::furnace_cube, .name = "furnace_cube", .kind = resource_kind::image_cube, .scope = resource_scope::device_wide, .lifetime = resource_lifetime::persistent},
        {.id = resource_id::scene_textures, .name = "scene_textures", .kind = resource_kind::image2d, .scope = resource_scope::device_wide, .lifetime = resource_lifetime::persistent},
        {.id = resource_id::white_texture, .name = "white_texture", .kind = resource_kind::image2d, .scope = resource_scope::device_wide, .lifetime = resource_lifetime::persistent},
        {.id = resource_id::ibl_env, .name = "ibl_env", .kind = resource_kind::image_cube, .scope = resource_scope::device_wide, .lifetime = resource_lifetime::persistent},
        {.id = resource_id::ibl_irradiance, .name = "ibl_irradiance", .kind = resource_kind::image_cube, .scope = resource_scope::device_wide, .lifetime = resource_lifetime::persistent},
        {.id = resource_id::brdf_lut, .name = "brdf_lut", .kind = resource_kind::image2d, .scope = resource_scope::device_wide, .lifetime = resource_lifetime::persistent},
        {.id = resource_id::top_level_structure, .name = "top_level_structure", .kind = resource_kind::accel_struct, .scope = resource_scope::device_wide, .lifetime = resource_lifetime::per_frame},
    }};

    /// @brief the schema entry for @p id, or nullptr when nothing declares it
    [[nodiscard]] constexpr resource_info const* find(resource_id const id) noexcept {
        for (resource_info const& info : resource_schema) {
            if (info.id == id) {
                return &info;
            }
        }
        return nullptr;
    }

    // =============================================================================================
    // 2. WHAT A PASS DOES WITH ONE - the usage half. The only shared token is `resource_id`.
    // =============================================================================================

    /// @brief how a binding is declared, which is what fixes its `VkDescriptorType`
    enum class binding_kind : uint8_t {
        sampled_image,
        storage_image,
        sampler,
        uniform_buffer,
        storage_buffer,
        input_attachment,
        acceleration_structure,
    };

    /**
     * @brief what the pass DOES with it, which is NOT derivable from the kind
     *
     * The spatial filter only READS its `gi_input` storage image while writing its output; the tracer writes
     * `gi_trace` and reads it back in the same dispatch. A barrier stage keys on this, not on the kind.
     */
    enum class binding_access : uint8_t {
        read,
        write,
        read_write,
    };

    /// @brief which of this renderer's samplers a binding wants (six exist; the choice is a field, not a ternary)
    enum class sampler_hint : uint8_t {
        none,
        gbuffer,
        probe_grid,
        taa,
        post,
        nearest,
        shadow,
    };

    /**
     * @brief the image layout a binding's DESCRIPTOR declares
     *
     * THIS FIELD EXISTS BECAUSE THE KIND DOES NOT IMPLY THE LAYOUT, which the first real conversion found
     * rather than assumed: the probe cache keeps ALL NINE of its own bindings in `GENERAL` - both ping-pong
     * sides and the per-cell geometry - because the propagation's barriers are same-layout ones for the whole
     * update, and a descriptor claiming `SHADER_READ_ONLY_OPTIMAL` for one of those images would be a lie the
     * validation layer rejects. Deriving the layout from "storage versus sampled" would have been wrong for
     * that pass on the first try.
     */
    enum class image_layout : uint8_t {
        sampled,          // SHADER_READ_ONLY_OPTIMAL: a sampled image that is only ever read in that layout
        general,          // GENERAL: a resource whose owner keeps it there for its whole life (a storage image,
                          // or a grid a propagation ping-pongs in place)
        color_attachment, // COLOR_ATTACHMENT_OPTIMAL: what a pass that RENDERS INTO this image leaves it in
    };

    /**
     * @brief who owns the SET a binding lives in
     *
     * `own` is the pass's private family, whose layout it declares in full. The other three name shared sets
     * whose layouts and descriptor counts belong to their owners - a pass declares only that it uses such a
     * binding, and cannot change it.
     */
    enum class set_owner : uint8_t {
        own,
        scene,
        gbuffer,
        post,
    };

    /// @brief shader stages as bits, so a binding can list more than one (the shared sets serve two)
    enum class stage_flag : uint8_t {
        none = 0u,
        vertex = 1u << 0u,
        fragment = 1u << 1u,
        compute = 1u << 2u,
    };

    [[nodiscard]] constexpr stage_flag operator|(stage_flag const a, stage_flag const b) noexcept {
        return static_cast<stage_flag>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
    }

    [[nodiscard]] constexpr bool has_stage(stage_flag const set, stage_flag const one) noexcept {
        return (static_cast<uint8_t>(set) & static_cast<uint8_t>(one)) != 0u;
    }

    /// @brief a name for each enum, so a validator message says "storage_image is not a buffer" not "1 is not 3"
    [[nodiscard]] constexpr std::string_view name_of(binding_kind const kind) noexcept {
        switch (kind) {
        case binding_kind::sampled_image:
            return "sampled_image";
        case binding_kind::storage_image:
            return "storage_image";
        case binding_kind::sampler:
            return "sampler";
        case binding_kind::uniform_buffer:
            return "uniform_buffer";
        case binding_kind::storage_buffer:
            return "storage_buffer";
        case binding_kind::input_attachment:
            return "input_attachment";
        case binding_kind::acceleration_structure:
            return "acceleration_structure";
        }
        return "?";
    }

    [[nodiscard]] constexpr std::string_view name_of(resource_kind const kind) noexcept {
        switch (kind) {
        case resource_kind::image2d:
            return "image2d";
        case resource_kind::image3d:
            return "image3d";
        case resource_kind::image_cube:
            return "image_cube";
        case resource_kind::buffer:
            return "buffer";
        case resource_kind::accel_struct:
            return "accel_struct";
        }
        return "?";
    }

    [[nodiscard]] constexpr std::string_view name_of(binding_access const access) noexcept {
        switch (access) {
        case binding_access::read:
            return "read";
        case binding_access::write:
            return "write";
        case binding_access::read_write:
            return "read_write";
        }
        return "?";
    }

    /// @brief one binding: one use. Everything the layout, the write and the pool count need is here.
    /// @ingroup vulkan_render_resource
    struct pass_binding {
        uint32_t set = 0;
        uint32_t binding = 0;
        set_owner owner = set_owner::own;
        binding_kind kind = binding_kind::sampled_image;
        resource_id resource = resource_id::none;
        /// which image of the resource FAMILY this binding names (0 for a single-image resource)
        uint16_t element = 0;
        /// how many descriptors this binding declares: 1, or the capacity of a bindless array. This is the
        /// number that must equal the layout's count - the one that was wrong once - and it is meaningful for
        /// an OWN set only: a shared set's counts belong to that set's owner.
        uint16_t descriptor_count = 1;
        binding_access access = binding_access::read;
        sampler_hint sampler = sampler_hint::none;
        /// the layout this binding's descriptor declares (see image_layout: NOT derivable from the kind)
        image_layout layout = image_layout::sampled;
        stage_flag stages = stage_flag::compute;
    };

    /// @brief a push-constant range: declared and pinned by `static_assert`, never computed from a declaration
    struct push_block {
        uint32_t offset = 0;
        uint32_t size = 0;
        stage_flag stages = stage_flag::compute;
    };

    /// @brief which slot of a rendering instance a declared target fills
    enum class target_kind : uint8_t {
        color, // a colour attachment (one of the G-buffer's targets, the HDR image a resolve writes)
        depth, // THE depth attachment, of which a rendering instance has exactly one
    };

    /**
     * @brief an image a pass RENDERS INTO, which is a use that cannot be a descriptor
     *
     * A fullscreen pass's colour attachment is not in any set: it is bound by `vkCmdBeginRendering`, so the
     * declaration has to name it somewhere else - and it has to name it SOMEWHERE, because otherwise a pass
     * could render into an image it never declared, which is the one property this layer enforces. Keeping it
     * in `bindings` would be worse than a second list: a binding is a descriptor, the layout generator walks
     * that list, and a colour attachment has no `VkDescriptorType` at all.
     *
     * The LOAD OP and the clear value are deliberately NOT declared: the pass that renders into the image is
     * the one that opens the rendering instance (see `behaviour_kind::fullscreen` and `graphics`), so it is the
     * one that says whether the old contents matter. What is declared is only what the layer has to know: which
     * resource, which image of its family, and whether it fills the colour or the DEPTH slot.
     */
    struct render_target {
        resource_id resource = resource_id::none;
        uint16_t element = 0; // which image of the resource's family (per-swapchain-image resources: the index)
        target_kind kind = target_kind::color;
        /**
         * HOW MANY CONSECUTIVE ELEMENTS of the family this ONE entry claims, rendered ONE INSTANCE PER ELEMENT.
         *
         * WHY IT IS NOT "several attachments of one instance", which is the obvious reading: a rendering instance
         * has exactly ONE depth attachment, so the N layers of one depth array cannot be one instance's targets.
         * The shadow pass renders its cascades layer by layer, each with its own transition, instance and
         * secondary - and what its declaration could not say before was that all N of those layers are elements
         * of ONE family, so the host had to hand them over itself (`runtime::resolve_shadow_pass`, now gone).
         *
         * A CEILING RATHER THAN A PROMISE: how many elements a family HAS this frame is the FRAME's fact (the
         * shadow map allocates the layers the cascade knob asks for - one to four - and `ensure_shadow_resources`
         * may keep spare layers a shrank count left behind), so resolution hands over the first @c count elements
         * that EXIST and the pass renders what it is given. `element + count` past the family's own count is a
         * declaration error the validator refuses, so this field is a claim about the SCHEMA, and the frame's
         * answer is only ever a shorter run.
         */
        uint16_t count = 1;
    };

    /**
     * @brief an image a pass must TRANSITION but does not bind as a descriptor
     *
     * WHY THIS IS A THIRD LIST, and it is the GI chain that forced it rather than a wish for symmetry: the SSGI
     * tracer writes `gi_trace` as a storage image, reads last frame's `gi_resolve` as a bounce feedback and
     * samples the probe grid - and NONE of those descriptors is in its own set. They live in the G-buffer set,
     * which its owner writes (see `pass_io::shared_sets`), so the pass cannot describe them without copying a
     * fact that is not its own. What it CAN say is which images it is responsible for moving between layouts,
     * because a layout transition names an IMAGE and no descriptor at all - the same argument that made
     * `render_target` a separate list from `bindings`.
     *
     * The LAYOUTS are deliberately not declared: which pair of layouts a transition uses is the pass's
     * knowledge (its own barriers, its own order), and a declaration that named them would have to be a
     * barrier generator rather than a description of what the pass touches. What is declared is only what the
     * layer must know to hand the pass a handle: which resource, and which image of its family.
     */
    struct barrier_image {
        resource_id resource = resource_id::none;
        /// which image of the resource's family: the FRAME's image index for a per-swapchain-image resource
        /// (resolved by the host from the frame, exactly as `render_target` does), or the element for a family
        /// a pass reaches by position (the probe grid's eight)
        uint16_t element = 0;
    };

    /**
     * @brief a BUFFER a pass orders around but never binds itself - the buffer twin of `barrier_image`
     *
     * WHY THIS EXISTS, and it is a measured need rather than symmetry for its own sake: the clustered-light
     * sort (`vulkan.pass.cluster`) writes two buffers that live in the SHARED scene set (bindings 11 and 12).
     * The pass binds the whole set, as it must, so its declaration names no binding for either buffer - and yet
     * its writes are not visible to the fragment stages that read them later in the same submission without a
     * buffer memory barrier, which only the writer can place. Before this field the renderer kept the whole
     * recording for exactly that reason: a pass could declare an IMAGE it moves and had no way to name a BUFFER
     * it moves. The pair of channel fields is now the same shape as `barrier_images`, one per resource class.
     *
     * The ACCESS masks are deliberately not declared, for the reason `barrier_image` gives about layouts: which
     * stages and accesses a barrier names is the pass's knowledge of its own ordering, and a declaration that
     * named them would be a barrier generator rather than a description of what the pass touches.
     */
    struct barrier_buffer {
        resource_id resource = resource_id::none;
        /// which buffer of the resource's family - the frame's slot for a per-frame-slot resource (resolved by
        /// the host from the frame, exactly as `barrier_image` does for images)
        uint16_t element = 0;
    };

    /**
     * @brief one SHARED set a pass binds: the set FAMILY, and which element of that family
     *
     * WHY AN ELEMENT AND NOT JUST AN INDEX, measured rather than anticipated: the post chain binds ONE family
     * whose five sets are one per STAGE - the four bloom levels and the composite/FXAA pair - so "set 2" does not
     * say which of them a pass binds, and six declarations could not be resolved from their own text because of
     * it. The FAMILY is the index the framework and the owner already use (0 the scene set, 1 the G-buffer set, 2
     * the post set in this renderer, which is also what `pass_context::shared_set_layout` is asked by), and the
     * ELEMENT selects within it: 0 for a family that holds one set, and the stage's own index in a family that
     * holds several. Only the OWNER can say what an element means - it is the one that created the sets - so this
     * field is carried, exactly like `behaviour::extent_of_element` is.
     */
    struct shared_set {
        uint16_t family = 0;
        uint16_t element = 0;
        [[nodiscard]] constexpr bool operator==(shared_set const&) const noexcept = default;
    };

    /**
     * @brief one pass's declared I/O
     *
     * `own_set` is the set index of the `set_owner::own` bindings; the validator requires every `own` binding
     * to live in that set and to be numbered contiguously from zero, because those are the ones the pass's
     * own descriptor set layout is generated from.
     * @ingroup vulkan_render_resource
     */
    struct pass_io {
        std::string_view name = {};
        uint32_t own_set = 1;
        std::span<pass_binding const> bindings = {};
        /**
         * The SHARED sets this pass binds, BY INDEX - the sets it does not own and must not describe.
         *
         * WHY THIS IS SEPARATE FROM `bindings`, and the reason is a measured one rather than a stylistic
         * preference: a pass that binds a whole set it does not own cannot enumerate that set's bindings
         * without copying a fact its OWNER owns. The G-buffer set is the case that proved it - it carries the
         * GI chain's four images, the world-space probe's four SH-2 coefficients, the post chain's depth and
         * normal, and the scene pass that binds it reads exactly one of those. A declaration that listed them
         * would be the drift this module exists to remove, in the one place where nothing can be GENERATED
         * from it (this pass builds no layout from that set; its owner does).
         *
         * The index is what a pipeline layout needs (the sets are bound in order), and a pass that binds only
         * such a set declares it here and declares no `own` binding at all - which is what the scene pass
         * does: it writes attachments and binds the shared scene set.
         */
        std::span<shared_set const> shared_sets = {};
        /// the images this pass renders into, in the order it uses them (a fullscreen pass has one)
        std::span<render_target const> targets = {};
        /**
         * The images this pass TRANSITIONS but never binds (see `barrier_image`), in the order the pass's own
         * record() indexes them. Empty for every pass whose resources are all its own or all descriptors.
         */
        std::span<barrier_image const> barrier_images = {};
        /**
         * The BUFFERS this pass orders around but never binds (see `barrier_buffer`), in the order the pass's
         * own record() indexes them. Empty for every pass but the clustered-light sort today.
         */
        std::span<barrier_buffer const> barrier_buffers = {};
        std::optional<push_block> push = std::nullopt;
    };

    // =============================================================================================
    // 3. THE INVARIANTS - the reason this is a module and not a comment
    // =============================================================================================

    /// @brief kind compatibility: which resource kinds a binding kind may name
    [[nodiscard]] constexpr bool compatible(binding_kind const kind, resource_kind const resource) noexcept {
        switch (kind) {
        case binding_kind::sampled_image:
        case binding_kind::storage_image:
        case binding_kind::input_attachment:
            return resource == resource_kind::image2d || resource == resource_kind::image3d || resource == resource_kind::image_cube;
        case binding_kind::sampler:
            return true; // a standalone sampler names no resource of its own
        case binding_kind::uniform_buffer:
        case binding_kind::storage_buffer:
            return resource == resource_kind::buffer;
        case binding_kind::acceleration_structure:
            return resource == resource_kind::accel_struct;
        }
        return false;
    }

    /// @brief access compatibility: what each binding kind can do
    [[nodiscard]] constexpr bool compatible(binding_kind const kind, binding_access const access) noexcept {
        switch (kind) {
        case binding_kind::sampled_image:
        case binding_kind::sampler:
        case binding_kind::uniform_buffer:
        case binding_kind::input_attachment:
        case binding_kind::acceleration_structure:
            return access == binding_access::read;
        case binding_kind::storage_image:
        case binding_kind::storage_buffer:
            return true; // writable, or readable: the spatial filter's input is a read-only storage image
        }
        return false;
    }

    /** @brief the schema's own invariants: no empty name, no empty family, and ONE entry per declared id */
    [[nodiscard]] inline std::expected<void, std::string> validate_schema() {
        for (resource_info const& info : resource_schema) {
            if (info.id == resource_id::none) {
                return std::unexpected("a schema entry has no id");
            }
            if (static_cast<uint32_t>(info.id) >= static_cast<uint32_t>(resource_id::count_)) {
                return std::unexpected("a schema entry uses an id outside the enumeration");
            }
            if (info.name.empty()) {
                return std::unexpected("a schema entry has no name");
            }
            if (info.count == 0) {
                return std::unexpected(std::string(info.name) + " declares zero images");
            }
            if (info.lifetime == resource_lifetime::imported && info.scope != resource_scope::per_swapchain_image) {
                return std::unexpected(std::string(info.name) + " is imported but not per-swapchain-image");
            }
        }
        // Completeness: every enumerator the renderer can name has exactly one entry. A forgotten entry is
        // the failure this catches, and it catches it without a second hand-kept list of ids.
        for (uint32_t raw = 1; raw < static_cast<uint32_t>(resource_id::count_); ++raw) {
            std::size_t seen = 0;
            for (resource_info const& info : resource_schema) {
                if (static_cast<uint32_t>(info.id) == raw) {
                    ++seen;
                }
            }
            if (seen != 1) {
                return std::unexpected("the schema has " + std::to_string(seen) + " entries for resource id " + std::to_string(raw));
            }
        }
        return {};
    }

    /**
     * @brief a pass's declaration against the schema
     *
     * THE INVARIANTS, each one either a bug this project has met or a class of bug it is built to avoid: a
     * named pass; every binding names a resource the schema DECLARES (usage is a subset of the schema); the
     * binding kind fits the resource kind (a `sampler3D` cannot be declared as a 2D image; an acceleration
     * structure is not a buffer); the access fits the kind (a sampled image cannot be written); `element` is
     * inside the family; `descriptor_count` is at least one; a sampled image names a sampler and nothing else
     * does; no two bindings share a (set, binding) pair, which would silently lose one of them in the layout;
     * and the pass's OWN bindings are exactly the set `own_set`, numbered contiguously from zero. A render
     * TARGET gets the same two checks a binding gets - the schema declares the resource, and the element is
     * inside its family - plus uniqueness, because two targets naming one image would be a pass rendering into
     * itself twice. A target that claims a RUN of elements (`render_target::count`) is checked as the run it
     * is: the last element has to be inside the family, and two runs of one resource may not overlap. And a
     * declaration that binds a SHARED set has to NAME one, which is the rule the probe cache's missing
     * `shared_sets` entry taught this function (a null descriptor set at bind time).
     * @ingroup vulkan_render_resource
     */
    [[nodiscard]] inline std::expected<void, std::string> validate(pass_io const& io) {
        if (io.name.empty()) {
            return std::unexpected("a pass declaration has no name");
        }
        std::string const who{io.name};
        std::size_t own_count = 0;
        std::size_t depth_targets = 0;
        for (render_target const& t : io.targets) {
            std::string const where = who + ": target " + std::to_string(t.element);
            resource_info const* const info = find(t.resource);
            if (info == nullptr) {
                return std::unexpected(where + " names a resource the schema does not declare");
            }
            if (info->kind != resource_kind::image2d && info->kind != resource_kind::image3d && info->kind != resource_kind::image_cube) {
                return std::unexpected(where + " names " + std::string(info->name) + ", which is not an image a pass can render into");
            }
            if (t.element >= info->count) {
                return std::unexpected(where + " names element " + std::to_string(t.element) + " of " + std::string(info->name) + ", which holds " +
                                       std::to_string(info->count));
            }
            if (t.count == 0u) {
                return std::unexpected(where + " claims no element of " + std::string(info->name) + ", so it renders nothing");
            }
            if (static_cast<uint32_t>(t.element) + t.count > info->count) {
                // A RUN that reaches past the family: `count` is a claim about the SCHEMA, and the frame can only
                // ever answer with a SHORTER run (see render_target::count).
                return std::unexpected(where + " claims elements " + std::to_string(t.element) + ".." + std::to_string(static_cast<uint32_t>(t.element) + t.count - 1u) + " of " +
                                       std::string(info->name) + ", which holds " + std::to_string(info->count));
            }
            if (t.kind == target_kind::depth && ++depth_targets > 1u) {
                // a rendering instance has exactly one depth attachment, so a second one cannot be recorded - and a
                // RUN of depth elements is still ONE entry here, because each element gets its own instance
                return std::unexpected(who + ": more than one DEPTH target is declared, and an instance has one");
            }
            for (render_target const& other : io.targets) {
                // Two entries whose element RUNS INTERSECT would have the pass render into one image twice - the
                // same defect the exact-match check below reports, one resource class over.
                if (&other != &t && other.resource == t.resource && t.element < static_cast<uint32_t>(other.element) + other.count && other.element < static_cast<uint32_t>(t.element) + t.count) {
                    return std::unexpected(where + " overlaps the entries " + std::to_string(other.element) + ".." + std::to_string(static_cast<uint32_t>(other.element) + other.count - 1u) + " of the same resource");
                }
            }
        }
        // THE BARRIER IMAGES, checked like a target: an image the pass will move between layouts has to exist
        // in the schema and has to BE an image (a transition names no descriptor, but it does name a resource).
        for (barrier_image const& t : io.barrier_images) {
            std::string const where = who + ": barrier image " + std::to_string(t.element);
            resource_info const* const info = find(t.resource);
            if (info == nullptr) {
                return std::unexpected(where + " names a resource the schema does not declare");
            }
            if (info->kind != resource_kind::image2d && info->kind != resource_kind::image3d && info->kind != resource_kind::image_cube) {
                return std::unexpected(where + " names " + std::string(info->name) + ", which is not an image");
            }
            if (t.element >= info->count) {
                return std::unexpected(where + " names element " + std::to_string(t.element) + " of " + std::string(info->name) + ", which holds " +
                                       std::to_string(info->count));
            }
            for (barrier_image const& other : io.barrier_images) {
                if (&other != &t && other.resource == t.resource && other.element == t.element) {
                    return std::unexpected(where + " is declared twice, and a pass indexes these by position");
                }
            }
        }
        // ... and THE BARRIER BUFFERS, the same three checks one resource class over: it has to exist in the
        // schema, it has to BE a buffer, and a pass indexes these by position so a duplicate is a defect.
        for (barrier_buffer const& t : io.barrier_buffers) {
            std::string const where = who + ": barrier buffer " + std::to_string(t.element);
            resource_info const* const info = find(t.resource);
            if (info == nullptr) {
                return std::unexpected(where + " names a resource the schema does not declare");
            }
            if (info->kind != resource_kind::buffer) {
                return std::unexpected(where + " names " + std::string(info->name) + ", which is not a buffer");
            }
            if (t.element >= info->count) {
                return std::unexpected(where + " names element " + std::to_string(t.element) + " of " + std::string(info->name) + ", which holds " +
                                       std::to_string(info->count));
            }
            for (barrier_buffer const& other : io.barrier_buffers) {
                if (&other != &t && other.resource == t.resource && other.element == t.element) {
                    return std::unexpected(where + " is declared twice, and a pass indexes these by position");
                }
            }
        }
        for (pass_binding const& b : io.bindings) {
            std::string const where = who + ": set " + std::to_string(b.set) + " binding " + std::to_string(b.binding);
            resource_info const* const info = find(b.resource);
            if (info == nullptr) {
                return std::unexpected(where + " names a resource the schema does not declare");
            }
            if (!compatible(b.kind, info->kind)) {
                return std::unexpected(where + " is declared " + std::string(name_of(b.kind)) + " but " + std::string(info->name) + " is " +
                                       std::string(name_of(info->kind)));
            }
            if (!compatible(b.kind, b.access)) {
                return std::unexpected(where + " is a " + std::string(name_of(b.kind)) + " used for " + std::string(name_of(b.access)));
            }
            if (b.kind == binding_kind::storage_image && b.layout != image_layout::general) {
                // a storage image is written, and a descriptor that claims SHADER_READ_ONLY_OPTIMAL for one
                // describes an image the pass is not allowed to write - validation rejects it at submit
                return std::unexpected(where + " is a storage image, whose descriptor must declare GENERAL");
            }
            if (b.element >= info->count) {
                return std::unexpected(where + " names element " + std::to_string(b.element) + " of " + std::string(info->name) +
                                       ", which holds " + std::to_string(info->count));
            }
            if (b.descriptor_count == 0) {
                return std::unexpected(where + " declares no descriptors");
            }
            if ((b.kind == binding_kind::sampled_image) != (b.sampler != sampler_hint::none)) {
                return std::unexpected(where + " must name a sampler exactly when it is a sampled image");
            }
            for (pass_binding const& other : io.bindings) {
                if (&other != &b && other.set == b.set && other.binding == b.binding) {
                    return std::unexpected(where + " is declared twice");
                }
            }
            if (b.owner == set_owner::own) {
                if (b.set != io.own_set) {
                    return std::unexpected(where + " is an own binding outside the pass's own set");
                }
                if (b.binding != own_count) {
                    return std::unexpected(who + ": the pass's own bindings are not contiguous from zero (expected " +
                                           std::to_string(own_count) + ", found " + std::to_string(b.binding) + ")");
                }
                ++own_count;
            }
        }
        if (io.push.has_value()) {
            if (io.push->size == 0 || (io.push->size % 4u) != 0u || io.push->offset + io.push->size > 128u) {
                return std::unexpected(who + ": the push block does not fit the 128-byte guaranteed minimum in 4-byte units");
            }
        }
        // THE SHARED SETS, checked after the bindings are counted: a set cannot be both the pass's own (whose
        // layout it generates and whose bindings it describes) and one it merely binds.
        //
        // ... AND A DECLARATION THAT BINDS A SHARED SET HAS TO DECLARE IT. This rule was ADDED because a pass got
        // it wrong in the one way nothing caught: the probe cache's seven scene-owned bindings give its pipeline
        // layout a set 0, but its declaration listed no shared set at all - so the framework had nothing to
        // resolve for set 0, the pass bound a NULL descriptor set, and the only scenario that runs the cache
        // reported `pDBindDescriptorSets(): pDescriptorSets[0] (VkDescriptorSet 0x0) is not a valid
        // VkDescriptorSet`. The bindings describe the LAYOUT; `shared_sets` is what tells the framework which sets
        // the owner has to FILL, and only the declaration can say that.
        std::size_t const non_own_bindings = static_cast<std::size_t>(std::count_if(io.bindings.begin(), io.bindings.end(), [](pass_binding const& b) { return b.owner != set_owner::own; }));
        if (non_own_bindings > 0 && io.shared_sets.empty()) {
            return std::unexpected(who + ": " + std::to_string(non_own_bindings) + " of its bindings belong to a SHARED set, and the declaration names none - the framework would resolve nothing for them");
        }
        for (std::size_t i = 0; i < io.shared_sets.size(); ++i) {
            shared_set const set = io.shared_sets[i];
            if (own_count != 0 && set.family == io.own_set) {
                return std::unexpected(who + ": set " + std::to_string(set.family) + " is declared as both its own and a shared set");
            }
            for (std::size_t j = i + 1; j < io.shared_sets.size(); ++j) {
                if (io.shared_sets[j] == set) {
                    return std::unexpected(who + ": shared set " + std::to_string(set.family) + " element " + std::to_string(set.element) + " is declared twice");
                }
            }
        }
        return {};
    }

    /** @brief the number of descriptors of each kind one set declares - what a pool must be sized from */
    struct descriptor_counts {
        uint32_t sampled_image = 0;
        uint32_t storage_image = 0;
        uint32_t sampler = 0;
        uint32_t uniform_buffer = 0;
        uint32_t storage_buffer = 0;
        uint32_t input_attachment = 0;
        uint32_t acceleration_structure = 0;

        [[nodiscard]] constexpr uint32_t total() const noexcept {
            return sampled_image + storage_image + sampler + uniform_buffer + storage_buffer + input_attachment + acceleration_structure;
        }
    };

    /// @brief count the descriptors one SET of a declaration needs, per kind
    /// @ingroup vulkan_render_resource
    [[nodiscard]] constexpr descriptor_counts descriptor_counts_for(pass_io const& io, uint32_t const set) noexcept {
        descriptor_counts counts;
        for (pass_binding const& b : io.bindings) {
            if (b.set != set) {
                continue;
            }
            switch (b.kind) {
            case binding_kind::sampled_image:
                counts.sampled_image += b.descriptor_count;
                break;
            case binding_kind::storage_image:
                counts.storage_image += b.descriptor_count;
                break;
            case binding_kind::sampler:
                counts.sampler += b.descriptor_count;
                break;
            case binding_kind::uniform_buffer:
                counts.uniform_buffer += b.descriptor_count;
                break;
            case binding_kind::storage_buffer:
                counts.storage_buffer += b.descriptor_count;
                break;
            case binding_kind::input_attachment:
                counts.input_attachment += b.descriptor_count;
                break;
            case binding_kind::acceleration_structure:
                counts.acceleration_structure += b.descriptor_count;
                break;
            }
        }
        return counts;
    }

    // =============================================================================================
    // 4. THE FIRST DECLARATION - the probe cache, read off shaders/gi_probe.comp
    // =============================================================================================

    /**
     * @brief the world-space probe cache's I/O, as its shader actually declares it
     *
     * Set 1 is the pass's own: bindings 0..3 are the ping-pong side being READ and 4..7 the side being
     * WRITTEN, both in coefficient order - which is exactly the `side*4+coefficient` indexing the schema's
     * `probe_grid` counts, and the reason a binding names an ELEMENT of a resource rather than a resource.
     * Binding 8 is the per-cell surface geometry the propagation tests visibility with.
     *
     * Set 0 is the SHARED scene set: a cell's own ray needs the top level structure to trace, and the shared
     * hit shading needs the material table, the texture array, the light UBO and the environment cubes to
     * shade what it finds. Those bindings reference the SAME schema the pass's own do - which is the point of
     * keeping the schema in one place and the usage with the pass.
     */
    inline constexpr std::array<pass_binding, 16> gi_probe_bindings = {{
        // ALL NINE OF THE PASS'S OWN BINDINGS DECLARE `GENERAL`, and that is not tidiness: the two ping-pong
        // sides stay in GENERAL for the whole update (which is what makes the propagation's barriers
        // same-layout ones), and the per-cell geometry is written by the injection and read by the propagation
        // in the same dispatch sequence. A descriptor claiming SHADER_READ for a sampled one of them would be a
        // lie validation rejects at submit.
        {.set = 1, .binding = 0, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::probe_grid, .element = 0, .access = binding_access::read, .sampler = sampler_hint::probe_grid, .layout = image_layout::general},
        {.set = 1, .binding = 1, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::probe_grid, .element = 1, .access = binding_access::read, .sampler = sampler_hint::probe_grid, .layout = image_layout::general},
        {.set = 1, .binding = 2, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::probe_grid, .element = 2, .access = binding_access::read, .sampler = sampler_hint::probe_grid, .layout = image_layout::general},
        {.set = 1, .binding = 3, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::probe_grid, .element = 3, .access = binding_access::read, .sampler = sampler_hint::probe_grid, .layout = image_layout::general},
        {.set = 1, .binding = 4, .owner = set_owner::own, .kind = binding_kind::storage_image, .resource = resource_id::probe_grid, .element = 4, .access = binding_access::write, .layout = image_layout::general},
        {.set = 1, .binding = 5, .owner = set_owner::own, .kind = binding_kind::storage_image, .resource = resource_id::probe_grid, .element = 5, .access = binding_access::write, .layout = image_layout::general},
        {.set = 1, .binding = 6, .owner = set_owner::own, .kind = binding_kind::storage_image, .resource = resource_id::probe_grid, .element = 6, .access = binding_access::write, .layout = image_layout::general},
        {.set = 1, .binding = 7, .owner = set_owner::own, .kind = binding_kind::storage_image, .resource = resource_id::probe_grid, .element = 7, .access = binding_access::write, .layout = image_layout::general},
        {.set = 1, .binding = 8, .owner = set_owner::own, .kind = binding_kind::storage_image, .resource = resource_id::probe_surface, .element = 0, .access = binding_access::write, .layout = image_layout::general},
        {.set = 0, .binding = 1, .owner = set_owner::scene, .kind = binding_kind::sampled_image, .resource = resource_id::scene_textures, .access = binding_access::read, .sampler = sampler_hint::post},
        {.set = 0, .binding = 2, .owner = set_owner::scene, .kind = binding_kind::sampled_image, .resource = resource_id::ibl_env, .access = binding_access::read, .sampler = sampler_hint::post},
        {.set = 0, .binding = 3, .owner = set_owner::scene, .kind = binding_kind::sampled_image, .resource = resource_id::ibl_irradiance, .access = binding_access::read, .sampler = sampler_hint::post},
        {.set = 0, .binding = 4, .owner = set_owner::scene, .kind = binding_kind::sampled_image, .resource = resource_id::brdf_lut, .access = binding_access::read, .sampler = sampler_hint::post},
        {.set = 0, .binding = 5, .owner = set_owner::scene, .kind = binding_kind::storage_buffer, .resource = resource_id::material_table, .access = binding_access::read},
        {.set = 0, .binding = 7, .owner = set_owner::scene, .kind = binding_kind::uniform_buffer, .resource = resource_id::light_ubo, .access = binding_access::read},
        {.set = 0, .binding = 16, .owner = set_owner::scene, .kind = binding_kind::acceleration_structure, .resource = resource_id::top_level_structure, .access = binding_access::read},
    }};

    /// @brief set 0 is the shared scene set (a cell's ray needs the top level structure, the light UBO and the
    ///        material table; the propagation's hit shading needs the textures and the environment cubes in it too)
    inline constexpr std::array<shared_set, 1> gi_probe_shared_sets = {{{.family = 0}}};

    /// @brief the probe cache's declaration
    /// @ingroup vulkan_render_resource
    inline constexpr pass_io gi_probe_io = {
        .name = "gi_probe",
        .own_set = 1,
        .bindings = gi_probe_bindings,
        // THE SHARED SET HAD TO BE DECLARED EXPLICITLY, and the gate is what found out: the seven scene-owned
        // bindings above give the pipeline layout its set 0, but a set nothing DECLARES is a set the framework
        // resolves nothing for - so the pass bound a null descriptor set and `sponza_gi` (the one scenario with the
        // cache on) reported `vkCmdBindDescriptorSets(): pDescriptorSets[0] (VkDescriptorSet 0x0) is not a valid
        // VkDescriptorSet`. The validator refuses that shape now (see `validate`).
        .shared_sets = gi_probe_shared_sets,
        // The pass's push block, DECLARED here because it is the range its pipeline layout is built with and
        // the size its host must compose. It is the pass's own struct
        // (`vulkan.pass.gi_probe::gi_probe_pass::push_constants`), and `vulkan.pass.gi_probe` carries the
        // `static_assert` that ties this number to that struct - a fact in two units that the compiler keeps
        // in agreement is the next best thing to a fact in one.
        .push = push_block{.offset = 0, .size = 56, .stages = stage_flag::compute},
    };

    // =============================================================================================
    // 5. THE SECOND DECLARATION - the TAA resolve, read off shaders/taa.frag
    // =============================================================================================

    /**
     * @brief the temporal resolve's I/O, as its shader declares it
     *
     * This is the first declaration whose bindings live in set 0 with no shared set beside them: the four
     * inputs (this frame's colour, the reprojected history, the motion vectors and the depth the
     * disocclusion guard reads) are all per-swapchain-image and all this pass's. `stages` is FRAGMENT and has
     * to be said: the generated layout's stage flags come from the declaration, and a resolve whose bindings
     * were declared COMPUTE would be a layout the fragment stage cannot see - which is the drift this
     * generator exists to make impossible, now in the other direction (the field is easy to forget when every
     * other declaration so far was a compute pass).
     */
    inline constexpr std::array<pass_binding, 4> taa_bindings = {{
        {.set = 0, .binding = 0, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::scene_color, .access = binding_access::read, .sampler = sampler_hint::taa, .layout = image_layout::sampled, .stages = stage_flag::fragment},
        {.set = 0, .binding = 1, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::taa_history, .access = binding_access::read, .sampler = sampler_hint::taa, .layout = image_layout::sampled, .stages = stage_flag::fragment},
        {.set = 0, .binding = 2, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::velocity, .access = binding_access::read, .sampler = sampler_hint::taa, .layout = image_layout::sampled, .stages = stage_flag::fragment},
        {.set = 0, .binding = 3, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::gbuffer_depth, .access = binding_access::read, .sampler = sampler_hint::taa, .layout = image_layout::sampled, .stages = stage_flag::fragment},
    }};

    /// @brief the resolve RENDERS INTO the frame's HDR target, which is why it needs a target and not a binding
    inline constexpr render_target taa_target = {.resource = resource_id::hdr, .element = 0};
    inline constexpr std::array<render_target, 1> taa_targets = {taa_target};

    /// @brief the temporal resolve's declaration
    /// @ingroup vulkan_render_resource
    inline constexpr pass_io taa_io = {
        .name = "taa",
        .own_set = 0,
        .bindings = taa_bindings,
        .targets = taa_targets,
        // eight floats: the history flag, the two blend weights, the texel size, and the projection's two
        // depth terms (see vulkan.pass.taa::taa_pass::push_constants, which static_asserts this number)
        .push = push_block{.offset = 0, .size = 32, .stages = stage_flag::fragment},
    };

    // =============================================================================================
    // 6. THE THIRD DECLARATION - the scene pass, which binds a whole set it does not own
    // =============================================================================================

    /**
     * @brief the scene pass's I/O: five colour attachments, one depth attachment, and the shared scene set
     *
     * THIS IS THE FIRST DECLARATION WITH NO OWN BINDINGS AT ALL, and that is what the pass IS: it draws the
     * scene's primitives into the frame's surface targets. The material table, the texture array, the camera
     * and light UBOs, the shadow map, the instance table and the top level structure all arrive through the
     * SHARED scene set (set 0), whose layout and contents its owner decides - so this declaration names the SET
     * and not its bindings (see `pass_io::shared_sets` for why that is the honest statement rather than a
     * shortcut). The pass builds no set layout of its own, which is why `own_set` is unused here.
     *
     * THE TARGETS ARE THE FRAME'S SURFACE, in the order the rendering instance needs them: the three stored
     * G-buffer targets, the motion-vector target, the scene colour target the lighting stage adds on top of,
     * and the G-buffer's own depth. The per-leaf push constants are the LEAVES' (a model matrix, a material
     * index), pushed through the draw path the primitive owns, so this declaration has no push block.
     *
     * `scene_color` IS AN ALIAS, and deliberately: the resolver maps it to the image the frame's scene pass
     * accumulates emissive into, which is `runtime::scene_target_view()` - the HDR target normally, and the
     * TAA resolve's input while that resolve runs (it takes the HDR target for its own output). That is a
     * FRAME decision, not a resource fact, so the declaration names what it means and the renderer decides
     * which image that is. It also means the scene pass and the TAA resolve agree by construction about which
     * image the scene writes and the resolve reads.
     */
    inline constexpr std::array<render_target, 6> scene_targets = {{
        {.resource = resource_id::gbuffer_targets, .element = 0},
        {.resource = resource_id::gbuffer_targets, .element = 1},
        {.resource = resource_id::gbuffer_targets, .element = 2},
        {.resource = resource_id::velocity, .element = 0},
        {.resource = resource_id::scene_color, .element = 0},
        {.resource = resource_id::gbuffer_depth, .element = 0, .kind = target_kind::depth},
    }};

    /// @brief set 0 is the shared scene set: the scene pass binds it and owns nothing of it
    inline constexpr std::array<shared_set, 1> scene_shared_sets = {{{.family = 0}}};

    /// @brief the scene pass's declaration
    /// @ingroup vulkan_render_resource
    inline constexpr pass_io scene_io = {
        .name = "scene",
        .own_set = 1, // unused: this pass has no own bindings (see the declaration's note)
        .bindings = {},
        .shared_sets = scene_shared_sets,
        .targets = scene_targets,
        .push = std::nullopt,
    };

    /**
     * @brief the transparent pass's declaration: the same surface, entered with LOAD
     *
     * TWO TARGETS, not six: the alpha-blended leaves composite over the shaded frame, so the pass draws into
     * the scene colour target and depth-tests against the surface depth - and it LOADs both, because what it
     * composites over must survive. That decision is the pass's (it opens the instance), which is why the
     * declaration names the images and not the load ops (see render_target).
     *
     * It binds the shared scene set like the scene pass does - a blended surface reads the same materials, the
     * same camera and the same shadow map - so it declares the set by index and owns nothing.
     */
    inline constexpr std::array<render_target, 2> transparent_targets = {{
        {.resource = resource_id::scene_color, .element = 0},
        {.resource = resource_id::gbuffer_depth, .element = 0, .kind = target_kind::depth},
    }};

    /// @brief the transparent pass's declaration
    /// @ingroup vulkan_render_resource
    inline constexpr pass_io transparent_io = {
        .name = "transparent",
        .own_set = 1, // unused: this pass has no own bindings either
        .bindings = {},
        .shared_sets = scene_shared_sets,
        .targets = transparent_targets,
        .push = std::nullopt,
    };

    // =============================================================================================
    // 6. THE GI CHAIN'S DECLARATIONS - the passes that reach their images without binding them
    // =============================================================================================

    /**
     * @brief the images the SSGI tracer transitions, in the order its record() indexes them
     *
     * EVERY ONE OF THESE IS IN THE G-BUFFER SET, not in the tracer's own: the tracer binds the shared scene set
     * (0) and the shared G-buffer set (1) and owns no descriptor at all - so before this list existed it could
     * not name a single image it is responsible for moving between layouts, and those barriers had to live in
     * the renderer. See `barrier_image` for why a transition is a use that needs no descriptor.
     *
     * THE ORDER IS THE INTERFACE. The pass indexes this list by position (a compile-time constant per slot, not
     * a search), because the layout pair each entry needs is the pass's own knowledge and the declaration
     * deliberately does not carry it:
     *
     *   0: `gi_trace`          this frame's raw trace, written as a storage image
     *   1: `gi_spec_resolve`   the reflection's accumulation, sampled by the G-buffer set at binding 15
     *   2: `gi_resolve`        last frame's diffuse accumulation (the bounce feedback the tracer samples)
     *   3-6: `probe_grid` 0-3  the cache's four SH-2 coefficient volumes, sampled by the tracer
     *   7-10: `probe_grid` 4-7 the cache's four scratch volumes, written by the propagation
     *   11: `probe_surface`    the per-cell geometry, written by the injection
     */
    inline constexpr std::array<barrier_image, 12> ssgi_trace_barriers = {{
        {.resource = resource_id::gi_trace, .element = 0},
        {.resource = resource_id::gi_spec_resolve, .element = 0},
        {.resource = resource_id::gi_resolve, .element = 0},
        {.resource = resource_id::probe_grid, .element = 0},
        {.resource = resource_id::probe_grid, .element = 1},
        {.resource = resource_id::probe_grid, .element = 2},
        {.resource = resource_id::probe_grid, .element = 3},
        {.resource = resource_id::probe_grid, .element = 4},
        {.resource = resource_id::probe_grid, .element = 5},
        {.resource = resource_id::probe_grid, .element = 6},
        {.resource = resource_id::probe_grid, .element = 7},
        {.resource = resource_id::probe_surface, .element = 0},
    }};

    /// @brief set 0 is the shared scene set and set 1 the shared G-buffer set: the tracer binds both
    inline constexpr std::array<shared_set, 2> ssgi_trace_shared_sets = {{{.family = 0}, {.family = 1}}};

    /**
     * @brief the SSGI tracer's declaration: a half-resolution compute dispatch over two shared sets
     *
     * The first pass on this branch that binds TWO shared sets, and the first with no own binding whose
     * resources still have to be named - which is what `barrier_images` was added for. Its push block is the
     * full 128 bytes the specification guarantees, which the shader's comments explain lane by lane.
     */
    inline constexpr pass_io ssgi_trace_io = {
        .name = "ssgi_trace",
        .own_set = 2, // unused: the tracer has no own bindings (its images come from the shared sets)
        .bindings = {},
        .shared_sets = ssgi_trace_shared_sets,
        .targets = {},
        .barrier_images = ssgi_trace_barriers,
        .push = push_block{.offset = 0, .size = 128, .stages = stage_flag::compute},
    };

    /**
     * @brief the images the glossy lobe transitions, in the order its record() indexes them
     *
     * The lobe binds the tracer's own two shared sets and writes the SAME raw trace the tracer just wrote (it
     * adds its reflection to it, see shaders/ssgi_spec.comp), so `gi_trace` is here for the compute-to-compute
     * ordering barrier that makes the read-after-write legal - and again at the end, where the completed trace
     * is handed to the denoiser. Its own two outputs are the other entries: they are storage images whose
     * descriptors in the G-buffer set declare GENERAL, so the lobe is their only writer and their first-use
     * transition is per image (see the pass's state).
     *
     *   0: `gi_trace`         the raw trace, read AND written (both writers of it are here and in the tracer)
     *   1: `gi_spec_trace`    this frame's reflection correction
     *   2: `gi_spec_reproject` where the reflected surface was, which the resolve reprojects by
     */
    inline constexpr std::array<barrier_image, 3> ssgi_spec_barriers = {{
        {.resource = resource_id::gi_trace, .element = 0},
        {.resource = resource_id::gi_spec_trace, .element = 0},
        {.resource = resource_id::gi_spec_reproject, .element = 0},
    }};

    /**
     * @brief the glossy lobe's declaration: the tracer's two shared sets, no binding of its own, three images
     *
     * The second pass to reach its resources through `barrier_images`, and the reason that channel is per-pass
     * rather than a special case for the tracer: the lobe writes images it does not own and binds no descriptor
     * to reach them either.
     */
    inline constexpr pass_io ssgi_spec_io = {
        .name = "ssgi_spec",
        .own_set = 2, // unused, exactly as in the tracer
        .bindings = {},
        .shared_sets = ssgi_trace_shared_sets,
        .targets = {},
        .barrier_images = ssgi_spec_barriers,
        .push = push_block{.offset = 0, .size = 96, .stages = stage_flag::compute},
    };

    /**
     * @brief the temporal resolve's own bindings: the first GI stage whose inputs are its OWN set
     *
     * The first GI declaration with `own` bindings at all - the tracer and the lobe reach everything through
     * the two shared sets, while the denoiser needs a set nobody else has, because it groups four things no
     * other pass puts together (the raw trace, the accumulated history, the motion vectors and the depth) plus
     * the image it writes and two more it samples. The layout is GENERATED from this list
     * (`bindings::make_set_layout`), which is why the seven bindings and the family's pool count cannot drift:
     * they were out of step once, by hand, and the validation layer is what caught it.
     *
     * BINDING 6 IS ONE SLOT FOR TWO IMAGES, and that is the one place this declaration describes a KIND rather
     * than an exact resource: mode 1 (the reflection) samples the lobe's reprojection there, while mode 0 (the
     * diffuse bounce, which ignores the value) is given the depth target instead - always readable on a frame
     * that resolves anything, where the lobe's image is not (it fails on exactly the frames the lobe is off).
     * The layout is identical either way, and which image goes in is the pass's own write.
     */
    inline constexpr std::array<pass_binding, 7> ssgi_temporal_bindings = {{
        {.set = 0, .binding = 0, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::gi_trace, .access = binding_access::read, .sampler = sampler_hint::gbuffer, .layout = image_layout::sampled},
        {.set = 0, .binding = 1, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::gi_history, .access = binding_access::read, .sampler = sampler_hint::gbuffer, .layout = image_layout::sampled},
        {.set = 0, .binding = 2, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::velocity, .access = binding_access::read, .sampler = sampler_hint::gbuffer, .layout = image_layout::sampled},
        {.set = 0, .binding = 3, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::gbuffer_depth, .access = binding_access::read, .sampler = sampler_hint::gbuffer, .layout = image_layout::sampled},
        {.set = 0, .binding = 4, .owner = set_owner::own, .kind = binding_kind::storage_image, .resource = resource_id::gi_resolve, .access = binding_access::write, .layout = image_layout::general},
        {.set = 0, .binding = 5, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::gbuffer_targets, .element = 1, .access = binding_access::read, .sampler = sampler_hint::gbuffer, .layout = image_layout::sampled},
        {.set = 0, .binding = 6, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::gi_spec_reproject, .access = binding_access::read, .sampler = sampler_hint::gbuffer, .layout = image_layout::sampled},
    }};

    /**
     * @brief the images the temporal resolve transitions
     *
     * The DIFFUSE signal's pair, and only those: this declaration belongs to the pass that resolves the diffuse
     * bounce. The reflection's accumulation and history are the second family's, resolved by the renderer today
     * because the two signals share ONE layout and one pipeline - which is exactly why a single declaration can
     * only ever describe one of them (the reasoning is in docs/pass_chain_plan.md).
     *
     *   0: `gi_resolve`   the diffuse accumulation this dispatch WRITES (also binding 4)
     *   1: `gi_history`   the diffuse history it reads and then copies into (also binding 1)
     */
    inline constexpr std::array<barrier_image, 2> ssgi_temporal_barriers = {{
        {.resource = resource_id::gi_resolve, .element = 0},
        {.resource = resource_id::gi_history, .element = 0},
    }};

    /**
     * @brief the temporal resolve's declaration: the denoiser that turns the raw trace into an accumulation
     *
     * The FIRST GI declaration with a set of its own AND the first whose `barrier_images` and `own` bindings
     * name the same two resources - deliberately, because the pass both binds them and moves them, and the two
     * lists answer different questions (what the descriptor says, and what the barriers take).
     */
    inline constexpr pass_io ssgi_temporal_io = {
        .name = "ssgi_temporal",
        .own_set = 0,
        .bindings = ssgi_temporal_bindings,
        .shared_sets = {},
        .targets = {},
        .barrier_images = ssgi_temporal_barriers,
        .push = push_block{.offset = 0, .size = 48, .stages = stage_flag::compute},
    };

    /**
     * @brief the image the spatial filter transitions
     *
     * One entry, and it is the filter's OUTPUT: `gi_spatial` lives in the G-buffer set as a storage image, so the
     * pass needs its handle for the two barriers (UNDEFINED -> GENERAL, then GENERAL -> SHADER_READ for the
     * composite) and no descriptor of its own. Its INPUT needs nothing - the temporal resolve handed it to
     * SHADER_READ with a transition that names COMPUTE as well as FRAGMENT - which is why this list is one entry
     * where the tracer's is twelve.
     */
    inline constexpr std::array<barrier_image, 1> ssgi_spatial_barriers = {{
        {.resource = resource_id::gi_spatial, .element = 0},
    }};

    /**
     * @brief the spatial filter's declaration: the joint-bilateral filter that ENDS the GI chain
     *
     * The third pass on the shared-sets + barrier-images shape (the tracer and the lobe are the others): it binds
     * the shared scene set and the shared G-buffer set - which already carries every binding it uses, the normal,
     * the depth, the accumulation it reads and the image it writes - and owns no descriptor at all. What the
     * composite samples is ITS output, which is why the renderer's `gi_resolved` is set from whether this pass
     * recorded: a frame whose filter did not run has nothing to add and must weigh 0 rather than show whatever
     * that image happens to hold.
     */
    inline constexpr pass_io ssgi_spatial_io = {
        .name = "ssgi_spatial",
        .own_set = 2, // unused: no own bindings, like the tracer and the lobe
        .bindings = {},
        .shared_sets = ssgi_trace_shared_sets, // the same two: the scene set (0) and the G-buffer set (1)
        .targets = {},
        .barrier_images = ssgi_spatial_barriers,
        .push = push_block{.offset = 0, .size = 48, .stages = stage_flag::compute},
    };

    /**
     * @brief the image the stochastic punctual lighting pass rewrites, and the only resource it names
     *
     * It reaches the image through the SHARED G-buffer set (bindings 16 and 17, see
     * `pipelines::make_gbuffer_set_layout`), so - like the tracer - this pass owns no descriptor at all and
     * declares no binding: what it has to say is which image it moves, because only the writer can place the
     * transitions (UNDEFINED -> GENERAL to write it as a storage image, GENERAL -> SHADER_READ to hand it to
     * the lighting stage that adds it, both recorded by this pass).
     */
    inline constexpr std::array<barrier_image, 1> megalights_trace_barriers = {{
        {.resource = resource_id::ml_trace, .element = 0},
    }};

    /**
     * @brief the stochastic punctual lighting pass's declaration: a half-resolution compute dispatch
     *
     * The tracer's shape exactly - two shared sets, no own binding, one image named through
     * `barrier_images` - because its inputs (the camera, the light UBO, the cluster lists, the G-buffer
     * surface) and its output (a half-resolution storage image) are all reached through those two sets.
     *
     * The push block is 96 bytes: the camera's inverse view-projection, the estimator's parameters (samples
     * per pixel, the minimum sample weight, the ray tmin, the frame counter) and the two origin-bias terms.
     */
    inline constexpr pass_io megalights_trace_io = {
        .name = "megalights_trace",
        .own_set = 2, // unused: no own bindings (everything it reads and writes is in the shared sets)
        .bindings = {},
        .shared_sets = ssgi_trace_shared_sets, // the scene set (0) and the G-buffer set (1), like the tracer
        .targets = {},
        .barrier_images = megalights_trace_barriers,
        .push = push_block{.offset = 0, .size = 96, .stages = stage_flag::compute},
    };

    /**
     * @brief the temporal resolve's own bindings, in the order its shader declares them
     *
     * THREE, and the depth and the velocity it also needs come from the SHARED G-buffer set (bindings 3 and 4)
     * instead - which is the difference from `ssgi_temporal_io`'s seven own bindings and it is a deliberate
     * simplification: a pass that reaches the G-buffer through the shared set has its depth and velocity
     * transitions published by the STAGE's prepare (the same two calls the GI denoise stage makes), so this
     * chain needs no second stage of its own for that ordering rule. What has to be per-image views is only
     * what lives in the stochastic chain's own families.
     *
     *   0: `ml_trace`   this frame's raw estimate (the temporal pass's input)
     *   1: `ml_history` last frame's accumulation, radiance in rgb and the frame count in alpha
     *   2: `ml_resolve` the accumulation this dispatch WRITES (a storage image)
     */
    inline constexpr std::array<pass_binding, 5> megalights_temporal_bindings = {{
        {.set = 0, .binding = 0, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::ml_trace, .access = binding_access::read, .sampler = sampler_hint::gbuffer},
        {.set = 0, .binding = 1, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::ml_history, .access = binding_access::read, .sampler = sampler_hint::gbuffer},
        {.set = 0, .binding = 2, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::velocity, .access = binding_access::read, .sampler = sampler_hint::gbuffer},
        {.set = 0, .binding = 3, .owner = set_owner::own, .kind = binding_kind::sampled_image, .resource = resource_id::gbuffer_depth, .access = binding_access::read, .sampler = sampler_hint::gbuffer},
        {.set = 0, .binding = 4, .owner = set_owner::own, .kind = binding_kind::storage_image, .resource = resource_id::ml_resolve, .access = binding_access::write, .layout = image_layout::general},
    }};

    /// @brief the images the temporal resolve moves, in the order its record() indexes them
    ///
    /// Its five OWN bindings include the two G-buffer targets it reads, so those are its transitions too: the
    /// pass is the first sampler of this image's depth and velocity this frame (see the stage's prepare).
    inline constexpr std::array<barrier_image, 4> megalights_temporal_barriers = {{
        {.resource = resource_id::ml_resolve, .element = 0},
        {.resource = resource_id::ml_history, .element = 0},
        {.resource = resource_id::gbuffer_depth, .element = 0},
        {.resource = resource_id::velocity, .element = 0},
    }};

    /**
     * @brief the temporal resolve's declaration: a half-resolution compute dispatch over its own set
     *
     * Its own set is index 1 and the SHARED G-buffer set is index 0 - the `gi_probe` shape, and the reason is
     * mechanical rather than aesthetic: a pipeline layout needs a descriptor set layout for every index up to
     * the highest one used, so a pass whose own bindings sit at set 2 would have to declare something at set 1
     * as well. Set 0 carries the depth it rejects on and the velocity it reprojects with; set 1 carries the
     * three images of its own chain. Its push block is the projection's linearization pair, the three accumulation bounds
     * and the extents.
     */
    inline constexpr pass_io megalights_temporal_io = {
        .name = "megalights_temporal",
        .own_set = 0,
        .bindings = megalights_temporal_bindings,
        .shared_sets = {},
        .targets = {},
        .barrier_images = megalights_temporal_barriers,
        .push = push_block{.offset = 0, .size = 32, .stages = stage_flag::compute},
    };

    /**
     * @brief the image the ray-traced shadow pass rewrites, and the only resource it has to name
     *
     * `rt_shadow` is per FRAME SLOT rather than per swapchain image (the rays are traced once per frame, not once
     * per presented image), so the host resolves this one from the frame's slot - the declaration says WHICH
     * resource, and the element is 0 because there is one family entry to choose from there. It is a storage
     * image the pass writes (UNDEFINED -> GENERAL) and the lighting stage samples (GENERAL -> SHADER_READ), both
     * transitions recorded by the pass itself.
     */
    inline constexpr std::array<barrier_image, 1> rt_shadow_barriers = {{
        {.resource = resource_id::rt_shadow_visibility, .element = 0},
    }};

    /**
     * @brief the ray-traced sun shadow pass's declaration: a full-resolution compute dispatch over two shared sets
     *
     * The same shape as the tracer's, with the frame's resolution instead of half: it binds the shared scene set
     * (the camera, the light UBO, the top level structure at binding 16) and the shared G-buffer set (the surface
     * each ray starts from), owns no descriptor at all, and names the one image it rewrites through
     * `barrier_images`. Its push block is the camera's inverse view-projection and four ray-offset terms.
     */
    inline constexpr pass_io rt_shadow_io = {
        .name = "rt_shadow",
        .own_set = 2, // unused: no own bindings (everything it reads is in the shared sets)
        .bindings = {},
        .shared_sets = ssgi_trace_shared_sets, // the same two: the scene set (0) and the G-buffer set (1)
        .targets = {},
        .barrier_images = rt_shadow_barriers,
        .push = push_block{.offset = 0, .size = 80, .stages = stage_flag::compute},
    };

    /**
     * @brief the two buffers the clustered-light sort writes, and the ONLY resources it has to name
     *
     * They are bindings 11 and 12 of the SHARED scene set - so this pass binds them as part of that set and
     * declares no binding of its own - but a pass's writes are not visible to the fragment stages reading them
     * later in the same submission without a BUFFER memory barrier, and only the writer can place it. That is
     * what `pass_io::barrier_buffers` exists for (see `barrier_buffer`), and this declaration is why the field
     * was added: before it, a pass could name an image it moves and had no way to name a buffer it moves.
     *
     * Both are per FRAME SLOT (the slot's count and index arrays), which the host resolves from the frame.
     */
    inline constexpr std::array<barrier_buffer, 2> cluster_barriers = {{
        {.resource = resource_id::cluster_counts, .element = 0},
        {.resource = resource_id::cluster_indices, .element = 0},
    }};

    /// @brief set 0 is the shared scene set: the camera, the light UBO and the two cluster buffers it writes
    inline constexpr std::array<shared_set, 1> cluster_shared_sets = {{{.family = 0}}};

    /**
     * @brief the clustered-light sort's declaration: a one-dimensional compute dispatch over the cluster grid
     *
     * It has no push block at all (the shader reads the light UBO and writes the cluster buffers through the
     * scene set's own bindings), no own binding, no target and no image to transition - the whole declaration is
     * "the shared scene set, and the two buffers I write". Its dispatch size is neither the frame's nor half of
     * it: it is `tiles_x * tiles_y * slices`, which is why its behaviour declares `extent_rule::none` and the
     * host hands the count over in the pass's frame instead.
     */
    inline constexpr pass_io cluster_io = {
        .name = "cluster",
        .own_set = 1, // unused: no own bindings
        .bindings = {},
        .shared_sets = cluster_shared_sets,
        .targets = {},
        .barrier_images = {},
        .barrier_buffers = cluster_barriers,
        .push = std::nullopt,
    };

    /// @brief the deferred lighting stage's two shared sets: the scene set (0) and the G-buffer set (1)
    inline constexpr std::array<shared_set, 2> deferred_shared_sets = {{{.family = 0}, {.family = 1}}};

    /// @brief the resource the deferred lighting stage RENDERS INTO, by declaration
    inline constexpr std::array<render_target, 1> deferred_targets = {{
        {.resource = resource_id::scene_color, .element = 0, .kind = target_kind::color},
    }};

    /**
     * @brief the deferred lighting stage's declaration: a fullscreen triangle that shades every pixel
     *
     * Every binding it uses belongs to one of its TWO SHARED SETS (the scene set: the camera, the IBL, the light
     * UBO, the shadow map; the G-buffer set: the three surface targets, the depth, the velocity), so it declares
     * no binding of its own - the shape the tracer, the lobe and the spatial filter have, with one render target
     * instead of a compute dispatch.
     *
     * THE TARGET IS A RECORDED DEVIATION: it names `scene_color`, and the host hands over whichever image the frame
     * actually lights - `scene_color` while the TAA resolve runs and `hdr` when it does not (the accessor the
     * renderer always used). A `render_target` names one resource today, and a "frame-dependent target" is a
     * framework decision this pass does not get to make on its own; the deviation is written down here and in
     * docs/pass_chain_plan.md rather than hidden.
     *
     * The push block is 92 bytes - a mat4, a vec4 (the SSAO knobs) and three floats (the render mode, whether the
     * traced chain replaces the ambient this frame, and whether the stochastic punctual lighting pass answered it) -
     * and the attachment is LOADed, because the lighting ADDS to the emissive the G-buffer pass already wrote (which
     * is also why the pass builds its pipeline with the additive blend).
     */
    inline constexpr pass_io deferred_io = {
        .name = "deferred",
        .own_set = 2, // unused: no own bindings (every binding it uses lives in the two shared sets)
        .bindings = {},
        .shared_sets = deferred_shared_sets,
        .targets = deferred_targets,
        .barrier_images = {},
        .barrier_buffers = {},
        .push = push_block{.offset = 0, .size = 88, .stages = stage_flag::fragment},
    };

    // =============================================================================================
    // THE POST CHAIN - the composite and the bloom chain's four levels
    // =============================================================================================

    /// @brief the POST set the composite writes the LDR image through and FXAA reads it back from: family 2, its
    ///        fifth set (element 4) - the one stage whose set is not a bloom level
    inline constexpr std::array<shared_set, 1> post_composite_shared_sets = {{{.family = 2, .element = 4}}};

    /// @brief each bloom level's OWN set of the post family: element == level (the set that reads the level before it)
    inline constexpr std::array<std::array<shared_set, 1>, 4> post_bloom_shared_sets = {{
        {{shared_set{.family = 2, .element = 0}}},
        {{shared_set{.family = 2, .element = 1}}},
        {{shared_set{.family = 2, .element = 2}}},
        {{shared_set{.family = 2, .element = 3}}},
    }};

    /// @brief the push block the whole post chain shares, in bytes; its shape is `vulkan.pass.post`'s
    inline constexpr uint32_t post_push_bytes = 52;

    /**
     * @brief the bloom chain's four levels, as ONE declaration per level
     *
     * WHY FOUR DECLARATIONS AND NOT ONE PARAMETERIZED PASS: the framework hands a pass ONE descriptor set per
     * shared owner (`resolved_io::shared`), and each of the four stages binds a DIFFERENT one of the post family's
     * five sets (set `level` reads level `level - 1` as its input), so the LEVEL is the pass boundary. Each entry
     * here is one pass's whole I/O:
     *
     *  * `targets` is the level it writes (element `level` of the `bloom` family, which the schema declares with
     *    four elements - see its `count`);
     *  * `barrier_images` is the level it READS, which is also the image the pass has to move to a sampled layout
     *    before it renders. Level 0 reads the HDR target instead, and that transition is deliberately NOT here:
     *    the host owns it, because the composite reads HDR too and because it is needed on the frames the whole
     *    bloom chain is skipped (see `post_composite_io`);
     *  * the LAST level's pass also hands its own output back to a sampled layout, from its target - the
     *    composite samples all four levels, and level 3 has no successor to do it (the writer's hand-back the GI
     *    chain's passes also use).
     *
     * The push block is the post chain's ONE block (`post_push_bytes`): the shader declares it whole, every stage
     * writes a different `mode` into it, and the bytes the stages do not read are pushed with the same defaults
     * they always were - the prefilter passes `mode = 0`, the three downsamples `mode = 1`.
     */
    /// the levels' named target and source lists: a `pass_io` holds SPANS, so every list it names has to outlive
    /// it (a brace-initialized array inside the initializer would be a temporary the span dangles on)
    inline constexpr std::array<render_target, 1> post_bloom_0_target = {{render_target{.resource = resource_id::bloom, .element = 0, .kind = target_kind::color}}};
    inline constexpr std::array<render_target, 1> post_bloom_1_target = {{render_target{.resource = resource_id::bloom, .element = 1, .kind = target_kind::color}}};
    inline constexpr std::array<render_target, 1> post_bloom_2_target = {{render_target{.resource = resource_id::bloom, .element = 2, .kind = target_kind::color}}};
    inline constexpr std::array<render_target, 1> post_bloom_3_target = {{render_target{.resource = resource_id::bloom, .element = 3, .kind = target_kind::color}}};
    inline constexpr std::array<barrier_image, 1> post_bloom_1_source = {{barrier_image{.resource = resource_id::bloom, .element = 0}}};
    inline constexpr std::array<barrier_image, 1> post_bloom_2_source = {{barrier_image{.resource = resource_id::bloom, .element = 1}}};
    inline constexpr std::array<barrier_image, 1> post_bloom_3_source = {{barrier_image{.resource = resource_id::bloom, .element = 2}}};

    inline constexpr std::array<pass_io, 4> post_bloom_io = {{
        {.name = "post_bloom_0",
         .own_set = 0, // unused: no own bindings (everything it reads is in the post set)
         .bindings = {},
         .shared_sets = post_bloom_shared_sets[0],
         .targets = post_bloom_0_target,
         .barrier_images = {},
         .barrier_buffers = {},
         .push = push_block{.offset = 0, .size = post_push_bytes, .stages = stage_flag::fragment}},
        {.name = "post_bloom_1",
         .own_set = 0,
         .bindings = {},
         .shared_sets = post_bloom_shared_sets[1],
         .targets = post_bloom_1_target,
         .barrier_images = post_bloom_1_source,
         .barrier_buffers = {},
         .push = push_block{.offset = 0, .size = post_push_bytes, .stages = stage_flag::fragment}},
        {.name = "post_bloom_2",
         .own_set = 0,
         .bindings = {},
         .shared_sets = post_bloom_shared_sets[2],
         .targets = post_bloom_2_target,
         .barrier_images = post_bloom_2_source,
         .barrier_buffers = {},
         .push = push_block{.offset = 0, .size = post_push_bytes, .stages = stage_flag::fragment}},
        {.name = "post_bloom_3",
         .own_set = 0,
         .bindings = {},
         .shared_sets = post_bloom_shared_sets[3],
         .targets = post_bloom_3_target,
         .barrier_images = post_bloom_3_source,
         .barrier_buffers = {},
         .push = push_block{.offset = 0, .size = post_push_bytes, .stages = stage_flag::fragment}},
    }};

    /// @brief what the composite RENDERS INTO by declaration: the swapchain (the FXAA-off case; see its comment)
    inline constexpr std::array<render_target, 1> post_composite_targets = {{render_target{.resource = resource_id::swapchain_image, .element = 0, .kind = target_kind::color}}};

    /**
     * @brief the composite's declaration: HDR plus the weighted bloom levels, tonemapped to the display
     *
     * It renders into the SWAPCHAIN by declaration and into the LDR image on the frames FXAA runs - a RECORDED
     * DEVIATION, the same one `deferred_io` records for `scene_color` and for the same reason: `render_target`
     * names one resource, and the composite's target decides its PIPELINE too (the swapchain's format against the
     * LDR image's R16F), so the host hands over both together.
     *
     * NO BARRIER IMAGE IS DECLARED, and that is the host's half rather than an omission: the HDR target's
     * transition to a sampled layout happens before the chain on EVERY frame (the composite reads it whether or
     * not bloom runs, and the bloom chain's prefilter reads it as well), so it has exactly one owner and that
     * owner has to be the frame loop. The four bloom levels arrive already sampled - the levels' own passes
     * moved them, or the host's off path did on a frame the chain was skipped.
     */
    inline constexpr pass_io post_composite_io = {
        .name = "post_composite",
        .own_set = 0, // unused: no own bindings (everything it reads is in the post set)
        .bindings = {},
        .shared_sets = post_composite_shared_sets,
        .targets = post_composite_targets,
        .barrier_images = {},
        .barrier_buffers = {},
        .push = push_block{.offset = 0, .size = post_push_bytes, .stages = stage_flag::fragment},
    };

    /// @brief what the FXAA pass RENDERS INTO by declaration: the swapchain, which it is the last writer of
    inline constexpr std::array<render_target, 1> fxaa_targets = {{render_target{.resource = resource_id::swapchain_image, .element = 0, .kind = target_kind::color}}};

    /// @brief the image the FXAA pass reads and therefore has to move to a sampled layout: the composite's LDR output
    inline constexpr std::array<barrier_image, 1> fxaa_barriers = {{barrier_image{.resource = resource_id::ldr, .element = 0}}};

    /**
     * @brief the FXAA pass's declaration: the gamma-encoded LDR image -> the anti-aliased swapchain
     *
     * It binds the SAME shared post set the composite does (set 4 of the family - the composite writes the LDR
     * image through binding 5 and FXAA reads it back through it, which is why FXAA cannot be folded into
     * `post.frag`: a descriptor may not name the image the pipeline is rendering into, and that would be a
     * different statically-used binding set).
     *
     * ITS INPUT IS DECLARED, unlike the composite's, and the difference is the frame: the LDR image is written by
     * the composite and read here, so this pass is the one that moves it - there is no frame where it has to be
     * moved and no pass runs, because a frame without FXAA never touches it at all.
     *
     * The push block is the chain's own (`post_push_bytes`): FXAA is mode 3 of the same shader's block.
     */
    inline constexpr pass_io fxaa_io = {
        .name = "fxaa",
        .own_set = 0, // unused: no own bindings (everything it reads is in the post set)
        .bindings = {},
        .shared_sets = post_composite_shared_sets,
        .targets = fxaa_targets,
        .barrier_images = fxaa_barriers,
        .barrier_buffers = {},
        .push = push_block{.offset = 0, .size = post_push_bytes, .stages = stage_flag::fragment},
    };

    /// @brief set 1 is the shared G-buffer set, whose LAYOUT the debug view's pass owns (it is the one that
    ///        generates it) and whose per-image SETS the renderer writes (six consumers bind them)
    inline constexpr std::array<shared_set, 1> gbuffer_debug_shared_sets = {{{.family = 1}}};

    /// @brief what the debug view RENDERS INTO: the HDR target, which is the image it actually writes - so unlike
    ///        the deferred stage's and the composite's, this declaration carries no deviation
    inline constexpr std::array<render_target, 1> gbuffer_debug_targets = {{render_target{.resource = resource_id::hdr, .element = 0, .kind = target_kind::color}}};

    /// @brief the four images it moves to a sampled layout: the three stored surface targets and the motion vectors
    inline constexpr std::array<barrier_image, 4> gbuffer_debug_barriers = {{barrier_image{.resource = resource_id::gbuffer_targets, .element = 0},
                                                                             barrier_image{.resource = resource_id::gbuffer_targets, .element = 1},
                                                                             barrier_image{.resource = resource_id::gbuffer_targets, .element = 2},
                                                                             barrier_image{.resource = resource_id::velocity, .element = 0}}};

    /**
     * @brief the G-buffer debug view's declaration: the stored surface, one channel at a time, into the HDR target
     *
     * Everything it READS is in the shared G-buffer set (the three targets, the depth, the velocity), so it declares
     * no binding of its own - and the DEPTH is deliberately not among its barrier images: its old layout depends on
     * whether the G-buffer instance rendered this frame, which is shared per-image bookkeeping the host owns (the
     * same hand-back the deferred stage's frame carries as a callback). The velocity target IS declared, because the
     * debug view runs INSTEAD of the lighting stage and is therefore the only stage that hands it to a sampler on
     * those frames.
     *
     * The push block is 16 bytes - the channel, the two projection terms the depth channel linearizes with, and the
     * motion gain - and all four values are the frame's (the renderer's knob and camera).
     */
    inline constexpr pass_io gbuffer_debug_io = {
        .name = "gbuffer-debug",
        .own_set = 0, // unused: no own bindings (everything it reads is in the shared G-buffer set)
        .bindings = {},
        .shared_sets = gbuffer_debug_shared_sets,
        .targets = gbuffer_debug_targets,
        .barrier_images = gbuffer_debug_barriers,
        .barrier_buffers = {},
        .push = push_block{.offset = 0, .size = 16, .stages = stage_flag::fragment},
    };
    /// @brief set 0 is the shared scene set: the shadow pass binds it for the light UBO, the material table and the
    ///        texture array its depth-only draw reads, through the same scene pipeline layout the leaves use
    inline constexpr std::array<shared_set, 1> shadow_shared_sets = {{{.family = 0}}};

    /// @brief the layers the shadow pass renders into: a RUN of elements, one rendering instance per cascade
    ///
    /// THIS USED TO BE A RECORDED DEVIATION, and the vocabulary that closed it is `render_target::count`: the
    /// declaration named the image's FIRST layer and the host handed over the layers this frame has
    /// (`runtime::resolve_shadow_pass`). The deviation was forced by the framework - the validator refuses a
    /// second DEPTH target, because a rendering instance has one depth attachment - but it was never a fact about
    /// the shadow pass: that pass renders 1..`max_shadow_cascades` LAYERS of ONE array image, one instance per
    /// cascade, which is exactly a run of elements. The count is the schema's own four (the family's count), and
    /// the FRAME caps it: `ensure_shadow_resources` allocates the layers the cascade knob asks for, and only
    /// those layers are published - so a one-cascade frame resolves one target and a three-cascade frame three.
    inline constexpr std::array<render_target, 1> shadow_targets = {{render_target{.resource = resource_id::shadow_map, .element = 0, .kind = target_kind::depth, .count = 4}}};
    /**
     * @brief the shadow pass's declaration: the scene's depth from the light, one cascade at a time
     *
     * The PUSH BLOCK is four bytes at offset 96 - the cascade index, pushed at `vulkan::scene_cascade_push_offset`
     * (`scene_push_constant_size`), which is where the scene's own push block ends: this pass draws the scene's
     * leaves through the SCENE pipeline layout, so its push shares that layout's range rather than owning one. The
     * literal is checked against the constant where both are visible (`vulkan.pass.shadow`'s static_assert), the
     * same "two copies, one size" rule the other pass blocks follow.
     *
     * NO BARRIER IMAGES: each layer is transitioned to a depth attachment immediately before the instance that
     * renders it, so the pass reaches every image it moves through `targets` - the shape the composite and the
     * debug view have. The targets are a RUN (`shadow_targets`), which is what makes that reach enough: the run
     * is however many layers this frame's map has, in cascade order.
     */
    inline constexpr pass_io shadow_io = {
        .name = "shadow",
        .own_set = 0, // unused: no own bindings (the scene set carries everything the draw reads)
        .bindings = {},
        .shared_sets = shadow_shared_sets,
        .targets = shadow_targets,
        .barrier_images = {},
        .barrier_buffers = {},
        .push = push_block{.offset = 96, .size = 4, .stages = stage_flag::vertex | stage_flag::fragment},
    };
} // namespace vulkan::render_resource
