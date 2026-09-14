// module version: 0.2.0  (independent of the app version in CMakeLists project(VERSION))

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
    inline constexpr std::array<resource_info, 37> resource_schema = {{
        {.id = resource_id::swapchain_image, .name = "swapchain_image", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::imported},
        {.id = resource_id::hdr, .name = "hdr", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
        {.id = resource_id::bloom, .name = "bloom", .kind = resource_kind::image2d, .scope = resource_scope::per_swapchain_image, .lifetime = resource_lifetime::per_frame},
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
        {.id = resource_id::shadow_map, .name = "shadow_map", .kind = resource_kind::image2d, .scope = resource_scope::per_frame_slot, .lifetime = resource_lifetime::per_frame},
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
     * the one that opens the rendering instance (see `behaviour_kind::fullscreen`), so it is the one that says
     * whether the old contents matter. What is declared is only what the layer has to know: which resource,
     * and which image of its family.
     */
    struct render_target {
        resource_id resource = resource_id::none;
        uint16_t element = 0; // which image of the resource's family (per-swapchain-image resources: the index)
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
        /// the images this pass renders into, in the order it uses them (a fullscreen pass has one)
        std::span<render_target const> targets = {};
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
     * itself twice.
     * @ingroup vulkan_render_resource
     */
    [[nodiscard]] inline std::expected<void, std::string> validate(pass_io const& io) {
        if (io.name.empty()) {
            return std::unexpected("a pass declaration has no name");
        }
        std::string const who{io.name};
        std::size_t own_count = 0;
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
            for (render_target const& other : io.targets) {
                if (&other != &t && other.resource == t.resource && other.element == t.element) {
                    return std::unexpected(where + " is declared twice");
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

    /// @brief the probe cache's declaration
    /// @ingroup vulkan_render_resource
    inline constexpr pass_io gi_probe_io = {
        .name = "gi_probe",
        .own_set = 1,
        .bindings = gi_probe_bindings,
        // The pass's push block, DECLARED here because it is the range its pipeline layout is built with and
        // the size its host must compose. It is the pass's own struct
        // (`vulkan.pass.gi_probe::gi_probe_pass::push_constants`), and `vulkan.pass.gi_probe` carries the
        // `static_assert` that ties this number to that struct - a fact in two units that the compiler keeps
        // in agreement is the next best thing to a fact in one.
        .push = push_block{.offset = 0, .size = 56, .stages = stage_flag::compute},
    };

} // namespace vulkan::render_resource
