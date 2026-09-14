// module version: 0.7.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/bindings/bindings.cppm
 * @defgroup vulkan_bindings Descriptor Set Families
 * @brief The descriptor-set side of the runtime: the per-swapchain-image set families and the
 *        per-frame-slot scene sets, with the pool lifetime rule that belongs to them.
 *
 * Extracted from vulkan.runtime. What made this worth its own module is that the three per-image
 * families - the post chain's, the G-buffer debug view's and TAA's - are THE SAME ALGORITHM read line
 * by line, differing only in how many sets each image needs (5 for post: the prefilter, three
 * downsamples and the composite; 1 for the other two), which views each set points at, and which
 * sampler it uses. So they become one class instead of three functions, and the property that keeps
 * descriptor writes off the per-frame path - "if the bound views are unchanged, do nothing" - moves
 * from being an incidental early return in each of them into the structure of that class.
 *
 * The pool lifetime rule travels with the pools, because it is only about them: a pool must not be
 * destroyed while a recorded command buffer still names one of its sets (VUID-vkDestroyDescriptorPool
 * -descriptorPool-00303), so a pool that is replaced is RETIRED rather than destroyed. Callers retire
 * their families when the swapchain is recreated (runtime::on_swapchain_recreated).
 *
 * The interface takes handles, never the runtime: the core's per-image views and the runtime's buffers,
 * images and samplers arrive as parameters, so nothing here can reach into either.
 */

module;

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

export module vulkan.bindings;

import vulkan.core;
import vulkan.render_resource;
import vulkan.render_resource.shared;

namespace vulkan::bindings {

    // =============================================================================================
    // A DECLARATION -> A DESCRIPTOR SET LAYOUT
    //
    // The half of `vulkan.render_resource` that needs Vulkan: the description layer is pure CPU on purpose
    // (so its own invariants are testable without a device), and everything that has to name a
    // `VkDescriptorType` or call `vkCreateDescriptorSetLayout` lives here instead.
    //
    // WHY IT IS WORTH GENERATING AT ALL, in this project's own history: the layout and the descriptor WRITES
    // were two hand-written halves kept in agreement by discipline, and both drifts that pair can have have
    // already happened - a pool sized for four descriptors per set while the layout asked for five, and a
    // binding whose type changed without its writer noticing. Both were found by the validation layer rather
    // than by review. One declaration, from which the layout and the writes are generated, removes the pair.
    // =============================================================================================

    /// @brief the `VkDescriptorType` a declared binding kind means
    /// @ingroup vulkan_bindings
    export [[nodiscard]] constexpr VkDescriptorType descriptor_type_of(render_resource::binding_kind const kind) noexcept {
        switch (kind) {
        case render_resource::binding_kind::sampled_image:
            return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        case render_resource::binding_kind::storage_image:
            return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case render_resource::binding_kind::sampler:
            return VK_DESCRIPTOR_TYPE_SAMPLER;
        case render_resource::binding_kind::uniform_buffer:
            return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case render_resource::binding_kind::storage_buffer:
            return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        case render_resource::binding_kind::input_attachment:
            return VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        case render_resource::binding_kind::acceleration_structure:
            return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        }
        return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }

    /// @brief the shader stages a declared stage set means (a shared set serves two, hence the union)
    /// @ingroup vulkan_bindings
    export [[nodiscard]] constexpr VkShaderStageFlags stage_flags_of(render_resource::stage_flag const stages) noexcept {
        VkShaderStageFlags flags = 0u;
        if (render_resource::has_stage(stages, render_resource::stage_flag::vertex)) {
            flags |= VK_SHADER_STAGE_VERTEX_BIT;
        }
        if (render_resource::has_stage(stages, render_resource::stage_flag::fragment)) {
            flags |= VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        if (render_resource::has_stage(stages, render_resource::stage_flag::compute)) {
            flags |= VK_SHADER_STAGE_COMPUTE_BIT;
        }
        return flags;
    }

    /// @brief how many bindings one set of one declaration may hold (the generator's fixed buffer)
    /// @ingroup vulkan_bindings
    export inline constexpr uint32_t max_set_bindings = 32;

    /**
     * @brief build a `VkDescriptorSetLayout` from a pass's declaration
     * @param vk the device to create it on
     * @param io the declaration; its bindings for @p set become the layout, and everything else is ignored
     * @param set which set to generate - the pass's own, normally `io.own_set`
     * @return the layout, or a message naming the pass and the set it was building
     * @ingroup vulkan_bindings
     *
     * THE BINDINGS ARE EMITTED IN DECLARATION ORDER, which is why `vulkan.render_resource::validate` requires
     * a pass's own bindings to be numbered contiguously from zero: the generated layout is then the same thing
     * the shader declares, and the number in the shader and the number in the declaration cannot drift.
     */
    export [[nodiscard]] inline std::expected<VkDescriptorSetLayout, std::string> make_set_layout(core const& vk, render_resource::pass_io const& io, uint32_t const set) {
        std::array<VkDescriptorSetLayoutBinding, max_set_bindings> bindings = {};
        uint32_t count = 0;
        for (render_resource::pass_binding const& b : io.bindings) {
            if (b.set != set) {
                continue;
            }
            if (count == bindings.size()) {
                return std::unexpected(std::string(io.name) + ": set " + std::to_string(set) + " declares more than " + std::to_string(max_set_bindings) + " bindings");
            }
            bindings[count].binding = b.binding;
            bindings[count].descriptorType = descriptor_type_of(b.kind);
            bindings[count].descriptorCount = b.descriptor_count;
            bindings[count].stageFlags = stage_flags_of(b.stages);
            bindings[count].pImmutableSamplers = nullptr;
            ++count;
        }
        VkDescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = count;
        layout_info.pBindings = bindings.data();
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(vk.device, &layout_info, nullptr, &layout) != VK_SUCCESS) {
            return std::unexpected(std::string(io.name) + ": set " + std::to_string(set) + " layout creation failed");
        }
        return layout;
    }

    /// @brief the `VkImageLayout` a declared binding layout means
    /// @ingroup vulkan_bindings
    export [[nodiscard]] constexpr VkImageLayout image_layout_of(render_resource::image_layout const layout) noexcept {
        switch (layout) {
        case render_resource::image_layout::sampled:
            return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        case render_resource::image_layout::general:
            return VK_IMAGE_LAYOUT_GENERAL;
        }
        return VK_IMAGE_LAYOUT_UNDEFINED;
    }

    // The six samplers a declaration chooses between now live in `vulkan.render_resource.shared`, with the rest
    // of the shared handles, for the reason that module's header gives: the description layer stays pure CPU so
    // its invariants are testable on a machine with no GPU, and every handle lives on this side of it.

    /**
     * @brief write a set's descriptors from a pass's declaration
     * @param vk the device that owns @p target
     * @param io the declaration whose bindings for @p set are written
     * @param set which set to write
     * @param target the descriptor set to write into
     * @param views the view for each binding NUMBER of that set (only the image kinds read it)
     * @param buffers the buffer for each binding NUMBER of that set (only the buffer kinds read it)
     * @param samplers the renderer's six samplers, chosen by each binding's declared hint
     * @return nothing, or a message naming the pass and the binding that could not be written
     * @ingroup vulkan_bindings
     *
     * THE OTHER HALF OF THE GENERATOR, and the half that removes the drift this project actually suffered: the
     * layout (above) and the writes (here) come from one declaration, so a binding's type, its descriptor
     * count, its layout and its sampler cannot disagree with the layout they are written against. Two cases
     * are refused rather than guessed, and both are refused LOUDLY because guessing them is how a descriptor
     * silently points at the wrong thing: an ARRAY binding (`descriptor_count > 1`, the bindless texture
     * array) is written element by element by the owner of that capacity, and an ACCELERATION STRUCTURE needs
     * a `pNext` chain this function does not build - the top level structure is bound by the scene set's own
     * owner today, which is where it belongs until a pass declares one of its own.
     */
    export [[nodiscard]] inline std::expected<void, std::string> write_set(core const& vk, render_resource::pass_io const& io, uint32_t const set, VkDescriptorSet const target,
                                                                           std::span<VkImageView const> const views, std::span<VkBuffer const> const buffers, render_resource::shared::sampler_set const& samplers) {
        std::array<VkDescriptorImageInfo, max_set_bindings> image_infos = {};
        std::array<VkDescriptorBufferInfo, max_set_bindings> buffer_infos = {};
        std::array<VkWriteDescriptorSet, max_set_bindings> writes = {};
        uint32_t count = 0;
        for (render_resource::pass_binding const& b : io.bindings) {
            if (b.set != set) {
                continue;
            }
            std::string const where = std::string(io.name) + ": set " + std::to_string(set) + " binding " + std::to_string(b.binding);
            if (count == writes.size()) {
                return std::unexpected(std::string(io.name) + ": set " + std::to_string(set) + " declares more than " + std::to_string(max_set_bindings) + " bindings");
            }
            if (b.descriptor_count != 1) {
                return std::unexpected(where + " is an array binding, which its owner writes");
            }
            VkWriteDescriptorSet& write = writes[count];
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = target;
            write.dstBinding = b.binding;
            write.descriptorCount = b.descriptor_count;
            write.descriptorType = descriptor_type_of(b.kind);
            switch (b.kind) {
            case render_resource::binding_kind::sampled_image:
            case render_resource::binding_kind::storage_image:
            case render_resource::binding_kind::input_attachment:
            case render_resource::binding_kind::sampler: {
                if (b.binding >= views.size()) {
                    return std::unexpected(where + " has no view to bind");
                }
                image_infos[count].sampler = samplers.of(b.sampler);
                image_infos[count].imageView = b.kind == render_resource::binding_kind::sampler ? VK_NULL_HANDLE : views[b.binding];
                image_infos[count].imageLayout = image_layout_of(b.layout);
                write.pImageInfo = &image_infos[count];
                break;
            }
            case render_resource::binding_kind::uniform_buffer:
            case render_resource::binding_kind::storage_buffer: {
                if (b.binding >= buffers.size()) {
                    return std::unexpected(where + " has no buffer to bind");
                }
                buffer_infos[count].buffer = buffers[b.binding];
                buffer_infos[count].offset = 0;
                buffer_infos[count].range = VK_WHOLE_SIZE;
                write.pBufferInfo = &buffer_infos[count];
                break;
            }
            case render_resource::binding_kind::acceleration_structure:
                return std::unexpected(where + " is an acceleration structure, which the owner of its set writes");
            }
            ++count;
        }
        if (count != 0) {
            vkUpdateDescriptorSets(vk.device, count, writes.data(), 0, nullptr);
        }
        return {};
    }

    /**
     * @brief the descriptor sets of one per-image family (post chain, G-buffer debug view, TAA)
     * @ingroup vulkan_bindings
     *
     * ensure() is called once per frame and is a no-op unless the views it would bind have changed
     * (a swapchain recreation, a resize, or the first call after creation): it compares the signature
     * view of every family member against what it bound last time. That comparison is what keeps a
     * vkUpdateDescriptorSets off the frame path, so it is deliberately part of the class rather than a
     * caller's responsibility.
     *
     * @note the sets are flattened: set(image, which) is the @c which -th set of @c image, with
     *       @c sets_per_image entries per image (post: 0=prefilter, 1..3=downsamples, 4=composite).
     */
    /**
     * @brief the frame slots' scene descriptor sets: one set per slot, never re-pointed
     * @ingroup vulkan_bindings
     *
     * The scene set is what every pipeline that draws the scene binds first (camera UBO, IBL, material
     * table, instance transforms, per-slot light UBO and shadow map, per-slot skin/morph and cluster
     * buffers). Its shape differs from image_set_family in two ways that matter: there is one set per
     * FRAME SLOT rather than per swapchain image, and its sets come from the core
     * (core::make_descriptor_set), so there is no pool of its own to size and retire - which is why
     * this class is small and the pool lifetime rule stays entirely with image_set_family.
     *
     * What it does own is the rule the old scattered arrays also encoded: a slot's set always points
     * at that slot's own camera / shadow / skin resources, so an in-flight frame never observes the
     * next frame's descriptors, and update_all() applies one batch of writes to every slot rather
     * than to one.
     *
     * The bindings that need the runtime's buffer details (the camera / material / instance / skin /
     * morph / cluster buffers, binding 0 and 5-12) still get written by the runtime: only the ones
     * that take plain handles - the IBL views (2-4) - are written here.
     */
    export class scene_bindings {
    public:
        /// create one set per frame slot from the scene layout; a no-op once they exist
        void create(core const& vk, VkDescriptorSetLayout layout);

        /// whether the sets exist (before that, set() is null and the writers are no-ops)
        [[nodiscard]] bool created() const noexcept;

        /// the scene set of @p slot, or VK_NULL_HANDLE when there is none
        [[nodiscard]] VkDescriptorSet set(uint32_t slot) const noexcept;

        /// apply one batch of writes to every slot's set: dstSet is replaced per slot
        void update_all(core const& vk, VkWriteDescriptorSet const* writes, uint32_t write_count) const;

        /// bindings 2-4: the three environment views, or the white placeholder while IBL is not loaded
        void write_ibl(core const& vk, bool ibl_ready, std::span<VkImageView const> ibl_views, VkSampler env_sampler, VkImageView placeholder_view, VkSampler placeholder_sampler) const;

    private:
        std::array<vk_descriptor_set, core::MAX_FRAMES_IN_FLIGHT> sets = {};
        bool created_flag = false;
    };

    export class image_set_family {
    public:
        /// describes one image's sets to the family: the caller writes them (bindings depend on the
        /// pass's own view layout, which only the call site knows)
        using write_sets_fn = std::function<void(core const& vk, uint32_t image_index, std::span<VkDescriptorSet const> sets)>;

        /**
         * @brief make sure this family's sets exist, are allocated for @p signature_views.size() images
         *        and point at @p signature_views; returns false when the inputs are not usable yet
         * @param vk the core (device, and the per-image views the caller passes in the callback)
         * @param layout the set layout the family allocates from (owned by the pass, see vulkan.pipelines)
         * @param image_count how many images the family must serve (the swapchain generation's count)
         * @param sets_per_image how many sets each image needs (1 unless a chain like post's needs more)
         * @param descriptors_per_set how many combined-image-sampler descriptors one of those sets holds
         *        (the caller decides the pool size, so a wrong count shows up as an allocation failure)
         * @param signature_views the views whose identity decides whether a rebind is needed: one per
         *        image, or - as the G-buffer debug view passes it - the few views of image 0 that
         *        identify the generation. It says NOTHING about how many images there are (image_count
         *        does), which is exactly the distinction the first version of this class got wrong
         * @param write called for each image that needs (re)binding, with that image's sets
         */
        /// one fingerprint per thing the family's bindings depend on (the post chain compares the HDR,
        /// the bloom and the LDR view lists; the others need one)
        using signatures_t = std::span<std::span<VkImageView const> const>;

        [[nodiscard]] bool ensure(core const& vk,
                                  VkDescriptorSetLayout layout,
                                  uint32_t image_count,
                                  uint32_t sets_per_image,
                                  uint32_t descriptors_per_set,
                                  std::span<VkImageView const> signature_views,
                                  write_sets_fn const& write);

        /**
         * @brief ensure() for a family whose rebind depends on more than one fingerprint: every list is
         *        compared and a rebind happens when any of them changed
         * @param signatures the fingerprints (empty is rejected like a null layout)
         */
        [[nodiscard]] bool ensure_all(core const& vk,
                                      VkDescriptorSetLayout layout,
                                      uint32_t image_count,
                                      uint32_t sets_per_image,
                                      uint32_t descriptors_per_set,
                                      signatures_t signatures,
                                      write_sets_fn const& write);

        /// every set of image @p image_index, or an empty span when the family is not ready
        [[nodiscard]] std::span<VkDescriptorSet const> sets(uint32_t image_index) const noexcept;

        /// the @c which -th set of image @p image_index (VK_NULL_HANDLE when out of range)
        [[nodiscard]] VkDescriptorSet set(uint32_t image_index, uint32_t which) const noexcept;

        /// whether the sets are allocated and bound
        [[nodiscard]] bool ready() const noexcept;

        /// how many images the family is currently allocated for (0 before the first ensure)
        [[nodiscard]] uint32_t image_count() const noexcept;

        /// retire the current pool instead of destroying it: a pool must outlive every recorded command
        /// buffer that names one of its sets (see the class comment). Called on swapchain recreation.
        void retire_all();

        /// destroys every pool this family owns, the retired ones included: they outlived their
        /// generation on purpose, so they may only go away once the runtime itself does
        ~image_set_family();

    private:
        VkDevice device = VK_NULL_HANDLE; // remembered by ensure() so the destructor can destroy the pools
        VkDescriptorPool pool = VK_NULL_HANDLE;
        uint32_t pool_capacity = 0;
        uint32_t sets_per_image = 0;
        std::vector<VkDescriptorSet> flat_sets = {};                 // sets_per_image entries per image
        std::vector<std::vector<VkImageView>> bound_signatures = {}; // what the current sets point at
        uint32_t images = 0;                                         // how many images they serve
        std::vector<VkDescriptorPool> retired = {};                  // destroyed with the runtime, never earlier
    };
    // ---- scene_bindings ----------------------------------------------------------------------------
    void scene_bindings::create(core const& vk, VkDescriptorSetLayout const layout) {
        if (this->created_flag || layout == VK_NULL_HANDLE) {
            return;
        }
        for (std::size_t slot = 0; slot < this->sets.size(); ++slot) {
            this->sets[slot] = vk.make_descriptor_set(layout);
        }
        this->created_flag = true;
    }

    bool scene_bindings::created() const noexcept {
        return this->created_flag;
    }

    VkDescriptorSet scene_bindings::set(uint32_t const slot) const noexcept {
        if (!this->created_flag || slot >= this->sets.size()) {
            return VK_NULL_HANDLE;
        }
        return *this->sets[slot];
    }

    void scene_bindings::update_all(core const& vk, VkWriteDescriptorSet const* writes, uint32_t const write_count) const {
        if (!this->created_flag || write_count == 0) {
            return;
        }
        std::vector<VkWriteDescriptorSet> per_set(writes, writes + write_count);
        for (vk_descriptor_set const& scene_set : this->sets) {
            for (VkWriteDescriptorSet& write : per_set) {
                write.dstSet = *scene_set;
            }
            vkUpdateDescriptorSets(vk.device, write_count, per_set.data(), 0, nullptr);
        }
    }

    void scene_bindings::write_ibl(core const& vk, bool const ibl_ready, std::span<VkImageView const> const ibl_views, VkSampler const env_sampler,
                                   VkImageView const placeholder_view, VkSampler const placeholder_sampler) const {
        if (!this->created_flag) {
            return;
        }
        std::array<VkDescriptorImageInfo, 3> image_infos = {};
        for (std::size_t i = 0; i < image_infos.size(); ++i) {
            bool const use_env = ibl_ready && i < ibl_views.size();
            image_infos[i] = {
                .sampler = use_env ? env_sampler : placeholder_sampler,
                .imageView = use_env ? ibl_views[i] : placeholder_view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
        }
        for (vk_descriptor_set const& scene_set : this->sets) {
            std::array<VkWriteDescriptorSet, 3> writes = {};
            for (std::size_t i = 0; i < writes.size(); ++i) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = *scene_set;
                writes[i].dstBinding = static_cast<uint32_t>(2 + i);
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[i].pImageInfo = &image_infos[i];
            }
            vkUpdateDescriptorSets(vk.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }

    // ---- image_set_family --------------------------------------------------------------------------
    // The algorithm the three per-image families shared: ready check, image count check, signature
    // comparison, pool (re)creation with retirement, allocation, then the caller's writes. Everything a
    // family knows that the class cannot is a parameter: how many sets an image needs, how many
    // descriptors a set holds (which sizes the pool), which views decide a rebind, and what to write.
    namespace {
        [[nodiscard]] bool same_views(std::vector<VkImageView> const& bound, std::span<VkImageView const> const wanted) {
            return bound.size() == wanted.size() && std::equal(bound.begin(), bound.end(), wanted.begin());
        }

        [[nodiscard]] bool same_signatures(std::vector<std::vector<VkImageView>> const& bound, std::span<std::span<VkImageView const> const> const wanted) {
            if (bound.size() != wanted.size()) {
                return false;
            }
            for (std::size_t i = 0; i < bound.size(); ++i) {
                if (!same_views(bound[i], wanted[i])) {
                    return false;
                }
            }
            return true;
        }
    } // namespace

    image_set_family::~image_set_family() {
        for (VkDescriptorPool const pooled : this->retired) {
            vkDestroyDescriptorPool(this->device, pooled, nullptr);
        }
        this->retired.clear();
        if (this->pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(this->device, this->pool, nullptr);
            this->pool = VK_NULL_HANDLE;
        }
    }

    bool image_set_family::ensure(core const& vk, VkDescriptorSetLayout const layout, uint32_t const image_count, uint32_t const per_image,
                                  uint32_t const descriptors_per_set, std::span<VkImageView const> const signature_views, write_sets_fn const& write) {
        std::array<std::span<VkImageView const>, 1> const one = {signature_views};
        return this->ensure_all(vk, layout, image_count, per_image, descriptors_per_set, one, write);
    }

    bool image_set_family::ensure_all(core const& vk, VkDescriptorSetLayout const layout, uint32_t const image_count, uint32_t const per_image,
                                      uint32_t const descriptors_per_set, signatures_t const signatures, write_sets_fn const& write) {
        this->device = vk.device;
        if (layout == VK_NULL_HANDLE || !static_cast<bool>(write) || image_count == 0 || per_image == 0 || signatures.empty()) {
            return false;
        }
        // The property that keeps descriptor writes off the frame path: the sets stay allocated, and their
        // contents are rewritten only when the views they point at actually change.
        if (this->sets_per_image == per_image && this->flat_sets.size() == static_cast<std::size_t>(image_count) * per_image && same_signatures(this->bound_signatures, signatures)) {
            return true;
        }

        if (this->pool == VK_NULL_HANDLE || this->pool_capacity != image_count || this->sets_per_image != per_image) {
            // A pool cannot grow and its sets are still allocated, so a different image count needs a new
            // pool; the old one is RETIRED rather than destroyed, because recorded command buffers still
            // name its sets (VUID-vkDestroyDescriptorPool-descriptorPool-00303).
            if (this->pool != VK_NULL_HANDLE) {
                this->retired.push_back(this->pool);
                this->pool = VK_NULL_HANDLE;
            }
            VkDescriptorPoolSize pool_size = {};
            pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            pool_size.descriptorCount = static_cast<uint32_t>(static_cast<std::size_t>(image_count) * per_image * descriptors_per_set);
            // TWO sizes, because these families are not all sampler-only: the G-buffer set declares two
            // STORAGE images (the raw GI trace and the filtered GI) and the GI denoiser's set one, so a
            // pool that lists only COMBINED_IMAGE_SAMPLER is missing a type its own layouts declare.
            // Validation reports exactly that ("binding 6 was created with
            // VK_DESCRIPTOR_TYPE_STORAGE_IMAGE but VkDescriptorPool ... was not created with any
            // VkDescriptorPoolSize::type with VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"), and - as the message
            // itself warns - a driver is allowed to return VK_ERROR_OUT_OF_POOL_MEMORY for it instead of
            // tolerating it, which would fail the allocation and take the pass with it.
            //
            // The count is over-provisioned on purpose: a pool size is a capacity, not an allocation, so
            // giving both types the full budget costs nothing and keeps this independent of which family
            // happens to declare which mix.
            std::array<VkDescriptorPoolSize, 2> pool_sizes = {};
            pool_sizes[0] = pool_size;
            pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            pool_sizes[1].descriptorCount = pool_size.descriptorCount;
            VkDescriptorPoolCreateInfo pool_info = {};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = static_cast<uint32_t>(static_cast<std::size_t>(image_count) * per_image);
            pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
            pool_info.pPoolSizes = pool_sizes.data();
            if (vkCreateDescriptorPool(this->device, &pool_info, nullptr, &this->pool) != VK_SUCCESS) {
                this->pool = VK_NULL_HANDLE;
                this->pool_capacity = 0;
                this->sets_per_image = 0;
                this->flat_sets.clear();
                this->bound_signatures.clear();
                return false;
            }
            this->pool_capacity = image_count;
            this->sets_per_image = per_image;
            this->flat_sets.clear();
            this->bound_signatures.clear();
        }

        if (this->flat_sets.size() != image_count * per_image) {
            std::vector<VkDescriptorSetLayout> const layouts(image_count * per_image, layout);
            this->flat_sets.assign(image_count * per_image, VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo allocate_info = {};
            allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate_info.descriptorPool = this->pool;
            allocate_info.descriptorSetCount = static_cast<uint32_t>(layouts.size());
            allocate_info.pSetLayouts = layouts.data();
            if (vkAllocateDescriptorSets(this->device, &allocate_info, this->flat_sets.data()) != VK_SUCCESS) {
                this->flat_sets.clear();
                this->bound_signatures.clear();
                return false;
            }
        }

        for (std::size_t i = 0; i < image_count; ++i) {
            write(vk, static_cast<uint32_t>(i), std::span<VkDescriptorSet const>(this->flat_sets.data() + i * per_image, per_image));
        }
        this->bound_signatures.clear();
        this->bound_signatures.reserve(signatures.size());
        for (std::span<VkImageView const> const views : signatures) {
            this->bound_signatures.emplace_back(views.begin(), views.end());
        }
        this->images = image_count;
        return true;
    }

    std::span<VkDescriptorSet const> image_set_family::sets(uint32_t const image_index) const noexcept {
        if (this->sets_per_image == 0) {
            return {};
        }
        std::size_t const first = static_cast<std::size_t>(image_index) * this->sets_per_image;
        if (first + this->sets_per_image > this->flat_sets.size()) {
            return {};
        }
        return {this->flat_sets.data() + first, this->sets_per_image};
    }

    VkDescriptorSet image_set_family::set(uint32_t const image_index, uint32_t const which) const noexcept {
        std::span<VkDescriptorSet const> const all = this->sets(image_index);
        return which < all.size() ? all[which] : VK_NULL_HANDLE;
    }

    bool image_set_family::ready() const noexcept {
        return this->sets_per_image != 0 && this->images != 0 && this->flat_sets.size() == static_cast<std::size_t>(this->images) * this->sets_per_image;
    }

    uint32_t image_set_family::image_count() const noexcept {
        return this->images;
    }

    void image_set_family::retire_all() {
        if (this->pool != VK_NULL_HANDLE) {
            this->retired.push_back(this->pool);
            this->pool = VK_NULL_HANDLE;
        }
        this->pool_capacity = 0;
        this->sets_per_image = 0;
        this->images = 0;
        this->flat_sets.clear();
        this->bound_signatures.clear();
    }
} // namespace vulkan::bindings