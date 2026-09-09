module;

#include <cstdint> // ::uint64_t (used unqualified below)

export module vulkan.core.vma.handles;

import vstd;

/**
 * @defgroup vulkan_vma_handles Vulkan VMA Handles' RAII Wrapper
 * @file vma_handles.cppm
 * @brief RAII owners for GPU buffers / images allocated through vma_allocator
 *
 * The allocator hands out raw uint64 handles today, which makes ownership implicit: every
 * call site must remember to free exactly once, and shared (deduplicated) images rely on a
 * reference count nobody can see. These wrappers make ownership structural:
 *   - copying a vk_buffer / vk_image shares the underlying resource (the allocator's
 *     reference count is bumped through the injected retain callback);
 *   - destroying one releases exactly one reference (the release callback; the allocator
 *     really destroys the GPU object when the count reaches zero).
 * The retain/release callbacks are injected by vma_allocator when it creates the wrapper
 * (see vulkan.core.vma), which lets the allocator keep free/retain private - there is no
 * public raw-handle release path left to misuse.
 *
 * vk_buffer / vk_image deliberately know nothing about vma_allocator (they only store the
 * handle + the two callbacks), so this module does not depend on vulkan.core.vma.
 */
namespace vulkan {
    /**
     * @ingroup vulkan_vma_handles
     * @brief owning reference to one GPU buffer allocated through vma_allocator
     * @note copy = share (allocator refcount +1), destroy = release (refcount -1);
     *       default-constructed / moved-from wrappers are empty and own nothing
     */
    export class vk_buffer {
    public:
        vk_buffer() noexcept = default;

        /**
         * @brief adopt one reference to @p handle; @p retain / @p release adjust the
         *        allocator's reference count (injected by vma_allocator::create_buffer)
         */
        explicit vk_buffer(uint64_t handle,
                           std::function<void(uint64_t)> retain,
                           std::function<void(uint64_t)> release) noexcept
            : handle_{handle}
            , retain_{std::move(retain)}
            , release_{std::move(release)} {
        }

        ~vk_buffer() noexcept {
            this->release();
        }

        vk_buffer(vk_buffer const& other)
            : handle_{other.handle_}
            , retain_{other.retain_}
            , release_{other.release_} {
            if (this->handle_ != 0 && this->retain_) {
                this->retain_(this->handle_); // share: bump the allocator's reference count
            }
        }

        vk_buffer& operator=(vk_buffer const& other) {
            if (this != &other) {
                this->release();
                this->handle_ = other.handle_;
                this->retain_ = other.retain_;
                this->release_ = other.release_;
                if (this->handle_ != 0 && this->retain_) {
                    this->retain_(this->handle_);
                }
            }
            return *this;
        }

        vk_buffer(vk_buffer&& other) noexcept
            : handle_{other.handle_}
            , retain_{std::move(other.retain_)}
            , release_{std::move(other.release_)} {
            other.handle_ = 0; // moved-from owns nothing
        }

        vk_buffer& operator=(vk_buffer&& other) noexcept {
            if (this != &other) {
                this->release();
                this->handle_ = other.handle_;
                this->retain_ = std::move(other.retain_);
                this->release_ = std::move(other.release_);
                other.handle_ = 0;
            }
            return *this;
        }

        /** @brief the underlying allocator handle (0 = empty) */
        [[nodiscard]] uint64_t handle() const noexcept {
            return this->handle_;
        }

        /** @brief whether this wrapper owns a reference */
        [[nodiscard]] bool valid() const noexcept {
            return this->handle_ != 0;
        }

        /** @brief release this reference and become empty (idempotent) */
        void reset() noexcept {
            this->release();
        }

    private:
        void release() noexcept {
            if (this->handle_ != 0 && this->release_) {
                this->release_(this->handle_);
            }
            this->handle_ = 0;
            this->retain_ = {};
            this->release_ = {};
        }

        uint64_t handle_ = 0;
        std::function<void(uint64_t)> retain_ = {};
        std::function<void(uint64_t)> release_ = {};
    };

    /**
     * @ingroup vulkan_vma_handles
     * @brief owning reference to one GPU image allocated through vma_allocator
     * @note same ownership semantics as vk_buffer: copy = share, destroy = release
     */
    export class vk_image {
    public:
        vk_image() noexcept = default;

        explicit vk_image(uint64_t handle,
                          std::function<void(uint64_t)> retain,
                          std::function<void(uint64_t)> release) noexcept
            : handle_{handle}
            , retain_{std::move(retain)}
            , release_{std::move(release)} {
        }

        ~vk_image() noexcept {
            this->release();
        }

        vk_image(vk_image const& other)
            : handle_{other.handle_}
            , retain_{other.retain_}
            , release_{other.release_} {
            if (this->handle_ != 0 && this->retain_) {
                this->retain_(this->handle_);
            }
        }

        vk_image& operator=(vk_image const& other) {
            if (this != &other) {
                this->release();
                this->handle_ = other.handle_;
                this->retain_ = other.retain_;
                this->release_ = other.release_;
                if (this->handle_ != 0 && this->retain_) {
                    this->retain_(this->handle_);
                }
            }
            return *this;
        }

        vk_image(vk_image&& other) noexcept
            : handle_{other.handle_}
            , retain_{std::move(other.retain_)}
            , release_{std::move(other.release_)} {
            other.handle_ = 0;
        }

        vk_image& operator=(vk_image&& other) noexcept {
            if (this != &other) {
                this->release();
                this->handle_ = other.handle_;
                this->retain_ = std::move(other.retain_);
                this->release_ = std::move(other.release_);
                other.handle_ = 0;
            }
            return *this;
        }

        /** @brief the underlying allocator handle (0 = empty) */
        [[nodiscard]] uint64_t handle() const noexcept {
            return this->handle_;
        }

        /** @brief whether this wrapper owns a reference */
        [[nodiscard]] bool valid() const noexcept {
            return this->handle_ != 0;
        }

        /** @brief release this reference and become empty (idempotent) */
        void reset() noexcept {
            this->release();
        }

    private:
        void release() noexcept {
            if (this->handle_ != 0 && this->release_) {
                this->release_(this->handle_);
            }
            this->handle_ = 0;
            this->retain_ = {};
            this->release_ = {};
        }

        uint64_t handle_ = 0;
        std::function<void(uint64_t)> retain_ = {};
        std::function<void(uint64_t)> release_ = {};
    };
} // namespace vulkan
