module;

#include <mimalloc.h>

module utility.better_pmr;

void* mimalloc_memory_resource::do_allocate(std::size_t bytes, std::size_t align) {
    return mi_aligned_alloc(align, bytes);
}

void mimalloc_memory_resource::do_deallocate(void* p, [[maybe_unused]] std::size_t bytes, const std::size_t align) {
    mi_free_aligned(p, align);
}

bool mimalloc_memory_resource::do_is_equal(memory_resource const& other) const noexcept {
    return this == &other;
}

namespace utility {
    pmr_manager::pmr_manager() {
        this->memory_resource = std::make_unique<mimalloc_memory_resource>();
        std::pmr::set_default_resource(this->memory_resource.get());
    }

    pmr_manager::~pmr_manager() {
        std::pmr::set_default_resource(nullptr);
        this->memory_resource = nullptr;
    }

    pmr_manager& init_pmr() {
        static pmr_manager manager;
        return manager;
    }
} // namespace utility
