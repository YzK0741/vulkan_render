//
// Created by 23530 on 2026/8/18.
//
module;

#include <memory>
#include <memory_resource>

export module utility.better_pmr;

class mimalloc_memory_resource : public std::pmr::memory_resource //NOLINT
{
    void* do_allocate(std::size_t, std::size_t) override;
    void do_deallocate(void*, std::size_t, std::size_t) override;
    [[nodiscard]] bool do_is_equal(memory_resource const&) const noexcept override;
};

namespace utility
{
    export class pmr_manager
    {
        std::unique_ptr<mimalloc_memory_resource> memory_resource = nullptr;
        pmr_manager();
    public:
        ~pmr_manager();
        friend pmr_manager& init_pmr();
    };
    pmr_manager& init_pmr();
}