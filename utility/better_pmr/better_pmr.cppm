export module utility.better_pmr;
export import std;
/**
 * @defgroup better_pmr
 * @ingroup utility
 * @brief a submodule from utility helps to optimize pmr container's memory performance via mimalloc
 * @note the entry is utility::init_pmr()
 */

class mimalloc_memory_resource : public std::pmr::memory_resource // NOLINT
{
    void* do_allocate(std::size_t, std::size_t) override;
    void do_deallocate(void*, std::size_t, std::size_t) override;
    [[nodiscard]] bool do_is_equal(memory_resource const&) const noexcept override;
};

namespace utility {
    class pmr_manager // NOLINT
    {
        std::unique_ptr<mimalloc_memory_resource> memory_resource = nullptr;
        pmr_manager();

    public:
        pmr_manager(pmr_manager const&) = delete;
        pmr_manager& operator=(pmr_manager const&) = delete;
        ~pmr_manager();
        friend pmr_manager& init_pmr();
    };

    /**
     * @ingroup better_pmr
     * @brief call this function to replace default memory resource to mimalloc memory resource
     * @warning DO NOT use any pmr container (including static) before init_pmr(), it will lead memory fault
     */
    export pmr_manager& init_pmr();
} // namespace utility