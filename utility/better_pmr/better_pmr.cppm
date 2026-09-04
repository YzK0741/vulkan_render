export module utility.better_pmr;
export import std;
/**
 * @defgroup better_pmr
 * @ingroup utility
 * @brief a submodule from utility helps to optimize pmr container's memory performance via mimalloc
 * @note the entry is utility::init_pmr(); call it before creating any std::pmr container
 * @note in use: main.cpp and vulkan/runtime.cpp keep a file-scope
 *     @c [[maybe_unused]] static auto& pmr = utility::init_pmr(); whose dynamic
 *     initialization runs before main(), so the runtime's per-frame std::pmr vectors
 *     (cull_visible / frame_leaves / frame_visible) already allocate via mimalloc
 */

class mimalloc_memory_resource : public std::pmr::memory_resource // NOLINT
{
    void* do_allocate(std::size_t size, std::size_t alignment) override;
    void do_deallocate(void* p, std::size_t size, std::size_t alignment) override;
    [[nodiscard]] bool do_is_equal(memory_resource const& other) const noexcept override;
};

namespace utility {
    export class pmr_manager // NOLINT
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