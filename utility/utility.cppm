//
// Created by 小叶 on 2026/7/28.
//
module;

#include <functional>
#include <mutex>
#include <print>
#include <span>
#include <set>
#include <source_location>
#include <stack>

export module utility;

namespace utility {
    export class enable_handle_distribute {

        std::set<uint64_t> recycled_handlers = {};
        std::mutex access_mutex = {};
        uint64_t handle_upper_bound = 1;

    public:

        std::optional<uint64_t> distribute() noexcept;

        void recycle(uint64_t handle) noexcept;

    };

    export class enable_stack_destruct {
    public:
        using destruct_type = std::function<void()>;

    private:
        std::stack<destruct_type> destruct_stack = {};
        std::mutex access_mutex = {};

    public:
        void register_cleanup(std::function<void()> const& destructor) noexcept;

        void cleanup() noexcept;

        void clear() noexcept;
    };

    export void at_terminate(const std::function<void()>& task);

    export [[noreturn]] void panic(std::string_view msg = "", std::source_location source_location = std::source_location::current()) noexcept;

    export std::chrono::milliseconds time_test(std::function<void()> const &test) noexcept;

    export struct md5_digest {
        uint64_t a = 0;
        uint64_t b = 0;
        bool operator==(md5_digest const& other) const noexcept;
        bool operator!=(md5_digest const& other) const noexcept;
        [[nodiscard]] std::string to_hex_string() const noexcept;
    };

    export std::optional<md5_digest> md5(std::span<const unsigned char> data);
}