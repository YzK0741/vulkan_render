//
// Created by 小叶 on 2026/7/28.
//
module;

#include <algorithm>
#include <functional>
#include <mutex>
#include <print>
#include <span>
#include <set>
#include <source_location>
#include <stack>
#include <openssl/evp.h>

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

    export template<size_t S>
    struct data_block {
        constexpr static uint32_t size_byte = S;
        std::array<uint8_t, S> data;

        bool operator==(data_block<S> const& other) const {
            return std::ranges::equal(data, other.data);
        }

        bool operator!=(data_block<S> const& other) const {
            return !(*this == other);
        }

        bool operator<(const data_block<S>& other) const {
            return std::lexicographical_compare(
                data.begin(), data.end(),
                other.data.begin(), other.data.end()
            );
        }

        [[nodiscard]] std::string to_hex_string() const {
            std::string result;
            result.reserve(S * 2);
            std::ranges::for_each(this->data, [&result](auto const& byte){result += std::format("{:02x}", byte);});
            return result;
        }
    };

    export using md5_digest = data_block<16>;

    export std::optional<md5_digest> md5(std::span<const unsigned char> data);

    export using sha256_digest = data_block<32>;

    export std::optional<sha256_digest> sha256(std::span<const unsigned char> data);

    export using blake2_digest = data_block<64>;

    export std::optional<blake2_digest> blake2(std::span<const unsigned char> data);
}