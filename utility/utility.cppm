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


/**
 * @file utility.cppm
 * @defgroup utility
 */
namespace utility {
    /**
     * @ingroup utility
     * @brief a mixin-class to enable derived class distribute unique handles
     *
     * @note
     *     - uniqueness is only guaranteed in the class instance
     *     - thread-safe (by using std::mutex)
     *     - consider use it in private as a class feature
     *
     * @code{.cpp}
     * class derived : enable_handle_distribute {
     *     uint64_t derived::mem() {
     *         uint64_t handle = 0
     *         auto handle_opt = this->distribute();
     *         if (!handle_opt){
     *             //error process...
     *         }
     *         handle = handle_opt.value();
     *         //do sth...
     *         return handle;
     *     }
     *     //...
     * }
     * @endcode
     */
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

    /**
     * @ingroup utility
     * @brief stack-style safe the argument and invoke it when panic attached
     * @param task callable object you want invoke at panic
     * @note
     *     - thread safe
     *     - LIFO
     */
    export void at_panic(const std::function<void()>& task);

    /**
     * @ingroup utility
     * @brief use when program cause a terminating error, will invoke functions assigned by at_panic()
     * @param msg error message
     * @param source_location just use the default argument it will get call position info for better error print
     * @note thread safe
     */
    export [[noreturn]] void panic(std::string_view msg = "", std::source_location source_location = std::source_location::current()) noexcept;

    /**
     * @ingroup utility
     * @brief a simple time test function
     * @param test callable objects wants to get the invoke time cost
     * @return used time in invoking the argument
     */
    export std::chrono::milliseconds time_test(std::function<void()> const &test) noexcept;


    /**
     * @ingroup utility
     * @brief struct template creates a sized data type provides auto generated operator==/!= (use std::ranges::equal),
     *     operator<=>(use std::lexicographical_compare) and hex formatter (.to_hex_string())
     * @tparam S byte size of the struct
     */
    export template<size_t S>
    struct data_block {
        constexpr static uint32_t size_byte = S;
        std::array<uint8_t, S> data;

        constexpr static size_t size = S;

        bool operator==(data_block<S> const& other) const {
            return std::ranges::equal(data, other.data);
        }

        bool operator!=(data_block<S> const& other) const {
            return !(*this == other);
        }

        auto operator<=>(const data_block<S>& other) const {
            return std::lexicographical_compare_three_way(
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

    /**
     * @ingroup utility
     * @brief hash series functions wrapper of openssl
     * @note
     *     - digest supports operator==/!=/<=>, hex formatter (.to_hex_string())
     *     - returns std::optional wrappered digest
     *     - returns std::nullopt when failed (virtually impossible)
     */
    /// @{
    export using md5_digest = data_block<16>;

    export std::optional<md5_digest> md5(std::span<const unsigned char> data_view);

    export using sha256_digest = data_block<32>;

    export std::optional<sha256_digest> sha256(std::span<const unsigned char> data_view);

    export using blake2_digest = data_block<64>;

    export std::optional<blake2_digest> blake2(std::span<const unsigned char> data_view);
    /// @}
}