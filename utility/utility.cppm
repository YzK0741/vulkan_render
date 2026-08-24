module;

#include <cstdint>

export module utility;
export import std;
export import utility.data_block;
export import utility.bvh;
export import utility.better_pmr;

/**
 * @file utility.cppm
 * @defgroup utility utility functions, classes sets
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
     * @code {.cpp}
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

        std::set<uint64_t> recycled_handles = {};
        std::mutex access_mutex = {};
        uint64_t handle_upper_bound = 1;

    public:

        std::optional<uint64_t> distribute() noexcept;

        void recycle(uint64_t handle) noexcept;

    };

    /**
     * @ingroup utility
     * @brief a mixin class which enables derived class a stack-style destruct ability
     * @note
     *     - LIFO
     *     - consider use it in private as a class feature
     *     - thread safe
     *
     * @code {.cpp}
     * class sth : enable_stack_destruct{
     *     void mem(){
     *         //...
     *         this->register_cleanup(
     *             [this]{
     *                 // sth cleanup...
     *             });
     *     }
     *     ~sth(){
     *         this->do_cleanup();
     *         //sth cleanup without stack-style...
     *     }
     *     //...
     * }
     * @endcode
     */
    export class enable_stack_destruct {
    public:
        using destruct_type = std::function<void()>;

    private:
        std::stack<destruct_type> destruct_stack = {};
        std::mutex access_mutex = {};

    public:
        /**
         * @param destructor callable objects  wants to push in the destruct stack
         */
        void register_cleanup(std::function<void()> const& destructor) noexcept;
        /**
         * @note invoke this function will remove destructor on the stack top
         */
        void pop_destructor() noexcept;
        /**
         * @note pop and invoke all destructor in the stack
         */
        void do_cleanup() noexcept;
        /**
         * @note clean the stack without invoke
         */
        void clear_stack() noexcept;
    };

    /**
     * @ingroup utility
     * @brief stack-style save the argument and invoke it when panic attached
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
     * @defgroup hash
     * @ingroup utility
     * @brief hash series functions wrapper of openssl
     * @note
     *     - digest supports operator==/!=/<=>, hex formatter (.to_hex_string())
     *     - returns std::optional wrappered digest
     *     - returns std::nullopt when failed (virtually impossible)
     */

    /**
     * @typedef md5_digest
     * @relates data_block
     * @ingroup hash
     */
    export using md5_digest = data_block<16>;

    /**
     * @brief md5 hash function
     * @param data_view
     * @return md5 digest
     * @ingroup hash
     */
    export std::optional<md5_digest> md5(std::span<const unsigned char> data_view);

    /**
     * @typedef sha256_digest
     * @relates data_block
     * @ingroup hash
     */
    export using sha256_digest = data_block<32>;

    /**
     * @brief sha256 hash function
     * @param data_view
     * @return sha256 digest
     * @ingroup hash
     */
    export std::optional<sha256_digest> sha256(std::span<const unsigned char> data_view);

    /**
     * @typedef blake2_digest
     * @relates data_block
     * @ingroup hash
     */
    export using blake2_digest = data_block<64>;

    /**
     * @brief blake2 hash function
     * @param data_view
     * @return blake2 digest
     * @ingroup hash
     */
    export std::optional<blake2_digest> blake2(std::span<const unsigned char> data_view);
}