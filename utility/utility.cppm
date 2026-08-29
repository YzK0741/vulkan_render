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
    export std::chrono::milliseconds time_test(std::function<void()> const& test) noexcept;

    /**
     * @ingroup utility
     * @brief read the whole file in binary mode into a byte vector
     * @param path the file path
     * @return the file contents, or std::nullopt if the file cannot be read
     */
    export std::optional<std::vector<unsigned char>> read_binary_to_vector(const std::filesystem::path& path);

    /**
     * @ingroup utility
     * @brief read the whole file in binary mode into a string
     * @param path the file path
     * @return the file contents, or std::nullopt if the file cannot be read
     */
    export std::optional<std::string> read_binary_to_string(const std::filesystem::path& path);

    /**
     * @ingroup utility
     * @brief asynchronous logging sink (Meyer's singleton), internal implementation
     * @note
     *      - messages are pushed to a thread-safe queue; a background thread keeps popping
     *        them and writes each one: to the terminal in Debug builds (NDEBUG unset),
     *        to a debug.log file in Release builds (NDEBUG set)
     *      - not exported; use the utility::log() function template instead
     */
    class log_sink {
        std::mutex queue_mutex = {};
        std::condition_variable queue_cv = {};
        std::condition_variable drained_cv = {}; // 队列排空通知
        std::queue<std::string> messages = {};
        std::size_t pending = 0; // 待写消息数（排队中 + 正在输出）
        std::thread worker = {};
        std::atomic<bool> running = true;
        std::ofstream file = {}; // Release 构建输出到 debug.log

        log_sink();
        ~log_sink();
        void worker_loop() noexcept;

    public:
        log_sink(log_sink const&) = delete;
        log_sink& operator=(log_sink const&) = delete;

        static log_sink& instance() noexcept;
        void write(std::string message);
        void wait_all();
    };

    /**
     * @ingroup utility
     * @brief asynchronous log: formats the message like std::format and pushes it to the log singleton
     * @tparam Args argument types
     * @param fmt the format string (compile-time checked)
     * @param args arguments to format
     * @note output goes to the terminal in Debug builds, to a debug.log file in Release builds
     */
    export template <typename... Args>
    void log(std::format_string<Args...> fmt, Args&&... args) {
        log_sink::instance().write(std::format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup utility
     * @brief asynchronous log: writes a single pre-formatted string as-is
     * @param message the message (string literal, const char*, std::string or std::string_view)
     * @note
     *      - for runtime strings, which cannot construct the consteval std::format_string
     *      - for format-string usage prefer the template overload
     */
    export void log(std::string_view message) {
        log_sink::instance().write(std::string(message));
    }

    // 内部：错误消息输出——Debug 直接红色输出到 stderr（不经过日志队列，error 之后基本是 terminate），
    //       Release 交给日志线程写入 debug.log
    void error_message(std::string message);

    /**
     * @ingroup utility
     * @brief error log: in Debug builds prints directly to stderr in red (not queued);
     *        in Release builds hands the message to the log singleton with an [ERROR] prefix
     * @tparam Args argument types
     * @param fmt the format string (compile-time checked)
     * @param args arguments to format
     */
    export template <typename... Args>
    void error(std::format_string<Args...> fmt, Args&&... args) {
        error_message(std::format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup utility
     * @brief error log: in Debug builds prints directly to stderr in red (not queued);
     *        in Release builds hands the message to the log singleton with an [ERROR] prefix
     * @param message the message (string literal, const char*, std::string or std::string_view)
     */
    export void error(std::string_view message) {
        error_message(std::string(message));
    }

    /**
     * @ingroup utility
     * @brief block until all log messages queued so far have been written by the log thread
     * @note useful before shutdown or before reading output that must be complete
     */
    export void wait_log_all() {
        log_sink::instance().wait_all();
    }

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
} // namespace utility