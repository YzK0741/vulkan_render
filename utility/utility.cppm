// ============================================================================
// module: utility
// module version: 0.1.1  (independent of the app version in CMakeLists project(VERSION))
//
// Pure-CPU toolkit: data_block, BVH, thread_pool, frame_clock / frame_stats,
// better_pmr (mimalloc routing), content hashing. Standalone - no Vulkan or app
// dependency, link it into any host.
//
// evolve: bump MAJOR on breaking interface changes, MINOR on additive features,
//         PATCH on internal fixes - independently of the rest of the project.
// ============================================================================
module;

#include <cstdint>

export module utility;
export import vstd;
// Forward-export every utility submodule so consumers only need `import utility;`
// (frame_clock / frame_stats are the frame-loop time + fps helpers; data_block /
// bvh / better_pmr / thread_pool cover the rest). Submodules stay individually
// importable for callers that want only one of them.
export import utility.data_block;
export import utility.bvh;
export import utility.better_pmr;
export import utility.frame_clock;
export import utility.frame_stats;
export import utility.thread_pool;

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
    export void at_panic(std::function<void()> const& task);

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
     * @brief panic with a compile-time-checked format string (like std::format)
     * @tparam Args argument types
     * @param source_location the caller's location, pass std::source_location::current()
     * @param fmt the format string (compile-time checked)
     * @param args arguments to format
     * @note thread safe
     * @note the location is an explicit parameter because clang does not deduce a parameter
     *       pack that is followed by another parameter
     */
    export template <typename... Args>
    [[noreturn]] void panic(std::source_location source_location, std::format_string<Args...> fmt, Args&&... args) noexcept {
        panic(std::format(fmt, std::forward<Args>(args)...), source_location);
    }

    /**
     * @ingroup utility
     * @brief a simple time test function
     * @param test callable objects wants to get the invoke time cost
     * @return used time in invoking the argument
     */
    /**
     * @ingroup utility
     * @brief write an 8-bit RGBA image to a PNG file
     * @param path output file (overwritten)
     * @param width image width in pixels
     * @param height image height in pixels
     * @param rgba tightly packed RGBA rows (width * height * 4 bytes)
     * @return empty expected on success, an error message otherwise
     * @note no external dependency: a minimal PNG writer (CRC32 + zlib stream of uncompressed
     *       deflate blocks + adler32), so captures work without pulling in an image library
     */
    export std::expected<void, std::string> write_png(std::filesystem::path const& path, uint32_t width, uint32_t height, std::span<unsigned char const> rgba);
    export std::chrono::milliseconds time_test(std::function<void()> const& test) noexcept;

    /**
     * @ingroup utility
     * @brief read the whole file in binary mode into a byte vector
     * @param path the file path
     * @return the file contents, or std::nullopt if the file cannot be read
     */
    export std::optional<std::vector<unsigned char>> read_binary_to_vector(std::filesystem::path const& path);

    /**
     * @ingroup utility
     * @brief read the whole file in binary mode into a string
     * @param path the file path
     * @return the file contents, or std::nullopt if the file cannot be read
     */
    export std::optional<std::string> read_binary_to_string(std::filesystem::path const& path);

    /**
     * @ingroup utility
     * @brief asynchronous logging sink (Meyer's singleton), internal implementation
     * @note
     *      - messages are pushed to a thread-safe queue; a background thread keeps popping
     *        them and writes each one: to the terminal in Debug builds (NDEBUG unset),
     *        to a debug.log file in Release builds (NDEBUG set)
     *      - not exported; use the utility::log() function template instead
     */
    class log_sink { // NOLINT
        std::mutex queue_mutex = {};
        std::condition_variable queue_cv = {};
        std::condition_variable drained_cv = {}; // notifies when the queue has been drained
        std::queue<std::string> messages = {};
        std::size_t pending = 0; // messages pending write (queued + currently being written)
        std::thread worker = {};
        std::atomic<bool> running = true;
        std::ofstream file = {}; // Release builds write to debug.log

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

    // Internal: error message output — Debug writes directly to stderr in red (bypassing the log queue;
    //       error is usually followed by terminate), Release hands it to the log thread for debug.log
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
     * @defgroup hash Content Hashing
     * @ingroup utility
     * @brief xxHash-based 128-bit content hash (XXH3_128bits), returned as a data_block<16>
     * @note
     *     - non-cryptographic, extremely fast (used for content dedup)
     *     - 128-bit digest: two independent 64-bit lanes, so an accidental collision is
     *       negligible for content-addressed GPU-resource dedup (a wrong share would silently
     *       render the wrong texture / image)
     *     - digest supports operator==/!=/<=> and hex formatting (.to_hex_string())
     *     - cannot fail (no allocation / error state)
     */

    /**
     * @typedef xxh3_digest
     * @relates data_block
     * @ingroup hash
     */
    export using xxh3_digest = data_block<16>;

    /**
     * @brief xxh3_128bits hash function
     * @param data_view bytes to fingerprint
     * @return 16-byte digest of @p data_view (the raw 128-bit fingerprint)
     * @ingroup hash
     */
    export xxh3_digest xxh3_128bits(std::span<unsigned char const> data_view);
} // namespace utility