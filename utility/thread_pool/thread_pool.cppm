#ifndef UTILITY_THREAD_POOL_H
#define UTILITY_THREAD_POOL_H

#include <cstdint>
import std;

/**
 * @ingroup utility
 * @defgroup thread_pool
 * @file thread_pool.cppm
 * @brief a module provides raii thread pool (utility::thread_pool)
 * @note due to a clang bug std::jthread can't be used in module, so use the header-style, and requires link
 *     thread_pool separately
 * @note <b>reserved for future use</b>: no target imports this module yet (it is only built and linked
 *     into the executable); it is kept as a ready-to-use building block for upcoming parallel work,
 *     e.g. async glTF loading or BVH build tasks
 *
 * @code {.cpp}
 * #include "utility/thread_pool/thread_pool.cppm"
 *
 * int main{
 *     utility::thread_pool pool(4);
 *
 *     auto task = []{...};
 *     pool.post(task);
 *     pool.post(task);
 *     pool.post(task);
 *     //...
 *     pool.wait_until_free();
 * }
 *
 *
 * @endcode
 */

namespace utility {
    /**
     * @brief thread pool class
     */
    class thread_pool { // NOLINT
        enum class shutdown_policy : uint8_t {
            discard,
            wait,
        };

        struct task {
            int priority = 0;
            std::function<void()> action;
            bool operator<(task const& other) const noexcept;
        };

        std::vector<std::jthread> threads;
        std::priority_queue<task> tasks;
        std::condition_variable cv;
        std::condition_variable idle;
        std::mutex access_mutex;
        std::atomic_int active_thread = 0;
        shutdown_policy policy = shutdown_policy::wait;

        void worker_loop(std::stop_token const& token);

    public:
        /**
         * @brief thread_pool's constructor
         * @param threads thread number you want create
         * @param policy behavior when tasks remain at destruction
         * @note threads should <= std::thread::hardware_concurrency()
         */
        explicit thread_pool(int threads, shutdown_policy policy = shutdown_policy::wait);
        ~thread_pool();
        /**
         * @brief post a task to thread_pool, signature must be void()
         * @param task callable object
         * @param priority @see task::priority
         */
        void post(std::function<void()> task, int priority = 0);
        /**
         * @brief request all thread stop after finishing current task
         */
        void shutdown();
        /**
         * @brief check whether thread_pool is free
         * @return if free returns true, otherwise returns false
         */
        bool is_free() const;
        /**
         * @brief block current thread until all the tasks completed
         */
        void wait_until_free();
        /**
         * @brief get count of active thread
         * @return count of active thread
         */
        int get_active_thread() const;
    };
} // namespace utility

#endif