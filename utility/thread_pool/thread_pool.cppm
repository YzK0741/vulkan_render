module;

#include <cstdint>

export module utility.thread_pool;
export import vstd;

/**
 * @ingroup utility
 * @defgroup thread_pool Thread Pool
 * @file thread_pool.cppm
 * @brief a module provides raii thread pool (utility::thread_pool)
 *
 * @code {.cpp}
 * import utility.thread_pool;
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
    export class thread_pool { // NOLINT
        enum class shutdown_policy : uint8_t {
            discard,
            wait,
        };

        struct task {
            int priority = 0;
            std::function<void()> action;
            bool operator<(task const& other) const noexcept;
        };

        std::priority_queue<task> tasks;
        std::condition_variable cv;
        std::condition_variable idle;
        mutable std::mutex access_mutex; // mutable: is_free() const reads tasks under the lock
        std::atomic_int active_thread = 0;
        // per-priority pending count: tasks posted with a given priority that are still queued
        // or running. post/post_batch increment it under the lock, workers decrement it when a
        // task finishes (or is discarded at shutdown), and wait_until_priority_done() blocks on
        // it reaching zero - so a caller can wait for "its" priority group without waiting for
        // unrelated tasks posted by other users of a shared pool.
        std::unordered_map<int, std::size_t> pending_by_priority;
        shutdown_policy policy = shutdown_policy::wait;
        // Declared last so the jthreads are destroyed (auto-joined) FIRST, before the mutex /
        // condition variables above: worker threads still exiting would otherwise touch
        // already-destroyed synchronization state.
        std::vector<std::jthread> threads;

        void worker_loop(std::stop_token const& token);
        // lock-free helpers (caller holds access_mutex): bookkeeping for one posted/finished task
        void note_task_posted(int priority);
        void note_task_finished(int priority);

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
         * @return false when the pool is shut down and will never run the task (not queued)
         */
        bool post(std::function<void()> task, int priority = 0);
        /**
         * @brief post a batch of tasks at one priority in a single lock acquisition
         * @param tasks callable objects, all posted with @p priority (any order)
         * @param priority @see task::priority
         * @return false when the pool is shut down and none of the tasks were queued
         * @note pair with wait_until_priority_done(@p priority) to run a group synchronously:
         *       post_batch() once, then wait for exactly that priority's tasks to finish
         */
        bool post_batch(std::span<std::function<void()>> tasks, int priority = 0);
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
         * @brief block until every task posted with @p priority (before this call) has finished
         * @param priority the priority group to wait for
         * @note snapshot semantics: only tasks already posted with @p priority are waited on;
         *       tasks posted later with the same priority are not included. Callers that run a
         *       group synchronously should post_batch() first and then wait - posting while
         *       waiting may extend the wait past the intended group.
         * @note returns immediately when no task with @p priority is pending (posted or running)
         */
        void wait_until_priority_done(int priority);
        /**
         * @brief get count of worker threads
         * @return number of threads this pool runs
         */
        [[nodiscard]] int thread_count() const noexcept;
        /**
         * @brief get count of active thread
         * @return count of active thread
         */
        int get_active_thread() const;
    };
} // namespace utility
