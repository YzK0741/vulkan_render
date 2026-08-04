//
// Created by 小叶 on 2026/8/2.
//
#include <functional>
#include <thread>
#include <queue>
#include <condition_variable>

namespace utility {

    class thread_pool {
        enum class shutdown_policy {
            discard,
            wait
        };

        struct task {
            int priority = 0;
            std::function<void()> action;

            task& operator()() const noexcept;
            bool operator<(const task& other) const noexcept;
        };

        std::vector<std::jthread> threads;
        std::priority_queue<task> tasks;
        std::condition_variable cv;
        std::condition_variable idle;
        std::mutex access_mutex;
        std::atomic_int active_thread = 0;
        shutdown_policy policy = shutdown_policy::wait;

        void thread(const std::stop_token &token);

    public:
        explicit thread_pool(int threads, shutdown_policy policy = shutdown_policy::wait);
        ~thread_pool();
        void post(std::function<void()> task, int priority = 0);
        void shutdown();
        bool is_free() const;
        void wait_until_free();
        int get_active_thread() const;
    };
}