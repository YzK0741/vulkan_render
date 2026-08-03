//
// Created by 小叶 on 2026/8/2.
//

module;

#include <functional>
#include <thread>
#include <queue>

export module utility.thread_pool;

namespace utility {



    export class thread_pool {
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
        std::mutex access_mutex;
        shutdown_policy policy = shutdown_policy::wait;

        void thread(const std::stop_token &token);

    public:
        explicit thread_pool(int threads, shutdown_policy policy = shutdown_policy::wait);
        ~thread_pool();
        void post(std::function<void()> task, int priority = 0);
        void shutdown();

    };
}