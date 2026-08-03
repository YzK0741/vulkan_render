//
// Created by 小叶 on 2026/8/2.
//

module;

#include <thread>
#include <functional>
#include <mutex>
#include <vector>

module utility.thread_pool;

namespace utility {
    thread_pool::task &thread_pool::task::operator()() const noexcept {
        this->action();
        return const_cast<task &>(*this);
    }

    bool thread_pool::task::operator<(const task &other) const noexcept {
        return this->priority < other.priority;
    }

    void thread_pool::thread(const std::stop_token &token) {
        std::function<void()> current_task;
        while (true) {
            {
                std::unique_lock lock(this->access_mutex);
                cv.wait(lock, [this, &token]() {
                    return !this->tasks.empty() || token.stop_requested();
                });

                if (this->tasks.empty()) {
                    return;
                }

                if (token.stop_requested() && this->policy == shutdown_policy::discard) {
                    return;
                }

                current_task = std::move(this->tasks.top().action);
                this->tasks.pop();

            }
            current_task();
        }
    }


    thread_pool::thread_pool(const int threads, shutdown_policy policy) {
        this->threads.resize(threads);

        for (auto& thread : this->threads) {
            thread = std::jthread([this](const std::stop_token& stop_token) {
                thread_pool::thread(stop_token);
            });
        }
    }

    void thread_pool::post(std::function<void()> task, int priority) {
        std::unique_lock lock(this->access_mutex);
        this->tasks.emplace(priority, task);
        if (this->tasks.size() == 1) {
            lock.unlock();
            this->cv.notify_one();
        } else {
            lock.unlock();
            this->cv.notify_all();
        }
    }

    void thread_pool::shutdown() {
        for (auto& thread : this->threads) {
            thread.request_stop();
        }
        this->cv.notify_all();
    }

    thread_pool::~thread_pool() {
        this->shutdown();
    }
}
