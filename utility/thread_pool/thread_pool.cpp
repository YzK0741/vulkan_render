//
// Created by 小叶 on 2026/8/2.
//
#include <thread>
#include <functional>
#include <mutex>
#include <stop_token>
#include <vector>
#include "thread_pool.cppm"

namespace utility {
    thread_pool::task &thread_pool::task::operator()() const noexcept {
        this->action();
        return const_cast<task &>(*this);
    }

    bool thread_pool::task::operator<(const task &other) const noexcept {
        return this->priority < other.priority;
    }

    void thread_pool::thread(const std::stop_token &token) {
        this->active_thread.fetch_add(1);
        std::function<void()> current_task;
        while (true) {
            {
                std::unique_lock lock(this->access_mutex);
                this->active_thread.fetch_sub(1);
                cv.wait(lock, [this, &token]() {
                    return !this->tasks.empty() || token.stop_requested();
                });
                this->active_thread.fetch_add(1);

                if (this->tasks.empty()) {
                    this->active_thread.fetch_sub(1);
                    return;
                }

                if (token.stop_requested() && this->policy == shutdown_policy::discard) {
                    this->active_thread.fetch_sub(1);
                    return;
                }

                current_task = this->tasks.top().action;
                this->tasks.pop();

                if (!this->tasks.empty()) {
                    this->cv.notify_one();
                }

            }
            current_task();
        }
    }


    thread_pool::thread_pool(const int threads,[[maybe_unused]] shutdown_policy policy) {
        this->threads.resize(threads);
        this->policy = policy;

        for (auto& thread : this->threads) {
            thread = std::jthread([this](const std::stop_token& stop_token) {
                thread_pool::thread(stop_token);
            });
        }
    }

    void thread_pool::post(std::function<void()> task, int priority) {
        std::unique_lock lock(this->access_mutex);
        this->tasks.emplace(priority, task);
        this->cv.notify_one();
    }

    void thread_pool::shutdown() {
        for (auto& thread : this->threads) {
            thread.request_stop();
        }
        this->cv.notify_one();
    }

    thread_pool::~thread_pool() {
        this->shutdown();
    }

    bool thread_pool::is_free() const {
        return this->active_thread.load() == 0;
    }

    void thread_pool::wait_until_free() const {
        while (this->active_thread.load() > 0 || !this->tasks.empty()) {
            std::this_thread::yield();
        }
    }

    int thread_pool::get_active_thread() const {
        return this->active_thread.load();
    }
}
