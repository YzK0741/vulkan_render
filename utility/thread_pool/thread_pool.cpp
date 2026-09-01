#include "thread_pool.cppm"

import std;

namespace utility {
    bool thread_pool::task::operator<(task const& other) const noexcept {
        return this->priority < other.priority;
    }

    void thread_pool::worker_loop(std::stop_token const& token) {
        this->active_thread.fetch_add(1);
        std::function<void()> current_task;
        while (true) {
            {
                std::unique_lock lock(this->access_mutex);
                this->active_thread.fetch_sub(1);
                if (this->active_thread.load() < 1) {
                    this->idle.notify_one();
                }
                cv.wait(lock, [this, &token]() {
                    return !this->tasks.empty() || token.stop_requested();
                });
                this->active_thread.fetch_add(1);

                if (this->tasks.empty()) {
                    this->active_thread.fetch_sub(1);
                    return;
                }

                if (token.stop_requested()) {
                    if (this->tasks.empty()) {
                        this->active_thread.fetch_sub(1);
                        if (this->active_thread.load() == 0) {
                            this->idle.notify_one();
                        }
                        return;
                    }
                    if (this->policy == shutdown_policy::discard) {
                        this->active_thread.fetch_sub(1);
                        if (this->active_thread.load() == 0) {
                            this->idle.notify_one();
                        }
                        return;
                    }
                }

                current_task = this->tasks.top().action;
                this->tasks.pop();
            }
            current_task();
        }
    }

    thread_pool::thread_pool(int const threads, shutdown_policy const policy) {
        this->threads.resize(threads);
        this->policy = policy;

        for (auto& thread : this->threads) {
            thread = std::jthread([this](std::stop_token const& stop_token) {
                thread_pool::worker_loop(stop_token);
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
        this->cv.notify_all();
    }

    thread_pool::~thread_pool() {
        this->shutdown();
    }

    bool thread_pool::is_free() const {
        return this->active_thread.load() == 0;
    }

    void thread_pool::wait_until_free() {
        std::unique_lock lock(this->access_mutex);
        this->idle.wait(lock, [this] { return this->tasks.empty() && this->active_thread.load() == 0; });
    }

    int thread_pool::get_active_thread() const {
        return this->active_thread.load();
    }
} // namespace utility