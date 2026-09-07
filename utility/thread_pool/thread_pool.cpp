module utility.thread_pool;

namespace utility {
    bool thread_pool::task::operator<(task const& other) const noexcept {
        return this->priority < other.priority;
    }

    // lock-free bookkeeping helpers: the caller holds access_mutex
    void thread_pool::note_task_posted(int const priority) {
        ++this->pending_by_priority[priority];
    }

    void thread_pool::note_task_finished(int const priority) {
        auto const it = this->pending_by_priority.find(priority);
        if (it != this->pending_by_priority.end() && --it->second == 0) {
            this->pending_by_priority.erase(it); // a zero entry is indistinguishable from "never posted"
        }
    }

    void thread_pool::worker_loop(std::stop_token const& token) {
        this->active_thread.fetch_add(1);
        std::function<void()> current_task;
        while (true) {
            int current_priority = 0;
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
                    if (this->policy == shutdown_policy::discard) {
                        // every still-queued task is dropped without running: unwind their
                        // pending counts so priority waiters are not stuck forever
                        while (!this->tasks.empty()) {
                            this->note_task_finished(this->tasks.top().priority);
                            this->tasks.pop();
                        }
                        this->active_thread.fetch_sub(1);
                        this->idle.notify_all();
                        return;
                    }
                }

                current_task = this->tasks.top().action;
                current_priority = this->tasks.top().priority;
                this->tasks.pop();
            }
            current_task();
            {
                // finished (or dropped above): the task's priority group made progress
                std::lock_guard lock(this->access_mutex);
                this->note_task_finished(current_priority);
                this->idle.notify_all();
            }
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

    bool thread_pool::post(std::function<void()> task, int priority) {
        std::unique_lock lock(this->access_mutex);
        // after shutdown() every worker is leaving: queuing would sit forever unexecuted
        if (this->threads.empty() || this->threads.front().get_stop_source().stop_requested()) {
            return false;
        }
        this->note_task_posted(priority);
        this->tasks.emplace(priority, std::move(task));
        this->cv.notify_one();
        return true;
    }

    bool thread_pool::post_batch(std::span<std::function<void()>> const tasks, int const priority) {
        if (tasks.empty()) {
            return true;
        }
        std::unique_lock lock(this->access_mutex);
        if (this->threads.empty() || this->threads.front().get_stop_source().stop_requested()) {
            return false;
        }
        for (std::function<void()> const& task : tasks) {
            this->note_task_posted(priority);
            this->tasks.emplace(priority, task); // copies: the span is transient (const&)
        }
        this->cv.notify_all();
        return true;
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
        std::lock_guard lock(this->access_mutex);
        // same predicate as wait_until_free(): idle workers with queued tasks are not "free"
        return this->tasks.empty() && this->active_thread.load() == 0;
    }

    void thread_pool::wait_until_free() {
        std::unique_lock lock(this->access_mutex);
        this->idle.wait(lock, [this] { return this->tasks.empty() && this->active_thread.load() == 0; });
    }

    void thread_pool::wait_until_priority_done(int const priority) {
        std::unique_lock lock(this->access_mutex);
        this->idle.wait(lock, [this, priority] {
            return !this->pending_by_priority.contains(priority);
        });
    }

    int thread_pool::thread_count() const noexcept {
        return static_cast<int>(this->threads.size());
    }

    int thread_pool::get_active_thread() const {
        return this->active_thread.load();
    }
} // namespace utility
