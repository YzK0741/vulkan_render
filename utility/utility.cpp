module;

#include <openssl/evp.h>

module utility;

std::optional<uint64_t> utility::enable_handle_distribute::distribute() noexcept {
    std::lock_guard guard(this->access_mutex);
    if (!this->recycled_handles.empty()) {
        const auto it = recycled_handles.begin();
        uint64_t handle = *it;
        recycled_handles.erase(it);
        return handle;
    }
    if (this->handle_upper_bound < UINT64_MAX) {
        return this->handle_upper_bound++;
    }
    return std::nullopt;
}

void utility::enable_handle_distribute::recycle(const uint64_t handle) noexcept {
    std::lock_guard guard(this->access_mutex);
    if (handle < this->handle_upper_bound && !this->recycled_handles.contains(handle)) {
        this->recycled_handles.insert(handle);
    }
}

void utility::enable_stack_destruct::register_cleanup(std::function<void()> const& destructor) noexcept {
    std::lock_guard guard(this->access_mutex);
    this->destruct_stack.push(destructor);
}

void utility::enable_stack_destruct::do_cleanup() noexcept {
    std::lock_guard guard(this->access_mutex);
    while (!this->destruct_stack.empty()) {
        auto destructor = this->destruct_stack.top();
        destructor();
        this->destruct_stack.pop();
    }
}

void utility::enable_stack_destruct::pop_destructor() noexcept {
    this->destruct_stack.pop();
}

void utility::enable_stack_destruct::clear_stack() noexcept {
    this->destruct_stack = std::stack<destruct_type>();
}

namespace {
    std::stack<std::function<void()>> tasks = {};
    std::mutex access_mutex = {};
} // namespace

void utility::at_panic(std::function<void()> const& task) {
    std::lock_guard guard(access_mutex);
    tasks.push(task);
}

[[noreturn]] void utility::panic(std::string_view msg, std::source_location source_location) noexcept {
    std::lock_guard guard(access_mutex);
    // Route through error(): Debug prints to stderr, Release (GUI subsystem, no console) writes to debug.log
    error("program panic!");

    error("processing terminate tasks...");

    while (!tasks.empty()) {
        tasks.top()();
        tasks.pop();
    }

    if (!msg.empty()) {
        error("error info: {}", msg);
    }

    error("occurred at function [{}] line {}", source_location.function_name(), source_location.line());
    error("time point: {:%Y-%m-%d %H:%M:%S}", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));

    // Release writes logs through the async log thread; flush before terminating,
    // otherwise the panic messages above may be lost (std::terminate skips static destructors).
    wait_log_all();

    std::terminate();
}

std::chrono::milliseconds utility::time_test(std::function<void()> const& test) noexcept {
    const auto start = std::chrono::steady_clock::now();

    test();

    const auto end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

std::optional<std::vector<unsigned char>> utility::read_binary_to_vector(const std::filesystem::path& path) {
    std::error_code error;
    const uintmax_t file_size = std::filesystem::file_size(path, error);
    if (error) {
        return std::nullopt;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }
    // Preallocate based on file size to avoid repeated reallocation while reading
    std::vector<unsigned char> data;
    data.reserve(static_cast<size_t>(file_size));
    data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (file.bad()) {
        return std::nullopt;
    }
    return data;
}

std::optional<std::string> utility::read_binary_to_string(const std::filesystem::path& path) {
    std::error_code error;
    const uintmax_t file_size = std::filesystem::file_size(path, error);
    if (error) {
        return std::nullopt;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }
    // Preallocate based on file size to avoid repeated reallocation while reading
    std::string data;
    data.reserve(static_cast<size_t>(file_size));
    data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (file.bad()) {
        return std::nullopt;
    }
    return data;
}

// ---- Async logging (Meyer singleton, internal implementation) ----

namespace {
    // Startup rotation for the Release log file: move the previous session's debug.log content
    // aside to debug.log.old (with a session-end timestamp when the content carries none), then
    // truncate debug.log so the new session starts fresh. Only called in Release builds (NDEBUG).
    [[maybe_unused]] void rotate_previous_log() {
        // Text mode on both sides: the read translates CRLF to LF, the text-mode write
        // translates LF back to CRLF, so line endings stay consistent with debug.log
        std::ifstream current_log("debug.log");
        if (!current_log) {
            return; // no previous log yet
        }
        current_log.seekg(0, std::ios::end);
        if (current_log.tellg() <= 0) {
            return; // empty, nothing to rotate
        }
        current_log.seekg(0, std::ios::beg);

        const std::string content((std::istreambuf_iterator<char>(current_log)), std::istreambuf_iterator<char>());
        current_log.close();

        std::ofstream old_log("debug.log.old", std::ios::out | std::ios::app);
        if (!old_log) {
            return;
        }

        // Timestamp the rotated block so sessions are distinguishable in debug.log.old
        if (content.find("===== session") == std::string::npos) {
            old_log << std::format("===== session ended at {:%Y-%m-%d %H:%M:%S} =====\n",
                                   std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
        }

        // Normalize to one blank line after every log line, matching the worker's debug.log
        // format (idempotent: already double-spaced content stays unchanged)
        std::istringstream lines(content);
        std::string line;
        while (std::getline(lines, line)) {
            if (!line.empty()) {
                old_log << line << '\n'
                        << '\n';
            }
        }
        old_log.close();

        // Start the new session with an empty debug.log
        std::ofstream fresh_log("debug.log", std::ios::out | std::ios::trunc);
        fresh_log.close();
    }
} // namespace

utility::log_sink& utility::log_sink::instance() noexcept {
    static log_sink instance;
    return instance;
}

utility::log_sink::log_sink() {
#ifdef NDEBUG
    // Release builds: rotate the previous session's log aside, then append the new session
    rotate_previous_log();
    this->file.open("debug.log", std::ios::out | std::ios::app);
#endif
    this->worker = std::thread([this] { this->worker_loop(); });
}

utility::log_sink::~log_sink() {
    this->running = false;
    this->queue_cv.notify_all();
    if (this->worker.joinable()) {
        this->worker.join(); // wait for the worker to drain the queue before exiting
    }
#ifdef NDEBUG
    if (this->file.is_open()) {
        this->file.close();
    }
#endif
}

void utility::log_sink::worker_loop() noexcept {
    while (true) {
        std::string message;
        {
            std::unique_lock lock(this->queue_mutex);
            // Keep waiting for messages; drain the queue before exiting
            this->queue_cv.wait(lock, [this] { return !this->running || !this->messages.empty(); });
            if (this->messages.empty()) {
                if (!this->running) {
                    break;
                }
                continue;
            }
            message = std::move(this->messages.front());
            this->messages.pop();
        }
        // Write outside the lock to avoid blocking producers (a blank line follows every
        // message for readability; messages carry no \n)
#ifdef NDEBUG
        if (this->file.is_open()) {
            this->file << message << '\n'
                       << '\n'
                       << std::flush;
        } else {
            std::println("{}", message); // fall back to the terminal if the file cannot be opened
        }
#else
        std::println("{}", message);
#endif
        // Decrement pending only after the write finishes so wait_log_all also covers the message being written
        {
            std::lock_guard lock(this->queue_mutex);
            --this->pending;
            if (this->pending == 0) {
                this->drained_cv.notify_all();
            }
        }
    }
}

void utility::log_sink::write(std::string message) {
    {
        std::lock_guard lock(this->queue_mutex);
        ++this->pending;
        this->messages.push(std::move(message));
    }
    this->queue_cv.notify_one();
}

void utility::log_sink::wait_all() {
    std::unique_lock lock(this->queue_mutex);
    this->drained_cv.wait(lock, [this] { return this->pending == 0; });
}

void utility::error_message(std::string message) {
#ifdef NDEBUG
    // Release: hand to the log thread (writes to debug.log)
    log_sink::instance().write("[ERROR] " + std::move(message));
#else
    // Debug: print directly to stderr in red, no queueing (error is usually followed by terminate)
    std::print(stderr, "\x1b[31m[ERROR] {}\x1b[0m\n", message);
#endif
}

std::optional<utility::md5_digest> utility::md5(const std::span<const unsigned char> data_view) {
    md5_digest digest = {};
    unsigned int size_byte = sizeof(md5_digest);
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();

    if (!ctx) {
        return std::nullopt;
    }

    if (!EVP_DigestInit_ex(ctx, EVP_md5(), nullptr)) {
        EVP_MD_CTX_destroy(ctx);
        return std::nullopt;
    }
    if (!EVP_DigestUpdate(ctx, data_view.data(), data_view.size_bytes())) {
        EVP_MD_CTX_destroy(ctx);
        return std::nullopt;
    }

    if (const int code = EVP_DigestFinal_ex(ctx, digest.data.data(), &size_byte); code == 1) {
        EVP_MD_CTX_destroy(ctx);
        return digest;
    }
    EVP_MD_CTX_destroy(ctx);
    return std::nullopt;
}

std::optional<utility::sha256_digest> utility::sha256(const std::span<const unsigned char> data_view) {
    sha256_digest digest = {};
    unsigned int size_byte = sizeof(sha256_digest);
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();

    if (!ctx) {
        return std::nullopt;
    }

    if (!EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr)) {
        EVP_MD_CTX_destroy(ctx);
        return std::nullopt;
    }
    if (!EVP_DigestUpdate(ctx, data_view.data(), data_view.size_bytes())) {
        EVP_MD_CTX_destroy(ctx);
        return std::nullopt;
    }

    if (const int code = EVP_DigestFinal_ex(ctx, digest.data.data(), &size_byte); code == 1) {
        EVP_MD_CTX_destroy(ctx);
        return digest;
    }
    EVP_MD_CTX_destroy(ctx);
    return std::nullopt;
}

std::optional<utility::blake2_digest> utility::blake2(std::span<const unsigned char> data_view) {
    blake2_digest digest = {};

    unsigned int size_byte = blake2_digest::size_byte;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();

    if (!ctx) {
        return std::nullopt;
    }

    if (!EVP_DigestInit_ex(ctx, EVP_blake2b512(), nullptr)) {
        EVP_MD_CTX_destroy(ctx);
        return std::nullopt;
    }
    if (!EVP_DigestUpdate(ctx, data_view.data(), data_view.size_bytes())) {
        EVP_MD_CTX_destroy(ctx);
        return std::nullopt;
    }

    if (const int code = EVP_DigestFinal_ex(ctx, digest.data.data(), &size_byte); code == 1) {
        EVP_MD_CTX_destroy(ctx);
        return digest;
    }
    EVP_MD_CTX_destroy(ctx);
    return std::nullopt;
}