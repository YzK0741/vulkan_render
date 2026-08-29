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
    std::println(stderr, "program panic!");

    std::println(stderr, "processing terminate tasks...");

    while (!tasks.empty()) {
        tasks.top()();
        tasks.pop();
    }

    if (!msg.empty()) {
        std::println(stderr, "error info: {}", msg);
    }

    std::println(stderr, "occurred at function [{}] line {}", source_location.function_name(), source_location.line());
    std::println(stderr, "time point: {:%Y-%m-%d %H:%M:%S}", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
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
    // 按文件大小预分配，避免读取过程中反复扩容
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
    // 按文件大小预分配，避免读取过程中反复扩容
    std::string data;
    data.reserve(static_cast<size_t>(file_size));
    data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (file.bad()) {
        return std::nullopt;
    }
    return data;
}

// ---- 异步日志（Meyer 单例，内部实现） ----

utility::log_sink& utility::log_sink::instance() noexcept {
    static log_sink instance;
    return instance;
}

utility::log_sink::log_sink() {
#ifdef NDEBUG
    // Release 构建：写入 debug.log（追加模式）
    this->file.open("debug.log", std::ios::out | std::ios::app);
#endif
    this->worker = std::thread([this] { this->worker_loop(); });
}

utility::log_sink::~log_sink() {
    this->running = false;
    this->queue_cv.notify_all();
    if (this->worker.joinable()) {
        this->worker.join(); // 等 worker 排空队列后退出
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
            // 一直等待并尝试取消息；退出前把队列排空
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
        // 锁外输出，避免阻塞生产者（统一补换行，消息本身不带 \n）
#ifdef NDEBUG
        if (this->file.is_open()) {
            this->file << message << '\n'
                       << std::flush;
        } else {
            std::println("{}", message); // 文件打不开时回退终端
        }
#else
        std::println("{}", message);
#endif
        // 输出完成后再递减待写计数，wait_log_all 才能等到包括"正在输出"的最后一条
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
    // Release：交给日志线程（写入 debug.log）
    log_sink::instance().write("[ERROR] " + std::move(message));
#else
    // Debug：直接红色输出到 stderr，醒目不排队（error 之后基本是 terminate）
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