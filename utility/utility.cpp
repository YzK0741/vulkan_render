module;

#include <cstdio> // std::print(stderr, ...) below needs the stderr macro (not exportable via modules)
#include <cstring>
#include <xxhash.h>

module utility;

std::optional<uint64_t> utility::enable_handle_distribute::distribute() noexcept {
    std::lock_guard guard(this->access_mutex);
    if (!this->recycled_handles.empty()) {
        auto const it = recycled_handles.begin();
        uint64_t handle = *it;
        recycled_handles.erase(it);
        return handle;
    }
    if (this->handle_upper_bound < UINT64_MAX) {
        return this->handle_upper_bound++;
    }
    return std::nullopt;
}

void utility::enable_handle_distribute::recycle(uint64_t const handle) noexcept {
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
    // Swap the stack out under the lock, then run the destructors unlocked: a callback may
    // itself call register_cleanup() (it takes the same mutex) and would deadlock otherwise.
    std::stack<destruct_type> pending;
    {
        std::lock_guard guard(this->access_mutex);
        pending.swap(this->destruct_stack); // noexcept
    }
    while (!pending.empty()) {
        pending.top()(); // callbacks run without the lock held
        pending.pop();
    }
}

void utility::enable_stack_destruct::pop_destructor() noexcept {
    std::lock_guard guard(this->access_mutex);
    this->destruct_stack.pop();
}

void utility::enable_stack_destruct::clear_stack() noexcept {
    std::lock_guard guard(this->access_mutex);
    std::stack<destruct_type>{}.swap(this->destruct_stack); // swap (not assign): noexcept
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
    error("program panic!");

    error("processing terminate tasks...");

    // Snapshot the registered at_panic tasks under the lock and run them unlocked: a task may
    // itself call at_panic()/error() (both take this mutex) and would deadlock otherwise.
    std::stack<std::function<void()>> pending;
    {
        std::lock_guard guard(access_mutex);
        pending.swap(tasks); // noexcept
    }
    while (!pending.empty()) {
        pending.top()();
        pending.pop();
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
    auto const start = std::chrono::steady_clock::now();

    test();

    auto const end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

std::optional<std::vector<unsigned char>> utility::read_binary_to_vector(std::filesystem::path const& path) {
    std::error_code error;
    uintmax_t const file_size = std::filesystem::file_size(path, error);
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

std::optional<std::string> utility::read_binary_to_string(std::filesystem::path const& path) {
    std::error_code error;
    uintmax_t const file_size = std::filesystem::file_size(path, error);
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

        std::string const content((std::istreambuf_iterator<char>(current_log)), std::istreambuf_iterator<char>());
        current_log.close();

        // Cap debug.log.old: once it exceeds the cap, start it fresh (truncate) instead of
        // appending forever, so the archive stays bounded across many sessions.
        constexpr uintmax_t old_log_cap = 8ull * 1024ull * 1024ull; // 8 MiB
        std::ios::openmode const old_mode = [&] {
            std::error_code ec;
            uintmax_t const size = std::filesystem::file_size("debug.log.old", ec);
            return (!ec && size >= old_log_cap) ? (std::ios::out | std::ios::trunc) : (std::ios::out | std::ios::app);
        }();
        std::ofstream old_log("debug.log.old", old_mode);
        if (!old_log) {
            return;
        }

        // Timestamp the rotated block so sessions are distinguishable in debug.log.old
        if (!content.contains("===== session")) {
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

utility::xxh3_digest utility::xxh3_128bits(std::span<unsigned char const> const data_view) {
    // XXH3_128bits returns a {low64, high64} pair; store its bytes in the digest
    xxh3_digest digest = {};
    XXH128_hash_t const hash = XXH3_128bits(data_view.data(), data_view.size_bytes());
    std::memcpy(digest.data.data(), &hash, sizeof(hash));
    return digest;
}
namespace {
    // CRC32 (PNG chunk checksums) - table generated at compile time
    constexpr std::array<uint32_t, 256> make_crc_table() {
        std::array<uint32_t, 256> table = {};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) != 0u ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        return table;
    }
    constexpr std::array<uint32_t, 256> crc_table = make_crc_table();

    void append_u32_be(std::vector<unsigned char>& out, uint32_t const value) {
        out.push_back(static_cast<unsigned char>((value >> 24) & 0xFFu));
        out.push_back(static_cast<unsigned char>((value >> 16) & 0xFFu));
        out.push_back(static_cast<unsigned char>((value >> 8) & 0xFFu));
        out.push_back(static_cast<unsigned char>(value & 0xFFu));
    }

    void append_png_chunk(std::vector<unsigned char>& out, char const* type, std::span<unsigned char const> const payload) {
        append_u32_be(out, static_cast<uint32_t>(payload.size()));
        std::size_t const crc_begin = out.size();
        for (int i = 0; i < 4; ++i) {
            out.push_back(static_cast<unsigned char>(type[i]));
        }
        out.insert(out.end(), payload.begin(), payload.end());

        uint32_t crc = 0xFFFFFFFFu;
        for (std::size_t i = crc_begin; i < out.size(); ++i) {
            crc = crc_table[(crc ^ out[i]) & 0xFFu] ^ (crc >> 8);
        }
        append_u32_be(out, crc ^ 0xFFFFFFFFu);
    }
} // namespace

std::expected<void, std::string> utility::write_png(std::filesystem::path const& path, uint32_t const width, uint32_t const height, std::span<unsigned char const> const rgba) {
    std::size_t const expected = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
    if (width == 0 || height == 0 || rgba.size() < expected) {
        return std::unexpected(std::string("write_png: pixel data does not match the dimensions"));
    }

    // raw scanlines: one filter byte (0 = none) followed by the RGBA row
    std::vector<unsigned char> raw;
    raw.reserve(expected + height);
    for (uint32_t y = 0; y < height; ++y) {
        raw.push_back(0);
        auto const row = rgba.subspan(static_cast<std::size_t>(y) * static_cast<std::size_t>(width) * 4u, static_cast<std::size_t>(width) * 4u);
        raw.insert(raw.end(), row.begin(), row.end());
    }

    std::vector<unsigned char> zlib;
    zlib.reserve(raw.size() + raw.size() / 65535u * 5u + 16u);
    zlib.push_back(0x78); // CM = 8 (deflate), CINFO = 7 (32K window)
    zlib.push_back(0x01); // FCHECK so that (0x7801 % 31) == 0
    std::size_t offset = 0;
    while (offset < raw.size()) {
        std::size_t const block = std::min<std::size_t>(raw.size() - offset, 65535u);
        bool const last = offset + block >= raw.size();
        zlib.push_back(last ? 1u : 0u);
        zlib.push_back(static_cast<unsigned char>(block & 0xFFu));
        zlib.push_back(static_cast<unsigned char>((block >> 8) & 0xFFu));
        zlib.push_back(static_cast<unsigned char>(~block & 0xFFu));
        zlib.push_back(static_cast<unsigned char>((~block >> 8) & 0xFFu));
        zlib.insert(zlib.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset), raw.begin() + static_cast<std::ptrdiff_t>(offset + block));
        offset += block;
    }
    uint32_t adler_a = 1;
    uint32_t adler_b = 0;
    for (unsigned char const byte : raw) {
        adler_a = (adler_a + byte) % 65521u;
        adler_b = (adler_b + adler_a) % 65521u;
    }
    append_u32_be(zlib, (adler_b << 16) | adler_a);

    std::vector<unsigned char> png = {0x89u, 'P', 'N', 'G', 0x0Du, 0x0Au, 0x1Au, 0x0Au};
    std::vector<unsigned char> ihdr;
    append_u32_be(ihdr, width);
    append_u32_be(ihdr, height);
    ihdr.push_back(8); // bit depth
    ihdr.push_back(6); // color type: RGBA
    ihdr.push_back(0); // compression: deflate
    ihdr.push_back(0); // filter method: adaptive
    ihdr.push_back(0); // interlace: none
    append_png_chunk(png, "IHDR", ihdr);
    append_png_chunk(png, "IDAT", zlib);
    append_png_chunk(png, "IEND", {});

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        return std::unexpected(std::format("write_png: cannot open '{}'", path.string()));
    }
    file.write(reinterpret_cast<char const*>(png.data()), static_cast<std::streamsize>(png.size()));
    if (!file) {
        return std::unexpected(std::format("write_png: write failed for '{}'", path.string()));
    }
    return {};
}
