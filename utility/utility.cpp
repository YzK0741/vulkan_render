//
// Created by 小叶 on 2026/7/31.
//
module;

#include <openssl/evp.h>

module utility;

std::optional<uint64_t> utility::enable_handle_distribute::distribute() noexcept {
    std::lock_guard guard(this->access_mutex);
    if (!this->recycled_handlers.empty()) {
        const auto it = recycled_handlers.begin();
        uint64_t handle = *it;
        recycled_handlers.erase(it);
        return handle;
    }
    if (this->handle_upper_bound < UINT64_MAX) {
        return this->handle_upper_bound++;
    }
    return std::nullopt;
}

void utility::enable_handle_distribute::recycle(const uint64_t handle) noexcept {
    std::lock_guard guard(this->access_mutex);
    if (handle < this->handle_upper_bound && !this->recycled_handlers.contains(handle)) {
        this->recycled_handlers.insert(handle);
    }
}

void utility::enable_stack_destruct::register_cleanup(std::function<void()> const &destructor) noexcept {
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
}

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

std::chrono::milliseconds utility::time_test(std::function<void()> const &test) noexcept {
    const auto start = std::chrono::steady_clock::now();

    test();

    const auto end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
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

std::optional<utility::sha256_digest> utility::sha256(const std::span<const unsigned char> data_view){
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