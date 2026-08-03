//
// Created by 小叶 on 2026/7/31.
//
module;

#include <functional>
#include <stack>
#include <cstdint>
#include <chrono>
#include <source_location>
#include <exception>
#include <mutex>
#include <span>
#include <optional>
#include <print>
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

void utility::enable_stack_destruct::cleanup() noexcept {
    std::lock_guard guard(this->access_mutex);
    while (!this->destruct_stack.empty()) {
        auto destructor = this->destruct_stack.top();
        destructor();
        this->destruct_stack.pop();
    }
}

void utility::enable_stack_destruct::clear() noexcept {
    this->destruct_stack = std::stack<destruct_type>();
}

namespace {
    std::stack<std::function<void()>> tasks = {};
    std::mutex access_mutex = {};
}

void utility::at_terminate(std::function<void()> const& task) {
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

bool utility::md5_digest::operator==(const md5_digest &other) const noexcept {
    return (this->a == other.a && this->b == other.b);
}

bool utility::md5_digest::operator!=(md5_digest const &other) const noexcept {
    return !(*this == other);
}

std::string utility::md5_digest::to_hex_string() const noexcept {
    return std::format("{:016x}{:016x}", this->a, this->b);
}

std::optional<utility::md5_digest> utility::md5(const std::span<const unsigned char> data) {
    md5_digest result = {};
    unsigned int size_byte = sizeof(uint64_t) * 2;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();

    if (!ctx) {
        return std::nullopt;
    }

    if (!EVP_DigestInit_ex(ctx, EVP_md5(), nullptr)) {
        EVP_MD_CTX_destroy(ctx);
        return std::nullopt;
    }
    if (!EVP_DigestUpdate(ctx, data.data(), data.size_bytes())) {
        EVP_MD_CTX_destroy(ctx);
        return std::nullopt;
    }

    if (const int code = EVP_DigestFinal_ex(ctx, reinterpret_cast<unsigned char *>(&result), &size_byte); code == 1) {
        EVP_MD_CTX_destroy(ctx);
        return result;
    }
    EVP_MD_CTX_destroy(ctx);
    return std::nullopt;
}
