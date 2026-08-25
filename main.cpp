import std;
import gltf_loader;
import vulkan.runtime;
import vulkan.core.pipeline;

#include "utility/thread_pool/thread_pool.cppm"

int main() {
    constexpr int rounds = 4;
    utility::thread_pool pool(2);
    vulkan::runtime runtime;
    std::vector<uint8_t> data(1024ull * 1024ull * 512ull);

    // 预分配 + 原子索引，避免并发 push_back
    std::vector<uint64_t> v_buffer(rounds);
    std::vector<uint64_t> i_buffer(rounds);
    std::vector<uint64_t> u_buffer(rounds);
    std::atomic<int> v_idx{0};
    std::atomic<int> i_idx{0};
    std::atomic<int> u_idx{0};
    std::atomic<int64_t> v_ns{0};
    std::atomic<int64_t> i_ns{0};
    std::atomic<int64_t> u_ns{0};

    auto task_vertex = [&] {
        auto t0 = std::chrono::steady_clock::now();
        auto handle = runtime->vma.create_buffer(std::span(data), vulkan::buffer_type::vertex);
        auto t1 = std::chrono::steady_clock::now();
        v_buffer[v_idx.fetch_add(1)] = handle;
        v_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        std::println("{}b vertex data sent", data.size());
    };

    auto task_index = [&] {
        auto t0 = std::chrono::steady_clock::now();
        auto handle = runtime->vma.create_buffer(std::span(data), vulkan::buffer_type::index);
        auto t1 = std::chrono::steady_clock::now();
        i_buffer[i_idx.fetch_add(1)] = handle;
        i_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        std::println("{}b index data sent", data.size());
    };

    auto task_uniform = [&] {
        auto t0 = std::chrono::steady_clock::now();
        auto handle = runtime->vma.create_buffer(std::span(data), vulkan::buffer_type::uniform_coherent);
        auto t1 = std::chrono::steady_clock::now();
        u_buffer[u_idx.fetch_add(1)] = handle;
        u_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        std::println("{}b uniform data sent", data.size());
    };

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < rounds; i++) {
        pool.post(task_vertex);
        pool.post(task_index);
        pool.post(task_uniform);
    }

    pool.wait_until_free();

    auto end = std::chrono::steady_clock::now();

    std::println("{}b total size sent", 12 * data.size());
    std::println("total time cost {}", end - start);
    std::println("vertex  avg: {:.1f} ms/call (staging: memcpy + gpu copy)", static_cast<double>(v_ns.load()) / 1e6 / rounds);
    std::println("index   avg: {:.1f} ms/call (staging: memcpy + gpu copy)", static_cast<double>(i_ns.load()) / 1e6 / rounds);
    std::println("uniform avg: {:.1f} ms/call (direct : memcpy only)     ", static_cast<double>(u_ns.load()) / 1e6 / rounds);

    for (auto const& v : v_buffer) {
        runtime->vma.free_buffer(v);
    }

    for (auto const& i : i_buffer) {
        runtime->vma.free_buffer(i);
    }

    for (auto const& u : u_buffer) {
        runtime->vma.free_buffer(u);
    }
}
