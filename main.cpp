import std;
import gltf_loader;
import vulkan.runtime;
import vulkan.core.pipeline;

#include "utility/thread_pool/thread_pool.cppm"

int main() {
    utility::thread_pool pool(2);
    vulkan::runtime runtime;
    std::vector<uint8_t> data(1024 * 1024 * 512);
    std::vector<uint64_t> v_buffer;
    auto task_vertex = [&] {
        auto handle = runtime->vma.create_buffer(std::span(data), vulkan::buffer_type::vertex);
        v_buffer.push_back(handle);
        std::println("{}b vertex data sent", data.size());
    };

    std::vector<uint64_t> i_buffer;

    auto task_index = [&] {
        auto handle = runtime->vma.create_buffer(std::span(data), vulkan::buffer_type::index);
        i_buffer.push_back(handle);
        std::println("{}b index data sent", data.size());
    };

    std::vector<uint64_t> u_buffer;

    auto task_uniform = [&] {
        auto handle = runtime->vma.create_buffer(std::span(data), vulkan::buffer_type::uniform_coherent);
        u_buffer.push_back(handle);
        std::println("{}b uniform data sent", data.size());
    };

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 4; i++) {
        pool.post(task_vertex);
        pool.post(task_index);
        pool.post(task_uniform);
    }

    pool.wait_until_free();

    auto end = std::chrono::steady_clock::now();

    std::println("{}b total size sent", 12 * data.size());
    std::println("total time cost {}", end - start);

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
