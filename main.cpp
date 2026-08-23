import vulkan.runtime;
import utility;

import std;

#include "utility/thread_pool/thread_pool.cppm"

/*

vulkan::runtime runtime;
    std::println("init succeeded");


    std::atomic_uint64_t total_size = 0;

    const auto data_span = std::span(test_data.data(), test_data.size());

    std::println("buffer upload test:");

    auto vertex_test = [&] {
        const auto time = utility::time_test([&]{runtime->vma.create_buffer(data_span, vulkan::buffer_type::vertex);});

        std::println("{}B size vertex buffer upload succeeded in {}ms", test_data.size() * sizeof(int), time.count());

        total_size.fetch_add(test_data.size());
    };

    auto index_test = [&] {
        const auto time = utility::time_test([&]{runtime->vma.create_buffer(data_span, vulkan::buffer_type::index);});
        std::println("{}B size index buffer upload succeeded in {}ms", test_data.size() * sizeof(int), time.count());
        total_size.fetch_add(test_data.size());
    };

    auto uniform_test = [&] {
        const auto time = utility::time_test([&]{runtime->vma.create_buffer(data_span, vulkan::buffer_type::uniform_coherent);});
        std::println("{}B size coherent uniform buffer upload succeeded in {}ms", test_data.size() * sizeof(int), time.count());
        total_size.fetch_add(test_data.size());
    };


#define for_times(x) for(int a_##__LINE__ = 0; a_##__LINE__ < x; a_##__LINE__++)

    for_times(4) {
        pool.post(vertex_test);
        pool.post(index_test);
        pool.post(uniform_test);
    }

    std::println("{}B size total upload succeeded", total_size.load());


    pool.wait_until_free();

    std::println("wait end");

 */
int main() {

    vulkan::runtime runtime;

    utility::thread_pool pool(4);

    std::vector<uint8_t> test_data(1024 * 1024 * 256);

    auto span = std::span(test_data);
    std::vector<uint64_t> handles;

    auto task = [&] {
        const auto handle = runtime->vma.create_buffer<uint8_t>(span, vulkan::buffer_type::uniform_coherent);
        handles.push_back(handle);
        std::println("buffer created handle: {}", handle);
    };

    for (int a = 0; a < 16; a++) {
        pool.post(task);
    }


    pool.wait_until_free();
    std::println("{}", handles);
}
