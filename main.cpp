import vulkan.runtime;
import utility;

#include <print>
#include <span>




int main() {
    //utility::panic();

    vulkan::runtime runtime;
    std::println("init succeeded");
    std::vector<int> test_data = {1u, 2u, 3u, 4u};
    test_data.resize(1024 * 1024 * 96);

    const auto data_span = std::span(test_data.data(), test_data.size());

    std::println("buffer upload test:");

    auto time = utility::time_test([&]{runtime->vma.create_buffer(data_span, vulkan::buffer_type::vertex);});

    std::println("{}B size vertex buffer upload succeeded in {}ms", test_data.size() * sizeof(int), time.count());

    time = utility::time_test([&]{runtime->vma.create_buffer(data_span, vulkan::buffer_type::index);});
    std::println("{}B size index buffer upload succeeded in {}ms", test_data.size() * sizeof(int), time.count());

    time = utility::time_test([&]{runtime->vma.create_buffer(data_span, vulkan::buffer_type::uniform_coherent);});
    std::println("{}B size coherent uniform buffer upload succeeded in {}ms", test_data.size() * sizeof(int), time.count());

    std::println("{}B size total upload succeeded", test_data.size() * sizeof(int) * 3);
 }
