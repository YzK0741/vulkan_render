// Headless unit tests: utility module (pure CPU) ===============================
// Covers xxh3 content hashing (drives image dedup), data_block key semantics,
// the thread pool, and the BVH frustum culling used by the main pass.
#include "vk_test.h"

#include <atomic>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <span>
#include <vector>

import utility;

namespace {
    void test_xxh3_content_hash() {
        unsigned char const a[] = {1, 2, 3, 4, 5};
        unsigned char const b[] = {1, 2, 3, 4, 5};
        unsigned char const c[] = {1, 2, 3, 4, 6};
        utility::xxh3_digest const da = utility::xxh3_128bits(std::span<unsigned char const>(a));
        utility::xxh3_digest const db = utility::xxh3_128bits(std::span<unsigned char const>(b));
        utility::xxh3_digest const dc = utility::xxh3_128bits(std::span<unsigned char const>(c));
        CHECK(da == db); // deterministic
        CHECK(da != dc); // content-sensitive
        CHECK(utility::xxh3_digest::size_byte == 16);
    }

    void test_data_block_key_semantics() {
        utility::data_block<4> zeros{};
        utility::data_block<4> x{};
        utility::data_block<4> y{};
        x.data = {1, 2, 3, 4};
        y.data = {1, 2, 3, 4};
        CHECK(x == y);
        CHECK(x != zeros);
        CHECK(zeros < x); // lexicographic ordering for ordered containers
        CHECK(x.hash64() == y.hash64());
        CHECK(x.hash64() != zeros.hash64());
    }

    void test_thread_pool_runs_every_posted_task() {
        utility::thread_pool pool(2);
        std::atomic<int> counter = 0;
        for (int i = 0; i < 20; ++i) {
            bool const queued = pool.post([&counter] { counter.fetch_add(1, std::memory_order_relaxed); });
            CHECK(queued);
        }
        pool.wait_until_free(); // deterministic: every queued task has finished
        CHECK(counter.load(std::memory_order_relaxed) == 20);
    }

    void test_thread_pool_priority_group_wait() {
        utility::thread_pool pool(2);
        std::atomic<int> counter = 0;
        std::function<void()> const tick = [&counter] { counter.fetch_add(1, std::memory_order_relaxed); };
        // runtime pattern: post_batch() then wait_until_priority_done(priority)
        std::vector<std::function<void()>> batch(10, tick);
        CHECK(pool.post_batch(batch, 5));
        pool.wait_until_priority_done(5);
        CHECK(counter.load(std::memory_order_relaxed) == 10);
    }

    void test_bvh_frustum_cull_keeps_visible_boxes() {
        int ids[3] = {0, 1, 2};
        // camera at the origin looking down -z: boxes A and B are in front, C behind
        std::vector<utility::aabb_box<int>> boxes;
        boxes.push_back(utility::aabb_box<int>{.min = glm::vec3(-1.0f, -1.0f, -6.0f), .max = glm::vec3(1.0f, 1.0f, -4.0f), .extra_data = &ids[0]});
        boxes.push_back(utility::aabb_box<int>{.min = glm::vec3(-0.5f, -0.5f, -3.0f), .max = glm::vec3(0.5f, 0.5f, -2.0f), .extra_data = &ids[1]});
        boxes.push_back(utility::aabb_box<int>{.min = glm::vec3(-1.0f, -1.0f, 4.0f), .max = glm::vec3(1.0f, 1.0f, 6.0f), .extra_data = &ids[2]});

        auto const tree = utility::bvh<int>::make(boxes);
        CHECK_MSG(tree.has_value(), tree.error().c_str());
        if (!tree.has_value()) {
            return;
        }

        glm::mat4 const proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 20.0f);
        glm::mat4 const view = glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        utility::frustum const frustum = utility::make_frustum(proj * view);

        std::vector<utility::bvh_node<int>*> const inside = tree->frustum_cull(frustum);
        bool saw_a = false;
        bool saw_b = false;
        bool saw_c = false;
        for (utility::bvh_node<int>* node : inside) {
            saw_a |= node->extra_data == &ids[0];
            saw_b |= node->extra_data == &ids[1];
            saw_c |= node->extra_data == &ids[2];
        }
        CHECK(saw_a);
        CHECK(saw_b);
        CHECK(!saw_c); // behind the camera is culled
    }
} // namespace

int main() {
    test_xxh3_content_hash();
    test_data_block_key_semantics();
    test_thread_pool_runs_every_posted_task();
    test_thread_pool_priority_group_wait();
    test_bvh_frustum_cull_keeps_visible_boxes();
    return vk_test::finish("test_utility");
}
