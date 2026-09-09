// Headless unit tests: utility module (pure CPU) ===============================
// Covers xxh3 content hashing (drives image dedup), data_block key semantics,
// the thread pool, and the BVH frustum culling used by the main pass.
#include "vk_test.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <span>
#include <thread>
#include <vector>

import utility;

namespace {
    // Compile-time self-checks: data_block is fully constexpr (zero-init default + FNV-1a).
    // FNV-1a-64 golden vectors: {1,2,3,4} -> 13725386680924731485, {0,0,0,0} -> 5558979605539197941.
    constexpr unsigned char golden_bytes[] = {1, 2, 3, 4};
    static_assert(utility::data_block<4>(golden_bytes).hash64() == 13725386680924731485ull);
    static_assert(utility::data_block<4>().hash64() == 5558979605539197941ull); // default = zeroed
    static_assert(utility::data_block<4>(golden_bytes) == utility::data_block<4>(golden_bytes));

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
        // the C-array constructor copies element-for-element (also exercised at compile time above)
        utility::data_block<4> const from_c_array(golden_bytes);
        CHECK(from_c_array == x);
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

    // Regression: two threads waiting on the SAME pool + a wait_until_free waiter. With
    // notify_one on the idle/completion paths one waiter could consume a notification while its
    // predicate was already true and leave the other waiter sleeping forever (predicate true,
    // no further wake). Would hang this test under the old implementation.
    void test_thread_pool_two_concurrent_waiters() {
        utility::thread_pool pool(2);
        std::function<void()> const slow_tick = [] { std::this_thread::sleep_for(std::chrono::milliseconds(4)); };
        std::vector<std::function<void()>> batch(6, slow_tick);
        CHECK(pool.post_batch(batch, 11));

        std::atomic<bool> waiter_a_done = false;
        std::atomic<bool> waiter_b_done = false;
        std::atomic<bool> free_waiter_done = false;
        std::jthread waiter_a([&] {
            pool.wait_until_priority_done(11);
            waiter_a_done.store(true);
        });
        std::jthread waiter_b([&] {
            pool.wait_until_priority_done(11);
            waiter_b_done.store(true);
        });
        std::jthread free_waiter([&] {
            pool.wait_until_free();
            free_waiter_done.store(true);
        });
        waiter_a.join();
        waiter_b.join();
        free_waiter.join();
        CHECK(waiter_a_done.load());
        CHECK(waiter_b_done.load());
        CHECK(free_waiter_done.load());
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

    // Documented contract: add() new leaves, then rebuild(). Leaf storage must keep element
    // addresses stable across the growth (a vector would reallocate and dangle the internal
    // nodes until rebuild). Regression: repeated add()+rebuild() with a wide frustum must keep
    // returning every leaf.
    void test_bvh_add_rebuild_contract() {
        constexpr int initial = 12;
        constexpr int grown = 18;
        int ids[grown];
        std::vector<utility::aabb_box<int>> boxes;
        boxes.reserve(initial);
        for (int i = 0; i < initial; ++i) {
            ids[i] = i;
            float const x = static_cast<float>(i % 4) * 1.5f - 2.25f;
            boxes.push_back(utility::aabb_box<int>{.min = glm::vec3(x - 0.2f, -0.2f, -3.0f), .max = glm::vec3(x + 0.2f, 0.2f, -2.6f), .extra_data = &ids[i]});
        }
        auto tree = utility::bvh<int>::make(boxes);
        CHECK_MSG(tree.has_value(), tree.error().c_str());
        if (!tree.has_value()) {
            return;
        }
        // wide frustum: everything we place is inside
        auto const visible_count = [&tree]() -> std::size_t {
            glm::mat4 const proj = glm::perspective(glm::radians(150.0f), 1.0f, 0.1f, 60.0f);
            glm::mat4 const view = glm::lookAt(glm::vec3(0.0f, 0.0f, 12.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            return tree->frustum_cull(utility::make_frustum(proj * view)).size();
        };
        CHECK(visible_count() == initial);
        for (int i = initial; i < grown; ++i) {
            ids[i] = i;
            float const x = static_cast<float>(i) * 1.5f;
            CHECK(tree->add(utility::aabb_box<int>{.min = glm::vec3(x - 0.2f, -0.2f, -3.0f), .max = glm::vec3(x + 0.2f, 0.2f, -2.6f), .extra_data = &ids[i]}).has_value());
        }
        tree->rebuild();
        CHECK(visible_count() == grown);
    }
} // namespace

int main() {
    test_xxh3_content_hash();
    test_data_block_key_semantics();
    test_thread_pool_runs_every_posted_task();
    test_thread_pool_priority_group_wait();
    test_thread_pool_two_concurrent_waiters();
    test_bvh_frustum_cull_keeps_visible_boxes();
    test_bvh_add_rebuild_contract();
    return vk_test::finish("test_utility");
}
