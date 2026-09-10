// Headless unit tests: utility module (pure CPU) ===============================
// Covers xxh3 content hashing (drives image dedup), data_block key semantics,
// the thread pool, and the BVH frustum culling used by the main pass.
#include "vk_test.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iterator>
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

    // utility::write_png is pure CPU, so the dependency-free PNG encoder is testable headlessly:
    // the test walks the chunk list and re-checks every CRC32 (a wrong encoder would not survive
    // a real decoder, but structure + checksums already catch the usual mistakes).
    void test_write_png() {
        std::vector<unsigned char> const pixels = {255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255}; // 2x2
        std::filesystem::path const path = "test_write_png.png";
        auto const written = utility::write_png(path, 2, 2, pixels);
        CHECK(written.has_value());
        if (!written.has_value()) {
            return;
        }

        std::ifstream file(path, std::ios::binary);
        CHECK(file.good());
        std::vector<unsigned char> const bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        CHECK(bytes.size() > 16);

        unsigned char const signature[] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
        CHECK(std::equal(std::begin(signature), std::end(signature), bytes.begin()));

        auto const read_be32 = [&bytes](std::size_t const at) {
            return (static_cast<uint32_t>(bytes[at]) << 24) | (static_cast<uint32_t>(bytes[at + 1]) << 16) | (static_cast<uint32_t>(bytes[at + 2]) << 8) | static_cast<uint32_t>(bytes[at + 3]);
        };

        bool saw_ihdr = false;
        bool saw_idat = false;
        bool saw_iend = false;
        bool crc_ok = true;
        std::size_t offset = 8;
        while (offset + 12 <= bytes.size()) {
            uint32_t const length = read_be32(offset);
            if (offset + 12 + length > bytes.size()) {
                crc_ok = false;
                break;
            }
            std::string_view const type(reinterpret_cast<char const*>(bytes.data() + offset + 4), 4);

            uint32_t crc = 0xFFFFFFFFu;
            for (std::size_t i = offset + 4; i < offset + 8 + length; ++i) {
                crc ^= bytes[i];
                for (int k = 0; k < 8; ++k) {
                    crc = (crc & 1u) != 0u ? (0xEDB88320u ^ (crc >> 1)) : (crc >> 1);
                }
            }
            crc ^= 0xFFFFFFFFu;
            if (crc != read_be32(offset + 8 + length)) {
                crc_ok = false;
            }

            if (type == "IHDR") {
                saw_ihdr = true;
                CHECK(read_be32(offset + 8) == 2);  // width
                CHECK(read_be32(offset + 12) == 2); // height
            } else if (type == "IDAT") {
                saw_idat = true;
            } else if (type == "IEND") {
                saw_iend = true;
                break;
            }
            offset += 12 + length;
        }
        CHECK(saw_ihdr);
        CHECK(saw_idat);
        CHECK(saw_iend);
        CHECK(crc_ok);
        std::filesystem::remove(path);
    }
    // ---- binary writer (utility::write_binary / write_single) ----
    // The writer is the one primitive every binary format in the project goes through, so these
    // checks pin down its contract: byte order per tagged scalar, exactly size() bytes for a range
    // (no length prefix), native layout for a POD, and a hard stop at the first failure.
    void test_write_binary_scalar_byte_order() {
        std::ostringstream out;
        CHECK(utility::write_binary(out,
                                    utility::be(uint16_t{0x0102}),
                                    utility::le(uint16_t{0x0102}),
                                    utility::be(uint32_t{0x01020304}),
                                    utility::le(uint32_t{0x01020304}),
                                    utility::be(uint32_t{0xFFFFFFFE}), // a full-width value, no sign surprises
                                    utility::le(int32_t{-2}))
                  .has_value());
        std::string const bytes = out.str();
        std::string const expected = std::string("\x01\x02", 2) + std::string("\x02\x01", 2) +
                                     std::string("\x01\x02\x03\x04", 4) + std::string("\x04\x03\x02\x01", 4) +
                                     std::string("\xFF\xFF\xFF\xFE", 4) + std::string("\xFE\xFF\xFF\xFF", 4);
        CHECK_MSG(bytes == expected, "big/little endian bytes");
        CHECK(bytes.size() == 20);

        // floats go through the IEEE-754 bit pattern (1.0f == 0x3F800000)
        std::ostringstream float_out;
        CHECK(utility::write_binary(float_out, utility::be(1.0f), utility::le(1.0f)).has_value());
        CHECK(std::string("\x3F\x80\x00\x00", 4) + std::string("\x00\x00\x80\x3F", 4) == float_out.str());

        // a bare (untagged) scalar defaults to little-endian
        std::ostringstream plain;
        CHECK(utility::write_binary(plain, uint16_t{0x0102}).has_value());
        CHECK(std::string("\x02\x01", 2) == plain.str());
    }

    void test_write_binary_ranges_and_pod() {
        // a one-byte range is written as-is: exactly size() bytes, and an empty one writes nothing
        std::ostringstream out;
        unsigned char const raw[] = {0xDE, 0xAD, 0xBE, 0xEF};
        std::array<unsigned char, 4> const chunk = {'I', 'E', 'N', 'D'};
        std::vector<uint32_t> const words = {1u, 2u};
        std::span<unsigned char const> const empty = {};

        CHECK(utility::write_binary(out,
                                    std::string_view{"IHDR"}, // a fixed character sequence, no terminator
                                    raw,
                                    chunk,
                                    utility::le(words[0]),
                                    std::span{words}.subspan(1, 1),
                                    empty)
                  .has_value());
        std::string expected = "IHDR";
        expected += std::string("\xDE\xAD\xBE\xEF", 4);
        expected += "IEND";
        expected += std::string("\x01\x00\x00\x00", 4); // the single little-endian uint32
        expected += std::string("\x02\x00\x00\x00", 4);
        CHECK_MSG(out.str() == expected, "byte ranges write exactly their bytes, no length prefix");

        // The POD path writes the object's native representation. Padding bytes are UNSPECIFIED, so
        // this checks the contract rather than the bytes: sizeof(T) goes out and the members sit at
        // their real offsets. That is exactly why a portable file format writes its fields through
        // the writer (be()/le()) instead of memcpy-ing a struct.
        struct padded {
            uint8_t small = 0xAB;
            uint32_t large = 0x01020304u; // forces 3 padding bytes between the members
        };
        padded const pod = {};
        std::ostringstream pod_out;
        CHECK(utility::write_binary(pod_out, pod).has_value());
        CHECK(pod_out.str().size() == sizeof(padded));
        CHECK(static_cast<unsigned char>(pod_out.str()[offsetof(padded, small)]) == 0xABu);
        uint32_t restored = 0;
        std::memcpy(&restored, pod_out.str().data() + offsetof(padded, large), sizeof(restored));
        CHECK(restored == 0x01020304u);

        // a whole contiguous range of trivially copyable values is one block write, so N elements
        // cost exactly N * sizeof(element) bytes
        std::ostringstream block;
        std::array<float, 3> const xyz = {1.0f, 2.0f, 3.0f};
        CHECK(utility::write_binary(block, xyz).has_value());
        CHECK(block.str().size() == sizeof(float) * 3);
    }

    // a sink that can fail on demand (mirrors a full disk / a closed pipe)
    struct failing_sink {
        std::string bytes = {};
        bool ok = true;
        int writes = 0;
        void write(char const* data, std::size_t size) {
            ++writes;
            if (!ok) {
                return; // a real stream would set badbit; the writer must notice via operator bool
            }
            bytes.append(data, size);
        }
        explicit operator bool() const noexcept {
            return ok;
        }
    };

    void test_write_binary_failure_stops_the_fold() {
        failing_sink sink;
        sink.ok = false;
        auto const failed = utility::write_binary(sink, utility::be(uint32_t{1}), utility::be(uint32_t{2}));
        CHECK_MSG(!failed.has_value(), "a failed sink must be reported");
        CHECK(!failed.error().empty());
        CHECK_MSG(sink.writes == 1, "the fold stops at the first failure");

        failing_sink good;
        CHECK(utility::write_binary(good, utility::be(uint32_t{0x01020304})).has_value());
        CHECK(good.bytes.size() == 4);
    }

    void test_write_binary_file_round_trip() {
        std::filesystem::path const path = "test_write_binary.bin";
        std::vector<unsigned char> const payload = {0, 1, 2, 250, 251, 252};
        auto const written = utility::write_binary_file(path, utility::be(uint32_t{0x01020304}), payload, std::string_view{"END"});
        CHECK(written.has_value());
        auto const read_back = utility::read_binary_to_vector(path);
        CHECK(read_back.has_value());
        if (read_back.has_value()) {
            std::string const expected = std::string("\x01\x02\x03\x04", 4) + std::string(payload.begin(), payload.end()) + "END";
            CHECK_MSG(std::string(read_back->begin(), read_back->end()) == expected, "round trip");
        }
        std::filesystem::remove(path);

        // a path that cannot be opened is an error, not a crash
        CHECK(!utility::write_binary_file(std::filesystem::path{"no_such_dir/x.bin"}, uint32_t{1}).has_value());
    }

    // Compile-time contract of the accepted types: the set is closed on purpose (see binary_writable).
    struct not_writable {
        std::string text = {};
    };
    static_assert(utility::binary_writable<uint32_t>);
    static_assert(utility::binary_writable<utility::ordered<uint32_t, utility::endian::big>>);
    static_assert(utility::binary_writable<std::span<unsigned char const>>);
    static_assert(utility::binary_writable<std::array<float, 3>>);
    static_assert(utility::binary_writable<std::string_view>);
    static_assert(!utility::binary_writable<not_writable>);
    static_assert(!utility::binary_writable<char const*>); // no raw pointers: use a span

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
    test_write_png();
    test_write_binary_scalar_byte_order();
    test_write_binary_ranges_and_pod();
    test_write_binary_failure_stops_the_fold();
    test_write_binary_file_round_trip();
    test_data_block_key_semantics();
    test_thread_pool_runs_every_posted_task();
    test_thread_pool_priority_group_wait();
    test_thread_pool_two_concurrent_waiters();
    test_bvh_frustum_cull_keeps_visible_boxes();
    test_bvh_add_rebuild_contract();
    return vk_test::finish("test_utility");
}
