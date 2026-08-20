//
// Created by 23530 on 2026/8/9.
//
module;

#include <array>
#include <algorithm>
#include <string>
#include <format>

export module utility.data_block;
namespace utility
 {
    /**
     * @defgroup data_block
     * @ingroup utility
     * @brief struct template creates a sized data type provides auto generated operator==/!= (use std::ranges::equal),
     *     operator<=>(use std::lexicographical_compare_three_way) and hex formatter (.to_hex_string())
     * @tparam S byte size of the struct
     */
     export template<size_t S>
     struct data_block {
         constexpr static uint32_t size_byte = S;
         std::array<uint8_t, S> data;

         constexpr static size_t size = S;

         bool operator==(data_block<S> const& other) const {
             return std::ranges::equal(data, other.data);
         }

         bool operator!=(data_block<S> const& other) const {
             return !(*this == other);
         }

         auto operator<=>(const data_block<S>& other) const {
             return std::lexicographical_compare_three_way(
                 data.begin(), data.end(),
                 other.data.begin(), other.data.end()
             );
         }

         [[nodiscard]] std::string to_hex_string() const {
             std::string result;
             result.reserve(S * 2);
             std::ranges::for_each(this->data, [&result](auto const& byte){result += std::format("{:02x}", byte);});
             return result;
         }
     };
 }