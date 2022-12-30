#pragma once

#include <cstdlib>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>


namespace c10 {
namespace guts {
/**
 * is_fundamental<T> is true_type iff the lambda type T is a fundamental type
 * (that is, arithmetic type, void, or nullptr_t). Example: is_fundamental<int>
 * // true We define it here to resolve a MSVC bug. See
 * https://github.com/pytorch/pytorch/issues/30932 for details.
 */
 template <class T>
 struct is_fundamental : std::is_fundamental<T> {};

template <class T>
using void_t = std::void_t<T>;

namespace detail {

template <class T, class Enable = void>
struct to_string_ final {
    static std::string call(T value) {
            std::ostringstream str;
            str << value;
            return str.str();
    }
};
// If a std::to_string exists, use that instead
template <class T>
struct to_string_<T, void_t<decltype(std::to_string(std::declval<T>()))>> final {
    static std::string call(T value) {
        return std::to_string(value);
    }
};

} // namespace detail

template <class T>
inline std::string to_string(T value) {
    return detail::to_string_<T>::call(value);
}

template <class T>
constexpr const T& min(const T& a, const T& b) {
        return (b < a) ? b : a;
}

template <class T>
constexpr const T& max(const T& a, const T& b) {
        return (a < b) ? b : a;
}

} //namespace guts
} //namespace c10
