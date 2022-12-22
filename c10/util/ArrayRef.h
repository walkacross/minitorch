//===--- ArrayRef.h - Array Reference Wrapper -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// ATen: modified from llvm::ArrayRef.
// removed llvm-specific functionality
// removed some implicit const -> non-const conversions that rely on
// complicated std::enable_if meta-programming
// removed a bunch of slice variants for simplicity...

#pragma once
#include <vector>
#include <iostream>

namespace c10 {
/// ArrayRef - Represent a constant reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length.  It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the ArrayRef. For this reason, it is not in general
/// safe to store an ArrayRef.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
    template<typename T>
    class ArrayRef final {
    public:
        using iterator = const T *;
        using const_iterator = const T *;
        using size_type = size_t;
        using value_type = T;

    private:
        /// The start of the array, in an external buffer.
        const T* Data;

        /// The number of elements.
        size_type Length;

    public:
        /// @name Constructors
        /// @{

        /// Construct an empty ArrayRef.
        /* implicit */ constexpr ArrayRef() : Data(nullptr), Length(0) {}

        /// Construct an ArrayRef from a single element.
        // TODO Make this explicit
        constexpr ArrayRef(const T &OneElt) : Data(&OneElt), Length(1) {}

        /// Construct an ArrayRef from a pointer and length.
        constexpr ArrayRef(const T *data, size_t length)
                : Data(data), Length(length) {
        }

        /// Construct an ArrayRef from a std::vector.
        // The enable_if stuff here makes sure that this isn't used for
        // std::vector<bool>, because ArrayRef can't work on a std::vector<bool>
        // bitfield.
        template<typename A>
        /* implicit */ ArrayRef(const std::vector <T, A> &Vec)
                : Data(Vec.data()), Length(Vec.size()) {
            static_assert(
                    !std::is_same<T, bool>::value,
                    "ArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.");
        }

        /// Construct an ArrayRef from a std::initializer_list.
        /* implicit */ constexpr ArrayRef(const std::initializer_list <T> &Vec)
                : Data(
                std::begin(Vec) == std::end(Vec) ? static_cast<T *>(nullptr)
                                                 : std::begin(Vec)),
                  Length(Vec.size()) {}

        /// @}
        /// @name Simple Operations
        /// @{

        constexpr iterator begin() const {
            return Data;
        }

        constexpr iterator end() const {
            return Data + Length;
        }

        /// empty - Check if the array is empty.
        constexpr bool empty() const {
            return Length == 0;
        }

        constexpr const T* data() const {
            return Data;
        }

        /// size - Get the array size.
        constexpr size_t size() const {
            return Length;
        }

        /// front - Get the first element.
        constexpr const T& front() const {
            return Data[0];
        }

        /// back - Get the last element.
        constexpr const T& back() const {
            return Data[Length - 1];
        }

        /// equals - Check for element-wise equality.
        constexpr bool equals(ArrayRef RHS) const {
            return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
        }

        /// @}
        /// @name Operator Overloads
        /// @{
        constexpr const T& operator[](size_t Index) const {
            return Data[Index];
        }

        /// Vector compatibility
        constexpr const T& at(size_t Index) const {
            return Data[Index];
        }

        /// Disallow accidental assignment from a temporary.
        ///
        /// The declaration here is extra complicated so that "arrayRef = {}"
        /// continues to select the move assignment operator.
        template<typename U>
        typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type &
        operator=(U &&Temporary) = delete;

        /// Disallow accidental assignment from a temporary.
        ///
        /// The declaration here is extra complicated so that "arrayRef = {}"
        /// continues to select the move assignment operator.
        template<typename U>
        typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type &
        operator=(std::initializer_list <U>) = delete;

        /// @}
        /// @name Expensive Operations
        /// @{
        std::vector <T> vec() const {
            return std::vector<T>(Data, Data + Length);
        }

        /// @}
    };

    template<typename T>
    std::ostream &operator<<(std::ostream &out, ArrayRef<T> list) {
        int i = 0;
        out << "[";
        for (auto e: list) {
            if (i++ > 0)
                out << ", ";
            out << e;
        }
        out << "]";
        return out;
    }

    using IntArrayRef = ArrayRef<int64_t>;
}
