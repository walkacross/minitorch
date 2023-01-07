#pragma once

// Forward declarations of core ATen types used in dispatch functions
namespace c10 {

    class SymInt;
    struct Storage;
    template <typename T>
    class ArrayRef;
    struct Device;
}  // namespace c10

namespace at {

    class Tensor;
    using TensorList = c10::ArrayRef<Tensor>;
    using IntArrayRef = c10::ArrayRef<int64_t>;

    using SymInt = c10::SymInt;
    using Device = c10::Device;

}  // namespace at