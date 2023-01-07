#pragma once

#include <ATen/core/Tensor.h>
#include <optional>

namespace at {
namespace native {

        Tensor as_strided_impl(const Tensor& self, c10::IntArrayRef size, c10::IntArrayRef stride, std::optional<int64_t> storage_offset_);
        Tensor as_strided(const Tensor& self, c10::IntArrayRef size, c10::IntArrayRef stride, std::optional<int64_t> storage_offset_);

}
}