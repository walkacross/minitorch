#pragma once

// @generated by tools/codegen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>
#include <ATen/ops/avg_pool2d_meta.h>

namespace at {
namespace native {

struct TORCH_API structured_avg_pool2d_out_cpu : public at::meta::structured_avg_pool2d {
void impl(const at::Tensor & self, int64_t kH, int64_t kW, int64_t dH, int64_t dW, int64_t padH, int64_t padW, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, const at::Tensor & out);
};
struct TORCH_API structured_avg_pool2d_out_cuda : public at::meta::structured_avg_pool2d {
void impl(const at::Tensor & self, int64_t kH, int64_t kW, int64_t dH, int64_t dW, int64_t padH, int64_t padW, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, const at::Tensor & out);
};
TORCH_API at::Tensor mkldnn_avg_pool2d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride={}, at::IntArrayRef padding=0, bool ceil_mode=false, bool count_include_pad=true, c10::optional<int64_t> divisor_override=c10::nullopt);
TORCH_API at::Tensor & mkldnn_avg_pool2d_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out);
TORCH_API at::Tensor avg_pool2d_quantized_cpu(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride={}, at::IntArrayRef padding=0, bool ceil_mode=false, bool count_include_pad=true, c10::optional<int64_t> divisor_override=c10::nullopt);

} // namespace native
} // namespace at
