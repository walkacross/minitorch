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


namespace at {
namespace native {

TORCH_API ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv2d_backward_out_cpu(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias);
TORCH_API ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv2d_backward_out_cuda(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias);
TORCH_API ::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv2d_backward_cpu(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, ::std::array<bool,3> output_mask);
TORCH_API ::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv2d_backward_cuda(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, ::std::array<bool,3> output_mask);

} // namespace native
} // namespace at
