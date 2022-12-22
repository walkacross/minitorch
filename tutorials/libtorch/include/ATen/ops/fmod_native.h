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
#include <ATen/ops/fmod_meta.h>

namespace at {
namespace native {

TORCH_API at::Tensor fmod(const at::Tensor & self, const at::Scalar & other);
TORCH_API at::Tensor & fmod_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out);
TORCH_API at::Tensor & fmod_(at::Tensor & self, const at::Scalar & other);
struct TORCH_API structured_fmod_out : public at::meta::structured_fmod_Tensor {
void impl(const at::Tensor & self, const at::Tensor & other, const at::Tensor & out);
};

} // namespace native
} // namespace at
