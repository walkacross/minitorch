#pragma once

// @generated by tools/codegen/gen.py from Operator.h

#include <c10/core/QScheme.h>
#include <tuple>
#include <vector>


// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
namespace c10 {

template<typename T>
class optional;
template<typename T>
class List;
class Stream;
class Scalar;
struct Storage;
struct TensorOptions;

}

namespace at {

class Tensor;
struct Dimname;
struct Generator;
using TensorList = c10::ArrayRef<Tensor>;
using DimnameList = c10::ArrayRef<Dimname>;
using c10::Stream;
using c10::Storage;
using c10::QScheme;
using c10::Scalar;
using c10::TensorOptions;

namespace _ops {


struct TORCH_API adaptive_max_pool3d_out {
  using schema = ::std::tuple<at::Tensor &,at::Tensor &> (const at::Tensor &, at::IntArrayRef, at::Tensor &, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::adaptive_max_pool3d")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "adaptive_max_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))")
  static ::std::tuple<at::Tensor &,at::Tensor &> call(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices);
  static ::std::tuple<at::Tensor &,at::Tensor &> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices);
};

struct TORCH_API adaptive_max_pool3d {
  using schema = ::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, at::IntArrayRef);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::adaptive_max_pool3d")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)")
  static ::std::tuple<at::Tensor,at::Tensor> call(const at::Tensor & self, at::IntArrayRef output_size);
  static ::std::tuple<at::Tensor,at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size);
};

}} // namespace at::_ops
