#pragma once

// @generated by tools/codegen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/arccos_ops.h>

namespace at {


// aten::arccos(Tensor self) -> Tensor
TORCH_API inline at::Tensor arccos(const at::Tensor & self) {
    return at::_ops::arccos::call(self);
}

// aten::arccos_(Tensor(a!) self) -> Tensor(a!)
TORCH_API inline at::Tensor & arccos_(at::Tensor & self) {
    return at::_ops::arccos_::call(self);
}

// aten::arccos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & arccos_out(at::Tensor & out, const at::Tensor & self) {
    return at::_ops::arccos_out::call(self, out);
}

// aten::arccos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & arccos_outf(const at::Tensor & self, at::Tensor & out) {
    return at::_ops::arccos_out::call(self, out);
}

}
