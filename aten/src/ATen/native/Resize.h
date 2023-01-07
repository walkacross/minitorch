#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

namespace at { namespace native {

template <typename T>
inline void setStrided( const Tensor& self,
                        c10::ArrayRef<T> size,
                        c10::ArrayRef<T> stride,
                        T storage_offset) {

            for (auto val : stride) {
            }

            auto* self_ = self.unsafeGetTensorImpl();
            self_->set_sizes_and_strides(size, stride, std::make_optional(storage_offset));
        }

}}
