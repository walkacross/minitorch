#include <ATen/core/Tensor.h>
#include <ATen/native/TensorShape.h>
#include <ATen/core/dispatch/dispatcher.h>

namespace at {
namespace native {

    Tensor as_strided_impl(const Tensor& self, c10::IntArrayRef size, c10::IntArrayRef stride, std::optional<int64_t> storage_offset_){
        return DISPATCH_DEVICE_IMPL(as_strided_impl, self, size, stride, storage_offset_);
    }

    Tensor as_strided(const Tensor& self, c10::IntArrayRef size, c10::IntArrayRef stride, std::optional<int64_t> storage_offset_){
        return as_strided_impl(self, size, stride, storage_offset_);
    }

}//namespace native
}//namespace at