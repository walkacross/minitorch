#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <ATen/core/dispatch/dispatcher.h>

namespace at {
    namespace native {


        Tensor as_strided_catch_all(const Tensor& self, c10::IntArrayRef size, c10::IntArrayRef stride, std::optional<int64_t> storage_offset_) {
            //TORCH_INTERNAL_ASSERT(!self.is_mps(), "as_strided_tensorimpl does not work with MPS; call self.as_strided(...) instead");
            auto storage_offset = storage_offset_.value_or(self.storage_offset());
            auto result = at::detail::make_tensor<c10::TensorImpl>(c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(), self.dtype());
            setStrided(result, size, stride, storage_offset);
            return result;
        }

        REGISTER_DEVICE_IMPL(as_strided_impl, CPU, &as_strided_catch_all);
        REGISTER_DEVICE_IMPL(as_strided_impl, CUDA, &as_strided_catch_all);


    }//namespace native
}//namespace at