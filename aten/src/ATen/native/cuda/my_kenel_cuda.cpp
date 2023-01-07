#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/dispatcher.h>
#include <iostream>
#include <ATen/native/my_kernel.h>

namespace at {

    void my_kernel_cuda(at::Tensor self){
        std::cout << "my_kernel_cpu impl in CUDA" << std::endl;
        //auto result = at::detail::make_tensor<c10::TensorImpl>(c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(), self.dtype());
        //return result;
    }

    REGISTER_DEVICE_IMPL(my_kernel_impl, CUDA, &my_kernel_cuda);

}