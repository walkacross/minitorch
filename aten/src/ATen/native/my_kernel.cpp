#include <ATen/core/dispatch/dispatcher.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/my_kernel.h>

namespace at{

    void my_kernel_impl(at::Tensor self) {
        return DISPATCH_DEVICE_IMPL(my_kernel_impl, self);
    }

    void my_kernel(at::Tensor self) {
        return my_kernel_impl(self);
    }
}