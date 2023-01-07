#pragma once

#include <ATen/core/Tensor.h>

namespace at{
    void my_kernel_impl(Tensor self);
    void my_kernel(Tensor self);
}
