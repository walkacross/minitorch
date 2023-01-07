#pragma once

#include <c10/core/TensorImpl.h>

namespace at {

class TensorBase {
public:
    struct unsafe_borrow_t {
        explicit unsafe_borrow_t() = default;
    };

protected:
    c10::intrusive_ptr <c10::TensorImpl> impl_;

public:
    TensorBase() = default;

    // This constructor should not be used by end users and is an implementation
    // detail invoked by autogenerated code.
    explicit TensorBase(c10::intrusive_ptr <c10::TensorImpl> tensor_impl)
            : impl_(std::move(tensor_impl)) {
        if (impl_.get() == nullptr) {
            throw std::runtime_error("TensorImpl with nullptr is not supported");
        }
    }

    TensorBase(const TensorBase &) = default;

    TensorBase(TensorBase &&) = default;

    TensorBase &operator=(const TensorBase &x) &{
        impl_ = x.impl_;
        return *this;
    };

    TensorBase &operator=(TensorBase &&x) &{
        impl_ = std::move(x.impl_);
        return *this;
    }

public:
    int64_t dim() const {
        return impl_->dim();
    }

    int64_t storage_offset() const {
        return impl_->storage_offset();
    }

    c10::TensorImpl *unsafeGetTensorImpl() const {
        return impl_.get();
    }

    const c10::intrusive_ptr <c10::TensorImpl> &getIntrusivePtr() const {
        return impl_;
    }

    c10::IntArrayRef sizes() const {
        return impl_->sizes();
    }

    c10::IntArrayRef strides() const {
        return impl_->strides();
    }

    int64_t numel() const {
        return impl_->numel();
    }

    size_t nbytes() const {
        return impl_->numel() * impl_->itemsize();
    }

    c10::DispatchKeySet key_set() const {
        return impl_->key_set();
    }

    /// Returns a `Tensor`'s dtype (`TypeMeta`).
    caffe2::TypeMeta dtype() const {
        return impl_->dtype();
    }

    /// Returns a `Tensor`'s device.
    inline c10::Device device() const {
        return impl_->device();
    }
    const c10::Storage& storage() const {
        return impl_->storage();
    }
    /// Returns if a `Tensor` has CPU backend.
    bool is_cpu() const {
        // NB: this is not a native function to avoid dispatching overhead.
        return impl_->is_cpu();
    }

    /// Returns if a `Tensor` has CUDA backend.
    bool is_cuda() const {
        // NB: this is not a native function to avoid dispatching overhead.
        return impl_->is_cuda();
    }

    /// Returns if a `Tensor` is an inference tensor.
    bool is_inference() const {
        return impl_->is_inference();
    }
};

namespace detail {
// Helper creator for Tensor class which doesn't requires the users to pass
// in an intrusive_ptr instead it just converts the argument passed to
// requested intrusive_ptr type.
    template <typename T, typename... Args>
    TensorBase make_tensor_base(Args&&... args) {
        return TensorBase(c10::make_intrusive<T>(std::forward<Args>(args)...));
    }

} // namespace detail
} // namespace at