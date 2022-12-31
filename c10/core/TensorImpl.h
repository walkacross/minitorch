#pragma once

#include <c10/core/Storage.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/SizesAndStrides.h>
#include <c10/util/typeid.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/accumulate.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>

namespace c10 {

    struct TensorImpl : public c10::intrusive_ptr_target {

    protected:
    Storage storage_;
    c10::impl::SizesAndStrides sizes_and_strides_;
    int64_t storage_offset_ = 0;
    // If sizes and strides are empty, the numel is 1!!  However, most of the
    // time, we will immediately set sizes to {0} and reset numel to 0.
    // (Can't do that in the default initializers, because there's no way to
    // spell "allocate a one-element array" for strides_).
    int64_t numel_ = 1;
    caffe2::TypeMeta data_type_;
    std::optional<c10::Device> device_opt_;
    DispatchKeySet key_set_;

    public:
    TensorImpl() = delete;
    virtual ~TensorImpl() override;
    // Note [Enum ImplType]
    // This enum is temporary. In the followup refactor we should
    // think about how to specialize TensorImpl creation for view
    // tensors. Currently we only special case its key_set_ but
    // there's also potential to share version_counter_ directly
    // without creating first and then override in as_view.
    enum ImplType { VIEW };

    /**
     * Construct a 1-dim 0-size tensor backed by the given storage.
     */
    TensorImpl(
            Storage&& storage,
            DispatchKeySet,
            const caffe2::TypeMeta data_type);

    private:
    // This constructor is private, because the data_type is redundant with
    // storage.  Still, we pass it in separately because it's easier to write
    // the initializer list if we're not worried about storage being moved out
    // from under us.
    TensorImpl(
            Storage&& storage,
            DispatchKeySet,
            const caffe2::TypeMeta data_type,
            std::optional<c10::Device>);

    public:
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;
    TensorImpl(TensorImpl&&) = delete;
    TensorImpl& operator=(TensorImpl&&) = delete;

    public:
    /**
     * Return the DispatchKeySet corresponding to this Tensor, specifying
     * all of the DispatchKeys that this Tensor identifies as.  This is the
     * information used to dispatch operations on this tensor.
     */
    DispatchKeySet key_set() const {
        return key_set_;
    }

    public:
    /**
     * Return a reference to the sizes of this tensor.  This reference remains
     * valid as long as the tensor is live and not resized.
     */
    IntArrayRef sizes() const {
        return sizes_and_strides_.sizes_arrayref();
    }

    /**
     * Return a reference to the strides of this tensor.  This reference remains
    * valid as long as the tensor is live and not restrided.
    */
    IntArrayRef strides() const {
        return sizes_and_strides_.strides_arrayref();
    }

    int64_t numel() const {
        return numel_;
    }

    int64_t dim() const {
        return sizes_and_strides_.size();
    }

    int64_t storage_offset() const {
        return storage_offset_;
    }

    public:
    bool is_cpu() const {
        // Note: we cannot rely on dispatch keys to determine the device type
        // of a tensor, because "wrapper" tensors (like FunctionalTensorWrapper)
        // don't include backend dispatch keys.
        return device_opt_.has_value() && device_opt_->type() == kCPU;
    }

    bool is_cuda() const {
        return device_opt_.has_value() && device_opt_->type() == kCUDA;
    }


    bool is_inference() {
            bool no_ADInplaceOrView = !key_set_.has_any(c10::inplace_or_view_ks);
            bool no_Autograd = !key_set_.has_any(c10::autograd_dispatch_keyset);
            return no_ADInplaceOrView && no_Autograd;
        }

    Device device() const {
        return device_default();
    }

    protected:
    c10::Device device_default() const {
        //TORCH_CHECK(device_opt_.has_value(), "tensor does not have a device");
        // See NOTE [c10::optional operator usage in CUDA]
        return *device_opt_;
    }

    public:
    /**
    * Returns the TypeMeta of a tensor, which describes what data type
     * it is (e.g., int, float, ...)
    */
    const caffe2::TypeMeta dtype() const {
        return data_type_;
    }

    /**
     * Return the size of a single element of this tensor in bytes.
     */
    size_t itemsize() const {
        return data_type_.itemsize();
    }

    void set_sizes_and_strides(
            IntArrayRef new_size,
            IntArrayRef new_stride,
            std::optional<int64_t> storage_offset = std::nullopt) {

        const auto new_dim = new_size.size();

        sizes_and_strides_.set_sizes(new_size);

        if (new_dim > 0) {
            for (size_t dim = new_dim - 1;; dim--) {
                if (new_stride[dim] >= 0) {
                    sizes_and_strides_.stride_at_unchecked(dim) = new_stride[dim];
                } else {
                    // XXX: This behavior is surprising and may need to be removed to
                    // support negative strides. Some pytorch functions rely on it:
                    // for example, torch.cat (run TestTorch.test_cat_empty).
                    if (dim == new_dim - 1) {
                        sizes_and_strides_.stride_at_unchecked(dim) = 1;
                    } else {
                        // Keep stride monotonically increasing to match NumPy.
                        sizes_and_strides_.stride_at_unchecked(dim) =
                                std::max<int64_t>(
                                        sizes_and_strides_.size_at_unchecked(dim + 1), 1) *
                                sizes_and_strides_.stride_at_unchecked(dim + 1);
                    }
                }
                if (dim == 0)
                    break;
            }
        }

        refresh_numel();
        //refresh_contiguous();

        if (storage_offset.has_value()) {
            storage_offset_ = *storage_offset;
        }
    }

    /**
    * Compute the number of elements based on the sizes of a tensor.
    */
    // NB: This is ONLY called when sizes_and_strides_ is used directly; if
    // we are virtualizing, then numel calls are virtualized as well, and this
    // should never get called
    int64_t compute_numel() const {
        return c10::multiply_integers(sizes_and_strides_.sizes_arrayref());
    }

    protected:
    /**
     * Recompute the cached numel of a tensor.  Call this if you modify
     * sizes.
     *
     * For tensors with sparse layouts, use safe_refresh_numel() instead
     * because it will catch integer overflow that may occur for tensors
     * with sparse layouts and large dimensions.
     *
     * NB: We may uselessly recompute cached numel even in situations where
     * it is completely never used (e.g., if CustomSizes for Python).  However,
     * we still must keep it up to date in case the Python overload
     * returns None (in which case we will consult the field here).  This also
     * implies that sizes/strides will never be complete garbage; in the
     * very worst case scenario, it will reflect a 1-dim zero size tensor.
     */
    void refresh_numel() {
            numel_ = compute_numel();
        }

    };
} //namespace c10