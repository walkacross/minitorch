#include <c10/core/TensorImpl.h>

namespace c10 {

    TensorImpl::~TensorImpl() {
        //destroy_pyobj_if_needed();
    }

    TensorImpl::TensorImpl(
            Storage&& storage,
            DispatchKeySet key_set,
            const caffe2::TypeMeta data_type)
    // Use std::forward to suppress static analyzer false positive.
            : TensorImpl(
            std::forward<Storage>(storage),
            key_set,
            data_type,
            storage.device()) {}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    TensorImpl::TensorImpl(
            Storage&& storage,
            DispatchKeySet key_set,
            const caffe2::TypeMeta data_type,
            std::optional<c10::Device> device_opt)
            : storage_(std::move(storage)),
              storage_offset_(0),
              numel_(0),
              data_type_(data_type),
              device_opt_(device_opt) {

        if (!key_set.empty()) {
        }

        // XXX: if updating keyset logic here also update
        // _change_backend_component_keys
        bool inference_mode = false;//true;

        // TODO: be more explicit about the full key set at call sites so we
        // don't have to keep recomputing it here
        auto k = key_set.highestBackendKey();

        key_set = key_set | getAutocastRelatedKeySetFromBackend(k);

        // See [Note: Python key removal]
        key_set = key_set - c10::python_ks;

        // Inference tensor doesn't have autograd related keys.
        if (inference_mode) {
            // See Note [Expected TLS state in InferenceMode] for why we exclude
            // Autograd & ADInplaceOrView keys. Normally key_set only contains backend
            // keys but we do the substraction here to make sure.
            key_set_ = key_set - c10::autograd_dispatch_keyset_with_ADInplaceOrView;
        } else {
            // TODO: Ideally we only add AutogradBackend key when the tensor requires
            // grad.
            //       See Note [Dream: skip VariableType kernel when requires_grad=false]
            key_set_ = key_set | getAutogradRelatedKeySetFromBackend(k);
        }

        // Inference tensor doesn't have version counter.
        if (!is_inference()) {
            //version_counter_ = VariableVersion(/*version=*/0);
        }
        // we would also like to check that non-cpu devices have an index, but some
        // Caffe2 operators create Storages with default devices.
    }

} //namespace c10