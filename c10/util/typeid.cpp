#include <c10/util/typeid.h>

#include <algorithm>
#include <atomic>

using std::string;

namespace caffe2 {

uint16_t TypeMeta::nextTypeIndex(c10::NumScalarTypes);
// fixed length array of TypeMetaData instances
    detail::TypeMetaData* TypeMeta::typeMetaDatas() {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        static detail::TypeMetaData instances[MaxTypeIndex + 1] = {
#define SCALAR_TYPE_META(T, name)        \
  /* ScalarType::name */                 \
  detail::TypeMetaData(                  \
      sizeof(T),                         \
      detail::_PickNew<T>(),             \
      detail::_PickPlacementNew<T>(),    \
      detail::_PickCopy<T>(),            \
      detail::_PickPlacementDelete<T>(), \
      detail::_PickDelete<T>(),          \
      TypeIdentifier::Get<T>(),          \
      c10::util::get_fully_qualified_type_name<T>()),
                AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SCALAR_TYPE_META)
#undef SCALAR_TYPE_META
                // The remainder of the array is padded with TypeMetaData blanks.
                // The first of these is the entry for ScalarType::Undefined.
                // The rest are consumed by CAFFE_KNOWN_TYPE entries.
        };
        return instances;
    }
} // nsmespace for caffe2