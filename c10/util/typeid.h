#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/IdWrapper.h>
#include <c10/util/TypeIndex.h>
#include <c10/util/string_view.h>
#include <c10/util/irange.h>

#include <atomic>
#include <cassert>
#include <complex>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <typeinfo>
#include <iostream>


namespace caffe2 {
/**
 * A type id is a unique id for a given C++ type.
 * You need to register your types using CAFFE_KNOWN_TYPE(MyType) to be able to
 * use TypeIdentifier with custom types. This is for example used to store the
 * dtype of tensors.
 */
class TypeIdentifier final : public c10::IdWrapper<TypeIdentifier, c10::util::type_index> {
public:
    friend std::ostream &operator<<(std::ostream &stream, TypeIdentifier typeId);

    friend constexpr bool operator<(TypeIdentifier lhs, TypeIdentifier rhs);
/**
 * Returns the unique id for the given type T. The id is unique for the type T
 * in the sense that for any two different types, their ids are different; for
 * the same type T, the id remains the same over different calls of the
 * function. However, this is not guaranteed over different runs, as the id
 * is generated during run-time. Do NOT serialize the id for storage.
 */
    template <typename T>
    static constexpr TypeIdentifier Get() noexcept {
        return TypeIdentifier(c10::util::get_type_index<T>());
    }

    static constexpr TypeIdentifier uninitialized() {
        return TypeIdentifier(c10::util::type_index{0});
    }

    private:
        constexpr explicit TypeIdentifier(c10::util::type_index id) : IdWrapper(id) {}
};

// Allow usage in std::map / std::set
// TODO Disallow this and rather use std::unordered_map/set everywhere
inline constexpr bool operator<(TypeIdentifier lhs, TypeIdentifier rhs) {
return lhs.underlyingId() < rhs.underlyingId();
}

inline std::ostream& operator<<(
        std::ostream& stream,
        caffe2::TypeIdentifier typeId) {
    return stream << typeId.underlyingId();
}

} // namespace caffe2

C10_DEFINE_HASH_FOR_IDWRAPPER(caffe2::TypeIdentifier)

namespace caffe2 {
namespace detail {

// This struct holds the actual type information. There will be
// one allocated per type. TypeMeta objects will then point to the struct
// instance for the type they're configured for.
        struct TypeMetaData final {
            using New = void *();
            using PlacementNew = void(void *, size_t);
            using Copy = void(const void *, void *, size_t);
            using PlacementDelete = void(void *, size_t);
            using Delete = void(void *);

            constexpr TypeMetaData() noexcept
                    : itemsize_(0),
                    new_(nullptr),
                    placementNew_(nullptr),
                    copy_(nullptr),
                    placementDelete_(nullptr),
                    delete_(nullptr),
                    id_(TypeIdentifier::uninitialized()),
                    name_("nullptr (uninitialized)") {}

            constexpr TypeMetaData(
                    size_t itemsize,
                    New *newFn,
                    PlacementNew *placementNew,
                    Copy *copy,
                    PlacementDelete *placementDelete,
                    Delete *deleteFn,
                    TypeIdentifier id,
                    c10::string_view name) noexcept :
            itemsize_(itemsize),
            new_(newFn),
            placementNew_(placementNew),
            copy_(copy),
            placementDelete_(placementDelete),
            delete_(deleteFn),
            id_(id),
            name_(name) {}

            size_t itemsize_;
            New *new_;
            PlacementNew *placementNew_;
            Copy *copy_;
            PlacementDelete *placementDelete_;
            Delete *delete_;
            TypeIdentifier id_;
            c10::string_view name_;
        };

/**
 * Placement new function for the type.
 */
    template <typename T>
    inline void _PlacementNew(void* ptr, size_t n) {
        T* typed_ptr = static_cast<T*>(ptr);
        for (const auto i : c10::irange(n)) {
            new (typed_ptr + i) T;
        }
    }

    template <typename T>
    inline void _PlacementNewNotDefault(void* /*ptr*/, size_t /*n*/) {
    }

    template <
            typename T,
            std::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
    inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
        return (c10::guts::is_fundamental<T>::value || std::is_pointer<T>::value)
               ? nullptr
               : &_PlacementNew<T>;
    }

    template <
            typename T,
            std::enable_if_t<!std::is_default_constructible<T>::value>* = nullptr>
    inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
        static_assert(
                !c10::guts::is_fundamental<T>::value && !std::is_pointer<T>::value,
                "this should have picked the other SFINAE case");
        return &_PlacementNewNotDefault<T>;
    }

    template <typename T>
    inline void* _New() {
        return new T;
    }

    template <typename T>
    inline void* _NewNotDefault() {
    }

    template <
            typename T,
            std::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
    inline constexpr TypeMetaData::New* _PickNew() {
        return &_New<T>;
    }

    template <
            typename T,
            std::enable_if_t<!std::is_default_constructible<T>::value>* = nullptr>
    inline constexpr TypeMetaData::New* _PickNew() {
        return &_NewNotDefault<T>;
    }

/**
 * Typed copy function for classes.
 */
    template <typename T>
    inline void _Copy(const void* src, void* dst, size_t n) {
        const T* typed_src = static_cast<const T*>(src);
        T* typed_dst = static_cast<T*>(dst);
        for (const auto i : c10::irange(n)) {
            typed_dst[i] = typed_src[i];
        }
    }

/**
 * A placeholder function for types that do not allow assignment.
 */
    template <typename T>
    inline void _CopyNotAllowed(const void* /*src*/, void* /*dst*/, size_t /*n*/) {
    }

    template <
            typename T,
            std::enable_if_t<std::is_copy_assignable<T>::value>* = nullptr>
    inline constexpr TypeMetaData::Copy* _PickCopy() {
        return (c10::guts::is_fundamental<T>::value || std::is_pointer<T>::value)
               ? nullptr
               : &_Copy<T>;
    }

    template <
            typename T,
            std::enable_if_t<!std::is_copy_assignable<T>::value>* = nullptr>
    inline constexpr TypeMetaData::Copy* _PickCopy() {
        static_assert(
                !c10::guts::is_fundamental<T>::value && !std::is_pointer<T>::value,
                "this should have picked the other SFINAE case");
        return &_CopyNotAllowed<T>;
    }

/**
 * Destructor for non-fundamental types.
 */
    template <typename T>
    inline void _PlacementDelete(void* ptr, size_t n) {
        T* typed_ptr = static_cast<T*>(ptr);
        for (const auto i : c10::irange(n)) {
            typed_ptr[i].~T();
        }
    }

    template <typename T>
    inline constexpr TypeMetaData::PlacementDelete* _PickPlacementDelete() {
        return (c10::guts::is_fundamental<T>::value || std::is_pointer<T>::value)
               ? nullptr
               : &_PlacementDelete<T>;
    }

    template <typename T>
    inline void _Delete(void* ptr) {
        T* typed_ptr = static_cast<T*>(ptr);
        delete typed_ptr;
    }

    template <class T>
    inline constexpr TypeMetaData::Delete* _PickDelete() noexcept {
    return &_Delete<T>;
}

class _Uninitialized final {};

} //namespace for detail


    class TypeMeta final {

    private:
        uint16_t index_;

        // TypeMeta can only be created by Make, making sure that we do not
        // create incorrectly mixed up TypeMeta objects.
        explicit TypeMeta(const uint16_t index) noexcept : index_(index) {}

        // specializations return indexes into typeMetaDataInstances()
        template <class T>
        static uint16_t _typeMetaData() noexcept;

        #define MaxTypeIndex UINT8_MAX
        static uint16_t nextTypeIndex;
        static detail::TypeMetaData* typeMetaDatas();

        inline const detail::TypeMetaData& data() const {
            return typeMetaDatas()[index_];
        }
    public:
        template <typename T>
        static TypeMeta Make() {
            // The instance pointed to is declared here, but defined in a .cpp file.
            // We need to silence the compiler warning about using an undefined
            // variable template. '-Wpragmas' and '-Wunknown-warning-option' has to be
            // disabled for compilers that don't know '-Wundefined-var-template' and
            // would error at our attempt to disable it.
            return TypeMeta(_typeMetaData<T>());
        }

        static inline TypeMeta fromScalarType(c10::ScalarType scalar_type) {
            const auto index = static_cast<uint16_t>(scalar_type);
            return TypeMeta(index);
        }

    public:
        using New = detail::TypeMetaData::New;
        using PlacementNew = detail::TypeMetaData::PlacementNew;
        using Copy = detail::TypeMetaData::Copy;
        using PlacementDelete = detail::TypeMetaData::PlacementDelete;
        using Delete = detail::TypeMetaData::Delete;

        /** Create a dummy TypeMeta object. To create a TypeMeta object for a specific
         * type, use TypeMeta::Make<T>().
         */
        TypeMeta() noexcept;

        /**
         * Copy constructor.
         */
        TypeMeta(const TypeMeta &src) noexcept = default;

        /**
         * Assignment operators.
         */
        TypeMeta &operator=(const TypeMeta &src) noexcept = default;

        TypeMeta(TypeMeta &&rhs) noexcept = default;

        inline TypeMeta &operator=(c10::ScalarType scalar_type) noexcept {
            index_ = static_cast<uint16_t>(scalar_type);
            return *this;
        }
    public:
        /**
         * Returns the type id.
         */
        TypeIdentifier id() const noexcept {
            return data().id_;
        }

        inline bool isScalarType() const noexcept {
            return index_ < c10::NumScalarTypes;
        }
        /**
         * true if we represent ScalarType scalar_type
         */
        inline bool isScalarType(c10::ScalarType scalar_type) const noexcept {
            return index_ == static_cast<uint16_t>(scalar_type);
        }
        /**
         * Returns the size of the item.
         */
        inline size_t itemsize() const noexcept {
            return data().itemsize_;
        }
        /**
        * Returns a printable name for the type.
        */
        c10::string_view name() const noexcept {
            return data().name_;
        }
        friend bool operator==(const TypeMeta lhs, const TypeMeta rhs) noexcept;

        // Below are static functions that can be called by passing a specific type.

        template <class T>
        static constexpr TypeIdentifier Id() noexcept {
            return TypeIdentifier::Get<T>();
        }

        template <class T>
        static c10::string_view TypeName() noexcept {
            return c10::util::get_fully_qualified_type_name<T>();
        }

        template <class T>
        static constexpr size_t ItemSize() noexcept {
            return sizeof(T);
        }

    };

// specializations of TypeMeta::_typeMetaData for ScalarType types
#define DEFINE_SCALAR_METADATA_INSTANCE(T, name)             \
  template <>                                                \
  constexpr uint16_t TypeMeta::_typeMetaData<T>() noexcept { \
    return static_cast<uint16_t>(c10::ScalarType::name);          \
  }
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_METADATA_INSTANCE)
#undef DEFINE_SCALAR_METADATA_INSTANCE

template <>
constexpr uint16_t TypeMeta::_typeMetaData<
            detail::_Uninitialized>() noexcept {
    return static_cast<uint16_t>(c10::ScalarType::Undefined);
}

inline TypeMeta::TypeMeta() noexcept
: index_(_typeMetaData<detail::_Uninitialized>()) {}

inline bool operator==(const TypeMeta lhs, const TypeMeta rhs) noexcept {
    return (lhs.index_ == rhs.index_);
}
inline bool operator!=(const TypeMeta lhs, const TypeMeta rhs) noexcept {
return !operator==(lhs, rhs);
}

inline std::ostream& operator<<(
        std::ostream& stream,
        caffe2::TypeMeta typeMeta) {
    return stream << typeMeta.name();
}

} //namespace for caffe2


