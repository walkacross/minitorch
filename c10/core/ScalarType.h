#pragma once

#include <cstdint>
#include <ostream>

namespace c10 {

// For the macros below:
// NB: If you want to macro some code for all non-QInt scalar types (i.e. types
// with complete information, you probably want one of the
// AT_FORALL_SCALAR_TYPES / AT_FORALL_SCALAR_TYPES_AND
// macros below, which are designed to behave similarly to the Dispatch macros
// with the same name.

// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_) \
  _(uint8_t, Byte) /* 0 */                               \
  _(int8_t, Char) /* 1 */                                \
  _(int16_t, Short) /* 2 */                              \
  _(int, Int) /* 3 */                                    \
  _(int64_t, Long) /* 4 */                               \
  _(float, Float) /* 6 */                                \
  _(double, Double) /* 7 */                              \


enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1, n) n,
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ENUM)
#undef DEFINE_ENUM
        Undefined,
        NumOptions
};

constexpr uint16_t NumScalarTypes = static_cast<uint16_t>(ScalarType::NumOptions);

static inline size_t elementSize(ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype, name) \
  case ScalarType::name:                   \
    return sizeof(ctype);

        switch (t) {
            AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(CASE_ELEMENTSIZE_CASE)
            }
#undef CASE_ELEMENTSIZE_CASE
        }

static inline const char* toString(ScalarType t) {
#define DEFINE_CASE(_, name) \
  case ScalarType::name:     \
    return #name;

        switch (t) {
            AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CASE)
                default:
                return "UNKNOWN_SCALAR";
            }
#undef DEFINE_CASE
        }

inline std::ostream& operator<<(std::ostream& stream, c10::ScalarType scalar_type) {
        return stream << toString(scalar_type);
}

} // namespace c10