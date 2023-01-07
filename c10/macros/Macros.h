#pragma once

// macros
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)

#define CONSTEXPR_EXCEPT_WIN_CUDA constexpr
#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA __host__

#define STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(field, val) \
  static constexpr const char* field = val;
#define STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cls, field, val)