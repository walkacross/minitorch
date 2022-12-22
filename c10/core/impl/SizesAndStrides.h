#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <c10/util/ArrayRef.h>

// macros
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)

#define C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE 5

namespace c10 { namespace impl {

    class SizesAndStrides {

    public:
        // TODO: different iterator types for sizes & strides to prevent
        // mixing the two accidentally.
        using sizes_iterator = int64_t*;
        using sizes_const_iterator = const int64_t*;
        using strides_iterator = int64_t*;
        using strides_const_iterator = const int64_t*;

    private:
        size_t size_;
        union {
            int64_t* outOfLineStorage_;
            int64_t inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE * 2]{};
        };
        bool isInline() const noexcept {
            return size_ <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE;
        }

    public:
        SizesAndStrides() : size_(1) {
            size_at_unchecked(0) = 0;
            stride_at_unchecked(0) = 1;
        }

        ~SizesAndStrides() {
            if (C10_UNLIKELY(!isInline())) {
                free(outOfLineStorage_);
            }
        }

        // size related
        size_t size() const noexcept {
            return size_;
        }

        const int64_t* sizes_data() const noexcept {
            if (C10_LIKELY(isInline())) {
                return &inlineStorage_[0];
            } else {
                return &outOfLineStorage_[0];
            }
        }

        int64_t* sizes_data() noexcept {
            if (C10_LIKELY(isInline())) {
                return &inlineStorage_[0];
            } else {
                return &outOfLineStorage_[0];
            }
        }
        sizes_iterator sizes_begin() noexcept {
            return sizes_data();
        }

        sizes_iterator sizes_end() noexcept {
            return sizes_begin() + size();
        }

        IntArrayRef sizes_arrayref() const noexcept {
            return IntArrayRef{sizes_data(), size()};
        }

        void set_sizes(IntArrayRef newSizes) {
            resize(newSizes.size());
            std::copy(newSizes.begin(), newSizes.end(), sizes_begin());
        }

        // stride related
        const int64_t* strides_data() const noexcept {
            if (C10_LIKELY(isInline())) {
                return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
            } else {
                return &outOfLineStorage_[size()];
            }
        }

        int64_t* strides_data() noexcept {
            if (C10_LIKELY(isInline())) {
                return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
            } else {
                return &outOfLineStorage_[size()];
            }
        }

        strides_iterator strides_begin() noexcept {
            if (C10_LIKELY(isInline())) {
                return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
            } else {
                return &outOfLineStorage_[size()];
            }
        }

        strides_iterator strides_end() noexcept {
            return strides_begin() + size();
        }

        IntArrayRef strides_arrayref() const noexcept {
            return IntArrayRef{strides_data(), size()};
        }

        void set_strides(IntArrayRef strides) {
            std::copy(strides.begin(), strides.end(), strides_begin());
        }
        //

        int64_t size_at_unchecked(size_t idx) const noexcept {
            return sizes_data()[idx];
        }

        int64_t& size_at_unchecked(size_t idx) noexcept {
            return sizes_data()[idx];
        }

        int64_t stride_at_unchecked(size_t idx) const noexcept {
            return strides_data()[idx];
        }

        int64_t& stride_at_unchecked(size_t idx) noexcept {
            return strides_data()[idx];
        }

        static size_t storageBytes(size_t size) noexcept {
            return size * 2 * sizeof(int64_t);
        }

        void resize(size_t newSize) {
            const auto oldSize = size();
            if (newSize == oldSize) {
                return;
            }
            if (C10_LIKELY(
                    newSize <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE && isInline())) {
                if (oldSize < newSize) {
                    const auto bytesToZero = (newSize - oldSize) * sizeof(inlineStorage_[0]);
                    memset(&inlineStorage_[oldSize], 0, bytesToZero);
                    memset(
                            &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE + oldSize],
                            0,
                            bytesToZero);
                }
                size_ = newSize;
            } else {
                resizeSlowPath(newSize, oldSize);
            }
        }

        void resizeSlowPath(size_t newSize, size_t oldSize);

        void resizeOutOfLineStorage(size_t newSize) {
            outOfLineStorage_ = static_cast<int64_t*>(realloc(outOfLineStorage_, storageBytes(newSize)));
        }

    };
} // for impl
} // for c10