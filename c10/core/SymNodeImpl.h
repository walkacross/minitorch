
#pragma once

#include <c10/util/intrusive_ptr.h>
#include <memory>
#include <mutex>
#include <vector>
#include <iostream>

namespace c10 {

    class SymNodeImpl;
    using SymNode = c10::intrusive_ptr<SymNodeImpl>;

    class SymNodeImpl : public c10::intrusive_ptr_target {
    public:
    virtual ~SymNodeImpl(){};

    template <typename T>
    c10::intrusive_ptr<T> dyn_cast() const {
        return c10::intrusive_ptr<T>::reclaim_copy(dynamic_cast<T*>(this));
    }

    // these could be pure virtual when we implement LTC versions
    virtual bool is_int() {
    }
    virtual bool is_float() {
    };
    virtual SymNode add(const SymNode& other) {
    };
    virtual SymNode sub(const SymNode& other) {
    };
    virtual SymNode mul(const SymNode& other) {
    };
    virtual SymNode truediv(const SymNode& other) {
    };
    virtual SymNode pow(const SymNode& other) {
    };
    virtual SymNode floordiv(const SymNode& other) {
    };
    virtual SymNode mod(const SymNode& other) {
    };
    virtual SymNode eq(const SymNode& other) {
    };
    virtual SymNode ne(const SymNode& other) {
    };
    virtual SymNode gt(const SymNode& other) {
    };
    virtual SymNode lt(const SymNode& other) {
    };
    virtual SymNode le(const SymNode& other) {
    };
    virtual SymNode ge(const SymNode& other) {
    };
    virtual SymNode ceil() {
    }
    virtual SymNode floor() {
    }
    virtual SymNode neg() {
    }
    virtual SymNode min(const SymNode& other) {
    }
    virtual SymNode max(const SymNode& other) {
    }
    virtual SymNode clone() {
    }
    virtual SymNode sym_float() {
    }
    virtual SymNode wrap_int(int64_t num) {
    }
    virtual SymNode wrap_float(double num) {
    }
    virtual int64_t guard_int(const char* file, int64_t line) {
    }
    virtual double guard_float(const char* file, int64_t line) {
    }
    virtual int64_t int_() {
    }
    virtual bool bool_() {
    }
    virtual std::string str() {
    }
    std::ostream& operator<<(std::ostream& os) {
        os << str();
        return os;
    };
};
} // namespace c10