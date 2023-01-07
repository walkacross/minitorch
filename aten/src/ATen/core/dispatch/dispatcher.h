#pragma once

#include <cassert>
#include <functional>
#include <map>
#include <type_traits>
#include <ATen/core/ATen_fwd.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>

namespace at {

    using Device = c10::Device;
    using DeviceType = c10::DeviceType;

}  // namespace at

inline std::string GetDeviceStr(const at::Device& device) {
    std::string str = c10::DeviceTypeName(device.type(), true);
    if (device.has_index()) {
        str.push_back(':');
        str.append(std::to_string(device.index()));
    }
    return str;
}

// Registry
template <typename F, F f>
class DeviceRegistry;

template <typename Ret, typename... Args, Ret (*f)(Args...)>
class DeviceRegistry<Ret (*)(Args...), f> {
public:
    using FunctionType = Ret (*)(Args...);
    static const int MAX_DEVICE_TYPES = int8_t(at::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);

    void Register(at::DeviceType device, FunctionType function) {
        funcs_[int8_t(device)] = function;
    }

    FunctionType Find(at::DeviceType device) const {
        return funcs_[int8_t(device)];
    }

    static DeviceRegistry& instance() {
        static DeviceRegistry inst;
        return inst;
    }

private:
    DeviceRegistry() {
        for (size_t i = 0; i < MAX_DEVICE_TYPES; ++i) {
            funcs_[i] = nullptr;
        }
    };
    FunctionType funcs_[MAX_DEVICE_TYPES];
};

// get device of first tensor param

template <typename T, typename... Args,
        std::enable_if_t<std::is_same<std::decay_t<T>, at::Tensor>::value,
bool> = true>
at::Device GetFirstTensorDevice(T&& t, Args&&... args) {
    return std::forward<T>(t).device();
}
template <typename T, typename... Args,
        std::enable_if_t<!std::is_same<std::decay_t<T>, at::Tensor>::value,
bool> = true>
at::Device GetFirstTensorDevice(T&& t, Args&&... args) {
    return GetFirstTensorDevice(std::forward<Args>(args)...);
}

// check device consistency

inline std::pair<int, at::Device> CheckDeviceConsistency(
        const at::Device& device, int index) {
    return {index, device};
}

template <typename T, typename... Args,
        std::enable_if_t<!std::is_same<std::decay_t<T>, at::Tensor>::value,
bool> = true>
std::pair<int, at::Device> CheckDeviceConsistency(const at::Device& device,
                                                  int index, T&& t,
                                                  Args&&... args);

template <typename T, typename... Args,
        std::enable_if_t<std::is_same<std::decay_t<T>, at::Tensor>::value,
bool> = true>
std::pair<int, at::Device> CheckDeviceConsistency(const at::Device& device,
                                                  int index, T&& t,
                                                  Args&&... args) {
    auto new_device = std::forward<T>(t).device();
    if (new_device.type() != device.type() ||
        new_device.index() != device.index()) {
        return {index, new_device};
    }
    return CheckDeviceConsistency(device, index + 1, std::forward<Args>(args)...);
}

template <
        typename T, typename... Args,
        std::enable_if_t<!std::is_same<std::decay_t<T>, at::Tensor>::value, bool>>
std::pair<int, at::Device> CheckDeviceConsistency(const at::Device& device,
                                                  int index, T&& t,
                                                  Args&&... args) {
    return CheckDeviceConsistency(device, index + 1, std::forward<Args>(args)...);
}

// dispatch

template <typename R, typename... Args>
auto Dispatch(const R& registry, const char* name, Args&&... args) {
    auto device = GetFirstTensorDevice(std::forward<Args>(args)...);
    auto inconsist = CheckDeviceConsistency(device, 0, std::forward<Args>(args)...);
    auto f_ptr = registry.Find(device.type());
    return f_ptr(std::forward<Args>(args)...);
}

// helper macro
#define DEVICE_REGISTRY(key) DeviceRegistry<decltype(&(key)), key>::instance()

#define REGISTER_DEVICE_IMPL(key, device, value)           \
  struct key##_##device##_registerer {                     \
    key##_##device##_registerer() {                        \
      DEVICE_REGISTRY(key).Register(c10::k##device, value); \
    }                                                      \
  };                                                       \
  static key##_##device##_registerer _##key##_##device##_registerer;

#define DISPATCH_DEVICE_IMPL(key, ...) \
  Dispatch(DEVICE_REGISTRY(key), #key, __VA_ARGS__)
