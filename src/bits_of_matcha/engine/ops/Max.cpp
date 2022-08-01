#include "bits_of_matcha/engine/ops/Max.h"

#include <limits>
#include <execution>
#include <algorithm>

namespace matcha::engine::ops {

Max::Max(Tensor* a)
  : AxiswiseFoldOp(a)
{}

Max::Max(Tensor* a, int axis)
  : AxiswiseFoldOp(a, axis)
{}

Reflection<Max> Max::reflection {
  .name = "Max",
};

template <class T>
inline T foo(T* begin, size_t stride, T* end) {
  T buffer = std::numeric_limits<T>::lowest();
  if (stride != 1 or true) {
    for (T* iter = begin; iter != end; iter += stride) {
      if (*iter > buffer) {
        buffer = *iter;
      }
    }
  } else {
    auto result = std::max_element(std::execution::par_unseq, begin, end);
  }

  return buffer;
}

template <class T>
inline std::complex<T> fooc(std::complex<T>* begin, size_t stride, std::complex<T>* end) {
  std::complex<T> buffer = std::numeric_limits<T>::lowest();
  std::complex<T>* pos;
  for (auto iter = begin; iter != end; iter += stride) {
    if (iter->real() < buffer.real()) {
      buffer = *iter;
    }
  }

  return buffer;
}

void Max::run() {
  if (isReal(inputs[0]))
    runCpuReal([](auto a, auto b, auto c) { return foo(a, b, c); });
  else
    runCpuComplex([](auto a, auto b, auto c) { return fooc(a, b, c); });
}

}