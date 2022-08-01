#include "bits_of_matcha/engine/ops/Min.h"

#include <limits>
#include <execution>
#include <algorithm>

namespace matcha::engine::ops {

Min::Min(Tensor* a)
  : AxiswiseFoldOp(a)
{}

Min::Min(Tensor* a, int axis)
  : AxiswiseFoldOp(a, axis)
{}

Reflection<Min> Min::reflection {
  .name = "Min",
};

template <class T>
inline T fold(T* begin, size_t stride, T* end) {
  T buffer = std::numeric_limits<T>::max();
  if (stride != 1) {
    for (T* iter = begin; iter != end; iter += stride) {
      buffer = std::min(buffer, *iter);
    }
  } else {
    auto result = std::min_element(std::execution::par_unseq, begin, end);
    buffer = *result;
  }

  return buffer;
}

template <class T>
inline std::complex<T> foldc(std::complex<T>* begin, size_t stride, std::complex<T>* end) {
  std::complex<T> buffer = std::numeric_limits<T>::max();
  for (auto iter = begin; iter != end; iter += stride) {
    if (buffer.real() < iter->real()) buffer = *iter;
  }
  return buffer;
}

void Min::run() {
  if (isReal(inputs[0]))
    runCpuReal([](auto a, auto b, auto c) { return fold(a, b, c);});
  else
    runCpuComplex([](auto a, auto b, auto c) { return foldc(a, b, c);});
}

}
