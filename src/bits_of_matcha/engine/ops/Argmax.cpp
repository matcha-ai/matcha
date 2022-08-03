#include "bits_of_matcha/engine/ops/Argmax.h"

#include <limits>
#include <execution>
#include <algorithm>

namespace matcha::engine::ops {

Argmax::Argmax(Tensor* a, bool keep_dims)
  : AxiswiseFoldOp(a, keep_dims)
{}

Argmax::Argmax(Tensor* a, int axis, bool keep_dims)
  : AxiswiseFoldOp(a, axis, keep_dims)
{}

Reflection<Argmax> Argmax::reflection {
  .name = "Argmax",
};

template <class T>
inline T foo(T* begin, size_t stride, T* end) {
  T buffer = std::numeric_limits<T>::min();
  T* pos = begin;
  if (stride != 1) {
    for (T* iter = begin; iter != end; iter += stride) {
      if (*iter > buffer) {
        buffer = *iter;
        pos = iter;
      }
    }
  } else {
    auto result = std::max_element(std::execution::par_unseq, begin, end);
    pos = &*result;
  }

  return (T) ((pos - begin) / stride);

}

template <class T>
inline T fooc(std::complex<T>* begin, size_t stride, std::complex<T>* end) {
  std::complex<T> buffer = std::numeric_limits<T>::min();
  std::complex<T>* pos;
  for (auto iter = begin; iter != end; iter += stride) {
    if (iter->real() > buffer.real()) {
      buffer = *iter;
      pos = iter;
    }
  }

  return ((pos - begin) / stride);

}

void Argmax::run() {
  if (isReal(inputs[0]))
    runCpuReal([](auto a, auto b, auto c) { return foo(a, b, c); });
  else
    runCpuComplex([](auto a, auto b, auto c) { return fooc(a, b, c); });
}

}
