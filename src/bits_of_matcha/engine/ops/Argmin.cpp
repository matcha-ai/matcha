#include "bits_of_matcha/engine/ops/Argmin.h"

#include <limits>
#include <execution>
#include <algorithm>

namespace matcha::engine::ops {

Argmin::Argmin(Tensor* a)
  : AxiswiseFoldOp(a)
{}

Argmin::Argmin(Tensor* a, int axis)
  : AxiswiseFoldOp(a, axis)
{}

OpMeta<Argmin> Argmin::meta {
  .name = "Argmin",
};

template <class T>
inline T foo(T* begin, size_t stride, T* end) {
  T buffer = std::numeric_limits<T>::min();
  T* pos;
  if (stride != 1) {
    for (T* iter = begin; iter != end; iter += stride) {
      if (*iter < buffer) {
        buffer = *iter;
        pos = iter;
      }
    }
  } else {
    auto result = std::min_element(std::execution::par_unseq, begin, end);
    pos = &*result;
  }

  return (T) ((pos - begin) / stride);

}

void Argmin::run() {
  outputs[0]->malloc();

  runCPU<float>(foo<float>);
}

}
