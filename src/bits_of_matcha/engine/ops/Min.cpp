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

OpMeta<Min> Min::meta {
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

void Min::run() {
  runCPU<float>(fold<float>);
}

}
