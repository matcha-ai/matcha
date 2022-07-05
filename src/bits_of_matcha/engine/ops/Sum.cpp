#include "bits_of_matcha/engine/ops/Sum.h"

#include <algorithm>
#include <numeric>
#include <execution>


namespace matcha::engine::ops {

Dtype promoteDtypesSum(Tensor* a) {
  if (a->dtype() == Bool)
    return Int;
  else
    return a->dtype();
}

Sum::Sum(Tensor* a)
  : AxiswiseFoldOp(a, promoteDtypesSum(a))
{}

Sum::Sum(Tensor* a, int axis)
  : AxiswiseFoldOp(a, axis, promoteDtypesSum(a))
{}

OpMeta<Sum> Sum::meta {
  .name = "Sum"
};

template <class T>
inline T foo(T* begin, size_t stride, T* end) {
  T buffer = 0;
  if (stride != 1) {
    for (T* iter = begin; iter != end; iter += stride) {
      buffer += *iter;
    }
  } else {
    buffer = std::accumulate(begin, end, (T) 0, std::plus<T>());
  }
  return buffer;
}

void Sum::run() {
  runCpu([](auto begin, auto stride, auto end) { return foo(begin, stride, end); });
}

}