#include "bits_of_matcha/engine/ops/Sum.h"
#include "bits_of_matcha/engine/ops/Broadcast.h"
#include "bits_of_matcha/engine/ops/Reshape.h"

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

Sum::Sum(Tensor* a, bool keep_dims)
  : AxiswiseFoldOp(a, keep_dims, promoteDtypesSum(a))
{}

Sum::Sum(Tensor* a, int axis, bool keep_dims)
  : AxiswiseFoldOp(a, axis, keep_dims, promoteDtypesSum(a))
{}

Reflection<Sum> Sum::reflection {
  .name = "Sum",
  .back = [](const BackCtx& ctx) {
    auto fold = dynamic_cast<AxiswiseFoldOp*>(ctx.forward);
    auto a = ctx.forward->inputs[0];
    auto gb = ctx.vals[0];

    Tensor* temp = nullptr;

    if (fold->keepDims()) {
      temp = gb;
    } else if (fold->global()) {
      temp = dispatch<Reshape>(gb, std::vector(a->rank(), 1))[0];
    } else {
      std::vector<int> dims(a->shape().begin(), a->shape().end());
      dims[fold->axis()] = 1;
      temp = dispatch<Reshape>(gb, dims)[0];
    }

    auto outs = dispatch<Broadcast>(temp, a->shape());
    return outs;
  },
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