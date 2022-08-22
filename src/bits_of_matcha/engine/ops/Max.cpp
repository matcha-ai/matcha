#include "bits_of_matcha/engine/ops/Max.h"
#include "bits_of_matcha/engine/cpu/kernels/axiswiseFoldBack.h"

#include <limits>
#include <execution>
#include <algorithm>

namespace matcha::engine::ops {

Max::Max(Tensor* a, bool keep_dims)
  : AxiswiseFoldOp(a, keep_dims)
{}

Max::Max(Tensor* a, int axis, bool keep_dims)
  : AxiswiseFoldOp(a, axis, keep_dims)
{}

Reflection<Max> Max::reflection {
  .name = "Max",
  .back = [](auto& ctx) { return dispatch<MaxBack>(ctx); },
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

MaxBack::MaxBack(const BackCtx& ctx)
  : OpBack(ctx)
{
  auto fold = dynamic_cast<AxiswiseFoldOp*>(ctx.forward);
  iter_ = fold->iter();
  inputs.push_back(forward_->inputs[0]);
  inputs.push_back(forward_->outputs[0]);
  forward_->inputs[0]->req();
  forward_->outputs[0]->req();
}

Reflection<MaxBack> MaxBack::reflection {
  .name = "MaxBack"
};

void MaxBack::run() {
  auto gy = inputs.front();
  auto ga = outputs.front();
  auto a = inputs[1];
  auto y = inputs[2];

  ga->malloc();

  auto begin_gy = gy->buffer().as<float*>();
  auto begin_ga = ga->buffer().as<float*>();
  auto begin_a = a->buffer().as<float*>();
  auto begin_y = y->buffer().as<float*>();

  switch (gy->dtype()) {
  case Float:

    cpu::axiswiseFoldBack<float>([=](auto* begin, auto stride, auto* end, auto& g) {
      size_t count = 0;
      auto offset_begin = begin - begin_ga;
      auto offset_end = end - begin_ga;
      auto m = begin_y + (&g - begin_gy);
      for (auto i = begin_a + offset_begin; i != begin_a + offset_end; i += stride)
        if (*i == *m) count++;

      auto g_normed = g / count;

      auto j = begin_a + offset_begin;
      for (auto i = begin; i != end; i += stride) {
        *i = (*j == *m) ? g_normed : 0;
        j += stride;
      }

    }, ga->malloc(), gy->buffer(), iter_);

    break;
  default:
    throw std::runtime_error("unsupported dtype");
  }
}

}