#include "bits_of_matcha/error/BroadcastError.h"
#include "bits_of_matcha/engine/ops/Broadcast.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinary.h"


namespace matcha::engine::ops {

Broadcast::Broadcast(Tensor* a, const Shape& shape)
  : Op{a}
  , iter_(a->shape(), shape)
{
  if (Shape(iter_.dims_c) != shape)
    throw BroadcastError(a->shape(), shape);

  addOutput(a->dtype(), shape);
}

Reflection<Broadcast> Broadcast::reflection {
  .name = "Broadcast"
};

void Broadcast::run() {
  Dtype dtype = inputs[0]->dtype();

  switch (dtype) {
  case Float: return cpu::elementwiseBinary<float>([](auto a, auto b) { return a; }, inputs[0]->buffer(), outputs[0]->malloc(), outputs[0]->malloc(), iter_);
  default:
    throw std::runtime_error("unsupported dtype");
  }
}

}