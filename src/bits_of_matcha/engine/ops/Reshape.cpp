#include "bits_of_matcha/engine/ops/Reshape.h"
#include "bits_of_matcha/error/IncompatibleShapesError.h"


namespace matcha::engine::ops {

Reshape::Reshape(Tensor* a, const Shape::Reshape& dims)
  : Op{a}
{
  Shape shape = dims(a->shape());
  if (a->size() != shape.size()) {
    throw IncompatibleShapesError(a->shape(), shape);
  }
  outputs.add(this, a->dtype(), shape);
}

OpMeta<Reshape> Reshape::meta {
  .name = "Reshape",
  .back = [](auto& ctx) {
    return new ReshapeBack(ctx);
  },
};

void Reshape::run() {
  outputs[0]->share(inputs[0]);
}

ReshapeBack::ReshapeBack(const BackCtx& ctx)
  : OpBack(ctx)
{}

OpMeta<ReshapeBack> ReshapeBack::meta {
  .name = "ReshapeBack"
};

void ReshapeBack::run() {
  outputs[0]->share(inputs[0]);
}

}
