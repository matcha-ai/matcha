#include "bits_of_matcha/engine/ops/Reshape.h"
#include "bits_of_matcha/error/IncompatibleShapesError.h"


namespace matcha::engine::ops {

Reshape::Reshape(Tensor* a, const Shape::Reshape& dims)
  : Op{a}
{
  Shape shape = dims(a->shape());
  if (a->size() != shape.size())
    throw IncompatibleShapesError(a->shape(), shape);

  addOutput(a->dtype(), shape);
}

Reflection<Reshape> Reshape::reflection {
  .name = "Reshape",
  .back = [](auto& ctx) { return dispatch<Reshape>(ctx.vals[0], ctx.forward->inputs[0]->shape()); },
};

void Reshape::run() {
  outputs[0]->share(inputs[0]);
}

}
