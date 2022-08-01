#include "bits_of_matcha/engine/ops/Cast.h"
#include "bits_of_matcha/engine/memory/cast.h"
#include "bits_of_matcha/engine/op/BackCtx.h"


namespace matcha::engine::ops {

Cast::Cast(Tensor* a, const Dtype& dtype)
  : Op{a}
{
  addOutput(dtype, a->shape());
}

Reflection<Cast> Cast::reflection {
  .name = "Cast",
  .back = [](const BackCtx& ctx) { return dispatch<Cast>(ctx.vals[0], Float); },
};

void Cast::run() {
  if (inputs[0]->dtype() == outputs[0]->dtype()) {
    outputs[0]->share(inputs[0]);
    return;
  }

  engine::cast(inputs[0]->buffer(), outputs[0]->malloc(),
               inputs[0]->dtype(), outputs[0]->dtype(), outputs[0]->size());
}

}