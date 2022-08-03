#include "bits_of_matcha/engine/ops/Identity.h"


namespace matcha::engine::ops {

Identity::Identity(Tensor* a)
  : Op{a}
{
  addOutput(a->frame());
}

Identity::Identity(Tensor* a, Tensor* target)
  : Op{a}
{
  if (a->frame() != target->frame())
    throw std::runtime_error("frame mismatch");
  addOutput(target);
}

Reflection<Identity> Identity::reflection {
  .name = "Identity",
  .back = [](const BackCtx& ctx) { return dispatch<Identity>(ctx.vals[0]); },
};

void Identity::run() {
  outputs[0]->share(inputs[0]);
}

}
