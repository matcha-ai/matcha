#include "bits_of_matcha/engine/ops/Identity.h"


namespace matcha::engine::ops {

Identity::Identity(Tensor* a)
  : Op{a}
{
  outputs.add(this, a->frame());
}

Identity::Identity(Tensor* a, Tensor* target)
  : Op{a}
{
  if (a->frame() != target->frame())
    throw std::runtime_error("frame mismatch");
  outputs.add(this, target);
}

OpMeta<Identity> Identity::meta {
  .name = "Identity",
  .back = [](auto& ctx) {
    return new Identity(ctx.vals[0]);
  },
};

void Identity::run() {
  outputs[0]->share(inputs[0]);
}

}
