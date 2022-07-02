#include "bits_of_matcha/engine/ops/Identity.h"


namespace matcha::engine::ops {

Identity::Identity(Tensor* a)
  : Op{a}
{
  outputs.add(this, a->frame());
}

OpMeta<Identity> Identity::meta {
  .name = "Identity",
  .back = [](auto& ctx) {
    return new IdentityBack(ctx);
  },
};

void Identity::run() {
  outputs[0]->share(inputs[0]);
}

IdentityBack::IdentityBack(const BackCtx& ctx)
  : OpBack(ctx)
{}

OpMeta<IdentityBack> IdentityBack::meta {
  .name = "IdentityBack"
};

void IdentityBack::run() {
  outputs[0]->share(inputs[0]);
}

}
