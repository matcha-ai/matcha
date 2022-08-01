#include "bits_of_matcha/engine/ops/Cast.h"
#include "bits_of_matcha/engine/memory/cast.h"
#include "bits_of_matcha/engine/op/BackCtx.h"


namespace matcha::engine::ops {

Cast::Cast(Tensor* a, const Dtype& dtype)
  : Op{a}
{
  outputs.add(this, dtype, a->shape());
}

OpMeta<Cast> Cast::meta {
  .name = "Cast",
  .back = [](const BackCtx& ctx) {
    return new Cast(ctx.vals[0], Float);
  },
};

void Cast::run() {
  engine::cast(inputs[0]->buffer(), outputs[0]->malloc(),
               inputs[0]->dtype(), outputs[0]->dtype(), outputs[0]->size());
}

}