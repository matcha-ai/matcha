#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine {

OpBack::OpBack(const BackCtx& ctx)
  : Op(ctx.vals)
{
  for (int i = 0; i < ctx.wrts.size(); i++) {
    if (!ctx.wrts[i]) {
      outputs.add(this, nullptr);
      continue;
    }
    auto wrt = ctx.forward->inputs[i];
    outputs.add(this, Float, wrt->shape());
  }
  forward = ctx.forward;
}

}