#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine {

OpBack::OpBack(const BackCtx& ctx)
  : Op(ctx.vals)
{
  forward_ = ctx.forward;
  for (int i = 0; i < ctx.wrts.size(); i++) {
    if (!ctx.wrts[i])
      addOutput(nullptr);
    else
      addOutput(Float, ctx.forward->inputs[i]->shape());
  }
}

}