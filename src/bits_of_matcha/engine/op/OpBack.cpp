#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine {

OpBack::OpBack(const BackCtx& ctx)
  : Op(ctx.vals)
{
  for (auto wrt: ctx.wrts) {
    outputs.add(this, wrt);
  }
  forward = ctx.forward;
}

}