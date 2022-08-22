#include "bits_of_matcha/engine/ops/Exp.h"
#include "bits_of_matcha/engine/ops/Multiply.h"

#include <cmath>


namespace matcha::engine::ops {

Exp::Exp(Tensor* a)
  : ElementwiseUnaryOp(a, promoteDtypes(a->dtype(), Float))
{}

Reflection<Exp> Exp::reflection {
  .name = "Exp",
  .back = [](const BackCtx& ctx) {
    // y = e^a
    // dy/da = e^a * da = y * da
    return dispatch<Multiply>(ctx.forward->outputs[0], ctx.vals[0]);
  },
};

void Exp::run() {
  runCpu([](auto x) { return std::exp(x); });
}

}