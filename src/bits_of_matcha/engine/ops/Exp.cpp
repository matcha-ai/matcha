#include "bits_of_matcha/engine/ops/Exp.h"

#include <cmath>


namespace matcha::engine::ops {

Exp::Exp(Tensor* a)
  : ElementwiseUnaryOp(a, promoteDtypes(a->dtype(), Float))
{}

Reflection<Exp> Exp::reflection {
  .name = "Exp",
  .back = [](auto& ctx) { return dispatch<Exp>(ctx.vals[0]); },
};

void Exp::run() {
  runCpu([](auto x) { return std::exp(x); });
}

}