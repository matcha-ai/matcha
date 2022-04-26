#include "bits_of_matcha/engine/ops/Exp.h"

#include <cmath>


namespace matcha::engine::ops {

Exp::Exp(Tensor* a)
  : ElementwiseUnaryOp(a)
{}

OpMeta<Exp> Exp::meta {
  .name = "Exp",
  .back = [](auto& ctx) { return new ExpBack(ctx); }
};

void Exp::run() {
  runCPU([](float x) { return exp(x); });
}


ExpBack::ExpBack(const BackCtx& ctx)
  : OpBack(ctx)
{}

void ExpBack::run() {

}

}