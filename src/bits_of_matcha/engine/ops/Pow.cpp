#include "bits_of_matcha/engine/ops/Pow.h"

#include <cmath>


namespace matcha::engine::ops {

Pow::Pow(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b)
{}

OpMeta<Pow> Pow::meta {
  .name = "Pow",
  .back = [](auto ctx) { return new PowBack(ctx); }
};

void Pow::run() {
  outputs[0]->malloc();
  runCPU(std::pow<float, float>);
}


PowBack::PowBack(const BackCtx& ctx)
  : OpBack(ctx)
{
}

OpMeta<PowBack> PowBack::meta {
  .name = "PowBack",
};

}
