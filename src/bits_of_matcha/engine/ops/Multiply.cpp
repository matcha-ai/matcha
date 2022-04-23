#include "bits_of_matcha/engine/ops/Multiply.h"


namespace matcha::engine::ops {

Multiply::Multiply(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b)
{}

OpMeta<Multiply> Multiply::meta {
  .name = "Multiply",
  .back = [](auto ctx) { return new MultiplyBack(ctx); }
};

void Multiply::run() {
  outputs[0]->malloc();
  runCPU(std::multiplies<float>());
}


MultiplyBack::MultiplyBack(const BackCtx& ctx)
  : OpBack(ctx)
{
}

OpMeta<MultiplyBack> MultiplyBack::meta {
  .name = "MultiplyBack",
};

}
