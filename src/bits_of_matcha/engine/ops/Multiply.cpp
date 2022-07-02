#include "bits_of_matcha/engine/ops/Multiply.h"
#include "bits_of_matcha/engine/cpu/kernels/fill.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinaryBack.h"


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
  runCPU(std::multiplies());
}


MultiplyBack::MultiplyBack(const BackCtx& ctx)
  : OpBack(ctx)
  , iter_(forward->inputs[0]->shape(), forward->inputs[1]->shape())
{
}

OpMeta<MultiplyBack> MultiplyBack::meta {
  .name = "MultiplyBack",
};

void MultiplyBack::run() {
//  print(inputs[0], " ", outputs[0], " ", outputs[1]);
//  print(inputs[0]->buffer());
  if (outputs[0]) {
    cpu::fill(outputs[0]->malloc(), outputs[0]->size(), 0);

    cpu::elementwiseBinaryBack(
      [](float& a, float& b, float& c) {
//        print(a, b, c);
        a += b * c;
      },
      outputs[0]->buffer(),
      forward->inputs[1]->buffer(),
      inputs[0]->buffer(),
      iter_
    );
  }
  if (outputs[1]) {
    cpu::fill(outputs[1]->malloc(), outputs[1]->size(), 0);

    cpu::elementwiseBinaryBack(
      [](float& a, float& b, float& c) {
        b += a * c;
      },
      forward->inputs[0]->buffer(),
      outputs[1]->buffer(),
      inputs[0]->buffer(),
      iter_
    );
  }

}

}
