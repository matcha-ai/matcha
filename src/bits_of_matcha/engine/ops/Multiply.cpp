#include "bits_of_matcha/engine/ops/Multiply.h"
#include "bits_of_matcha/engine/cpu/kernels/fill.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinaryBack.h"
#include "bits_of_matcha/engine/ops/Broadcast.h"


namespace matcha::engine::ops {

Multiply::Multiply(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b)
{}

Reflection<Multiply> Multiply::reflection {
  .name = "Multiply",
  .back = [](auto& ctx) { return dispatch<MultiplyBack>(ctx); },
};

void Multiply::run() {
  outputs[0]->malloc();
  runCpu(std::multiplies());
}


MultiplyBack::MultiplyBack(const BackCtx& ctx)
  : ElementwiseBinaryOpBack(ctx)
{
}

Reflection<MultiplyBack> MultiplyBack::reflection {
  .name = "MultiplyBack",
};

void MultiplyBack::run() {
//  print(inputs[0], " ", outputs[0], " ", outputs[1]);
//  print(inputs[0]->buffer());
  if (outputs[0]) {
    cpu::fill(outputs[0]->malloc(), outputs[0]->size(), 0);

    cpu::elementwiseBinaryBack(
      [](float& a, float& b, float& c) {
//        print(a, " ", b, " ", c);
        a += b * c;
      },
      outputs[0]->buffer(),
      forwardInput(1)->buffer(),
      inputs[0]->buffer(),
      iter_
    );
  }
  if (outputs[1]) {
    cpu::fill(outputs[1]->malloc(), outputs[1]->size(), 0);

    cpu::elementwiseBinaryBack(
      [](float& a, float& b, float& c) {
//        print(b);
        b += a * c;
      },
      forwardInput(0)->buffer(),
      outputs[1]->buffer(),
      inputs[0]->buffer(),
      iter_
    );
  }

}

}
