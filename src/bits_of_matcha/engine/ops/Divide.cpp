#include "bits_of_matcha/engine/ops/Divide.h"
#include "bits_of_matcha/engine/cpu/kernels/fill.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinaryBack.h"


namespace matcha::engine::ops {

Dtype promoteDtypesDivide(Dtype a, Dtype b) {
  Dtype c = promoteDtypes(a, b);
  return promoteDtypes(c, Float);
}

Divide::Divide(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b, promoteDtypesDivide(a->dtype(), b->dtype()))
{}

Reflection<Divide> Divide::reflection {
  .name = "Divide",
  .back = [](auto& ctx) { return dispatch<DivideBack>(ctx); },
};

void Divide::run() {
  runCpu(std::divides());
}


DivideBack::DivideBack(const BackCtx& ctx)
  : ElementwiseBinaryOpBack(ctx)
{
}

Reflection<DivideBack> DivideBack::reflection {
  .name = "DivideBack",
};

void DivideBack::run() {
  if (outputs[0]) {
    cpu::fill<float>(outputs[0]->malloc(), outputs[0]->size(), 0);

    cpu::elementwiseBinaryBack(
      [](float& a, float& b, float& c) {
        a += c / b;
      },
      outputs[0]->buffer(),
      forwardInput(1)->buffer(),
      inputs[0]->buffer(),
      iter_
    );
  }
  if (outputs[1]) {
    cpu::fill<float>(outputs[1]->malloc(), outputs[1]->size(), 0);
    auto bForw = forward_->inputs[1]->buffer().as<float*>();
    auto bBack = outputs[1]->buffer().as<float*>();

    cpu::elementwiseBinaryBack(
      [=](float& a, float& b, float& c) {
        // c = a / b = a b^-1
        // dc/db = - a b^-2
        float bf = bForw[&b - bBack];
        bf *= bf;
        b -= a * c / bf;
      },
      forwardInput(0)->buffer(),
      outputs[1]->buffer(),
      inputs[0]->buffer(),
      iter_
    );
  }

}

}
