#include "bits_of_matcha/engine/ops/Divide.h"
#include "bits_of_matcha/engine/cpu/kernels/fill.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinaryBack.h"


namespace matcha::engine::ops {

Dtype promoteDtypesDivide(Dtype a, Dtype b) {
  Dtype c = promoteDtypes(a, b);

  switch (c) {
  case Int:
  case Uint:
  case Long:
  case Ulong:
  case Double:
    return Double;
  default:
    return Float;
  }
}

Divide::Divide(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b, promoteDtypesDivide(a->dtype(), b->dtype()))
{}

OpMeta<Divide> Divide::meta {
  .name = "Divide",
  .back = [](auto ctx) { return new DivideBack(ctx); }
};

void Divide::run() {
  runCPU(std::divides());
}


DivideBack::DivideBack(const BackCtx& ctx)
  : OpBack(ctx)
  , iter_(forward->inputs[0]->shape(), forward->inputs[1]->shape())
{
}

OpMeta<DivideBack> DivideBack::meta {
  .name = "DivideBack",
};

void DivideBack::run() {
  if (outputs[0]) {
    cpu::fill(outputs[0]->malloc(), outputs[0]->size(), 0);

    cpu::elementwiseBinaryBack(
      [](float& a, float& b, float& c) {
        a += c / b;
      },
      outputs[0]->buffer(),
      forward->inputs[1]->buffer(),
      inputs[0]->buffer(),
      iter_
    );
  }
  if (outputs[1]) {
    cpu::fill(outputs[1]->malloc(), outputs[1]->size(), 0);
    auto bForw = forward->inputs[1]->buffer().as<float*>();
    auto bBack = outputs[1]->buffer().as<float*>();

    cpu::elementwiseBinaryBack(
      [=](float& a, float& b, float& c) {
        // c = a / b = a b^-1
        // dc/db = - a b^-2
        float bf = bForw[&b - bBack];
        bf *= bf;
        b -= a * c / bf;
      },
      forward->inputs[0]->buffer(),
      outputs[1]->buffer(),
      inputs[0]->buffer(),
      iter_
    );
  }

}

}
