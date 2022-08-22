#include "bits_of_matcha/engine/ops/Minimum.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinaryBack.h"
#include "bits_of_matcha/engine/cpu/kernels/fill.h"

namespace matcha::engine::ops {

Minimum::Minimum(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b)
{}

Reflection<Minimum> Minimum::reflection {
  .name = "Minimum",
};

void Minimum::run() {
  Dtype dtype = outputs[0]->dtype();

  if (isReal(dtype))
    runCpuReal([](auto a, auto b) { return a < b ? a : b; });
  else
    runCpuComplex([](auto a, auto b) { return a.real() < b.real() ? a : b; });
}


MinimumBack::MinimumBack(const BackCtx& ctx)
  : OpBack(ctx)
  , iter_(forward_->inputs[0]->shape(), forward_->inputs[1]->shape())
{}

Reflection<MinimumBack> MinimumBack::reflection {
  .name = "MinimumBack",
};

void MinimumBack::run() {
//  print("MinimumBack");
//  print("", inputs[0] ," -> ", outputs[0], " ", outputs[1]);
//  print();
//  return;
  auto forwA = forward_->inputs[0]->buffer().as<float*>();
  auto forwB = forward_->inputs[1]->buffer().as<float*>();

  if (outputs[0]) {
    cpu::fill(outputs[0]->malloc(), outputs[0]->size(), 0);

    cpu::elementwiseBinaryBack(
    [=](float& a, float& b, float& c) {
        if (*(forwA + std::distance(forwB, &b)) <= b) {
          a += c;
        }
      },
    outputs[0]->buffer(),
    forward_->inputs[1]->buffer(),
    inputs[0]->buffer(),
    iter_
    );
  }
  if (outputs[1]) {
    cpu::fill(outputs[1]->malloc(), outputs[1]->size(), 0);

    cpu::elementwiseBinaryBack(
    [=](float& a, float& b, float& c) {
        if (*(forwB + std::distance(forwA, &a)) <= a) {
          b += c;
        }
      },
    forward_->inputs[0]->buffer(),
    outputs[1]->buffer(),
    inputs[0]->buffer(),
    iter_
    );
  }
}

}
