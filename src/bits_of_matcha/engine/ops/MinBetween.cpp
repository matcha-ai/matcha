#include "bits_of_matcha/engine/ops/MinBetween.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinaryBack.h"
#include "bits_of_matcha/engine/cpu/kernels/fill.h"

namespace matcha::engine::ops {

MinBetween::MinBetween(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b)
{}

Reflection<MinBetween> MinBetween::reflection {
  .name = "MinBetween",
};

void MinBetween::run() {
  Dtype dtype = outputs[0]->dtype();

  if (isReal(dtype))
    runCpuReal([](auto a, auto b) { return a < b ? a : b; });
  else
    runCpuComplex([](auto a, auto b) { return a.real() < b.real() ? a : b; });
}


MinBetweenBack::MinBetweenBack(const BackCtx& ctx)
  : OpBack(ctx)
  , iter_(forward_->inputs[0]->shape(), forward_->inputs[1]->shape())
{}

Reflection<MinBetweenBack> MinBetweenBack::reflection {
  .name = "MinBetweenBack",
};

void MinBetweenBack::run() {
//  print("MinBetweenBack");
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
