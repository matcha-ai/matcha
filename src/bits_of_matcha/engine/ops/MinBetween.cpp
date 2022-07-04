#include "bits_of_matcha/engine/ops/MinBetween.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinaryBack.h"
#include "bits_of_matcha/engine/cpu/kernels/fill.h"

namespace matcha::engine::ops {

MinBetween::MinBetween(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b)
{}

OpMeta<MinBetween> MinBetween::meta {
  .name = "MinBetween",
  .back = [](auto ctx) { return new MinBetweenBack(ctx); }
};

void MinBetween::run() {
  outputs[0]->malloc();
  runCPU([] (auto a, auto b) { return a < b ? a : b; });
}


MinBetweenBack::MinBetweenBack(const BackCtx& ctx)
  : OpBack(ctx)
  , iter_(forward->inputs[0]->shape(), forward->inputs[1]->shape())
{}

OpMeta<MinBetweenBack> MinBetweenBack::meta {
  .name = "MinBetweenBack",
};

void MinBetweenBack::run() {
//  print("MinBetweenBack");
//  print("", inputs[0] ," -> ", outputs[0], " ", outputs[1]);
//  print();
//  return;
  auto forwA = forward->inputs[0]->buffer().as<float*>();
  auto forwB = forward->inputs[1]->buffer().as<float*>();

  if (outputs[0]) {
    cpu::fill(outputs[0]->malloc(), outputs[0]->size(), 0);

    cpu::elementwiseBinaryBack(
      [=](float& a, float& b, float& c) {
        if (*(forwA + std::distance(forwB, &b)) <= b) {
          a += c;
        }
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
      [=](float& a, float& b, float& c) {
        if (*(forwB + std::distance(forwA, &a)) <= a) {
          b += c;
        }
      },
      forward->inputs[0]->buffer(),
      outputs[1]->buffer(),
      inputs[0]->buffer(),
      iter_
    );
  }
}

}
