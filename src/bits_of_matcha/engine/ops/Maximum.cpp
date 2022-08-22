#include "bits_of_matcha/engine/ops/Maximum.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinaryBack.h"
#include "bits_of_matcha/engine/cpu/kernels/fill.h"

namespace matcha::engine::ops {

Maximum::Maximum(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b)
{}

Reflection<Maximum> Maximum::reflection {
  .name = "Maximum",
  .back = [](auto& ctx) { return dispatch<MaximumBack>(ctx); },
};

void Maximum::run() {
  Dtype dtype = outputs[0]->dtype();

  if (isReal(dtype))
    runCpuReal([](auto a, auto b) { return a > b ? a : b; });
  else
    runCpuComplex([](auto a, auto b) { return a.real() > b.real() ? a : b; });
}


MaximumBack::MaximumBack(const BackCtx& ctx)
  : ElementwiseBinaryOpBack(ctx)
{}

Reflection<MaximumBack> MaximumBack::reflection {
  .name = "MaximumBack",
};

void MaximumBack::run() {
//  print("MaximumBack");
//  print("", inputs[0] ," -> ", outputs[0], " ", outputs[1]);
//  print();
//  return;
  auto a = forwardInput(0);
  auto b = forwardInput(1);
  auto ga = outputs[0];
  auto gb = outputs[1];
  auto gy = inputs[0];

  auto forwA = forward_->inputs[0]->buffer().as<float*>();
  auto forwB = forward_->inputs[1]->buffer().as<float*>();

  if (ga) {
    cpu::fill(outputs[0]->malloc(), outputs[0]->size(), 0);

    cpu::elementwiseBinaryBack(
      [=](float& a, float& b, float& c) {
        if (*(forwA + std::distance(forwB, &b)) >= b) {
          a += c;
        }
      },
      ga->buffer(),
      b->buffer(),
      gy->buffer(),
      iter_
    );
  }
  if (outputs[1]) {
    cpu::fill(outputs[1]->malloc(), outputs[1]->size(), 0);

    cpu::elementwiseBinaryBack(
      [=](float& a, float& b, float& c) {
        if (*(forwB + std::distance(forwA, &a)) >= a) {
          b += c;
        }
      },
      a->buffer(),
      gb->buffer(),
      gy->buffer(),
      iter_
    );
  }
}

}
