#include "bits_of_matcha/engine/ops/Eq.h"

namespace matcha::engine::ops {

Eq::Eq(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b)
{}

OpMeta<Eq> Eq::meta {
  .name = "Eq",
};

void Eq::run() {
  outputs[0]->malloc();
  runCPU([](float a, float b) { return (float) a == b; });
}

}
