#include "bits_of_matcha/engine/ops/Neq.h"

namespace matcha::engine::ops {

Neq::Neq(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b)
{}

OpMeta<Neq> Neq::meta {
  .name = "Neq",
};

void Neq::run() {
  outputs[0]->malloc();
  runCPU([](float a, float b) { return (float) a != b; });
}

}
