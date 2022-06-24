#include "bits_of_matcha/engine/ops/Gt.h"

namespace matcha::engine::ops {

Gt::Gt(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b)
{}

OpMeta<Gt> Gt::meta {
  .name = "Gt",
};

void Gt::run() {
  outputs[0]->malloc();
  runCPU(std::greater<float>());
}

}
