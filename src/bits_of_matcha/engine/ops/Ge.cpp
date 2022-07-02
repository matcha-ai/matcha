#include "bits_of_matcha/engine/ops/Ge.h"

namespace matcha::engine::ops {

Ge::Ge(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b)
{}

OpMeta<Ge> Ge::meta {
  .name = "Ge",
};

void Ge::run() {
  runCPU(std::greater_equal());
}

}