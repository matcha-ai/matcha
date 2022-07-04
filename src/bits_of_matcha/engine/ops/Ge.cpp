#include "bits_of_matcha/engine/ops/Ge.h"

namespace matcha::engine::ops {

Ge::Ge(Tensor* a, Tensor* b)
  : ElementwiseBinaryLogicalOp(a, b)
{}

OpMeta<Ge> Ge::meta {
  .name = "Ge",
};

void Ge::run() {
  runCPU([](auto a, auto b) { return a >= b; });
}

}
