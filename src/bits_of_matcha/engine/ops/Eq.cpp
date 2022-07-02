#include "bits_of_matcha/engine/ops/Eq.h"

namespace matcha::engine::ops {

Eq::Eq(Tensor* a, Tensor* b)
  : ElementwiseBinaryLogicalOp(a, b)
{
}

OpMeta<Eq> Eq::meta {
  .name = "Eq",
};

void Eq::run() {
  runCPU(std::equal_to());
}

}
