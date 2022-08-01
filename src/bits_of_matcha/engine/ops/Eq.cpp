#include "bits_of_matcha/engine/ops/Eq.h"

namespace matcha::engine::ops {

Eq::Eq(Tensor* a, Tensor* b)
  : ElementwiseBinaryLogicalOp(a, b)
{
}

Reflection<Eq> Eq::reflection {
  .name = "Eq",
};

void Eq::run() {
  runCpu(std::equal_to());
}

}
