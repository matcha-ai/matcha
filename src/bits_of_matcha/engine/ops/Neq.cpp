#include "bits_of_matcha/engine/ops/Neq.h"

namespace matcha::engine::ops {

Neq::Neq(Tensor* a, Tensor* b)
  : ElementwiseBinaryLogicalOp(a, b)
{}

Reflection<Neq> Neq::reflection {
  .name = "Neq",
};

void Neq::run() {
  runCpu(std::not_equal_to());
}

}
