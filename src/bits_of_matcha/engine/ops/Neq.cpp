#include "bits_of_matcha/engine/ops/Neq.h"

namespace matcha::engine::ops {

Neq::Neq(Tensor* a, Tensor* b)
  : ElementwiseBinaryLogicalOp(a, b)
{}

OpMeta<Neq> Neq::meta {
  .name = "Neq",
};

void Neq::run() {
  runCPU(std::not_equal_to());
}

}
