#include "bits_of_matcha/engine/ops/Ge.h"

namespace matcha::engine::ops {

Ge::Ge(Tensor* a, Tensor* b)
  : ElementwiseBinaryLogicalOp(a, b)
{}

OpMeta<Ge> Ge::meta {
  .name = "Ge",
};

void Ge::run() {
  if (isReal(inputs[0]))
    runCpuReal(std::greater_equal());
  else
    runCpuComplex([](auto a, auto b) { return a.real() >= b.real(); });
}

}
