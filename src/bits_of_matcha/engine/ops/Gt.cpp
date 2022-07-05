#include "bits_of_matcha/engine/ops/Gt.h"

namespace matcha::engine::ops {

Gt::Gt(Tensor* a, Tensor* b)
  : ElementwiseBinaryLogicalOp(a, b)
{}

OpMeta<Gt> Gt::meta {
  .name = "Gt",
};

void Gt::run() {
  if (isReal(inputs[0]))
    runCpuReal(std::greater());
  else
    runCpuComplex([](auto a, auto b) { return a.real() > b.real(); });
}

}
