#include "bits_of_matcha/engine/ops/Le.h"

namespace matcha::engine::ops {

Le::Le(Tensor* a, Tensor* b)
  : ElementwiseBinaryLogicalOp(a, b)
{}

OpMeta<Le> Le::meta {
  .name = "Le",
};

void Le::run() {
  if (isReal(inputs[0]))
    runCpuReal(std::less_equal());
  else
    runCpuComplex([](auto a, auto b) { return a.real() <= b.real(); });
}

}
