#include "bits_of_matcha/engine/ops/Lt.h"

namespace matcha::engine::ops {

Lt::Lt(Tensor* a, Tensor* b)
  : ElementwiseBinaryLogicalOp(a, b)
{}

OpMeta<Lt> Lt::meta {
  .name = "Lt",
};

void Lt::run() {
  if (isReal(inputs[0]))
    runCpuReal(std::less());
  else
    runCpuComplex([](auto a, auto b) { return a.real() < b.real(); });
}

}
