#include "bits_of_matcha/engine/ops/Le.h"

namespace matcha::engine::ops {

Le::Le(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b)
{}

OpMeta<Le> Le::meta {
  .name = "Le",
};

void Le::run() {
  outputs[0]->malloc();
  runCPU(std::less_equal<float>());
}

}
