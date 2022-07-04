#include "bits_of_matcha/engine/ops/Le.h"

namespace matcha::engine::ops {

Le::Le(Tensor* a, Tensor* b)
  : ElementwiseBinaryLogicalOp(a, b)
{}

OpMeta<Le> Le::meta {
  .name = "Le",
};

void Le::run() {
  runCPU([](auto a, auto b) { return a <= b; });
}

}
