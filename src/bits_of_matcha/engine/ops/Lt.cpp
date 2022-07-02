#include "bits_of_matcha/engine/ops/Lt.h"

namespace matcha::engine::ops {

Lt::Lt(Tensor* a, Tensor* b)
  : ElementwiseBinaryLogicalOp(a, b)
{}

OpMeta<Lt> Lt::meta {
  .name = "Lt",
};

void Lt::run() {
  runCPU(std::less());
}

}
