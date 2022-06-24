#include "bits_of_matcha/engine/ops/Lt.h"

namespace matcha::engine::ops {

Lt::Lt(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b)
{}

OpMeta<Lt> Lt::meta {
  .name = "Lt",
};

void Lt::run() {
  outputs[0]->malloc();
  runCPU(std::less<float>());
}

}
