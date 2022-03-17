#include "bits_of_matcha/engine/ElementwiseBinary.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine::fn {


ElementwiseBinary::ElementwiseBinary(Tensor* a, Tensor* b)
  : Node{a, b}
  , dev_{CPU}
{
  iter_ = ElementwiseBinaryIteration(a->shape(), b->shape());
  createOut(iter_.scalar == 1 ? *a->frame() : *b->frame());
}

ElementwiseBinary::~ElementwiseBinary() {
}

void ElementwiseBinary::use(const Device& device) {
  Computation comp {
    .type = Computation::ElementwiseBinary,
    .cost = iter_.size
  };
  auto dev = device.get(comp);
}

const Device::Concrete* ElementwiseBinary::device() const {
  return &dev_;
}

}