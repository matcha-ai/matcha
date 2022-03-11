#include "bits_of_matcha/engine/elementwiseUnary.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine::fn {


ElementwiseUnary::ElementwiseUnary(Tensor* a)
: Node{a}
, dev_{CPU}
{
  size_ = a->size();
  createOut(*a->frame());
}

ElementwiseUnary::~ElementwiseUnary() {
}

void ElementwiseUnary::use(const Device& device) {
  Computation comp {
    .type = Computation::ElementwiseUnary,
    .cost = size_
  };
  auto dev = device.get(comp);
}

const Device::Concrete* ElementwiseUnary::device() const {
  return &dev_;
}

}
