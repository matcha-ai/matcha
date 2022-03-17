#include "bits_of_matcha/engine/Fold.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine::fn {


Fold::Fold(Tensor* a)
: Node{a}
, dev_{CPU}
{
  size_ = a->size();
  createOut(Float, {});
}

Fold::~Fold() {
}

void Fold::use(const Device& device) {
  Computation comp {
  .type = Computation::Fold,
  .cost = size_
  };
  auto dev = device.get(comp);
}

const Device::Concrete* Fold::device() const {
  return &dev_;
}


}