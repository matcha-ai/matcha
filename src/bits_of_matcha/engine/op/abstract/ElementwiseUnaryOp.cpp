#include "bits_of_matcha/engine/op/abstract/ElementwiseUnaryOp.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {


ElementwiseUnaryOp::ElementwiseUnaryOp(Tensor* a)
: Op{a}
{
  size_ = a->size();
  outputs.add(this, a->frame());
}

ElementwiseUnaryOp::~ElementwiseUnaryOp() {
}

}
