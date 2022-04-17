#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {


ElementwiseBinaryOp::ElementwiseBinaryOp(Tensor* a, Tensor* b)
  : Op{a, b}
//  , dev_{CPU}
{
  iter_ = ElementwiseBinaryIteration(a->shape(), b->shape());
  outputs.add(this, iter_.scalar == 1 ? a->frame() : b->frame());
}

ElementwiseBinaryOp::~ElementwiseBinaryOp() {
}

}