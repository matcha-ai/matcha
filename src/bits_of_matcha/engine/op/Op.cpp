#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/flow/Tracer.h"


namespace matcha::engine {

Op::Op(std::initializer_list<Tensor*> inputs)
  : Op(std::vector(inputs))
{}

Op::Op(const std::vector<Tensor*>& inputs)
  : inputs(inputs)
  , ctx_(this)
{}

void Op::init() {
  for (auto out: outputs) out->malloc();
}

void Op::run() {

}

OpCtx& Op::ctx() {
  return ctx_;
}

void collect(Op* op) {
  if (Tracer::handleOp(op)) {

  } else {
    op->init();
    op->run();
    delete op;
  }
}

}