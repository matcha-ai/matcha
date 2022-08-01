#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/chain/Tracer.h"


namespace matcha::engine {

Op::Op(std::initializer_list<Tensor*> inputs)
  : Op(std::vector(inputs))
{}

Op::Op(const std::vector<Tensor*>& inputs)
  : inputs(inputs)
{}

Op::~Op() {
//  print("deleting op");
}

void Op::init() {}

void Op::run() {

}

void dispatch(Op* op) {
  if (!tracing()) {
    op->init();
    op->run();
    delete op;
  } else {
    Tracer::handleNewOp(op);
    for (auto in: op->inputs) Tracer::handleOldTensor(in);
  }
}

}