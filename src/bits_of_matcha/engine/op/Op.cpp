#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/flow/Tracer.h"


namespace matcha::engine {

Op::Op(std::initializer_list<Tensor*> inputs)
  : Op(std::vector(inputs))
{}

Op::Op(const std::vector<Tensor*>& inputs)
  : inputs(inputs)
{
  Tracer::handleNewOp(this);
  for (auto in: inputs) Tracer::handleOldTensor(in);
}

Op::~Op() {
//  print("deleting op");
}

void Op::init() {
  for (auto out: outputs) out->malloc();
}

void Op::run() {

}

void send(Op* op) {
  if (!tracing()) {
    op->init();
    op->run();
    delete op;
  }
}

}