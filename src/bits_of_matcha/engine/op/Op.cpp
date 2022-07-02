#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/flow/Tracer.h"


namespace matcha::engine {

Op::Op(std::initializer_list<Tensor*> inputs)
  : Op(std::vector(inputs))
{}

Op::Op(const std::vector<Tensor*>& inputs)
  : inputs(inputs)
{
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
  if (Tracer::handleNewOp(op)) {
  } else {
    for (auto&& in: op->inputs) {
      auto&& preop = in->op();
      if (preop) {
        preop->init();
        preop->run();
        delete preop;
      }
    }
    op->init();
    op->run();
    delete op;
  }
}

}