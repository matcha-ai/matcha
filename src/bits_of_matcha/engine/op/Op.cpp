#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/lambda/Tracer.h"


namespace matcha::engine {

Op::Op(std::initializer_list<Tensor*> inputs)
  : Op(std::vector(inputs))
{}

Op::Op(const std::vector<Tensor*>& inputs)
  : inputs(inputs)
{
  for (auto&& in: inputs)
    if (in) in->req();
}

Op::~Op() {
  for (auto&& in: inputs)
    if (in) in->unreq();

  for (auto&& out: outputs)
    if (out) out->setOp(nullptr);
}

void Op::init() {}

void Op::run() {

}

Tensor* Op::addOutput(const Frame& frame) {
  return addOutput(new Tensor(frame));
}

Tensor* Op::addOutput(const Dtype& dtype, const Shape& shape) {
  return addOutput(new Tensor(dtype, shape));
}

Tensor* Op::addOutput(Tensor* tensor) {
  outputs.push_back(tensor);
  if (tensor) tensor->setOp(this);
  return tensor;
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