#include "bits_of_matcha/engine/chain/Tracer.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/ops/Print.h"


namespace matcha::engine {

thread_local  std::stack<Tracer*> Tracer::tracings_ = {};

bool tracing() {
  return Tracer::active();
}

void incept(Op* op, Op* preop) {
  auto in = preop->inputs[0];

  auto it = std::find(op->inputs.begin(), op->inputs.end(), in);
  *it = preop->outputs[0];
  in->unreq();
  preop->outputs[0]->req();

  dispatch(preop);
  if (tracing()) {
    Tracer::handleNewTensor(preop->outputs[0]);
  }
  /*
  if (preop->inputs.size() != 1 || preop->outputs.size() != 1)
    throw std::runtime_error("pre-operation must have 1 input and 1 output");

  auto in = preop->inputs[0];
  in->unreq();

  auto it = std::find(op->inputs.begin(), op->inputs.end(), in);
  *it = preop->outputs[0];

  preop->outputs[0]->req();
//  preop->outputs[0]->req();

  auto* tracer = Tracer::get();
  if (tracer) {
    auto& chain = tracer->chain_;
    auto& cops = chain.ops;
    auto& cop = cops[cops.size() - 2];
    auto& cpreop = cops[cops.size() - 1];
//    if (cop != op || cpreop != preop)
//      throw std::runtime_error("only pre-operations created inside operation constructor can be incepted");

//    std::swap(cop, cpreop);
  } else {
    preop->init();
    preop->run();
    delete preop;
  }
   */
}

Chain trace(const fn& function, const std::vector<Frame>& frames) {
  Tracer tracer;
  tuple ins = tracer.open(frames);
  tuple outs = function(ins);
  return tracer.close(outs);
}

bool Tracer::active() {
  return !tracings_.empty();
}

Tracer* Tracer::get() {
  if (!active()) return nullptr;
  return tracings_.top();
}

Tracer::Tracer() {}
Tracer::~Tracer() {}

tuple Tracer::open(const std::vector<Frame>& frames) {
  ops::Print::claimCout();
  tracings_.push(this);
  tuple inputs;
  inputs.reserve(frames.size());
  for (auto& frame: frames) {
    auto in = new Tensor(frame);
    chain_.inputs.push_back(in);
    inputs.push_back(ref(in));
  }
  return inputs;
}

Chain Tracer::close(const tuple& outputs) {
  ops::Print::unclaimCout();
  for (auto& output: outputs) {
    auto out = deref(output);
    chain_.outputs.push_back(out);
  }

  if (tracings_.top() != this)
    throw std::runtime_error("tracing stack got corrupted");

  tracings_.pop();
  return std::move(chain_);
}

bool Tracer::handleNewOp(Op* op) {
  auto tracer = get();
  if (!tracer) return false;
  return tracer->addNewOp(op);
}

bool Tracer::handleNewTensor(Tensor* tensor) {
  auto tracer = get();
  if (!tracer) return false;
  return tracer->addNewTensor(tensor);
}

bool Tracer::handleOldTensor(Tensor* tensor) {
  auto tracer = get();
  if (!tracer) return false;
  return tracer->addNewTensor(tensor);
}

bool Tracer::addNewOp(Op* op) {
  chain_.ops.push_back(op);
  return true;
}

bool Tracer::addNewTensor(Tensor* tensor) {
  if (addedTensors_.contains(tensor)) return true;
  addedTensors_.insert(tensor);
  chain_.tensors.push_back(tensor);
  tensor->req();
  return true;
}

bool Tracer::addOldTensor(Tensor* tensor) {
  if (addedTensors_.contains(tensor)) return true;
  addedTensors_.insert(tensor);
  chain_.tensors.push_back(tensor);
  tensor->req();
  return true;
}

}