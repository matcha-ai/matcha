#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/ops/Print.h"


namespace matcha::engine {

thread_local  std::stack<Tracer*> Tracer::tracings_ = {};

bool tracing() {
  return Tracer::active();
}

std::unique_ptr<Graph> trace(const AnyOp& op, const std::vector<Frame>& frames) {
  Tracer tracer;
  tuple ins = tracer.open(frames);
  tuple outs;

  if (std::holds_alternative<UnaryOp>(op)) {
    outs = {std::get<UnaryOp>(op)(ins[0])};
  } else if (std::holds_alternative<BinaryOp>(op)) {
    outs = {std::get<BinaryOp>(op)(ins[0], ins[1])};
  } else if (std::holds_alternative<TernaryOp>(op)) {
    outs = {std::get<TernaryOp>(op)(ins[0], ins[1], ins[2])};
  } else if (std::holds_alternative<NaryOp>(op)) {
    outs = std::get<NaryOp>(op)(ins);
  } else {
    throw std::runtime_error("unexpected op variant");
  }
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
  graph_ = std::make_unique<Graph>();
  tracings_.push(this);
  tuple inputs;
  inputs.reserve(frames.size());
  for (auto& frame: frames) {
    auto in = new Tensor(frame);
    graph_->inputs.push_back(in);
    inputs.push_back(ref(in));
  }
  return inputs;
}

std::unique_ptr<Graph> Tracer::close(const tuple& outputs) {
  ops::Print::unclaimCout();
  for (auto& output: outputs) {
    auto out = deref(output);
    graph_->outputs.push_back(out);
  }

  if (tracings_.top() != this)
    throw std::runtime_error("tracing stack got corrupted");

  tracings_.pop();
  return std::move(graph_);
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
  graph_->ops.push_back(op);
  return true;
}

bool Tracer::addNewTensor(Tensor* tensor) {
  if (addedTensors_.contains(tensor)) return true;
  addedTensors_.insert(tensor);
  graph_->tensors.push_back(tensor);
  tensor->req();
  return true;
}

bool Tracer::addOldTensor(Tensor* tensor) {
  if (addedTensors_.contains(tensor)) return true;
  addedTensors_.insert(tensor);
  graph_->tensors.push_back(tensor);
  tensor->req();
  return true;
}

}