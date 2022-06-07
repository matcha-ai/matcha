#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/ops/Print.h"


namespace matcha::engine {

bool Tracer::handleOp(Op* op) {
  if (!current()) return false;
  if (current()->ops_.contains(op)) return false;
  if (op->ctx().traced()) throw std::runtime_error("op is already traced");
  op->ctx().setTraced();
  current()->ops_.insert(op);
  current()->graph_.ops.push_back(op);
  for (auto in: op->inputs) handleOldTensor(in);
  return true;
}

bool Tracer::handleNewTensor(Tensor* tensor) {
  if (!current()) return false;
  if (current()->tensors_.contains(tensor)) return false;
  tensor->ctx().setMode(TensorCtx::Constant);
  current()->tensors_.insert(tensor);
  current()->graph_.tensors.push_back(tensor);
  tensor->ref();
  return true;
}

bool Tracer::handleOldTensor(Tensor* tensor) {
  if (!current()) return false;
  if (current()->tensors_.contains(tensor)) return false;

  switch (tensor->ctx().mode()) {
  case TensorCtx::Untraced:
    tensor->ctx().setMode(TensorCtx::Variable);
    break;
  case TensorCtx::Constant:
    throw std::runtime_error("tensor is a traced constant");
  case TensorCtx::Variable:
    break;
  }

  current()->tensors_.insert(tensor);
  current()->graph_.tensors.push_back(tensor);
  tensor->ref();
  return true;
}

Tracer* Tracer::current() {
  if (stack_.empty()) return nullptr;
  return stack_.top();
}

Tracer::Tracer()
{}

tuple Tracer::open(const std::vector<Frame>& frames) {
//  print("tracing open");
  stack_.push(this);

  tuple inputs;
  inputs.reserve(frames.size());
  for (auto&& frame: frames) {
    auto t = new Tensor(frame);
    inputs.push_back(ref(t));
    graph_.inputs.push_back(t);
  }

  ops::Print::claimCout();
  return inputs;
}

void Tracer::close(const tuple& outputs) {
//  print("tracing close");
  ops::Print::unclaimCout();
  auto printRest = new ops::Print("", false);
  engine::collect(printRest);

  for (auto& out: outputs) {
    auto t = deref(out);
    graph_.outputs.push_back(t);
  }

  stack_.pop();
}

Graph Tracer::collect() {
  return std::move(graph_);
}

std::stack<Tracer*> Tracer::stack_ = {};

Graph trace(const AnyOp& anyOp, const std::vector<Frame>& frames) {
  Tracer tracer;
  tuple inputs = tracer.open(frames);
  tuple outputs;

  if (std::holds_alternative<UnaryOp>(anyOp)) {

    if (inputs.size() != 1) throw std::runtime_error("wrong number of frames");
    auto op = std::get<UnaryOp>(anyOp);
    outputs = {op(inputs[0])};

  } else if (std::holds_alternative<BinaryOp>(anyOp)) {

    if (inputs.size() != 2) throw std::runtime_error("wrong number of frames");
    auto op = std::get<BinaryOp>(anyOp);
    outputs = {op(inputs[0], inputs[1])};

  } else if (std::holds_alternative<TernaryOp>(anyOp)) {

    if (inputs.size() != 3) throw std::runtime_error("wrong number of frames");
    auto op = std::get<TernaryOp>(anyOp);
    outputs = {op(inputs[0], inputs[1], inputs[2])};

  } else if (std::holds_alternative<NaryOp>(anyOp)) {

    auto op = std::get<NaryOp>(anyOp);
    outputs = {op(inputs)};

  }

  tracer.close(outputs);
  return tracer.collect();
}

}