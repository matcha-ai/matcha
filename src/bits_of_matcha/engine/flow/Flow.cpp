#include "bits_of_matcha/engine/flow/Flow.h"
#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/engine/flow/compiler/Compiler.h"
#include "bits_of_matcha/engine/flow/graph/Graph.h"
#include "bits_of_matcha/engine/flow/graph/TensorMask.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

Flow::Flow()
  : hasOp_(false)
  , compile_{true}
{}

Flow::Flow(const AnyOp& op)
  : op_(op)
  , hasOp_(true)
  , compile_{true}
{}

bool Flow::hasOp() const {
  return hasOp_;
}

void Flow::setOp(const AnyOp& op) {
  op_ = op;
}

bool Flow::built() const {
  return !graph_.tensors.empty();
}

void Flow::build(const std::vector<Frame>& frames) {
  graph_ = engine::trace(op_, frames);
  compileJit();
}

void Flow::updateTasks() {
  if (!compile_) return;
  compile();
  compile_ = false;
}

void Flow::compile() {
  tasks_ = engine::compile(graph_, grads_);
}

std::vector<Tensor*> Flow::forward(const std::vector<Tensor*>& inputs) {
  if (!built()) throw std::runtime_error("flow is not built");
  updateTasks();

  if (inputs.size() != graph_.inputs.size()) {
    throw std::invalid_argument("incorrect number of inputs");
  }

  for (int i = 0; i < inputs.size(); i++) {
    if (inputs[i]->frame() != graph_.inputs[i]->frame()) {
      throw std::invalid_argument("input frame mismatch");
    }
  }

  auto outs = tasks_.forward(inputs);
  return outs;
}

void Flow::requireGrad(tensor* wrt) {
  grads_[wrt] = deref(wrt);
  compileJit();
}

void Flow::requireGrad(const std::vector<tensor*>& wrts) {
  for (auto wrt: wrts) requireGrad(wrt);
}

void Flow::unrequireGrad(tensor* wrt) {
  grads_.erase(wrt);
  compileJit();
}

void Flow::unrequireGrad(const std::vector<tensor*>& wrts) {
  for (auto wrt: wrts) unrequireGrad(wrt);
}

std::set<tensor*> Flow::requiredGrad() {
  std::set<tensor*> keys;
  for (auto [external, internal]: grads_) {
    keys.insert(external);
  }
  return keys;
}

void Flow::setRequiredGrad(const std::vector<tensor*>& wrts) {
  grads_.clear();
  for (auto wrt: wrts) requireGrad(wrt);
  compileJit();
}

std::map<tensor*, tensor> Flow::backward(Tensor* delta) {
  if (delta->frame() != tasks_.delta->frame()) throw std::runtime_error("delta frame mismatch");
  auto result = tasks_.backward(delta);
  return result;
}

void Flow::compileJit() {
  compile_ = true;
}

TensorMask Flow::getGradMask() {
  auto ctx = graph_.ctx();
  TensorMask grads(graph_);
  for (auto [external, internal]: grads_) {
    grads[internal] = true;
  }

  return grads;
}

matcha::Flow ref(Flow* internal) {
  auto p = (matcha::Flow*) &internal;
  matcha::Flow f = *p;
  return f;
}

Flow* deref(const matcha::Flow* external) {
  const matcha::Flow** flowpp = &external;
  auto internalppp = (Flow***)flowpp;
  return **internalppp;
}

Flow* deref(const matcha::Flow& external) {
  return deref(&external);
}

}