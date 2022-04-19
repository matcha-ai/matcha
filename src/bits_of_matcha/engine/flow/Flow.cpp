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
  TensorMask grads = getGradMask();
  tasks_ = engine::compile(graph_, grads);
}

std::vector<Tensor*> Flow::forward(const std::vector<Tensor*>& inputs) {
  if (!built()) throw std::runtime_error("flow is not built");
  updateTasks();
  auto outs = tasks_.forward(inputs);
  return outs;
}

void Flow::requireGrad(Tensor* wrt) {
  grads_.insert(wrt);
  compileJit();
}

void Flow::requireGrad(const std::vector<Tensor*>& wrts) {
  for (auto wrt: wrts) requireGrad(wrt);
}

void Flow::unrequireGrad(Tensor* wrt) {
  grads_.erase(wrt);
  compileJit();
}

void Flow::unrequireGrad(const std::vector<Tensor*>& wrts) {
  for (auto wrt: wrts) unrequireGrad(wrt);
}

const std::set<Tensor*>& Flow::requiredGrad() {
  return grads_;
}

void Flow::setRequiredGrad(const std::vector<Tensor*>& wrts) {
  grads_.clear();
  for (auto wrt: wrts) requireGrad(wrt);
  compileJit();
}

void Flow::compileJit() {
  compile_ = true;
}

TensorMask Flow::getGradMask() {
  auto ctx = graph_.ctx();
  TensorMask grads(graph_);
  for (auto wrt: grads_) {
    grads[wrt] = true;
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