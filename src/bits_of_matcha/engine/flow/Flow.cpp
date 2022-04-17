#include "bits_of_matcha/engine/flow/Flow.h"
#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/engine/flow/compiler/Compiler.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

Flow::Flow()
  : hasOp_(false)
{}

Flow::Flow(const AnyOp& op)
  : op_(op)
  , hasOp_(true)
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
  TensorMask grads(graph_);

  {
    auto guard = graph_.ctx();
    grads[graph_.inputs[0]] = true;
  }
  tasks_ = engine::compile(graph_, grads);
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