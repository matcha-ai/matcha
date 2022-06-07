#include "bits_of_matcha/Flow.h"
#include "bits_of_matcha/engine/flow/Flow.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/flow/Tracer.h"

using matcha::engine::ref;
using matcha::engine::deref;
using Internal = matcha::engine::Flow*;

namespace matcha {

Flow::Flow(const AnyOp& op)
  : internal_(new engine::Flow(op))
{

}

Flow::Flow()
  : internal_(new engine::Flow())
{}

void Flow::requireGrad(tensor* wrt) {
  auto flow = Internal(internal_);
  flow->requireGrad(wrt);
}

void Flow::requireGrad(const std::vector<tensor*>& wrts) {
  for (auto& wrt: wrts) requireGrad(wrt);
}

void Flow::unrequireGrad(tensor* wrt) {
  auto flow = Internal(internal_);
  flow->unrequireGrad(wrt);
}

void Flow::unrequireGrad(const std::vector<tensor*>& wrts) {
  for (auto& wrt: wrts) unrequireGrad(wrt);
}

std::set<tensor*> Flow::requiredGrad() {
  auto flow = Internal(internal_);
  return flow->requiredGrad();
}

void Flow::setRequiredGrad(const std::vector<tensor*>& wrts) {
  auto flow = Internal(internal_);
  flow->setRequiredGrad(wrts);
}

std::map<tensor*, tensor> Flow::grad(const tensor& delta) {
  auto flow = Internal(internal_);
  return flow->backward(deref(delta));
}

tensor Flow::operator()(const tensor& a) {
  auto flow = Internal(internal_);

  if (flow->wantsOp()) {
    flow->setOp([&](const tensor& a) { init(a); return run(a); });
  }

  if (flow->wantsBuild()) {
    flow->build({a.frame()});
  }

  auto outs = flow->call({deref(a)});
  if (outs.size() != 1) throw std::runtime_error("incorrect output signature");
  return ref(outs[0]);
}

tensor Flow::operator()(const tensor& a, const tensor& b) {
  auto flow = Internal(internal_);

  if (flow->wantsOp()) {
    flow->setOp([&](const tensor& a, const tensor& b) { init(a); return run(a, b); });
  }

  if (flow->wantsBuild()) {
    flow->build({a.frame(), b.frame()});
  }

  auto outs = flow->call({deref(a), deref(b)});
  if (outs.size() != 1) throw std::runtime_error("incorrect output signature");
  return ref(outs[0]);
}

tensor Flow::operator()(const tensor& a, const tensor& b, const tensor& c) {
  auto flow = Internal(internal_);

  if (flow->wantsOp()) {
    flow->setOp([&](const tensor& a, const tensor& b, const tensor& c) { init(a); return run(a, b, c); });
  }

  if (flow->wantsBuild()) {
    flow->build({a.frame(), b.frame(), c.frame()});
  }

  auto outs = flow->call({deref(a), deref(b), deref(c)});
  if (outs.size() != 1) throw std::runtime_error("incorrect output signature");
  return ref(outs[0]);
}

tuple Flow::operator()(const tuple& inputs) {
  auto flow = Internal(internal_);

  if (flow->wantsOp()) {
    flow->setOp([&](const tuple& inputs) { init(inputs); return run(inputs); });
  }

  if (flow->wantsBuild()) {
    std::vector<Frame> frames;
    frames.reserve(inputs.size());
    for (auto&& input: inputs) frames.push_back(input.frame());
    flow->build(frames);
  }

  std::vector<engine::Tensor*> internals;
  internals.reserve(inputs.size());
  for (auto&& input: inputs) internals.push_back(deref(input));

  auto outs = flow->call(internals);

  tuple externals;
  externals.reserve(outs.size());
  for (auto&& out: outs) externals.push_back(ref(out));

  return externals;
}



void Flow::init(const tensor& a) {}
void Flow::init(const tensor& a, const tensor& b) {}
void Flow::init(const tensor& a, const tensor& b, const tensor& c) {}
void Flow::init(const tuple& inputs) {}

struct NotSubclassed : std::exception {};
tensor Flow::run(const tensor& a) { throw NotSubclassed(); }
tensor Flow::run(const tensor& a, const tensor& b) { throw NotSubclassed(); }
tensor Flow::run(const tensor& a, const tensor& b, const tensor& c) { throw NotSubclassed(); }
tuple Flow::run(const tuple& inputs) { throw NotSubclassed(); }

}