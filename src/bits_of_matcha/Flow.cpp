#include "bits_of_matcha/Flow.h"
#include "bits_of_matcha/engine/flow/Flow.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"

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

void Flow::requireGrad(const tensor& wrt) {
  auto flow = Internal(internal_);
  flow->requireGrad(deref(wrt));
}

void Flow::requireGrad(const tuple& wrts) {
  for (auto& wrt: wrts) requireGrad(wrt);
}

void Flow::unrequireGrad(const tensor& wrt) {
  auto flow = Internal(internal_);
  flow->unrequireGrad(deref(wrt));
}

void Flow::unrequireGrad(const tuple& wrts) {
  for (auto& wrt: wrts) unrequireGrad(wrt);
}

tensor Flow::operator()(const tensor& a) {
  auto flow = Internal(internal_);

  if (!flow->built()) {
    if (!flow->hasOp()) {
      flow->setOp([&](const tensor& a) { init(a); return run(a); });
    }
    flow->build({deref(a)->frame()});
  }

  auto outs = flow->forward({deref(a)});
  if (outs.size() != 1) throw std::runtime_error("incorrect output signature");
  return ref(outs[0]);
}

tensor Flow::operator()(const tensor& a, const tensor& b) {

}

tensor Flow::operator()(const tensor& a, const tensor& b, const tensor& c) {

}

tuple Flow::operator()(const tuple& inputs) {

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