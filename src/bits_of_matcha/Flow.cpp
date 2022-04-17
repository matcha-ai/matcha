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

tensor Flow::operator()(const tensor& a) {
  auto flow = Internal(internal_);

  if (!flow->built()) {
    if (!flow->hasOp()) {
      flow->setOp([&](const tensor& a) { return operator()(a); });
    }
    flow->build({deref(a)->frame()});
  }
  return 0;
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