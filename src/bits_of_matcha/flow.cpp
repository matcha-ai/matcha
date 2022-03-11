#include "bits_of_matcha/flow.h"
#include "bits_of_matcha/print.h"
#include "bits_of_matcha/engine/flow.h"
#include "bits_of_matcha/engine/flowContext.h"


namespace matcha {

Flow::FromFn::FromFn(const UnaryFn& fn)
  : fn_{fn}
  , ins_{1}
  , outs_{1}
{}

Flow::FromFn::FromFn(const BinaryFn& fn)
  : fn_{fn}
  , ins_{2}
  , outs_{1}
{}

Flow::FromFn::FromFn(const TernaryFn& fn)
  : fn_{fn}
  , ins_{3}
  , outs_{0}
{}


Flow::Flow(UnaryFn fn)
  : flow_{nullptr}
  , fn_{fn}
{
//  // create query tensor
  flow_ = Tensor::flowQuery(fn);
  if (!flow_) {
    throw std::invalid_argument("given Fn is not a Flow; you can create one from the function by using Flow::init with it");
  }
}

Flow::Flow(FromFn fn)
  : flow_{nullptr}
{

}

Flow Flow::init(UnaryFn fn) {
  auto* flow = Tensor::flowQuery(fn);
  if (flow) {
    return Flow(fn);
  }

  UnaryFn lambda = [&, fn](const Tensor& x) {
    flow_init(lambda, x);
    return fn(x);
  };

  flow = Tensor::flowQuery(lambda);
  if (flow) {
    return Flow(lambda);
  }

  throw std::invalid_argument("could not make it a flow");
}

Flow Flow::load(const std::string& file) {
  return Flow(0);
}

Tensor Flow::operator()(const Tensor& a) {
  return fn_(a);
}

bool Flow::built() {
  return flow_->built();
}

void Flow::save(const std::string& file) {
  if (!built()) throw std::runtime_error("flow has not been built yet");
}

float Flow::cost() {
  if (!built()) throw std::runtime_error("flow has not been built yet");
  return 0;
}

void Flow::use(const Device& device) {
  if (!built()) throw std::runtime_error("flow has not been built yet");
}

void Flow::use(const Device::Strategy& strategy) {
  use(Device{strategy});
}

}
