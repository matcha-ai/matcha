#include "bits_of_matcha/Flow.h"
#include "bits_of_matcha/Sequence.h"
#include "bits_of_matcha/engine/flow/Flow.h"
#include "bits_of_matcha/engine/flow/FlowTracer.h"
#include "bits_of_matcha/engine/Tensor.h"


namespace matcha {

Flow::Flow(Function function)
  : api_{API::Functional}
  , function_{std::move(function)}
  , internal_{nullptr}
{}

Flow::Flow(std::initializer_list<UnaryFn> sequence)
  : api_{API::Sequential}
  , function_{Sequence(sequence)}
  , internal_{nullptr}
{}

Flow::Flow()
  : api_{API::Subclassing}
  , internal_{nullptr}
{}

bool Flow::built() {
  return internal_;
}

void Flow::build(const std::vector<tensor> ins) {
  std::vector<Frame> frames;
  frames.reserve(ins.size());
  for (auto& in: ins) {
    frames.emplace_back(in.dtype(), in.shape());
  }
  build(frames);
}

void Flow::build(const std::vector<Frame> ins) {
  if (built()) throw std::runtime_error("flow has been already built");
}


tensor Flow::operator()(const tensor& a) {
  if (!built()) build({a});
  return getFunction<UnaryFn>()(a);
}

tensor Flow::operator()(const tensor& a, const tensor& b) {
  if (!built()) build({a, b});
  return getFunction<BinaryFn>()(a, b);
}

tensor Flow::operator()(const tensor& a, const tensor& b, const tensor& c) {
  if (!built()) build({a, b, c});
  return getFunction<TernaryFn>()(a, b, c);
}

Tuple Flow::operator()(const Tuple& tuple) {
  if (!built()) build(tuple);
  return getFunction<NaryFn>()(tuple);
}

void Flow::init(const tensor& a)                                    { throw InactiveInit(); }
void Flow::init(const tensor& a, const tensor& b)                   { throw InactiveInit(); }
void Flow::init(const tensor& a, const tensor& b, const tensor& c)  { throw InactiveInit(); }
void Flow::init(const Tuple& tuple)                                 { throw InactiveInit(); }
tensor Flow::run(const tensor& a)                                   { throw InactiveRun(); }
tensor Flow::run(const tensor& a, const tensor& b)                  { throw InactiveRun(); }
tensor Flow::run(const tensor& a, const tensor& b, const tensor& c) { throw InactiveRun(); }
Tuple Flow::run(const Tuple& tuple)                                 { throw InactiveRun(); }



}