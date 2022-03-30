#include "bits_of_matcha/Flow.h"
#include "bits_of_matcha/Sequence.h"
#include "bits_of_matcha/engine/flow/Flow.h"
#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/engine/Tensor.h"


namespace matcha {

Flow::Flow(Function function)
  : api_{API::Functional}
  , function_{std::move(function)}
  , internal_{nullptr}
  , grad{this}
{}

Flow::Flow()
  : api_{API::Subclassing}
  , internal_{nullptr}
  , grad{this}
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

  engine::Tracer tracer;
  Tuple tracerIns = tracer.open(ins);
  Tuple tracerOuts;

  try {
    switch (api_) {
    case API::Functional:
      tracerOuts = tracingFunctional(tracerIns);
      break;
    case API::Subclassing:
      tracerOuts = tracingSubclassing(tracerIns);
      break;
    }
  } catch (NotImplemented&) {
    throw std::invalid_argument("couldn't build flow for given signature");
  }

  tracer.close(tracerOuts);
  internal_ = tracer.collect();

  if (!internal_) throw std::runtime_error("flow build failed");
  internal_->compile();
}

Tuple Flow::tracingFunctional(const Tuple& ins) {
  if (std::holds_alternative<UnaryFn>(function_)) {
    if (ins.size() != 1) throw NotImplemented();
    return {std::get<UnaryFn>(function_)(ins[0])};
  }
  if (std::holds_alternative<BinaryFn>(function_)) {
    if (ins.size() != 2) throw NotImplemented();
    return {std::get<BinaryFn>(function_)(ins[0], ins[1])};
  }
  if (std::holds_alternative<TernaryFn>(function_)) {
    if (ins.size() != 3) throw NotImplemented();
    return {std::get<TernaryFn>(function_)(ins[0], ins[1], ins[2])};
  }
  if (ins.size() <= 3) throw NotImplemented();
  return std::get<NaryFn>(function_)(ins);
}

Tuple Flow::tracingSubclassing(const Tuple& ins) {
  try {
    init(ins);
    return run(ins);
  } catch (NotImplemented&) {
  }

  switch (ins.size()) {
  case 1:
    init(ins[0]);
    return {run(ins[0])};
  case 2:
    init(ins[0], ins[1]);
    return {run(ins[0], ins[1])};
  case 3:
    init(ins[0], ins[1], ins[2]);
    return {run(ins[0], ins[1], ins[2])};
  default:
    throw NotImplemented();
  }
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

Flow flow(const Flow::Function& function) {
  return Flow(function);
}

void Flow::init(const tensor& a)                                    {}
void Flow::init(const tensor& a, const tensor& b)                   {}
void Flow::init(const tensor& a, const tensor& b, const tensor& c)  {}
void Flow::init(const Tuple& tuple)                                 {}
tensor Flow::run(const tensor& a)                                   { throw NotImplemented(); }
tensor Flow::run(const tensor& a, const tensor& b)                  { throw NotImplemented(); }
tensor Flow::run(const tensor& a, const tensor& b, const tensor& c) { throw NotImplemented(); }
Tuple Flow::run(const Tuple& tuple)                                 { throw NotImplemented(); }



}