#include "bits_of_matcha/engine/flow/FlowFunctionContext.h"
#include "bits_of_matcha/engine/flow/FlowTracer.h"
#include "bits_of_matcha/engine/flow/Flow.h"
#include "bits_of_matcha/engine/Tensor.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {


FlowFunctionContext::FlowFunctionContext(const Function& function)
  : function_{function}
  , internal_{nullptr}
{}

bool FlowFunctionContext::built() {
  return internal_;
}

bool FlowFunctionContext::building() {
  return FlowTracer::current();
}

void FlowFunctionContext::build(const std::vector<const tensor*>& ins) {
  std::vector<Frame> frames;
  frames.reserve(ins.size());
  for (auto in: ins) {
    frames.emplace_back(in->dtype(), in->shape());
  }

  FlowTracer tracker;
  auto trackerIns = tracker.open(frames);
  Tuple outs;

  if (std::holds_alternative<UnaryFn>(function_)) {
    outs = {std::get<UnaryFn>(function_)(trackerIns[0])};
  } else if (std::holds_alternative<BinaryFn>(function_)){
    outs = {std::get<BinaryFn>(function_)(trackerIns[0], trackerIns[1])};
  } else if (std::holds_alternative<TernaryFn>(function_)){
    outs = {std::get<TernaryFn>(function_)(trackerIns[0], trackerIns[1], trackerIns[2])};
  } else if (std::holds_alternative<NaryFn>(function_)){
    outs = {std::get<NaryFn>(function_)(trackerIns)};
  }

  tracker.close(outs);
  internal_ = tracker.get();
}

std::vector<tensor> FlowFunctionContext::run(const std::vector<const tensor*>& inputs) {
  if (!built()) throw std::runtime_error("flow has not been built yet");

  std::vector<Tensor*> ins;
  ins.reserve(inputs.size());
  for (auto input: inputs) {
    ins.push_back(deref(input));
  }

  auto outs = internal_->run(ins);

  std::vector<tensor> outputs;
  outputs.reserve(outs.size());
  for (auto out: outs) {
    outputs.emplace_back(out);
  }
  return outputs;
}



}