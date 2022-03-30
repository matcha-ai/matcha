#include "bits_of_matcha/engine/flow/FlowFunctionContext.h"
#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/engine/flow/Flow.h"
#include "bits_of_matcha/engine/Tensor.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {


FlowFunctionContext::FlowFunctionContext(const Function& function)
  : flow_{function}
{}

bool FlowFunctionContext::built() {
  return flow_.built();
}

bool FlowFunctionContext::building() {
  return Tracer::current();
}

void FlowFunctionContext::build(const std::vector<const tensor*>& ins) {
  std::vector<Frame> frames;
  frames.reserve(ins.size());
  for (auto in: ins) {
    frames.emplace_back(in->dtype(), in->shape());
  }
  flow_.build(frames);
}

std::vector<tensor> FlowFunctionContext::run(const std::vector<const tensor*>& inputs) {
  Flow* internal = flow_.internal_;
  return {};
}



}