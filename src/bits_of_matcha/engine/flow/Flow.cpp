#include "bits_of_matcha/engine/flow/Flow.h"
#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/engine/flow/module/Module.h"
#include "bits_of_matcha/engine/flow/module/ModuleForw.h"

namespace matcha::engine {

Flow::Flow(const AnyOp& preimage)
  : preimage_(preimage)
{}

Flow::Flow()
  : lastCalledModule_(nullptr)
{
}

bool Flow::hasPreimage() const {
  if (std::holds_alternative<UnaryOp>(preimage_)) {
    return (bool) std::get<UnaryOp>(preimage_);
  } else if (std::holds_alternative<BinaryOp>(preimage_)) {
    return (bool) std::get<BinaryOp>(preimage_);
  } else if (std::holds_alternative<TernaryOp>(preimage_)) {
    return (bool) std::get<TernaryOp>(preimage_);
  } else if (std::holds_alternative<NaryOp>(preimage_)) {
    return (bool) std::get<NaryOp>(preimage_);
  } else {
    return false;
  }
}

void Flow::setPreimage(const AnyOp& op) {
  preimage_= op;
}

std::vector<Tensor*> Flow::call(const std::vector<Tensor*>& ins) {
  auto m = module(ins);
  if (tracing()) {
    auto mop = new ModuleForw(m, ins);
    auto outs = mop->outputs;
    send(mop);
    return outs.stdVector();
  } else {
    lastCalledModule_ = m;
    auto outs = m->forward(ins);
    return outs;
  }
}

std::string Flow::getId(const std::vector<Frame>& frames) {
  std::string buffer;
  for (auto& frame: frames) buffer += frame.string();
  return buffer;
}

Module* Flow::module(const std::vector<Tensor*>& tensors) {
  std::vector<Frame> frames;
  frames.reserve(tensors.size());
  for (auto tensor: tensors) frames.push_back(tensor->frame());
  return module(frames);
}

Module* Flow::module(const std::vector<Frame>& frames) {
  std::string id = getId(frames);
  try {
    return modules_.at(id);
  } catch (std::out_of_range&) {
  }

  if (!hasPreimage()) throw std::runtime_error("missing Flow preimage");
  auto graph = trace(preimage_, frames);
  auto m = new Module(std::move(graph));
  modules_[id] = m;
  return m;
}

void Flow::build(const std::vector<Frame>& frames) {
  module(frames);
}


}