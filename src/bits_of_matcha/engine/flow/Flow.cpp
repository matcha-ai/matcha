#include "bits_of_matcha/engine/flow/Flow.h"
#include "bits_of_matcha/engine/chain/Tracer.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/engine/flow/Module.h"
#include "bits_of_matcha/engine/flow/ModuleForw.h"
#include "bits_of_matcha/engine/chain/passes/reduceToEffects.h"
#include "bits_of_matcha/engine/chain/passes/contractIdentities.h"
#include "bits_of_matcha/engine/chain/passes/flatten.h"
#include "bits_of_matcha/engine/chain/passes/check.h"

namespace matcha::engine {

Flow::Flow(const AnyOp& preimage)
  : preimage_(preimage)
  , refs_(0)
{}

Flow::Flow()
  : lastCalledModule_(nullptr)
  , refs_(0)
{}


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
  lastCalledModule_ = m;
  if (tracing()) {
    auto mop = new ModuleForw(m, ins);
    auto outs = mop->outputs.stdVector();
    dispatch(mop);
    return outs;
  } else {
    return m->run(ins);
  }
}

std::string Flow::getId(const std::vector<Frame>& frames) {
  std::string buffer;
  for (auto& frame: frames) buffer += frame.string();
  return buffer;
}

std::shared_ptr<Module> Flow::module(const std::vector<Tensor*>& tensors) {
  std::vector<Frame> frames;
  frames.reserve(tensors.size());
  for (auto tensor: tensors) frames.push_back(tensor->frame());
  return module(frames);
}

std::shared_ptr<Module> Flow::module(const std::vector<Frame>& frames) {
  std::string id = getId(frames);
  try {
    return modules_.at(id);
  } catch (std::out_of_range&) {
  }

  if (!hasPreimage()) throw std::runtime_error("missing Flow preimage");
  auto m = std::make_shared<Module>(preimage_, frames);
  m->pass(optimizer);

  modules_[id] = m;

  return m;
}

void Flow::optimizer(Chain& chain) {
//  check(chain);
//  flatten(chain);
//  check(chain);
  reduceToEffects(chain);
  contractIdentities(chain);
}

void Flow::build(const std::vector<Frame>& frames) {
  module(frames);
}

void Flow::ref() {
  refs_++;
}

void Flow::unref() {
  if (refs_ == 0)
    throw std::runtime_error("refs are already 0");
  refs_--;
  if (refs_ == 0)
    delete this;
}


}