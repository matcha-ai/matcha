#include "bits_of_matcha/Backprop.h"
#include "bits_of_matcha/engine/chain/Tracer.h"
#include "bits_of_matcha/engine/flow/Module.h"
#include "bits_of_matcha/engine/autograd/Autograd.h"

namespace matcha {

struct Internal {
  Internal(const std::vector<tensor*>& wrt)
    : wrt_(wrt)
  {}

  std::vector<tensor*> wrt_;
  engine::Tracer tracer_;
};

Backprop::Backprop(std::initializer_list<tensor*> wrt)
  : Backprop(std::vector(wrt.begin(), wrt.end()))
{}

Backprop::Backprop(const std::vector<tensor*>& wrt) {\
  auto internal = new Internal(wrt);
  internal_ = internal;
  internal->tracer_.open({});
}

std::map<tensor*, tensor> Backprop::operator()(const tensor& root) {
  if (!internal_)
    throw std::runtime_error("you must schedule a new Autograd first");

  auto internal = (Internal*) internal_;
  auto graph = internal->tracer_.close({root});

  auto& wrt = internal->wrt_;
  auto op = new engine::Autograd(std::move(graph), engine::deref(wrt));

  if (op->outputs.size() != wrt.size())
    throw std::runtime_error("can't pair gradients");

  std::map<tensor*, tensor> gradients;
  for (int i = 0; i < wrt.size(); i++)
    gradients[wrt[i]] = ref(op->outputs[i]);

  engine::dispatch(op);
  internal_ = nullptr;
  delete internal;
  return gradients;
}

Backprop::~Backprop() {
  if (!internal_) return;
  auto internal = (Internal*) internal_;
  auto graph = internal->tracer_.close({});
  auto module = new engine::Module(std::move(graph));
  module->run({});

  delete internal;
}

}