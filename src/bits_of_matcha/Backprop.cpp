#include "bits_of_matcha/Backprop.h"
#include "bits_of_matcha/engine/chain/Tracer.h"
#include "bits_of_matcha/engine/chain/Module.h"
#include "bits_of_matcha/engine/autograd/backprop.h"
#include "bits_of_matcha/engine/chain/passes/check.h"
#include "bits_of_matcha/engine/chain/executors/SinglecoreExecutor.h"

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
  auto chain = internal->tracer_.close({root});

  auto&& wrt = internal->wrt_;
  engine::backprop(chain, engine::deref(wrt));
//  check(chain);

  auto executor = std::make_shared<engine::SinglecoreExecutor>(std::move(chain));
  auto op = new engine::Module({}, std::move(executor));

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
  auto chain = internal->tracer_.close({});
  auto executor = std::make_shared<engine::SinglecoreExecutor>(std::move(chain));
  executor->run({});

  delete internal;
}

}