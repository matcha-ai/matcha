#include "bits_of_matcha/engine/autograd/Autograd.h"
#include "bits_of_matcha/engine/flow/Module.h"
#include "bits_of_matcha/engine/autograd/GradientTape.h"


namespace matcha::engine {

Tensor* getRoot(const std::shared_ptr<Module>& module) {
  auto& chain = module->chain();
  if (chain.outputs.size() != 1)
    throw std::runtime_error("chain to be differentiated must have exactly one output");
  return chain.outputs[0];
}

Autograd::Autograd(std::shared_ptr<Module> module, const std::vector<Tensor*>& wrt)
  : Op{getRoot(module)}
  , module_(std::move(module))
  , wrt_(wrt)
{
  for (auto&& w: wrt)
    outputs.add(this, Float, w->shape());
}


Autograd::Autograd(Chain chain, const std::vector<Tensor*>& wrt)
  : Autograd(std::make_shared<Module>(std::move(chain)), wrt)
{}

OpMeta<Autograd> Autograd::meta {
  .name = "Autograd"
};

void Autograd::run() {
//  GradientTape gt(module_->chain(), wrt_);
//  auto altitudes = gt.forward({});
//  gt.backward(outputs.stdVector());
  auto altitudes = module_->run({});
  for (auto&& alt: altitudes) delete alt;
  for (auto&& out: outputs) out->malloc();
}

}