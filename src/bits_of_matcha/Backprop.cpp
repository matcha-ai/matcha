#include "bits_of_matcha/Backprop.h"
#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/engine/flow/module/Module.h"
#include "bits_of_matcha/engine/tensor/factories.h"

namespace matcha {

Backprop::Backprop(std::initializer_list<tensor*> wrt)
  : Backprop(std::vector(wrt))
{}

Backprop::Backprop(const std::vector<tensor*>& wrt)
  : interal_(nullptr)
  , wrt_(wrt)
{
  auto tracer = new engine::Tracer;
  interal_ = tracer;
  tracer->open({});
}

std::map<tensor*, tensor> Backprop::operator()(const tensor& root) {
  auto tracer = (engine::Tracer*) interal_;
  auto graph = tracer->close({root});
  delete tracer;
  interal_ = nullptr;

  auto module = new engine::Module(std::move(graph));

  auto wrtInternal = engine::deref(wrt_);
  std::map<engine::Tensor*, engine::Module::Partial> partials;
  auto ys = module->forward({}, partials, wrtInternal);

  auto d = engine::ones(ys[0]->shape());
  module->backward(partials, {d});

  std::map<tensor*, tensor> result;
  for (int i = 0; i < wrt_.size(); i++) {
    auto we = wrt_[i];
    auto wi = wrtInternal[i];
    auto& partial = partials[wi];
    engine::Module::accumulateGrads(partial, wi->shape());
    result[we] = ref(partial.first);
  }

  delete module;
  return result;
}

Backprop::~Backprop() {
  auto tracer = (engine::Tracer*) interal_;
  if (!tracer) return;
  auto graph = tracer->close({});
  auto module = new engine::Module(std::move(graph));
  module->forward({});
  delete module;
}

}