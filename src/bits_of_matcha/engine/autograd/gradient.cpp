#include "bits_of_matcha/engine/autograd/gradient.h"
#include "bits_of_matcha/engine/autograd/Autograd.h"
#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/engine/flow/module/Module.h"
#include "bits_of_matcha/engine/tensor/factories.h"


namespace matcha::engine {

std::vector<Tensor*> gradient(const std::function<tensor ()>& function, const std::vector<Tensor*>& wrt) {
  auto fn = [&] (const tuple& ins) {
    return tuple{ function() };
  };

  auto graph = trace(fn, {});
  auto module = new Module(std::move(graph));
  std::map<Tensor*, Module::Partial> partials;
  auto ys = module->forward({}, partials, wrt);
  auto d = engine::ones(ys[0]->shape());
  module->backward(partials, {d});
  std::vector<Tensor*> result;
  result.reserve(wrt.size());
  for (auto w: wrt) {
    auto& partial = partials[w];
    Module::accumulateGrads(partial, w->shape());
    result.push_back(partial.first);
  }

  delete module;
  return result;
}

}