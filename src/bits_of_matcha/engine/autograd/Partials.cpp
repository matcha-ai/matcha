#include "bits_of_matcha/engine/autograd/Partials.h"
#include "bits_of_matcha/engine/autograd/AccumulateGrads.h"
#include "bits_of_matcha/engine/ops/Require.h"
#include "bits_of_matcha/engine/tensor/factories.h"


namespace matcha::engine {

Partials::Partials(Lambda& lambda, const std::vector<Tensor*>& wrt)
  : lambda_(lambda)
{
  for (auto&& w: wrt) {
//    std::cerr << "WRT: " << w << std::endl;
    partials_[w] = {nullptr, {}};
  }

  for (auto&& op: lambda.ops) {
    if (!needs(op)) continue;
    for (auto&& out: op->outputs) {
      partials_[out] = {nullptr, {}};
//      std::cerr << "WRT: " << out << std::endl;
    }
  }

  auto req = new ops::Require(lambda.outputs);
  lambda_.ops.push_back(req);
  for (auto&& out: lambda.outputs) {
    auto root = engine::ones(out->shape());
    partials_[out] = {root, {}};
    root->req();
    lambda_.tensors.push_back(root);
  }
}

auto Partials::needs(Op* op) const -> bool {
  for (auto&& in: op->inputs)
    if (needs(in))
      return true;

  return false;
}

auto Partials::needs(Tensor* tensor) const -> bool {
  return partials_.contains(tensor);
}

auto Partials::needs(const std::vector<Op*>& ops) const -> std::vector<bool> {
  std::vector<bool> result;
  result.reserve(ops.size());
  for (auto&& op: ops) result.push_back(needs(op));
  return result;
}

auto Partials::needs(const std::vector<Tensor*>& tensors) const -> std::vector<bool> {
  std::vector<bool> result;
  result.reserve(tensors.size());
  for (auto&& t: tensors) result.push_back(needs(t));
  return result;
}

auto Partials::accumulateGrads(Tensor* t) -> Tensor* {
  if (partials_.contains(t)) {
    auto& partial = partials_[t];

    if (!partial.first)
      partial.first = new Tensor(Float, t->shape());

    if (partial.first->op() || partial.first->buffer())
      return partial.first;

    auto acc = new AccumulateGrads(partial.second, partial.first);
    lambda_.ops.push_back(acc);
    lambda_.tensors.push_back(acc->outputs[0]);
    acc->outputs[0]->req();

    return partial.first;

  } else {
    throw std::out_of_range("no such partial");
  }
}

auto Partials::accumulateGrads(const std::vector<Tensor*>& tensors) -> std::vector<Tensor*> {
  std::vector<Tensor*> result;
  result.reserve(tensors.size());
  for (auto&& t: tensors)
    result.push_back(accumulateGrads(t));
  return result;
}

void Partials::addGrads(Tensor* tensor, Tensor* grad) {
  if (!partials_.contains(tensor))
//    throw std::runtime_error("added gradient is not needed");
    partials_[tensor] = {nullptr, {}};

  partials_[tensor].second.push_back(grad);
}

void Partials::addGrads(const std::vector<Tensor*>& tensors, const std::vector<Tensor*>& grads) {
  for (int i = 0; i < tensors.size(); i++)
    addGrads(tensors[i], grads[i]);
}

}