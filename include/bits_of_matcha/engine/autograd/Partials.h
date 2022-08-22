#pragma once

#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

using Partial = std::pair<Tensor*, std::vector<Tensor*>>;

class Partials {
public:
  explicit Partials(Lambda& lambda, const std::vector<Tensor*>& wrt);

  auto needs(Op* op) const -> bool;
  auto needs(const std::vector<Op*>& ops) const -> std::vector<bool>;
  auto needs(Tensor* tensor) const -> bool;
  auto needs(const std::vector<Tensor*>& tensors) const -> std::vector<bool>;

  auto accumulateGrads(Tensor* tensor) -> Tensor*;
  auto accumulateGrads(const std::vector<Tensor*>& tensors) -> std::vector<Tensor*>;

  void addGrads(Tensor* tensor, Tensor* grad);
  void addGrads(const std::vector<Tensor*>& tensors, const std::vector<Tensor*>& grads);

private:
  Lambda& lambda_;
  std::map<Tensor*, Partial> partials_;
};

}