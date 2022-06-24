#pragma once

#include "bits_of_matcha/engine/tensor/Tensor.h"

#include <memory>
#include <vector>

namespace matcha::engine {

struct Graph;

std::unique_ptr<Graph> autograd(const std::unique_ptr<Graph>& graph, const std::vector<Tensor*>& wrt);

class Autograd {
  friend std::unique_ptr<Graph> autograd(const std::unique_ptr<Graph>&);

  explicit Autograd(const std::unique_ptr<Graph>& graph);
  void run();

  const std::unique_ptr<Graph>& graph_;
  std::unique_ptr<Graph> adjoint_;
};

}