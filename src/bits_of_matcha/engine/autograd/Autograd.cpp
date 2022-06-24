#include "bits_of_matcha/engine/autograd/Autograd.h"
#include "bits_of_matcha/engine/flow/Graph.h"

namespace matcha::engine {

std::unique_ptr<Graph> autograd(const std::unique_ptr<Graph>& graph) {
  Autograd ag(graph);
  ag.run();
  return std::move(ag.adjoint_);
}

Autograd::Autograd(const std::unique_ptr<Graph>& graph)
  : graph_(graph)
{}

void Autograd::run() {
  adjoint_ = std::make_unique<Graph>();
}

}