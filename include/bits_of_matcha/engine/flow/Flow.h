#pragma once

#include "bits_of_matcha/engine/Tensor.h"
#include "bits_of_matcha/engine/flow/Graph.h"
#include "bits_of_matcha/engine/flow/Instructions.h"


namespace matcha::engine {

class Instructions;

class Flow {
public:
  Flow(Graph graph);
  std::vector<Tensor*> run(const std::vector<Tensor*> ins);

  std::tuple<std::vector<Frame>, std::vector<Frame>> signature() const;

  void compile();
  std::vector<Tensor*> forward();
  std::vector<Tensor*> backward();

private:
  void createBackwardFlow();

private:
  Graph graph_;
  Instructions instructions_;

private:
};

}