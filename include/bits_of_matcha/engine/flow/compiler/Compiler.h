#pragma once

#include "bits_of_matcha/engine/flow/Tasks.h"
#include "bits_of_matcha/engine/flow/graph/Graph.h"
#include "bits_of_matcha/engine/flow/graph/OpDict.h"
#include "bits_of_matcha/engine/flow/graph/OpMask.h"
#include "bits_of_matcha/engine/flow/graph/TensorDict.h"
#include "bits_of_matcha/engine/flow/graph/TensorMask.h"
#include "bits_of_matcha/engine/flow/graph/AdjointGraph.h"

#include <vector>
#include <map>

namespace matcha::engine {

Tasks compile(Graph* graph, const std::map<tensor*, Tensor*>& grads);
Tasks compile(Graph& graph, const std::map<tensor*, Tensor*>& grads);

class Compiler {
private:
  Compiler(Graph* graph, const std::map<tensor*, Tensor*>& grads);
  Tasks run();

private:
  AdjointGraph buildBackwardGraph();
  TensorMask findGradientFlow();
  OpMask findEffects();
  Tasks generateTasks(const AdjointGraph& back);
  TensorDict<unsigned> getTotalTensorReqs(const AdjointGraph& back);

private:
  Graph* graph_;
  std::map<tensor*, Tensor*> grads_;
  TensorMask gradsMask_;

private:
  friend Tasks compile(Graph*, const std::map<tensor*, Tensor*>&);
};

}