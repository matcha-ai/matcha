#pragma once

#include "bits_of_matcha/engine/flow/Tasks.h"
#include "bits_of_matcha/engine/flow/graph/Graph.h"
#include "bits_of_matcha/engine/flow/graph/OpDict.h"
#include "bits_of_matcha/engine/flow/graph/OpMask.h"
#include "bits_of_matcha/engine/flow/graph/TensorDict.h"
#include "bits_of_matcha/engine/flow/graph/TensorMask.h"

#include <vector>

namespace matcha::engine {

Tasks compile(Graph* graph, const TensorMask& grads);
Tasks compile(Graph& graph, const TensorMask& grads);

class Compiler {
private:
  Compiler(Graph* graph, const TensorMask& grads);
  Tasks run();

private:
  Graph* buildBackwardGraph();
  TensorMask findGradientFlow();

private:
  Graph* graph_;
  TensorMask grads_;

private:
  friend Tasks compile(Graph*, const TensorMask&);
};

}