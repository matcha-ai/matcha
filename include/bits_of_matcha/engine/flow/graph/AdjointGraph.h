#pragma once

#include "bits_of_matcha/engine/flow/graph/TensorDict.h"
#include "bits_of_matcha/engine/flow/graph/OpDict.h"

namespace matcha::engine {

struct AdjointGraph {
  Graph* adjointGraph;
  TensorDict<Tensor*> adjointTensors;
  OpDict<Op*> adjointOps;
};


}