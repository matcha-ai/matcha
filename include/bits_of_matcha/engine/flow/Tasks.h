#pragma once

#include "bits_of_matcha/engine/flow/compiler/Instruction.h"
#include "bits_of_matcha/engine/flow/graph/TensorMask.h"

#include <vector>
#include <map>

namespace matcha::engine {

class Tensor;
class Op;

struct Tasks {
  std::vector<Instruction> instructionsForward;
  std::vector<Instruction> instructionsBackward;

  std::vector<Tensor*> inputs;
  std::vector<Tensor*> outputs;

  std::vector<Tensor*> deltas;
  std::map<tensor*, Tensor*> grads;

  void init();

  std::vector<Tensor*> forward(const std::vector<Tensor*>& inputs);
  std::map<tensor*, tensor> backward(const std::vector<Tensor*>& delta);


};

}