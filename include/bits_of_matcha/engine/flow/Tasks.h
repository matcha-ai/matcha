#pragma once

#include <vector>

namespace matcha::engine {

class Tensor;
class Op;

struct Tasks {
  std::vector<Op*> opsForward;
  std::vector<Op*> opsBackward;
  std::vector<Tensor*> inputs;
  std::vector<Tensor*> outputs;

  void init();

  std::vector<Tensor*> forward(const std::vector<Tensor*>& inputs);
  std::vector<Tensor*> backward(Tensor* delta);


};

}