#pragma once

#include <vector>
#include <memory>
#include <set>

namespace matcha::engine {

class Tensor;
class Op;

struct Graph {
  std::vector<Tensor*> tensors;
  std::vector<Tensor*> inputs;
  std::vector<Tensor*> outputs;
  std::vector<Op*> ops;


};

}