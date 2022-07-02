#pragma once

#include <vector>
#include <memory>
#include <tuple>
#include <map>

namespace matcha::engine {

class Tensor;
class Block;
class Graph;

struct Backprop {
  using Partial = std::pair<Tensor*, std::vector<Tensor*>>;
  std::map<Tensor*, Partial> partials;
  std::vector<Tensor*> targets;
};

}