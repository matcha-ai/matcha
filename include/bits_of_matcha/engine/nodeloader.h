#pragma once

#include "bits_of_matcha/engine/tensor.h"

#include <string>
#include <vector>
#include <iostream>
#include <functional>


namespace matcha {
namespace engine {

class Node;

struct NodeLoader {
  using Action = std::function<Node* (std::istream& is, const std::vector<Tensor*>& ins)>;

  std::string type;
  Action load;
};


}
}
