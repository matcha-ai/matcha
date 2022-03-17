#pragma once

#include <vector>


namespace matcha::engine {

class Node;

class Instructions {
public:
  void run();

private:
  std::vector<Node*> nodes_;

};

}