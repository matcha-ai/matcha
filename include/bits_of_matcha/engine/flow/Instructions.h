#pragma once

#include <vector>


namespace matcha::engine {

class Node;

struct Instructions {

  void run();

  void prepareBuffers();
  void clearBuffers();
  void runNodes();

  std::vector<Node*> nodes;

};

}