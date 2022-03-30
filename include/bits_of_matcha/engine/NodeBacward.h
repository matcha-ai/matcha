#include "bits_of_matcha/engine/Node.h"


namespace matcha::engine {

class NodeBackward : public Node {
public:
  using ArgPartial = int;
  using ArgPartials = std::vector<ArgPartial>;
  NodeBackward(std::vector<Tensor*> chainIns, ArgPartials argPartials, Node* forward);

  // inputs are: Node outputs
  // outputs are: Node inputs?
  //
  virtual void run();

protected:
  Node* forward_;
  ArgPartials argPartials_;

};


}