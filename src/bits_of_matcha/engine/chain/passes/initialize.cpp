#include "bits_of_matcha/engine/chain/passes/initialize.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

void initialize(Chain& chain) {
  for (auto&& op: chain.ops) if (op) op->init();
}

}