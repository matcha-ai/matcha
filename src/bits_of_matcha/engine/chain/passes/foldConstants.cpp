#include "bits_of_matcha/engine/chain/passes/foldConstants.h"


namespace matcha::engine {

void foldConstants(Chain& chain) {
  std::set<Tensor*> non_constant;

  for (auto&& in: chain.inputs)
    non_constant.insert(in);

  for (auto&& [in, binding]: chain.side_inputs)
    non_constant.insert(in);

  std::vector<Op*> ops;
  for (auto&& op: chain.ops) {

  }
}

}