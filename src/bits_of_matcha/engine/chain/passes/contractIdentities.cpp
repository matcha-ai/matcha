#include "bits_of_matcha/engine/chain/passes/contractIdentities.h"
#include "bits_of_matcha/engine/ops/Identity.h"

namespace matcha::engine {

void contractIdentities(Chain& chain) {
  std::map<Tensor*, std::vector<Op*>> dependencies;
  std::vector<Op*> ops;
  std::set<Tensor*> orphans;

  // find dependencies
  for (auto&& op: chain.ops) {
    for (auto&& in: op->inputs) {
      if (!dependencies.contains(in)) dependencies[in] = {};
      dependencies[in].push_back(op);
    }
  }

  // find Identities
  for (auto&& op: chain.ops) {
    if (typeid(*op) != typeid(ops::Identity)) {
      ops.push_back(op);
      continue;
    }

    Tensor* source = op->inputs[0];
    Tensor* target = op->outputs[0];

    delete op;
    orphans.insert(target);

    // relink target dependencies to source
    for (auto&& dep: dependencies[target]) {
      for (auto& in: dep->inputs) {
        if (in != target) continue;
        in->unreq();
        in = source;
        source->req();
      }
    }

    // relink chain outputs
    for (auto& out: chain.outputs) {
      if (out == target) out = source;
    }
  }

  std::vector<Tensor*> tensors;
  for (auto&& t: chain.tensors) {
    if (!orphans.contains(t)) tensors.push_back(t);
    else t->unreq();
  }
  chain.ops = ops;
  chain.tensors = tensors;
}

}