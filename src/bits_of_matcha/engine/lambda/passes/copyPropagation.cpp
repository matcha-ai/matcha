#include "bits_of_matcha/engine/lambda/passes/copyPropagation.h"
#include "bits_of_matcha/engine/ops/Identity.h"

namespace matcha::engine {

void copyPropagation(Lambda& lambda) {
  std::map<Tensor*, std::vector<Op*>> dependencies;
  std::map<Tensor*, Tensor*> propagations;
  std::vector<Op*> ops;

  // Find tensor dependencies for later relinking.
  for (auto&& op: lambda.ops) {
    for (auto&& in: op->inputs) {
      if (!dependencies.contains(in)) dependencies[in] = {};
      dependencies[in].push_back(op);
    }
  }

  // Find and contract Identities.
  for (auto&& op: lambda.ops) {
    if (typeid(*op) != typeid(ops::Identity)) {
      ops.push_back(op);
      continue;
    }

    // Ops are sorted topologically. Therefore. the identity
    // source is guaranteed not to be changed by later relinking.
    Tensor* source = op->inputs[0];
    Tensor* target = op->outputs[0];
    propagations[target] = source;

    delete op;

    // Relink all target dependencies to source, including side-effects.
    for (auto&& dep: dependencies[target]) {
      for (auto& in: dep->inputs) {
        if (in != target) continue;
        in->unreq();
        in = source;
        in->req();
      }
    }
  }

  // Relink lambda outputs.
  for (auto& out: lambda.outputs) {
    if (! propagations.contains(out)) continue;
    out = propagations[out];
  }

  // Delete pruned tensors.
  std::vector<Tensor*> tensors;
  for (auto&& t: lambda.tensors) {
    if (propagations.contains(t))
      t->unreq();
    else
      tensors.push_back(t);
  }

  lambda.ops = ops;
  lambda.tensors = tensors;
}

}