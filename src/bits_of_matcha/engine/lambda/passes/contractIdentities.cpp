#include "bits_of_matcha/engine/lambda/passes/copyPropagation.h"
#include "bits_of_matcha/engine/ops/Identity.h"

namespace matcha::engine {

void copyPropagation(Lambda& lambda) {
  std::map<Tensor*, std::vector<Op*>> dependencies;
  std::vector<Op*> ops;
  std::set<Tensor*> orphans;

  // find dependencies
  for (auto&& op: lambda.ops) {
    for (auto&& in: op->inputs) {
      if (!dependencies.contains(in)) dependencies[in] = {};
      dependencies[in].push_back(op);
    }
  }

  // find Identities
  for (auto&& op: lambda.ops) {
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

    // relink lambda outputs
    for (auto& out: lambda.outputs) {
      if (out == target) out = source;
    }
  }

  std::vector<Tensor*> tensors;
  for (auto&& t: lambda.tensors) {
    if (!orphans.contains(t)) tensors.push_back(t);
    else t->unreq();
  }
  lambda.ops = ops;
  lambda.tensors = tensors;
}

}