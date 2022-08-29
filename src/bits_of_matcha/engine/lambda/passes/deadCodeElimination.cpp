#include "bits_of_matcha/engine/lambda/passes/deadCodeElimination.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

void deadCodeElimination(Lambda& lambda) {
  std::set<Op*> alive;

  std::function<void (Op*)> dfs = [&](Op* op) {
    // recursively get all alive ops
    if (!op || alive.contains(op)) return;
    alive.insert(op);
    for (auto&& in: op->inputs)
      if (in) dfs(in->op());
  };

  // require lambda outputs
  for (auto&& out: lambda.outputs)
    dfs(out->op());

  // require lambda side-effects
  for (auto&& op: lambda.ops)
    if (ops::isSideEffect(op))
      dfs(op);

  std::set<Tensor*> orphans;
  std::vector<Op*> ops;
  ops.reserve(alive.size());

  // retain alive ops, eliminate rest
  for (auto&& op: lambda.ops) {
    if (alive.contains(op)) {
      ops.push_back(op);
    } else {
      for (auto&& out: op->outputs)
        orphans.insert(out);
      delete op;
    }
  }

  // eliminate orphan tensors
  std::vector<Tensor*> tensors;
  for (auto&& t: lambda.tensors) {
    if (orphans.contains(t))
      t->unreq();
    else
      tensors.push_back(t);
  }

  lambda.ops = ops;
  lambda.tensors = tensors;
}

}