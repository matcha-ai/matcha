#include "bits_of_matcha/engine/lambda/passes/deadCodeElimination.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

void deadCodeElimination(Lambda& lambda) {
  std::set<Op*> eff;

  std::function<void (Op*)> dfs = [&](Op* op) {
    if (!op) return;
    if (eff.contains(op)) return;
    eff.insert(op);
    for (auto&& in: op->inputs) {
      if (!in || !in->op()) continue;
      dfs(in->op());
    }
  };

  for (auto&& out: lambda.outputs)
    dfs(out->op());

  for (auto&& op: lambda.ops)
    if (ops::isSideEffect(op))
      dfs(op);

  std::vector<Op*> ops;
  ops.reserve(eff.size());

  std::set<Tensor*> orphans;
  for (auto&& op: lambda.ops) {
    if (eff.contains(op)) {
      ops.push_back(op);
    } else {
      for (auto&& out: op->outputs) orphans.insert(out);
      delete op;
    }
  }

  std::vector<Tensor*> tensors;

  for (auto&& t: lambda.tensors) {
    if (orphans.contains(t)) delete t;
    else tensors.push_back(t);
  }

  lambda.ops = ops;
  lambda.tensors = tensors;
}

}