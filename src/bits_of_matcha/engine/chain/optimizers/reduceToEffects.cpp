#include "bits_of_matcha/engine/chain/optimizers/reduceToEffects.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

void reduceToEffects(Chain& chain) {
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

  for (auto&& out: chain.outputs)
    dfs(out->op());

  for (auto&& op: chain.ops)
    if (ops::isSideEffect(op))
      dfs(op);

  std::vector<Op*> ops;
  ops.reserve(eff.size());

  std::set<Tensor*> orphans;
  for (auto&& op: chain.ops) {
    if (eff.contains(op)) {
      ops.push_back(op);
    } else {
      for (auto&& out: op->outputs) orphans.insert(out);
      delete op;
    }
  }

  std::vector<Tensor*> tensors;

  for (auto&& t: chain.tensors) {
    if (orphans.contains(t)) delete t;
    else tensors.push_back(t);
  }

  chain.ops = ops;
  chain.tensors = tensors;
}

}