#include "bits_of_matcha/engine/lambda/passes/constantPropagation.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

void constantPropagation(Lambda& lambda) {
  std::set<Tensor*> runtime; // tensors that must be computed in the runtime
  std::set<Tensor*> keep, fold;
  std::vector<Op*> ops;

  for (auto&& in: lambda.inputs)
    runtime.insert(in);

  for (auto&& [in, binding]: lambda.side_inputs)
    runtime.insert(in);

  for (auto&& op: lambda.ops) {
    bool deterministic = ops::isDeterministic(op);
    bool side_effect = ops::isSideEffect(op);

    bool r = std::any_of(op->inputs.begin(),
                         op->inputs.end(),
                         [&](Tensor* in) { return runtime.contains(in); });

    if (!deterministic || r || side_effect) {
      // propagate runtime scheduling
      for (auto&& out: op->outputs)
        runtime.insert(out);

      for (auto&& in: op->inputs)
        keep.insert(in);

      ops.push_back(op);
      continue;
    }

    // Pre-compute.
    op->init();
    op->run();

    for (auto&& in: op->inputs)
      fold.insert(in);

    delete op;
  }

  // Prune unnecessary tensors.
  std::vector<Tensor*> tensors;
  for (auto&& t: lambda.tensors) {
    if (fold.contains(t) && !keep.contains(t))
      t->unreq();
    else
      tensors.push_back(t);
  }

  lambda.ops = ops;
  lambda.tensors = tensors;
}

}