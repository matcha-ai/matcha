#include "bits_of_matcha/engine/lambda/passes/constantPropagation.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

void constantPropagation(Lambda& lambda) {
  std::set<Tensor*> runtime; // tensors that must be computed in the runtime
  std::vector<Op*> ops;

  for (auto&& in: lambda.inputs)
    runtime.insert(in);

  for (auto&& [in, binding]: lambda.side_inputs)
    runtime.insert(in);

  for (auto&& op: lambda.ops) {
    bool deterministic = true;
    bool side_effect = ops::isSideEffect(op);

    bool r = std::any_of(op->inputs.begin(),
                         op->inputs.end(),
                         [&](Tensor* in) { return runtime.contains(in); });

    if (!deterministic || r || side_effect) {
      // propagate runtime scheduling
      for (auto&& out: op->outputs)
        runtime.insert(out);
      ops.push_back(op);
      continue;
    }

    // pre-compute
    op->run();
    delete op;
  }

  std::vector<Tensor*> tensors;
  for (auto&& t: lambda.tensors) {
    if (runtime.contains(t))
      tensors.push_back(t);
    else
      t->unreq();
  }
  lambda.ops = ops;
  lambda.tensors = tensors;
}

}