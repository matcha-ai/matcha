#include "bits_of_matcha/engine/lambda/passes/inlineExpansion.h"
#include "bits_of_matcha/engine/lambda/passes/debug.h"
#include "bits_of_matcha/engine/lambda/Module.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/ops/Identity.h"


namespace matcha::engine {

void inlineExpansion(Lambda& lambda) {
  std::vector<Op*> ops;
  for (auto&& op: lambda.ops) {

    // keep normal Op
    if (typeid(*op) != typeid(Module)) {
      ops.push_back(op);
      continue;
    }

    // recursively inlineExpansion all nested Lambdas
    auto c = clone(dynamic_cast<Module*>(op)->lambda());
    inlineExpansion(c);

    // relink lambda inputs
    for (int i = 0; i < c.inputs.size(); i++) {
      auto cin = c.inputs[i];
      auto oin = op->inputs[i];

      auto id = new ops::Identity(oin, cin);
      ops.push_back(id);
    }

    // include lambda
    for (auto&& opp: c.ops)
      ops.push_back(opp);
    for (auto&& tt: c.tensors)
      lambda.tensors.push_back(tt);

    // relink lambda outputs
    for (int i = 0; i < c.outputs.size(); i++) {
      auto& cout = c.outputs[i];
      auto& oout = op->outputs[i];

      auto id = new ops::Identity(cout, oout);
      ops.push_back(id);
      oout = nullptr;
    }

    // TODO: relink lambda side inputs

    for (auto&& [in, binding]: c.side_inputs) {
      lambda.side_inputs[in] = binding;
    }

    // TODO: inlineExpansion lambda side outputs

    delete op;

    // don't deallocate the copies
    c = {};
  }
  lambda.ops = ops;
//  std::cerr << "FLATTENED:" << std::endl;
//  check(lambda);
}

}
