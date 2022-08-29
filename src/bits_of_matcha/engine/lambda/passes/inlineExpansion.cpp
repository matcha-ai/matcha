#include "bits_of_matcha/engine/lambda/passes/inlineExpansion.h"
#include "bits_of_matcha/engine/lambda/passes/debug.h"
#include "bits_of_matcha/engine/lambda/Module.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/ops/Identity.h"
#include "bits_of_matcha/engine/ops/SideOutput.h"


namespace matcha::engine {

void inlineExpansion(Lambda& lambda) {
  std::vector<Op*> ops;
  std::map<tensor*, Tensor*> side_outs;
  std::map<Tensor*, const tensor*> side_ins;

  for (auto&& op: lambda.ops) {
//    std::cerr << ops::name(op) << std::endl;

    for (auto&& in: op->inputs) {
      if (!lambda.side_inputs.contains(in)) continue;
      auto source = const_cast<tensor*>(lambda.side_inputs[in]);
      lambda.side_inputs.erase(in);
      if (side_outs.contains(source)) {
        auto id = new ops::Identity(side_outs[source], in);
        ops.push_back(id);
      } else {
        side_ins[in] = source;
      }
    }

    if (typeid(*op) == typeid(ops::SideOutput)) {
      // Additional relinking and potential pruning necessary for side-outs.
      auto side_out = dynamic_cast<ops::SideOutput*>(op);
      side_outs[side_out->target()] = side_out->inputs[0];
      delete side_out;
      continue;
    }

    // Keep normal Ops.
    if (typeid(*op) != typeid(Module)) {
      ops.push_back(op);
      continue;
    }

    // Recursively inline all nested Lambdas.
    Lambda l = dynamic_cast<Module*>(op)->lambda();
    inlineExpansion(l);

    // Relink lambda inputs via Identity.
    for (int i = 0; i < l.inputs.size(); i++) {
      auto cin = l.inputs[i];
      auto oin = op->inputs[i];

      auto id = new ops::Identity(oin, cin);
      ops.push_back(id);
    }

    // Relink inner lambda side-inputs to outer side-outputs
    // or propagate them to outer side-inputs.
    for (auto&& [target, source]: l.side_inputs) {
      auto s = const_cast<tensor*>(source);
      if (side_outs.contains(s)) {
        auto id = new ops::Identity(side_outs[s], target);
        ops.push_back(id);
      } else {
        side_ins[target] = source;
      }
    }

    // Unpack inner lambda ops and tensors.
    for (auto&& tt: l.tensors)
      lambda.tensors.push_back(tt);

    for (auto&& opp: l.ops) {
      if (typeid(*opp) != typeid(ops::SideOutput)) {
        ops.push_back(opp);
        continue;
      }

      // Additional relinking and potential pruning necessary for side-outs.
      auto side_out = dynamic_cast<ops::SideOutput*>(opp);
      side_outs[side_out->target()] = side_out->inputs[0];
      delete side_out;
    }

    // Relink lambda outputs via Identity.
    for (int i = 0; i < l.outputs.size(); i++) {
      auto& lout = l.outputs[i];
      auto& oout = op->outputs[i];

      auto id = new ops::Identity(lout, oout);
      ops.push_back(id);
      oout = nullptr;
    }

    delete op;

    // don't deallocate the copies
    l = {};
  }
  // Configure side-outputs.
  for (auto&& [target, source]: side_outs) {
    auto s = new ops::SideOutput(source, target);
    ops.push_back(s);
  }

  // And side-inputs.
  for (auto&& [target, source]: lambda.side_inputs) {
    auto s = const_cast<tensor*>(source);
    if (side_outs.contains(s)) {
      auto id = new ops::Identity(side_outs[s], target);
      ops.push_back(id);
    } else {
      side_ins[target] = source;
    }
  }

//  std::cerr << side_ins.size() << std::endl;
  lambda.side_inputs = side_ins;
  lambda.ops = ops;

}

}
