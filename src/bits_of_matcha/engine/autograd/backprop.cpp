#include "bits_of_matcha/engine/autograd/backprop.h"
#include "bits_of_matcha/engine/autograd/Partials.h"
#include "bits_of_matcha/engine/lambda/passes/inlineExpansion.h"
#include "bits_of_matcha/engine/lambda/passes/copyPropagation.h"
#include "bits_of_matcha/engine/lambda/passes/deadCodeElimination.h"
#include "bits_of_matcha/engine/lambda/passes/init.h"
#include "bits_of_matcha/engine/lambda/passes/debug.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/tensor/factories.h"


namespace matcha::engine {


void backprop(Lambda& lambda, const std::vector<Tensor*>& wrt) {
  if (lambda.outputs.size() != 1)
    throw std::runtime_error("there must be exactly one root to back-propagate from");

//  debug(lambda);

  // extend lambda by backprop ops
  Partials partials(lambda, wrt);

  lambda.outputs.clear();

  constexpr bool disable = false;
  if constexpr (disable) {
    for (auto&& w: wrt)
      lambda.outputs.push_back(engine::ones(w->shape()));
    return;
  }

  for (int i = (int) lambda.ops.size() - 1; i >= 0; i--) {
    auto op = lambda.ops[i];
    if (!partials.needs(op)) continue;

    BackCtx ctx {
      .forward = op,
      .vals = partials.accumulateGrads(op->outputs),
      .wrts = partials.needs(op->inputs),
    };
    auto back = ops::back(ctx);
//    auto back = Lambda();

    for (auto&& bop: back.ops) {
      if (!bop) continue;
      lambda.ops.push_back(bop);
      for (auto&& out: bop->outputs) {
        if (!out) continue;
        lambda.tensors.push_back(out);
      }
    }

    for (int j = 0; j < back.outputs.size(); j++) {
      auto&& bout = back.outputs[j];
      if (!bout) continue;
      partials.addGrads(op->inputs[j], bout);
    }

    back = Lambda{};
  }

  lambda.outputs.clear();
  for (auto&& g: partials.accumulateGrads(wrt))
    lambda.outputs.push_back(g);
}

}
