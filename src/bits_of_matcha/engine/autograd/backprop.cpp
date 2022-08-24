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

  constexpr bool disable = false;
  if constexpr (disable) {
    lambda.outputs.clear();
    for (auto&& w: wrt) {
      auto g = engine::ones(w->shape());
      lambda.tensors.push_back(g);
      lambda.outputs.push_back(g);
      g->req();
    }
    return;
  }

  // extend lambda by backprop ops

  std::set<Tensor*> tensors(lambda.tensors.begin(), lambda.tensors.end());
  Partials partials(lambda, wrt);

  for (int i = (int) lambda.ops.size() - 1; i >= 0; i--) {
    auto op = lambda.ops[i];
    if (!partials.needs(op)) continue;

    BackCtx ctx {
      .forward = op,
      .vals = partials.accumulateGrads(op->outputs),
      .wrts = partials.needs(op->inputs),
    };
    Lambda back = ops::back(ctx);

    for (auto&& bop: back.ops)
      lambda.ops.push_back(bop);

    for (auto&& t: back.tensors) {
      if (tensors.contains(t)) {
        t->unreq();
      } else {
        lambda.tensors.push_back(t);
        tensors.insert(t);
      }
    }

    for (int j = 0; j < back.outputs.size(); j++) {
      auto&& bout = back.outputs[j];
      if (!bout) continue;
      partials.addGrads(op->inputs[j], bout);
    }

    back = {};
  }

  lambda.outputs.clear();
  for (auto&& g: partials.accumulateGrads(wrt))
    lambda.outputs.push_back(g);
}

}
