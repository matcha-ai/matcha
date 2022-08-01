#include "bits_of_matcha/engine/autograd/backprop.h"
#include "bits_of_matcha/engine/autograd/Partials.h"
#include "bits_of_matcha/engine/chain/passes/flatten.h"
#include "bits_of_matcha/engine/chain/passes/contractIdentities.h"
#include "bits_of_matcha/engine/chain/passes/reduceToEffects.h"
#include "bits_of_matcha/engine/chain/passes/check.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {


void backprop(Chain& chain, const std::vector<Tensor*>& wrt) {
  if (chain.outputs.size() != 1)
    throw std::runtime_error("there must be exactly one root to back-propagate from");

  // make the job easier
  flatten(chain);
  reduceToEffects(chain);
  contractIdentities(chain);
//  check(chain);

  // extend chain by backprop ops
  Partials partials(chain, wrt);

  chain.outputs.clear();
  for (auto&& g: partials.accumulateGrads(wrt))
    chain.outputs.push_back(g);
  return;

  for (int i = (int) chain.ops.size() - 1; i >= 0; i--) {
    auto op = chain.ops[i];
    if (!partials.needs(op)) continue;

    BackCtx ctx {
      .forward = op,
      .vals = partials.accumulateGrads(op->outputs),
      .wrts = partials.needs(op->inputs),
    };
    auto back = ops::back(ctx);

    for (auto&& bop: back.ops) {
      if (!bop) continue;
      chain.ops.push_back(bop);
      for (auto&& out: bop->outputs) {
        if (!out) continue;
        chain.tensors.push_back(out);
        out->req();
      }
    }

    for (int j = 0; j < back.outputs.size(); j++) {
      auto&& bout = back.outputs[j];
      if (!bout) continue;
      partials.addGrads(op->inputs[j], bout);
    }

    back = {};
  }

  // modify chain outputs
  chain.outputs.clear();
  for (auto&& g: partials.accumulateGrads(wrt))
    chain.outputs.push_back(g);

  flatten(chain);
  reduceToEffects(chain);
  contractIdentities(chain);
  check(chain);
}

}
