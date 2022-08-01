#include "bits_of_matcha/engine/chain/passes/flatten.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/flow/Module.h"
#include "bits_of_matcha/engine/flow/ModuleForw.h"
#include "bits_of_matcha/engine/ops/Identity.h"


namespace matcha::engine {

void unfold(Chain& chain) {
  std::vector<Op*> ops;
  for (auto&& op: chain.ops) {
    if (typeid(*op) != typeid(ModuleForw)) {
      ops.push_back(op);
      continue;
    }

    // unfold recursively
    auto c = copy(dynamic_cast<ModuleForw*>(op)->module().chain());
    unfold(c);

    // relink chain inputs
    for (int i = 0; i < c.inputs.size(); i++) {
      auto& cin = c.inputs[i];
      auto& oin = op->inputs[i];
//      std::cout << cin->frame() << " " << oin->frame() << std::endl;

      auto id = new ops::Identity(oin, cin);
      ops.push_back(id);
    }

    // include chain
    for (auto&& opp: c.ops)
      ops.push_back(opp);
    for (auto&& tt: c.tensors)
      chain.tensors.push_back(tt);

    // relink chain outputs
    for (int i = 0; i < c.outputs.size(); i++) {
      auto& cout = c.outputs[i];
      auto& oout = op->outputs[i];
//      std::cout << cout->frame() << " " << oout->frame() << std::endl;

      auto id = new ops::Identity(cout, oout);
      ops.push_back(id);

    }
    // don't deallocate the copies
    c.ops = {};
    c.tensors = {};
  }
  chain.ops = ops;
//  std::cerr << "unfolding done" << std::endl;
//  exit(0);
}

}
