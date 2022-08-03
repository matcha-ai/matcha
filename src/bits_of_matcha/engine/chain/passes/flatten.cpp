#include "bits_of_matcha/engine/chain/passes/flatten.h"
#include "bits_of_matcha/engine/chain/passes/check.h"
#include "bits_of_matcha/engine/chain/Module.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/ops/Identity.h"


namespace matcha::engine {

void flatten(Chain& chain) {
  std::vector<Op*> ops;
  for (auto&& op: chain.ops) {

    // keep normal Op
    if (typeid(*op) != typeid(Module)) {
      ops.push_back(op);
      continue;
    }

    // flatten Module recursively
//    std::cerr << "ORIGINAL:" << std::endl;
//    check(dynamic_cast<ModuleForw*>(op)->module().chain());
    auto c = copy(dynamic_cast<Module*>(op)->chain());
//    std::cerr << "COPY:" << std::endl;
//    check(c);
    flatten(c);

    // relink chain inputs
    for (int i = 0; i < c.inputs.size(); i++) {
      auto cin = c.inputs[i];
      auto oin = op->inputs[i];

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

      auto id = new ops::Identity(cout, oout);
      ops.push_back(id);
      oout = nullptr;

    }
    delete op;

    // don't deallocate the copies
    c = {};
  }
  chain.ops = ops;
//  std::cerr << "FLATTENED:" << std::endl;
//  check(chain);
}

}