#include "bits_of_matcha/Backprop.h"
#include "bits_of_matcha/engine/chain/Tracer.h"
#include "bits_of_matcha/engine/chain/Module.h"
#include "bits_of_matcha/engine/autograd/backprop.h"
#include "bits_of_matcha/engine/chain/passes/debug.h"
#include "bits_of_matcha/engine/chain/executors/SinglecoreExecutor.h"
#include "bits_of_matcha/engine/chain/passes/flatten.h"
#include "bits_of_matcha/engine/chain/passes/reduceToEffects.h"
#include "bits_of_matcha/engine/chain/passes/contractIdentities.h"

namespace matcha {

struct Internal {
  Internal()
    : wrt_()
  {}

  std::vector<tensor*> wrt_;
  engine::Tracer tracer_;
};

//Backprop::Backprop(std::initializer_list<tensor*> wrt)
//  : Backprop(std::vector(wrt.begin(), wrt.end()))
//{}
//
//Backprop::Backprop(const std::vector<tensor*>& wrt) {\
//  auto internal = new Internal(wrt);
//  internal_ = internal;
//  internal->tracer_.open({});
//}

Backprop::Backprop() {
  auto internal = new Internal;
  internal_ = internal;
  internal->tracer_.open({});
}

std::map<tensor*, tensor> Backprop::operator()(const tensor& root, const std::vector<tensor*>& wrt) {
  if (!internal_)
    throw std::runtime_error("Backprop already used, initialize a new one");

  auto internal = (Internal*) internal_;

  // *MUST* emplace, or a temporary tensor will be recorded
  // and deleted after the Tracer's closed
  std::vector<tensor> temp;
  temp.emplace_back(root);

  auto chain = internal->tracer_.close(temp);

  engine::flatten(chain);
  engine::reduceToEffects(chain);
  engine::contractIdentities(chain);

  std::map<const tensor*, engine::Tensor*> side_inputs_inv;
//  std::cerr << std::endl;
  for (auto&& [in, binding]: chain.side_inputs) {
//    std::cerr << "SIDEIN: " << in << " <- " << binding << std::endl;
    side_inputs_inv[binding] = in;
  }

  std::vector<engine::Tensor*> wrt_internal;
  for (auto&& w: wrt) {
//    std::cerr << "NEED " << w << std::endl;
    engine::Tensor* translated;
    if (side_inputs_inv.contains(w)) {
      translated = side_inputs_inv[w];
//      std::cerr << translated << std::endl;
    } else {
      translated = engine::deref(w);
    }
    wrt_internal.push_back(translated);
  }

  engine::backprop(chain, wrt_internal);

  auto executor = std::make_shared<engine::SinglecoreExecutor>(std::move(chain));
  auto op = new engine::Module({}, std::move(executor));

  if (op->outputs.size() != wrt.size())
    throw std::runtime_error("can't pair gradients");

  std::map<tensor*, tensor> gradients;
  for (int i = 0; i < wrt.size(); i++)
    gradients[wrt[i]] = ref(op->outputs[i]);

  engine::dispatch(op);
  internal_ = nullptr;
  delete internal;
  return gradients;
}

Backprop::~Backprop() {
  if (!internal_) return;
  auto internal = (Internal*) internal_;
  auto chain = internal->tracer_.close({});
  auto executor = std::make_shared<engine::SinglecoreExecutor>(std::move(chain));
  executor->run({});

  delete internal;
}

}