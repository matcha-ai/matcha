#include "bits_of_matcha/Backprop.h"
#include "bits_of_matcha/engine/lambda/Tracer.h"
#include "bits_of_matcha/engine/lambda/Module.h"
#include "bits_of_matcha/engine/autograd/backprop.h"
#include "bits_of_matcha/engine/lambda/passes/debug.h"
#include "bits_of_matcha/engine/lambda/executors/SinglecoreExecutor.h"
#include "bits_of_matcha/engine/lambda/passes/inlineExpansion.h"
#include "bits_of_matcha/engine/lambda/passes/deadCodeElimination.h"
#include "bits_of_matcha/engine/lambda/passes/copyPropagation.h"

namespace matcha {

struct Internal {
  engine::Tracer tracer_;
};

Backprop::Backprop() {
  auto internal = new Internal;
  internal_ = internal;
  internal->tracer_.open({});
}

std::map<tensor*, tensor> Backprop::operator()(const tensor& root, const std::vector<tensor*>& wrt) {
  if (!internal_)
    throw std::runtime_error("Backprop already used, init a new one");

  auto internal = (Internal*) internal_;

  // *MUST* emplace, or a temporary tensor will be recorded
  // and deleted after the Tracer's closed, eventually causing a segfault
  std::vector<tensor> temp;
  temp.emplace_back(root);

  auto lambda = internal->tracer_.close(temp);

//  engine::debug(lambda);
  engine::inlineExpansion(lambda);

  std::map<const tensor*, engine::Tensor*> side_inputs_inv;
  for (auto&& [in, binding]: lambda.side_inputs)
    side_inputs_inv[binding] = in;

  std::vector<engine::Tensor*> wrt_internal;
  for (auto&& w: wrt) {
    engine::Tensor* translated;
    if (side_inputs_inv.contains(w)) {
      translated = side_inputs_inv[w];
    } else {
      translated = engine::deref(w);
    }
    wrt_internal.push_back(translated);
  }

  engine::backprop(lambda, wrt_internal);

  auto executor = std::make_shared<engine::SinglecoreExecutor>(std::move(lambda));
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
  auto lambda = internal->tracer_.close({});
  auto executor = std::make_shared<engine::SinglecoreExecutor>(std::move(lambda));
  executor->run({});

  delete internal;
}

}