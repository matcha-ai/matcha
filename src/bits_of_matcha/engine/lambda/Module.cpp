#include "bits_of_matcha/engine/lambda/Module.h"

namespace matcha::engine {

Module::Module(const std::vector<Tensor*>& inputs, std::shared_ptr<Executor> executor)
  : Op(inputs)
  , executor_(std::move(executor))
{
  for (auto&& out: executor_->lambda().outputs)
    addOutput(out->frame());
}

Reflection<Module> Module::reflection {
  .name = "Module",
  .side_effect = true,
};

auto Module::executor() -> std::shared_ptr<Executor>& {
  return executor_;
}

auto Module::lambda() -> Lambda& {
  return executor_->lambda();
}

void Module::run() {
  executor_->run(inputs, outputs);
}

}