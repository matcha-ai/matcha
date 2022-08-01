#include "bits_of_matcha/engine/chain/Module.h"

namespace matcha::engine {

Module::Module(const std::vector<Tensor*>& inputs, std::shared_ptr<Executor> executor)
  : Op(inputs)
  , executor_(std::move(executor))
{
  for (auto&& out: executor_->chain().outputs)
    outputs.add(this, new Tensor(out->frame()));
}

OpMeta<Module> Module::meta {
  .name = "Module",
  .sideEffect = true,
};

auto Module::executor() -> std::shared_ptr<Executor>& {
  return executor_;
}

auto Module::chain() -> Chain& {
  return executor_->chain();
}

void Module::run() {
  executor_->run(inputs.stdVector(), outputs.stdVector());
}

}