#pragma once

#include "bits_of_matcha/engine/chain/Executor.h"
#include "bits_of_matcha/engine/chain/Module.h"

namespace matcha::engine {

struct Module : Op {
  explicit Module(const std::vector<Tensor*>& inputs, std::shared_ptr<Executor> executor);
  static OpMeta<Module> meta;

  auto executor() -> std::shared_ptr<Executor>&;
  auto chain() -> Chain&;

  void run() override;

private:
  std::shared_ptr<Executor> executor_;
};

}