#pragma once

#include "bits_of_matcha/engine/lambda/Executor.h"
#include "bits_of_matcha/engine/lambda/Module.h"

namespace matcha::engine {

struct Module : Op {
  explicit Module(const std::vector<Tensor*>& inputs, std::shared_ptr<Executor> executor);
  static Reflection<Module> reflection;

  auto executor() -> std::shared_ptr<Executor>&;
  auto lambda() -> Lambda&;

  void run() override;

private:
  std::shared_ptr<Executor> executor_;
};

}