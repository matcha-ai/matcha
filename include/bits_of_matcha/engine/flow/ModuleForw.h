#pragma once

#include "bits_of_matcha/engine/op/Op.h"

#include <map>

namespace matcha::engine {

class Module;

struct ModuleForw : Op {
  ModuleForw(Module* module, const std::vector<Tensor*>& ins);
  static OpMeta<ModuleForw> meta;

  void run() override;

  auto module() -> Module&;
  auto module() const -> const Module&;

private:
  Module* module_;
};

}