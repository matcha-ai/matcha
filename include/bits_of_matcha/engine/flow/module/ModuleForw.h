#pragma once

#include "bits_of_matcha/engine/op/Op.h"

#include <map>

namespace matcha::engine {

class Module;

struct ModuleForw : Op {
  ModuleForw(Module* module, const std::vector<Tensor*>& ins);
  static OpMeta<ModuleForw> meta;

  static bool isModuleForw(Op* op);
  static ModuleForw asModuleForw(Op* op);

  void run() override;

  using Partial = std::pair<Tensor*, std::vector<Tensor*>>;
  void forward(std::map<Tensor*, Partial>& partials);

private:
  Module* module_;

  friend class Module;
  friend class ModuleBack;
};

}