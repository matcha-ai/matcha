#pragma once

#include "bits_of_matcha/engine/op/OpBack.h"

namespace matcha::engine {

class Module;

struct ModuleBack : OpBack {
  ModuleBack(const BackCtx& ctx);
  static OpMeta<ModuleBack> meta;

  void run() override;

private:
  Module* module;
};

}