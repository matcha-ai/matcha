#pragma once

#include "bits_of_matcha/engine/flow/Module.h"
#include "bits_of_matcha/engine/op/OpBack.h"

namespace matcha::engine {

struct ModuleBack : OpBack {
  ModuleBack(const BackCtx& ctx);
  static OpMeta<ModuleBack> meta;

  void run() override;

private:
  Tasks* tasks_;
};

}