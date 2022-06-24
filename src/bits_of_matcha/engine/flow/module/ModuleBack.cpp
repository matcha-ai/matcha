#include "bits_of_matcha/engine/flow/module/ModuleBack.h"


namespace matcha::engine {

ModuleBack::ModuleBack(const BackCtx& ctx)
  : OpBack(ctx)
{}

OpMeta<ModuleBack> ModuleBack::meta {
  .name = "ModuleBack",
};

void ModuleBack::run() {

}

}