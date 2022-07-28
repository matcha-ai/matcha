#include "bits_of_matcha/engine/flow/ModuleForw.h"
#include "bits_of_matcha/engine/flow/Module.h"


namespace matcha::engine {

ModuleForw::ModuleForw(Module* module, const std::vector<Tensor*>& ins)
  : Op{ins}
  , module_(module)
{
  for (auto&& out: module_->chain_.outputs)
    outputs.add(this, out->frame());
}

OpMeta<ModuleForw> ModuleForw::meta {
  .name = "Module",
  .sideEffect = true,
};

auto ModuleForw::module() -> Module& {
  return *module_;
}

auto ModuleForw::module() const -> const Module& {
  return *module_;
}

void ModuleForw::run() {
  module_->run(inputs.stdVector(), outputs.stdVector());
}

}