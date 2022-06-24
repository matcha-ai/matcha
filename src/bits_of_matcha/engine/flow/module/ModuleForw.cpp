#include "bits_of_matcha/engine/flow/module/ModuleForw.h"
#include "bits_of_matcha/engine/flow/module/ModuleBack.h"
#include "bits_of_matcha/engine/flow/module/Module.h"


namespace matcha::engine {

ModuleForw::ModuleForw(Module* module, const std::vector<Tensor*>& ins)
  : Op(ins)
  , module_(module)
{
  auto& graph = module_->graph_;
  for (auto out: graph->outputs)
    outputs.add(this, new Tensor(out->frame()));
}

OpMeta<ModuleForw> ModuleForw::meta {
  .name = "ModuleForw",
  .back = [](auto& ctx) { return new OpBack(ctx); }
};

void ModuleForw::run() {
  module_->forward(inputs.stdVector(), outputs.stdVector());
}

void ModuleForw::forward(std::map<Tensor*, Partial>& partials) {

}

}