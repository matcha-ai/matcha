#include "bits_of_matcha/engine/flow/ModuleBack.h"
#include "bits_of_matcha/engine/flow/compiler/Compiler.h"


namespace matcha::engine {

ModuleBack::ModuleBack(const BackCtx& ctx)
  : OpBack(ctx)
{
  auto module = (Module*) forward;
  Graph& graph = module->graph_;
  Tasks& tasks = module->tasks_;
  tasks = engine::compile(graph, {});
  tasks_ = &tasks;
}

void ModuleBack::run() {
//  tasks_->backward(nullptr);
}

}