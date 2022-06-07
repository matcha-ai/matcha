#include "bits_of_matcha/engine/flow/Module.h"
#include "bits_of_matcha/engine/flow/ModuleBack.h"
#include "bits_of_matcha/engine/flow/compiler/Compiler.h"


namespace matcha::engine {

Module::Module(const std::vector<Tensor*>& inputs, const Graph& graph)
  : Op(inputs)
  , graph_(graph)
{
  bool eql = std::equal(
    inputs.begin(), inputs.end(),
    graph.inputs.begin(),
    [](Tensor* a, Tensor* b) {
      return a->frame() == b->frame();
    }
  );
  if (!eql) throw std::invalid_argument("Module inputs mismatch");

  for (auto output: graph.outputs) {
    outputs.add(this, output->frame());
  }
}

OpMeta<Module> Module::meta {
  .name = "Module",
  .back = [](auto& ctx) {
    return new ModuleBack(ctx);
  },
};

void Module::init() {
  if (tasks_.inputs.empty() && tasks_.outputs.empty()) {
    engine::compile(graph_, {});
  }
}

void Module::run() {
  auto externalInIter = inputs.begin();
  for (auto internalIn: graph_.inputs) {
    internalIn->shareBuffer(*externalInIter++);
  }

  for (auto op: graph_.ops) op->run();

  auto externalOutIter = outputs.begin();
  for (auto internalOut: graph_.outputs) {
    auto externalOut = *externalOutIter++;
    externalOut->shareBuffer(internalOut);
  }
}

}