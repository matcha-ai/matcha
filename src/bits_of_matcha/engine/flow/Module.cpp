#include "bits_of_matcha/engine/flow/Module.h"
#include "bits_of_matcha/engine/flow/ModuleForw.h"
#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/engine/chain/executors/SinglecoreExecutor.h"
#include "bits_of_matcha/engine/chain/passes/check.h"

#include <ranges>


namespace matcha::engine {

Module::Module(Chain chain)
  : chain_(std::move(chain))
  , executor_(nullptr)
{
  for (auto op: chain.ops) op->init();
//  std::cerr << "Module" << std::endl;
}

Module::~Module() {
//  std::cerr << "~Module" << std::endl;
}

Chain& Module::chain() { return chain_; }
const Chain& Module::chain() const { return chain_; }

std::vector<Tensor*> Module::run(const std::vector<Tensor*>& ins) {
  return executor_->run(ins);
}

void Module::run(const std::vector<Tensor*>& ins, std::vector<Tensor*>& outs) {
  return executor_->run(ins, outs);
}


}