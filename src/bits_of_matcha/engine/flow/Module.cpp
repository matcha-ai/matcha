#include "bits_of_matcha/engine/flow/Module.h"
#include "bits_of_matcha/engine/flow/ModuleForw.h"
#include "bits_of_matcha/engine/tensor/factories.h"

#include <ranges>


namespace matcha::engine {

Module::Module(Chain chain)
  : chain_(std::move(chain))
{
  for (auto op: chain.ops) op->init();
}

Module::Module(const AnyOp& preimage, const std::vector<Frame>& frames)
  : chain_(engine::trace(preimage, frames))
{}

Chain& Module::chain() { return chain_; }
const Chain& Module::chain() const { return chain_; }

void Module::pass(const Pass& p) { p(chain_); }

std::vector<Tensor*> Module::run(const std::vector<Tensor*>& ins) {
  std::vector<Tensor*> result;
  result.reserve(chain_.outputs.size());
  for (auto gout: chain_.outputs)
    result.push_back(new Tensor(gout->frame()));

  run(ins, result);
  return result;
}

void Module::run(const std::vector<Tensor*>& ins, std::vector<Tensor*>& outs) {
  stream(ins, chain_.inputs);
  for (auto&& op: chain_.ops) runOp(op);
  stream(chain_.outputs, outs);
}

void Module::runOp(Op* op) {
  constexpr bool debug = false;
  if constexpr (debug) {
    std::string name;
    try {
      name = ops::name(op);
    } catch (...) {
      name = "Unknown";
    }
    print("Running ", name);
  }
  op->run();
}

void Module::stream(const std::vector<Tensor*>& source,
                    std::vector<Tensor*>& target)
{
  if (source.size() != target.size())
    throw std::runtime_error("source and target count mismatch");
  for (int i = 0; i < source.size(); i++)
    target[i]->share(source[i]);
}

}