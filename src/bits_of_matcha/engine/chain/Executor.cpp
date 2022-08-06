#include "bits_of_matcha/engine/chain/Executor.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"


namespace matcha::engine {

Executor::Executor(Chain&& chain)
  : chain_(std::move(chain))
{}

void stream(const std::vector<Tensor*>& source, std::vector<Tensor*>& target) {
  if (source.size() != target.size())
    throw std::runtime_error("source and target count mismatch");

  for (int i = 0; i < source.size(); i++) {
    target[i]->share(source[i]);
  }
}

void loadSideInputs(Chain& chain) {
  for (auto&& [in, binding]: chain.side_inputs) {
    in->share(deref(binding));
  }
}

auto Executor::run(const std::vector<Tensor*>& ins) -> std::vector<Tensor*> {
  stream(ins, chain_.inputs);
  loadSideInputs(chain_);
  runInternal();
  std::vector<Tensor*> outputs;
  for (auto&& cout: chain_.outputs) {
    auto out = new Tensor(cout->frame());
    out->share(cout);
    outputs.push_back(out);
  }
  return outputs;
}

void Executor::run(const std::vector<Tensor*>& ins, std::vector<Tensor*>& outs) {
  stream(ins, chain_.inputs);
  loadSideInputs(chain_);
  runInternal();
  stream(chain_.outputs, outs);
}

auto Executor::chain() -> Chain& { return chain_; }
auto Executor::chain() const -> const Chain& { return chain_; }

}