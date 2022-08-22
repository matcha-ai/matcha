#include "bits_of_matcha/engine/lambda/Executor.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"


namespace matcha::engine {

Executor::Executor(Lambda&& lambda)
  : lambda_(std::move(lambda))
{}

void stream(const std::vector<Tensor*>& source, std::vector<Tensor*>& target) {
  if (source.size() != target.size())
    throw std::runtime_error("source and target count mismatch");

  for (int i = 0; i < source.size(); i++) {
    target[i]->share(source[i]);
  }
}

void loadSideInputs(Lambda& lambda) {
  for (auto&& [in, binding]: lambda.side_inputs) {
    in->share(deref(binding));
  }
}

auto Executor::run(const std::vector<Tensor*>& ins) -> std::vector<Tensor*> {
  stream(ins, lambda_.inputs);
  loadSideInputs(lambda_);
  runInternal();
  std::vector<Tensor*> outputs;
  for (auto&& cout: lambda_.outputs) {
    auto out = new Tensor(cout->frame());
    out->share(cout);
    outputs.push_back(out);
  }
  return outputs;
}

void Executor::run(const std::vector<Tensor*>& ins, std::vector<Tensor*>& outs) {
  stream(ins, lambda_.inputs);
  loadSideInputs(lambda_);
  runInternal();
  stream(lambda_.outputs, outs);
}

auto Executor::lambda() -> Lambda& { return lambda_; }
auto Executor::lambda() const -> const Lambda& { return lambda_; }

}