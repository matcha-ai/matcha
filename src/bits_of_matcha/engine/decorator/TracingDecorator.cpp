#include "bits_of_matcha/engine/decorator/TracingDecorator.h"
#include "bits_of_matcha/engine/chain/Module.h"
#include "bits_of_matcha/engine/chain/executors/SinglecoreExecutor.h"


namespace matcha::engine {

TracingDecorator::TracingDecorator(const fn& function)
  : Decorator(function)
{}

std::vector<Tensor*> TracingDecorator::run(const std::vector<Tensor*>& inputs) {
  auto executor = cache(inputs);
  if (!tracing()) {
    return executor->run(inputs);
  } else {
    auto module = new Module(inputs, executor);
    auto outs = module->outputs.stdVector();
    dispatch(module);
    return outs;
  }
}

void TracingDecorator::build(const std::vector<Frame>& frames) {
  cache(frames);
}

void TracingDecorator::build(const std::vector<Tensor*>& tensors) {
  cache(tensors);
}

std::shared_ptr<Executor> TracingDecorator::cache(const std::vector<Tensor*>& tensors) {
  return cache(frames(tensors));
}

std::shared_ptr<Executor> TracingDecorator::cache(const std::vector<Frame>& frames) {
  std::string h = hash(frames);
  try {
    return cache_.at(h);
  } catch (std::out_of_range&) {
    auto chain = trace(preimage(), frames);
    auto executor = compile(std::move(chain));
    cache_[h] = executor;
    return executor;
  }
}

std::string TracingDecorator::hash(const std::vector<Frame>& frames) {
  std::string buffer;
  for (auto& frame: frames) buffer += frame.string();
  return buffer;
}

std::vector<Frame> TracingDecorator::frames(const std::vector<Tensor*>& tensors) {
  std::vector<Frame> result;
  result.reserve(tensors.size());
  for (auto&& t: tensors)
    result.push_back(t->frame());
  return result;
}

std::shared_ptr<Executor> TracingDecorator::compile(Chain chain) {
  return std::make_shared<SinglecoreExecutor>(std::move(chain));
}

}