#include "bits_of_matcha/engine/transform/TracingTransform.h"
#include "bits_of_matcha/engine/lambda/Module.h"
#include "bits_of_matcha/engine/lambda/executors/SinglecoreExecutor.h"


namespace matcha::engine {

TracingTransform::TracingTransform(const fn& function)
  : Transform(function)
{}

std::vector<Tensor*> TracingTransform::run(const std::vector<Tensor*>& inputs) {
  auto executor = cache(inputs);
  if (!tracing()) {
    return executor->run(inputs);
  } else {
    return dispatch<Module>(inputs, executor);
  }
}

void TracingTransform::build(const std::vector<Frame>& frames) {
  cache(frames);
}

void TracingTransform::build(const std::vector<Tensor*>& tensors) {
  cache(tensors);
}

std::shared_ptr<Executor> TracingTransform::cache(const std::vector<Tensor*>& tensors) {
  return cache(frames(tensors));
}

std::shared_ptr<Executor> TracingTransform::cache(const std::vector<Frame>& frames) {
  std::string h = hash(frames);
  try {
    auto& executor = cache_.at(h);

    for (auto&& [in, binding]: executor->lambda().side_inputs) {
      if (in->frame() != binding->frame()) {
        throw std::out_of_range("side input frame changed");
      }
    }

    return executor;
  } catch (std::out_of_range&) {
    auto lambda = trace(preimage(), frames);
    auto executor = compile(std::move(lambda));
    cache_[h] = executor;
    return executor;
  }
}

std::string TracingTransform::hash(const std::vector<Frame>& frames) {
  std::string buffer;
  for (auto& frame: frames) buffer += frame.string();
  return buffer;
}

std::vector<Frame> TracingTransform::frames(const std::vector<Tensor*>& tensors) {
  std::vector<Frame> result;
  result.reserve(tensors.size());
  for (auto&& t: tensors)
    result.push_back(t->frame());
  return result;
}

std::shared_ptr<Executor> TracingTransform::compile(Lambda lambda) {
  return std::make_shared<SinglecoreExecutor>(std::move(lambda));
}

}