#include "bits_of_matcha/engine/ops/Stack.h"

#include <numeric>
#include <algorithm>
#include <execution>


namespace matcha::engine::ops {

Stack::Stack(const std::vector<Tensor*>& ins)
  : Op(ins)
{
  if (inputs.empty())
    throw std::invalid_argument("expected at least one tensor to stack");
  auto& frame = inputs[0]->frame();
  for (int i = 1; i < inputs.size(); i++) {
    if (inputs[i]->frame() != frame)
      throw std::invalid_argument("can't stack tensors with different frames");
  }
  std::vector<unsigned> dims = {(unsigned) inputs.size()};
  dims.reserve(frame.shape().size() + 1);
  for (auto dim: frame.shape()) dims.push_back(dim);
  addOutput(frame.dtype(), dims);
}

void Stack::run() {
  auto begin = outputs[0]->malloc().as<float*>();
  auto end = begin + outputs[0]->size();
  size_t stride = inputs[0]->size();
  int i = 0;
  for (auto it = begin; it != end; it += stride) {
    auto source = inputs[i++]->buffer().as<float*>();
    std::copy(
      std::execution::par_unseq,
      source, source + stride,
      it
    );
  }
}


}