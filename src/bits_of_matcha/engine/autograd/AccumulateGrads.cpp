#include "bits_of_matcha/engine/autograd/AccumulateGrads.h"
#include "bits_of_matcha/engine/cpu/kernels/fill.h"


namespace matcha::engine {

AccumulateGrads::AccumulateGrads(const std::vector<Tensor*>& grads)
  : Op(grads)
{
  if (grads.empty()) throw std::runtime_error("grads are empty");
  addOutput(Float, grads[0]->shape());
}

AccumulateGrads::AccumulateGrads(const std::vector<Tensor*>& grads, Tensor* target)
  : Op(grads)
{
  addOutput(target);
  if (target->dtype() != Float)
    throw std::runtime_error("target is not of type Float");
}

Reflection<AccumulateGrads> AccumulateGrads::reflection {
  .name = "AccumulateGrads"
};

void AccumulateGrads::run() {
//  print("accumulate: ", inputs.size());
  if (inputs.size() == 1) {
//    print("just one");
    outputs[0]->share(inputs[0]);
    return;
  }

  auto& b = outputs[0]->malloc();
  cpu::fill(b, outputs[0]->size(), 0);

  for (auto in: inputs) {
    auto f = in->buffer().as<float*>();
    auto g = b.as<float*>();
    std::transform(std::execution::par_unseq,
                   f, f + in->size(),
                   g,
                   g,
                   std::plus<>());
  }
}

}