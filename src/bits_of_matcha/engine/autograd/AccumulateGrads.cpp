#include "bits_of_matcha/engine/autograd/AccumulateGrads.h"
#include "bits_of_matcha/engine/cpu/fill.h"

#include <numeric>
#include <execution>
#include <algorithm>


namespace matcha::engine::autograd {

AccumulateGrads::AccumulateGrads(const std::vector<Tensor*>& grads, Tensor* target)
  : Op(grads)
{
  outputs.add(this, target);
}

OpMeta<AccumulateGrads> AccumulateGrads::meta {
  .name = "AccumulateGrads",
};

void AccumulateGrads::run() {
//  print("AccumulateGrads");
//  print("", inputs[0] ," -> ", outputs[0]);
//  print();
//  print("input ", inputs[0]);
//  return;
//  if (inputs.size() == 1) {
//    outputs[0]->shareBuffer(inputs[0]);
//    return;
//  }

  auto buffer = outputs[0]->malloc();
  size_t size = outputs[0]->size();
  cpu::fill(buffer, size, 0);
  auto values = buffer->as<float*>();

  for (auto grad: inputs) {
//    print("grad: ", grad, " buffer: ",grad->buffer());
//    print(grad->buffer());
    std::transform(
      std::execution::par_unseq,
      values, values + size,
      grad->buffer()->as<float*>(),
      values,
      std::plus()
    );
  }
}

}