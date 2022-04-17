#include "bits_of_matcha/engine/autograd/AccumulateGrads.h"


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

}

}