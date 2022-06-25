#include "bits_of_matcha/engine/ops/Cast.h"


namespace matcha::engine::ops {

Cast::Cast(Tensor* a, const Dtype& dtype)
  : Op{a}
{
  outputs.add(this, dtype, a->shape());
}

void Cast::run() {
  outputs[0]->malloc();
}

}