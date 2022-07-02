#include "bits_of_matcha/engine/ops/Cast.h"
#include "bits_of_matcha/engine/memory/cast.h"


namespace matcha::engine::ops {

Cast::Cast(Tensor* a, const Dtype& dtype)
  : Op{a}
{
  outputs.add(this, dtype, a->shape());
}

void Cast::run() {
  engine::cast(inputs[0]->buffer(), outputs[0]->malloc(),
               inputs[0]->dtype(), outputs[0]->dtype(), outputs[0]->size());
}

}