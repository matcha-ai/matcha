#include "bits_of_matcha/engine/ops/Dot.h"
#include "bits_of_matcha/engine/ops/DotBack.h"


namespace matcha::engine::ops {

Dot::Dot(Tensor* a, Tensor* b)
  : Op{a, b}
  , iterA_(a->shape())
  , iterB_(b->shape())
{
  outputs.add(this, a->dtype(), {iterA_.rows, iterB_.cols});
}

OpMeta<Dot> Dot::meta {
  .name = "Dot",
  .back = [](auto& ctx) {
    return new DotBack(ctx);
  },
};

void Dot::run() {

}

}