#include "bits_of_matcha/engine/ops/Dot.h"


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

DotBack::DotBack(const BackCtx& ctx)
  : OpBack(ctx)
{}

OpMeta<DotBack> DotBack::meta {
  .name = "DotBack",
};

void DotBack::run() {
  auto a = inputs[0]->buffer()->as<float*>();
  auto b = inputs[1]->buffer()->as<float*>();
  auto c = outputs[0]->malloc()->as<float*>();
}

}