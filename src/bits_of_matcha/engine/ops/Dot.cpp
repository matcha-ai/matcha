#include "bits_of_matcha/engine/ops/Dot.h"
#include "bits_of_matcha/engine/cpu/kernels/mm.h"
#include "bits_of_matcha/error/IncompatibleShapesError.h"


namespace matcha::engine::ops {

Dot::Dot(Tensor* a, Tensor* b)
  : Op{a, b}
  , iter_(a->shape(), b->shape())
{
  if (a->dtype() != b->dtype()) {
    throw std::invalid_argument("dtype mismatch");
  }

  if (iter_.colsA != iter_.rowsB) {
    throw IncompatibleShapesError(a->shape(), b->shape());
  }

  std::vector dimsC = iter_.prefixDimsC;
  dimsC.push_back(iter_.rowsA);
  dimsC.push_back(iter_.colsB);

  outputs.add(this, a->dtype(), dimsC);

//  for (int i = 0; i < iter_.prefixStridesB.size(); i++) {
//    print(iter_.prefixStridesA[i], " ", iter_.prefixStridesB[i]);
//  }
}

OpMeta<Dot> Dot::meta {
  .name = "Dot",
  .back = [](auto& ctx) {
    return new DotBack(ctx);
  },
};

void Dot::run() {
  cpu::mm(inputs[0]->buffer(), inputs[1]->buffer(), outputs[0]->malloc(), iter_);
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