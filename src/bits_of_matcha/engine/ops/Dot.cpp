#include "bits_of_matcha/engine/ops/Dot.h"
#include "bits_of_matcha/engine/ops/Cast.h"
#include "bits_of_matcha/engine/cpu/kernels/mm.h"
#include "bits_of_matcha/error/IncompatibleShapesError.h"


namespace matcha::engine::ops {

Dot::Dot(Tensor* a, Tensor* b)
  : Op{a, b}
  , iter_(a->shape(), b->shape())
{
  Dtype dtype = promoteDtypes(a, b);
  switch (dtype) {
  case Float:
  case Double:
    break;
  default:
    throw std::runtime_error("Dot: dtype " + dtype.string() + " is not supported");
  }

  for (auto&& in: inputs) {
    if (in->dtype() == dtype) continue;
    auto preop = new ops::Cast(in, dtype);
    engine::incept(this, preop);
  }

  if (iter_.colsA != iter_.rowsB) {
    throw IncompatibleShapesError(a->shape(), b->shape());
  }

  std::vector dimsC = iter_.prefixDimsC;
  dimsC.push_back(iter_.rowsA);
  dimsC.push_back(iter_.colsB);

  outputs.add(this, dtype, dimsC);

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
  switch (outputs[0]->dtype()) {
  case Float:
    cpu::mm<float>(inputs[0]->buffer(), inputs[1]->buffer(), outputs[0]->malloc(), iter_); break;
  case Double:
    cpu::mm<double>(inputs[0]->buffer(), inputs[1]->buffer(), outputs[0]->malloc(), iter_); break;
  }
}

DotBack::DotBack(const BackCtx& ctx)
  : OpBack(ctx)
{}

OpMeta<DotBack> DotBack::meta {
  .name = "DotBack",
};

void DotBack::run() {
//  if (inputs[0]) {
//    inputs[0]->malloc().as<float*>();
//
//  }
//
//  if (inputs[1]) {
//    inputs[1]->malloc().as<float*>();
//
//  }
}

}