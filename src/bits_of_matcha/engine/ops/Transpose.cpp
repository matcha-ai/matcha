#include "bits_of_matcha/engine/ops/Transpose.h"
#include "bits_of_matcha/engine/cpu/kernels/transpose.h"

#include <cblas.h>


namespace matcha::engine::ops {

Transpose::Transpose(Tensor* a)
  : Op{a}
  , iter_(a->shape())
{
  auto& shapeA = a->shape();
  std::vector<unsigned> dims;

  if (shapeA.rank() >= 2) {
    dims = std::vector(shapeA.begin(), shapeA.end());
    std::swap(dims[dims.size() - 1], dims[dims.size() - 2]);
  } else {
    throw std::invalid_argument("can't transpose scalar or vector");
  }

  outputs.add(this, a->dtype(), dims);
}

OpMeta<Transpose> Transpose::meta {
  .name = "Transpose",
  .back = [](auto& ctx) {
    return new TransposeBack(ctx);
  },
};

void Transpose::run() {
  if (iter_.rows == 1 || iter_.cols == 1) {
    outputs[0]->share(inputs[0]);
    return;
  }

  cpu::transpose(inputs[0]->buffer(), outputs[0]->malloc(), iter_);
}

TransposeBack::TransposeBack(const BackCtx& ctx)
  : OpBack(ctx)
  , iter_(outputs[0]->shape())
{}

OpMeta<TransposeBack> TransposeBack::meta {
  .name = "TransposeBack",
};

void TransposeBack::run() {
  if (iter_.rows == 1 || iter_.cols == 1) {
    outputs[0]->share(inputs[0]);
    return;
  }

  cpu::transpose(inputs[0]->buffer(), outputs[0]->malloc(), iter_);
}


}
