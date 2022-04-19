#include "bits_of_matcha/engine/ops/Transpose.h"
#include "bits_of_matcha/engine/ops/TransposeBack.h"


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
    dims = {iter_.rows, iter_.cols};
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

}

}
