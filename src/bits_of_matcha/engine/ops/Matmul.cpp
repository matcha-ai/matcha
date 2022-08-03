#include "bits_of_matcha/engine/ops/Matmul.h"
#include "bits_of_matcha/engine/ops/Reshape.h"
#include "bits_of_matcha/engine/ops/Cast.h"
#include "bits_of_matcha/engine/ops/Transpose.h"
#include "bits_of_matcha/engine/ops/Sum.h"
#include "bits_of_matcha/engine/cpu/kernels/mm.h"
#include "bits_of_matcha/error/IncompatibleShapesError.h"


namespace matcha::engine::ops {

Matmul::Matmul(Tensor* a, Tensor* b)
  : Op{a, b}
  , iter_(a->shape(), b->shape())
{
  Dtype dtype = promoteDtypes(a, b);
  if (!isFloatingReal(dtype))
    throw std::runtime_error("Matmul: input dtypes must be floating real");

  for (auto&& in: inputs) {
    if (in->dtype() == dtype) continue;
    auto preop = new ops::Cast(in, dtype);
    engine::incept(this, preop);
  }

  if (iter_.cols_a != iter_.rows_b) {
    throw IncompatibleShapesError(a->shape(), b->shape());
  }

  std::vector dims_c = iter_.prefix_dims_c;
  dims_c.push_back(iter_.rows_a);
  dims_c.push_back(iter_.cols_b);

  addOutput(dtype, dims_c);

//  for (int i = 0; i < iter_.prefix_strides_b.size(); i++) {
//    print(iter_.prefix_strides_a[i], " ", iter_.prefix_strides_b[i]);
//  }
}

Reflection<Matmul> Matmul::reflection {
  .name = "Matmul",
  .back = [](const BackCtx& ctx) {
    std::vector<Tensor*> result = {nullptr, nullptr};

    auto a = ctx.forward->inputs[0];
    auto b = ctx.forward->inputs[1];
    auto gc = ctx.vals[0];

    if (ctx.wrts[0]) {
      auto bt = dispatch<Transpose>(b)[0];
      auto ga = dispatch<Matmul>(gc, bt)[0];

      result[0] = ga;
    }

    if (ctx.wrts[1]) {
      auto gct = dispatch<Transpose>(gc)[0];
      auto gbt = dispatch<Matmul>(gct, a)[0];
      auto gb = dispatch<Transpose>(gbt)[0];
      result[1] = gb;
    }

    for (int i = 0; i < 0; i++) {
      if (result[i]->rank() > ctx.forward->inputs[i]->rank()) {
        std::vector<int> dims = {-1};
        for (auto&& dim: ctx.forward->inputs[i]->shape())
          dims.push_back(dim);
        result[i] = dispatch<Reshape>(result[i], dims)[0];
        result[i] = dispatch<Sum>(result[i], 0, false)[0];
      }
    }

    return result;
  },
};

void Matmul::run() {
  switch (outputs[0]->dtype()) {
  case Float:
    cpu::mm<float>(inputs[0]->buffer(), inputs[1]->buffer(), outputs[0]->malloc(), iter_); break;
  case Double:
    cpu::mm<double>(inputs[0]->buffer(), inputs[1]->buffer(), outputs[0]->malloc(), iter_); break;
  }
}

}