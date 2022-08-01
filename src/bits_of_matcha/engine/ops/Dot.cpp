#include "bits_of_matcha/engine/ops/Dot.h"
#include "bits_of_matcha/engine/ops/Cast.h"
#include "bits_of_matcha/engine/ops/Transpose.h"
#include "bits_of_matcha/engine/ops/Sum.h"
#include "bits_of_matcha/engine/cpu/kernels/mm.h"
#include "bits_of_matcha/error/IncompatibleShapesError.h"


namespace matcha::engine::ops {

Dot::Dot(Tensor* a, Tensor* b)
  : Op{a, b}
  , iter_(a->shape(), b->shape())
{
  Dtype dtype = promoteDtypes(a, b);
  if (!isFloatingReal(dtype))
    throw std::runtime_error("Dot: input dtypes must be floating real");

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

Reflection<Dot> Dot::reflection {
  .name = "Dot",
  /*
  .back = [](const BackCtx& ctx) {
    BackOps bops;
    auto a = ctx.forward_->inputs[0];
    auto b = ctx.forward_->inputs[1];
    auto c = ctx.vals[0];

    if (ctx.wrts[0]) {
      auto bt = new Transpose(b);
      auto da = new Dot(c, bt->outputs[0]);
      bops.ops.push_back(bt);
      bops.ops.push_back(da);
      bops.outputs.push_back(da->outputs[0]);
    }
    if (ctx.wrts[1]) {
      auto ct = new Transpose(c);
      auto dbt = new Dot(ct->outputs[0], a);
      auto db = new Transpose(dbt->outputs[0]);
      bops.ops.push_back(ct);
      bops.ops.push_back(dbt);
      bops.ops.push_back(db);
      bops.outputs.push_back(db->outputs[0]);
    }
    return bops;
  },
  */
};

void Dot::run() {
  switch (outputs[0]->dtype()) {
  case Float:
    cpu::mm<float>(inputs[0]->buffer(), inputs[1]->buffer(), outputs[0]->malloc(), iter_); break;
  case Double:
    cpu::mm<double>(inputs[0]->buffer(), inputs[1]->buffer(), outputs[0]->malloc(), iter_); break;
  }
}

}