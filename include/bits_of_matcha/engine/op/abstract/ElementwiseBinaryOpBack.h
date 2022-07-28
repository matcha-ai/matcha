#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/ops/Cast.h"
#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"


namespace matcha::engine {

struct ElementwiseBinaryOpBack : Op {
  explicit ElementwiseBinaryOpBack(const BackCtx& ctx)
    : Op{ctx.vals}
    , forward_(ctx.forward)
    , iter_(forward_->inputs[0]->shape(), forward_->inputs[1]->shape())
  {
    Dtype dtype = Float;
    for (auto&& in: inputs) {
      if (in->dtype() == dtype) continue;
      auto preop = new ops::Cast(in, dtype);
      incept(this, preop);
    }

    for (int i = 0; i < ctx.wrts.size(); i++) {
      if (!ctx.wrts[i]) {
        outputs.add(this, nullptr);
        continue;
      }
      outputs.add(this, Float, forward_->inputs[i]->shape());
    }
  }

protected:
  Op* forward_;
  ElementwiseBinaryCtx iter_;
};

}