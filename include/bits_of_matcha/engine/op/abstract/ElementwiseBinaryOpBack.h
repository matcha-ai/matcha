#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/ops/Cast.h"
#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"
#include "bits_of_matcha/engine/utils/stdVector.h"


namespace matcha::engine {

struct ElementwiseBinaryOpBack : Op {
  explicit ElementwiseBinaryOpBack(const BackCtx& ctx)
    : Op{cat(ctx.vals, ctx.forward->outputs, ctx.forward->inputs)}
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
        addOutput(nullptr);
        continue;
      }
      addOutput(Float, forwardInput(i)->shape());
    }
  }

protected:
  Tensor* forwardInput(int idx) {
    int begin = (int) inputs.size() - forward_->inputs.size();
    return inputs[begin + idx];
  }
  Tensor* forwardOutput(int idx) {
    int begin = (int) inputs.size() - forward_->inputs.size() - forward_->outputs.size();
    return inputs[begin + idx];
  }

protected:
  Op* forward_;
  ElementwiseBinaryCtx iter_;
};

}