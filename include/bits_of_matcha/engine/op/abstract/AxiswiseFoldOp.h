#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/iterations/AxiswiseFoldCtx.h"
#include "bits_of_matcha/engine/cpu/kernels/axiswiseFold.h"


namespace matcha::engine {

struct AxiswiseFoldOp : Op {
  AxiswiseFoldOp(Tensor* a)
    : Op{a}
    , ctx_(a->shape())
  {
    outputs.add(this, a->dtype(), {});
  }

  AxiswiseFoldOp(Tensor* a, int axis)
    : Op{a}
    , ctx_(a->shape(), axis)
  {
    auto& shape = a->shape();
    if (axis < 0) axis += (int) shape.rank();
    std::vector<unsigned> outDims;
    for (int i = 0; i < shape.rank(); i++) {
      if (i == axis) continue;
      outDims.push_back(shape[i]);
    }

    outputs.add(this, a->dtype(), outDims);
  }

protected:
  AxiswiseFoldCtx ctx_;

  template <class Callback>
  void runCPU(const Callback& callback) {
    cpu::axiswiseFold(
      callback,
      inputs[0]->buffer(),
      outputs[0]->malloc(),
      ctx_
    );
  }
};

}