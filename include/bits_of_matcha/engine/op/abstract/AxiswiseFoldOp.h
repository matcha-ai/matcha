#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/iterations/AxiswiseFoldCtx.h"
#include "bits_of_matcha/engine/cpu/kernels/axiswiseFold.h"
#include "bits_of_matcha/engine/ops/Cast.h"


namespace matcha::engine {

struct AxiswiseFoldOp : Op {
  explicit AxiswiseFoldOp(Tensor* a)
    : Op{a}
    , ctx_(a->shape())
  {
    outputs.add(this, a->dtype(), {});
  }

  explicit AxiswiseFoldOp(Tensor* a, int axis)
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

  explicit AxiswiseFoldOp(Tensor* a, Dtype dtype)
    : Op{a}
    , ctx_(a->shape())
  {
    if (inputs[0]->dtype() != dtype)
      engine::incept(this, new ops::Cast(inputs[0], dtype));

    outputs.add(this, dtype, {});
  }

  explicit AxiswiseFoldOp(Tensor* a, int axis, Dtype dtype)
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

    if (inputs[0]->dtype() != dtype)
      engine::incept(this, new ops::Cast(inputs[0], dtype));

    outputs.add(this, dtype, outDims);
  }

protected:
  AxiswiseFoldCtx ctx_;

  template <class T, class Callback>
  void runCPU(const Callback& callback) {
    cpu::axiswiseFold<T, Callback>(
      callback,
      inputs[0]->buffer(),
      outputs[0]->malloc(),
      ctx_
    );
  }
};

}