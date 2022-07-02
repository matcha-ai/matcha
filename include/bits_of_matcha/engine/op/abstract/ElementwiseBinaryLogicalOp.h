#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinaryLogical.h"
#include "bits_of_matcha/engine/memory/implicitCast.h"
#include "bits_of_matcha/error/IncompatibleDtypesError.h"
#include "bits_of_matcha/engine/ops/Cast.h"

#include <algorithm>
#include <numeric>
#include <execution>
#include <iostream>


namespace matcha::engine {

struct ElementwiseBinaryLogicalOp : Op {
  explicit ElementwiseBinaryLogicalOp(Tensor* a, Tensor* b)
    : Op{a, b}
    , ctx_(a->shape(), b->shape())
  {
    Dtype dtype = promoteDtypes(a, b);
    outputs.add(this, Bool, ctx_.dimsC);
    for (auto&& in: inputs) {
      if (in->dtype() == dtype) continue;
      auto op = new ops::Cast(in, dtype);
      incept(this, op, in);
    }
  }

protected:
  ElementwiseBinaryCtx ctx_;

  template <class Callable>
  void runCPU(Callable callable) {
    auto a = inputs[0];
    auto b = inputs[1];
    auto c = outputs[0];

    switch (a->dtype()) {
    case Half:
    case Bool:
      cpu::elementwiseBinaryLogical<bool>(callable,
                                   a->buffer(), b->buffer(), c->malloc(),
                                   ctx_);
      break;
    case Float:
      cpu::elementwiseBinaryLogical<float>(callable,
                                    a->buffer(), b->buffer(), c->malloc(),
                                    ctx_);
      break;
    case Double:
      cpu::elementwiseBinaryLogical<double>(callable,
                                     a->buffer(), b->buffer(), c->malloc(),
                                     ctx_);
      break;
    case Sbyte:
      cpu::elementwiseBinaryLogical<int8_t>(callable,
                                     a->buffer(), b->buffer(), c->malloc(),
                                     ctx_);
      break;
    case Short:
      cpu::elementwiseBinaryLogical<int16_t>(callable,
                                      a->buffer(), b->buffer(), c->malloc(),
                                      ctx_);
      break;
    case Int:
      cpu::elementwiseBinaryLogical<int32_t>(callable,
                                      a->buffer(), b->buffer(), c->malloc(),
                                      ctx_);
      break;
    case Long:
      cpu::elementwiseBinaryLogical<int64_t>(callable,
                                      a->buffer(), b->buffer(), c->malloc(),
                                      ctx_);
      break;
    case Byte:
      cpu::elementwiseBinaryLogical<uint8_t>(callable,
                                            a->buffer(), b->buffer(), c->malloc(),
                                            ctx_);
      break;
    case Ushort:
      cpu::elementwiseBinaryLogical<uint16_t>(callable,
                                             a->buffer(), b->buffer(), c->malloc(),
                                             ctx_);
      break;
    case Uint:
      cpu::elementwiseBinaryLogical<uint32_t>(callable,
                                             a->buffer(), b->buffer(), c->malloc(),
                                             ctx_);
      break;
    case Ulong:
      cpu::elementwiseBinaryLogical<uint64_t>(callable,
                                             a->buffer(), b->buffer(), c->malloc(),
                                             ctx_);
      break;
    }
  }
};


}