#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinary.h"
#include "bits_of_matcha/engine/memory/implicitCast.h"
#include "bits_of_matcha/error/IncompatibleDtypesError.h"
#include "bits_of_matcha/engine/ops/Cast.h"
#include "bits_of_matcha/engine/flow/Tracer.h"

#include <algorithm>
#include <numeric>
#include <execution>
#include <iostream>


namespace matcha::engine {

struct ElementwiseBinaryOp : Op {
  explicit ElementwiseBinaryOp(Tensor* a, Tensor* b)
    : Op{a, b}
    , ctx_(a->shape(), b->shape())
  {
    Dtype dtype = promoteDtypes(a, b);
    outputs.add(this, dtype, ctx_.dimsC);
    for (auto&& in: inputs) {
      if (in->dtype() == dtype) continue;
      auto preop = new ops::Cast(in, dtype);
      engine::incept(this, preop);
    }
  }

  explicit ElementwiseBinaryOp(Tensor* a, Tensor* b, Dtype dtype)
    : Op{a, b}
    , ctx_(a->shape(), b->shape())
  {
    outputs.add(this, dtype, ctx_.dimsC);
    for (auto&& in: inputs) {
      if (in->dtype() == dtype) continue;
      auto preop = new ops::Cast(in, dtype);
      engine::incept(this, preop);
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
      cpu::elementwiseBinary<bool>(callable,
                                   a->buffer(), b->buffer(), c->malloc(),
                                   ctx_);
      break;
    case Float:
      cpu::elementwiseBinary<float>(callable,
                                    a->buffer(), b->buffer(), c->malloc(),
                                    ctx_);
      break;
    case Double:
      cpu::elementwiseBinary<double>(callable,
                                     a->buffer(), b->buffer(), c->malloc(),
                                     ctx_);
      break;
    case Sbyte:
      cpu::elementwiseBinary<int8_t>(callable,
                                     a->buffer(), b->buffer(), c->malloc(),
                                     ctx_);
      break;
    case Short:
      cpu::elementwiseBinary<int16_t>(callable,
                                      a->buffer(), b->buffer(), c->malloc(),
                                      ctx_);
      break;
    case Int:
      cpu::elementwiseBinary<int32_t>(callable,
                                      a->buffer(), b->buffer(), c->malloc(),
                                      ctx_);
      break;
    case Long:
      cpu::elementwiseBinary<int64_t>(callable,
                                      a->buffer(), b->buffer(), c->malloc(),
                                      ctx_);
      break;
    case Byte:
      cpu::elementwiseBinary<uint8_t>(callable,
                                     a->buffer(), b->buffer(), c->malloc(),
                                     ctx_);
      break;
    case Ushort:
      cpu::elementwiseBinary<uint16_t>(callable,
                                      a->buffer(), b->buffer(), c->malloc(),
                                      ctx_);
      break;
    case Uint:
      cpu::elementwiseBinary<uint32_t>(callable,
                                      a->buffer(), b->buffer(), c->malloc(),
                                      ctx_);
      break;
    case Ulong:
      cpu::elementwiseBinary<uint64_t>(callable,
                                      a->buffer(), b->buffer(), c->malloc(),
                                      ctx_);
      break;
    case Cint:
      cpu::elementwiseBinary<std::complex<int32_t>>(callable,
                                          a->buffer(), b->buffer(), c->malloc(),
                                          ctx_);
      break;
    case Cuint:
      cpu::elementwiseBinary<std::complex<uint32_t>>(callable,
                                                     a->buffer(), b->buffer(), c->malloc(),
                                                     ctx_);
      break;
    case Cfloat:
      cpu::elementwiseBinary<std::complex<float>>(callable,
                                                  a->buffer(), b->buffer(), c->malloc(),
                                                  ctx_);
      break;
    case Cdouble:
      cpu::elementwiseBinary<std::complex<double>>(callable,
                                                   a->buffer(), b->buffer(), c->malloc(),
                                                   ctx_);
      break;
    }
  }
};


}