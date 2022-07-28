#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinary.h"
#include "bits_of_matcha/engine/memory/implicitCast.h"
#include "bits_of_matcha/error/IncompatibleDtypesError.h"
#include "bits_of_matcha/engine/ops/Cast.h"
#include "bits_of_matcha/engine/chain/Tracer.h"

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
  inline void runCpu(const Callable& callable) {
    Dtype dtype = outputs[0]->dtype();
    switch (dtype) {

    case Float: return runCpuTyped<float>(callable);
    case Double: return runCpuTyped<double>(callable);

    case Sbyte: return runCpuTyped<int8_t>(callable);
    case Short: return runCpuTyped<int16_t>(callable);
    case Int: return runCpuTyped<int32_t>(callable);
    case Long: return runCpuTyped<int64_t>(callable);

    case Byte: return runCpuTyped<uint8_t>(callable);
    case Ushort: return runCpuTyped<uint16_t>(callable);
    case Uint: return runCpuTyped<uint32_t>(callable);
    case Ulong: return runCpuTyped<uint64_t>(callable);

    case Cint: return runCpuTyped<std::complex<int32_t>>(callable);
    case Cuint: return runCpuTyped<std::complex<uint32_t>>(callable);
    case Cfloat: return runCpuTyped<std::complex<float>>(callable);
    case Cdouble: return runCpuTyped<std::complex<double>>(callable);

    case Bool: return runCpuTyped<bool>(callable);

    }
  }

  template <class Callable>
  inline void runCpuReal(const Callable& callable) {
    Dtype dtype = outputs[0]->dtype();
    switch (dtype) {

    case Float: return runCpuTyped<float>(callable);
    case Double: return runCpuTyped<double>(callable);

    case Sbyte: return runCpuTyped<int8_t>(callable);
    case Short: return runCpuTyped<int16_t>(callable);
    case Int: return runCpuTyped<int32_t>(callable);
    case Long: return runCpuTyped<int64_t>(callable);

    case Byte: return runCpuTyped<uint8_t>(callable);
    case Ushort: return runCpuTyped<uint16_t>(callable);
    case Uint: return runCpuTyped<uint32_t>(callable);
    case Ulong: return runCpuTyped<uint64_t>(callable);

    case Bool: return runCpuTyped<bool>(callable);

    }
  }

  template <class Callable>
  inline void runCpuFloating(const Callable& callable) {
    Dtype dtype = outputs[0]->dtype();
    switch (dtype) {

    case Float: return runCpuTyped<float>(callable);
    case Double: return runCpuTyped<double>(callable);

    case Cfloat: return runCpuTyped<std::complex<float>>(callable);
    case Cdouble: return runCpuTyped<std::complex<double>>(callable);

    }
  }

  template <class Callable>
  inline void runCpuFloatingReal(const Callable& callable) {
    Dtype dtype = outputs[0]->dtype();
    switch (dtype) {

    case Float: return runCpuTyped<float>(callable);
    case Double: return runCpuTyped<double>(callable);

    }
  }

  template <class Callable>
  inline void runCpuSignedReal(const Callable& callable) {
    Dtype dtype = outputs[0]->dtype();
    switch (dtype) {

    case Sbyte: return runCpuTyped<int8_t>(callable);
    case Short: return runCpuTyped<int16_t>(callable);
    case Int: return runCpuTyped<int32_t>(callable);
    case Long: return runCpuTyped<int64_t>(callable);

    }
  }

  template <class Callable>
  inline void runCpuUnsignedReal(const Callable& callable) {
    Dtype dtype = outputs[0]->dtype();
    switch (dtype) {

    case Byte: return runCpuTyped<uint8_t>(callable);
    case Ushort: return runCpuTyped<uint16_t>(callable);
    case Uint: return runCpuTyped<uint32_t>(callable);
    case Ulong: return runCpuTyped<uint64_t>(callable);

    case Bool: return runCpuTyped<bool>(callable);
    }
  }

  template <class Callable>
  inline void runCpuComplex(const Callable& callable) {
    Dtype dtype = outputs[0]->dtype();
    switch (dtype) {

    case Cint: return runCpuTyped<std::complex<int32_t>>(callable);
    case Cuint: return runCpuTyped<std::complex<uint32_t>>(callable);
    case Cfloat: return runCpuTyped<std::complex<float>>(callable);
    case Cdouble: return runCpuTyped<std::complex<double>>(callable);

    }
  }

  template <class Callable>
  inline void runCpuFloatingComplex(const Callable& callable) {
    Dtype dtype = outputs[0]->dtype();
    switch (dtype) {

    case Cfloat: return runCpuTyped<std::complex<float>>(callable);
    case Cdouble: return runCpuTyped<std::complex<double>>(callable);

    }
  }

  template <class Callable>
  inline void runCpuSignedComplex(const Callable& callable) {
    Dtype dtype = outputs[0]->dtype();
    switch (dtype) {

    case Cint: return runCpuTyped<std::complex<int32_t>>(callable);

    }
  }

  template <class Callable>
  inline void runCpuUnsignedComplex(const Callable& callable) {
    Dtype dtype = outputs[0]->dtype();
    switch (dtype) {

    case Cuint: return runCpuTyped<std::complex<uint32_t>>(callable);

    }
  }

private:
  template <class Type, class Callable>
  inline void runCpuTyped(const Callable& callable) {
    auto a = inputs[0];
    auto b = inputs[1];
    auto c = outputs[0];

    cpu::elementwiseBinary<Type>(callable,
                                 a->buffer(), b->buffer(), c->malloc(),
                                 ctx_);
  }

};


}