#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseUnary.h"

#include <algorithm>
#include <numeric>
#include <execution>


namespace matcha::engine {


struct ElementwiseUnaryOp : public Op {
  explicit ElementwiseUnaryOp(Tensor* a)
    : Op{a}
  {
    size_ = a->size();
    addOutput(a->frame());
  }

protected:

  template <class Callable>
  inline void runCpu(const Callable& callable) {
    Dtype dtype = inputs[0]->dtype();
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
    Dtype dtype = inputs[0]->dtype();
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
    Dtype dtype = inputs[0]->dtype();
    switch (dtype) {

    case Float: return runCpuTyped<float>(callable);
    case Double: return runCpuTyped<double>(callable);

    case Cfloat: return runCpuTyped<std::complex<float>>(callable);
    case Cdouble: return runCpuTyped<std::complex<double>>(callable);

    }
  }

  template <class Callable>
  inline void runCpuFloatingReal(const Callable& callable) {
    Dtype dtype = inputs[0]->dtype();
    switch (dtype) {

    case Float: return runCpuTyped<float>(callable);
    case Double: return runCpuTyped<double>(callable);

    }
  }

  template <class Callable>
  inline void runCpuSignedReal(const Callable& callable) {
    Dtype dtype = inputs[0]->dtype();
    switch (dtype) {

    case Sbyte: return runCpuTyped<int8_t>(callable);
    case Short: return runCpuTyped<int16_t>(callable);
    case Int: return runCpuTyped<int32_t>(callable);
    case Long: return runCpuTyped<int64_t>(callable);

    }
  }

  template <class Callable>
  inline void runCpuUnsignedReal(const Callable& callable) {
    Dtype dtype = inputs[0]->dtype();
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
    Dtype dtype = inputs[0]->dtype();
    switch (dtype) {

    case Cint: return runCpuTyped<std::complex<int32_t>>(callable);
    case Cuint: return runCpuTyped<std::complex<uint32_t>>(callable);
    case Cfloat: return runCpuTyped<std::complex<float>>(callable);
    case Cdouble: return runCpuTyped<std::complex<double>>(callable);

    }
  }

  template <class Callable>
  inline void runCpuFloatingComplex(const Callable& callable) {
    Dtype dtype = inputs[0]->dtype();
    switch (dtype) {

    case Cfloat: return runCpuTyped<std::complex<float>>(callable);
    case Cdouble: return runCpuTyped<std::complex<double>>(callable);

    }
  }

  template <class Callable>
  inline void runCpuSignedComplex(const Callable& callable) {
    Dtype dtype = inputs[0]->dtype();
    switch (dtype) {

    case Cint: return runCpuTyped<std::complex<int32_t>>(callable);

    }
  }

  template <class Callable>
  inline void runCpuUnsignedComplex(const Callable& callable) {
    Dtype dtype = inputs[0]->dtype();
    switch (dtype) {

    case Cuint: return runCpuTyped<std::complex<uint32_t>>(callable);

    }
  }

private:

  template <class Type, class Callable>
  inline void runCpuTyped(const Callable& callable) {
    cpu::elementwiseUnary<Type>(callable, inputs[0]->buffer(), outputs[0]->malloc(), size_);
  }

protected:
  size_t size_;

};




}
