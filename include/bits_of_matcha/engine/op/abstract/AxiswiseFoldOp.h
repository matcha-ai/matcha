#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/iterations/AxiswiseFoldCtx.h"
#include "bits_of_matcha/engine/cpu/kernels/axiswiseFold.h"
#include "bits_of_matcha/engine/ops/Cast.h"


namespace matcha::engine {

struct AxiswiseFoldOp : Op {
  explicit AxiswiseFoldOp(Tensor* a, bool keep_dims)
    : Op{a}
    , ctx_(a->shape(), keep_dims)
  {
    if (keep_dims)
      addOutput(a->dtype(), std::vector<unsigned>(a->rank(), 1));
    else
      addOutput(a->dtype(), {});
  }

  explicit AxiswiseFoldOp(Tensor* a, int axis, bool keep_dims)
    : Op{a}
    , ctx_(a->shape(), axis, keep_dims)
  {
    auto& shape = a->shape();
    if (axis < 0) axis += (int) shape.rank();
    axis_ = axis;

    std::vector<unsigned> outDims;
    for (int i = 0; i < shape.rank(); i++) {
      if (i == axis) {
        if (keep_dims)
          outDims.push_back(1);

        continue;
      }
      outDims.push_back(shape[i]);
    }

    addOutput(a->dtype(), outDims);
  }

  explicit AxiswiseFoldOp(Tensor* a, bool keep_dims, Dtype dtype)
    : Op{a}
    , ctx_(a->shape(), keep_dims)
  {
    if (inputs[0]->dtype() != dtype)
      engine::incept(this, new ops::Cast(inputs[0], dtype));

    if (keep_dims)
      addOutput(a->dtype(), std::vector<unsigned>(a->rank(), 1));
    else
      addOutput(a->dtype(), {});
  }

  explicit AxiswiseFoldOp(Tensor* a, int axis, bool keep_dims, Dtype dtype)
    : Op{a}
    , ctx_(a->shape(), axis, keep_dims)
  {
    auto& shape = a->shape();
    if (axis < 0) axis += (int) shape.rank();
    axis_ = axis;

    std::vector<unsigned> outDims;
    for (int i = 0; i < shape.rank(); i++) {
      if (i == axis) {
        if (keep_dims)
          outDims.push_back(1);

        continue;
      }
      outDims.push_back(shape[i]);
    }

    if (inputs[0]->dtype() != dtype)
      engine::incept(this, new ops::Cast(inputs[0], dtype));

    addOutput(dtype, outDims);
  }

  const AxiswiseFoldCtx& iter() const {
    return ctx_;
  }

  int axis() const {
    return axis_;
  }

  bool global() const {
    return outputs[0]->rank() == 0;
  }

  bool keepDims() const {
    return inputs[0]->rank() == outputs[0]->rank();
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
    cpu::axiswiseFold<Type>(callable, inputs[0]->buffer(), outputs[0]->malloc(), ctx_);
  }

protected:
  AxiswiseFoldCtx ctx_;
  int axis_;
};

}