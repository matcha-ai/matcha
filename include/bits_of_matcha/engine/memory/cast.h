#pragma once

#include "bits_of_matcha/engine/memory/Buffer.h"
#include "bits_of_matcha/engine/cpu/kernels/cast.h"

#include <execution>
#include <numeric>
#include <algorithm>
#include <ccomplex>

namespace matcha::engine {

inline void cast(Buffer& in, Buffer& out, const Dtype& from, const Dtype& to, size_t size) {
  if (from == to) {
    out = in;
    return;
  }

  switch (from) {
  case Half:
    throw std::runtime_error("Half is not supported");
    break;
  case Float:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Double: cpu::cast<float, double>(in, out, size); break;

    case Sbyte: cpu::cast<float, int8_t>(in, out, size);
    case Short: cpu::cast<float, int16_t>(in, out, size);
    case Int: cpu::cast<float, int32_t>(in, out, size); break;
    case Long: cpu::cast<float, int64_t>(in, out, size); break;

    case Byte: cpu::cast<float, uint8_t>(in, out, size);
    case Ushort: cpu::cast<float, uint16_t>(in, out, size);
    case Uint: cpu::cast<float, uint32_t>(in, out, size); break;
    case Ulong: cpu::cast<float, uint64_t>(in, out, size); break;

    case Cint: cpu::cast<float, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::cast<float, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::cast<float, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::cast<float, std::complex<double>>(in, out, size); break;

    case Bool: cpu::cast<float, bool>(in, out, size); break;
    }
    break;
  case Double:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::cast<double, float>(in, out, size); break;

    case Sbyte: cpu::cast<double, int8_t>(in, out, size);
    case Short: cpu::cast<double, int16_t>(in, out, size);
    case Int: cpu::cast<double, int32_t>(in, out, size); break;
    case Long: cpu::cast<double, int64_t>(in, out, size); break;

    case Byte: cpu::cast<double, uint8_t>(in, out, size);
    case Ushort: cpu::cast<double, uint16_t>(in, out, size);
    case Uint: cpu::cast<double, uint32_t>(in, out, size); break;
    case Ulong: cpu::cast<double, uint64_t>(in, out, size); break;

    case Cint: cpu::cast<double, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::cast<double, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::cast<double, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::cast<double, std::complex<double>>(in, out, size); break;

    case Bool: cpu::cast<double, bool>(in, out, size); break;
    }
    break;

  case Sbyte:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::cast<int8_t, float>(in, out, size); break;
    case Double: cpu::cast<int8_t, double>(in, out, size); break;

    case Short: cpu::cast<int8_t, int16_t>(in, out, size);
    case Int: cpu::cast<int8_t, int32_t>(in, out, size); break;
    case Long: cpu::cast<int8_t, int64_t>(in, out, size); break;

    case Byte: cpu::cast<int8_t, uint8_t>(in, out, size);
    case Ushort: cpu::cast<int8_t, uint16_t>(in, out, size);
    case Uint: cpu::cast<int8_t, uint32_t>(in, out, size); break;
    case Ulong: cpu::cast<int8_t, uint64_t>(in, out, size); break;

    case Cint: cpu::cast<int8_t, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::cast<int8_t, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::cast<int8_t, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::cast<int8_t, std::complex<double>>(in, out, size); break;

    case Bool: cpu::cast<int8_t, bool>(in, out, size); break;
    }
    break;

  case Short:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::cast<int16_t, float>(in, out, size); break;
    case Double: cpu::cast<int16_t, double>(in, out, size); break;

    case Sbyte: cpu::cast<int16_t, int8_t>(in, out, size);
    case Int: cpu::cast<int16_t, int32_t>(in, out, size); break;
    case Long: cpu::cast<int16_t, int64_t>(in, out, size); break;

    case Byte: cpu::cast<int16_t, uint8_t>(in, out, size);
    case Ushort: cpu::cast<int16_t, uint16_t>(in, out, size);
    case Uint: cpu::cast<int16_t, uint32_t>(in, out, size); break;
    case Ulong: cpu::cast<int16_t, uint64_t>(in, out, size); break;

    case Cint: cpu::cast<int16_t, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::cast<int16_t, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::cast<int16_t, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::cast<int16_t, std::complex<double>>(in, out, size); break;

    case Bool: cpu::cast<int16_t, bool>(in, out, size); break;
    }
    break;

  case Int:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::cast<int32_t, float>(in, out, size); break;
    case Double: cpu::cast<int32_t, double>(in, out, size); break;

    case Sbyte: cpu::cast<int32_t, int8_t>(in, out, size);
    case Short: cpu::cast<int32_t, int16_t>(in, out, size); break;
    case Long: cpu::cast<int32_t, int64_t>(in, out, size); break;

    case Byte: cpu::cast<int32_t, uint8_t>(in, out, size);
    case Ushort: cpu::cast<int32_t, uint16_t>(in, out, size);
    case Uint: cpu::cast<int32_t, uint32_t>(in, out, size); break;
    case Ulong: cpu::cast<int32_t, uint64_t>(in, out, size); break;

    case Cint: cpu::cast<int32_t, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::cast<int32_t, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::cast<int32_t, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::cast<int32_t, std::complex<double>>(in, out, size); break;

    case Bool: cpu::cast<int32_t, bool>(in, out, size); break;
    }
    break;

  case Long:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::cast<int64_t, float>(in, out, size); break;
    case Double: cpu::cast<int64_t, double>(in, out, size); break;

    case Sbyte: cpu::cast<int64_t, int8_t>(in, out, size);
    case Short: cpu::cast<int64_t, int16_t>(in, out, size); break;
    case Int: cpu::cast<int64_t, int32_t>(in, out, size); break;

    case Byte: cpu::cast<int64_t, uint8_t>(in, out, size);
    case Ushort: cpu::cast<int64_t, uint16_t>(in, out, size);
    case Uint: cpu::cast<int64_t, uint32_t>(in, out, size); break;
    case Ulong: cpu::cast<int64_t, uint64_t>(in, out, size); break;

    case Cint: cpu::cast<int64_t, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::cast<int64_t, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::cast<int64_t, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::cast<int64_t, std::complex<double>>(in, out, size); break;

    case Bool: cpu::cast<int64_t, bool>(in, out, size); break;
    }
    break;

  case Byte:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::cast<uint8_t, float>(in, out, size); break;
    case Double: cpu::cast<uint8_t, double>(in, out, size); break;

    case Sbyte: cpu::cast<uint8_t, int8_t>(in, out, size);
    case Short: cpu::cast<uint8_t, int16_t>(in, out, size); break;
    case Int: cpu::cast<uint8_t, int32_t>(in, out, size); break;
    case Long: cpu::cast<uint8_t, int64_t>(in, out, size); break;

    case Ushort: cpu::cast<uint8_t, uint16_t>(in, out, size);
    case Uint: cpu::cast<uint8_t, uint32_t>(in, out, size); break;
    case Ulong: cpu::cast<uint8_t, uint64_t>(in, out, size); break;

    case Cint: cpu::cast<uint8_t, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::cast<uint8_t, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::cast<uint8_t, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::cast<uint8_t, std::complex<double>>(in, out, size); break;

    case Bool: cpu::cast<uint8_t, bool>(in, out, size); break;
    }

  case Ushort:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::cast<uint16_t, float>(in, out, size); break;
    case Double: cpu::cast<uint16_t, double>(in, out, size); break;

    case Sbyte: cpu::cast<uint16_t, int8_t>(in, out, size);
    case Int: cpu::cast<uint16_t, int32_t>(in, out, size); break;
    case Long: cpu::cast<uint16_t, int64_t>(in, out, size); break;

    case Byte: cpu::cast<uint16_t, uint8_t>(in, out, size);
    case Ushort: cpu::cast<uint16_t, uint16_t>(in, out, size);
    case Uint: cpu::cast<uint16_t, uint32_t>(in, out, size); break;
    case Ulong: cpu::cast<uint16_t, uint64_t>(in, out, size); break;

    case Cint: cpu::cast<uint16_t, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::cast<uint16_t, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::cast<uint16_t, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::cast<uint16_t, std::complex<double>>(in, out, size); break;

    case Bool: cpu::cast<uint16_t, bool>(in, out, size); break;
    }
    break;

  case Uint:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::cast<uint32_t, float>(in, out, size); break;
    case Double: cpu::cast<uint32_t, double>(in, out, size); break;

    case Sbyte: cpu::cast<uint32_t, int8_t>(in, out, size);
    case Short: cpu::cast<uint32_t, int16_t>(in, out, size); break;
    case Int: cpu::cast<uint32_t, int32_t>(in, out, size); break;
    case Long: cpu::cast<uint32_t, int64_t>(in, out, size); break;

    case Byte: cpu::cast<uint32_t, uint8_t>(in, out, size);
    case Ushort: cpu::cast<uint32_t, uint16_t>(in, out, size);
    case Ulong: cpu::cast<uint32_t, uint64_t>(in, out, size); break;

    case Cint: cpu::cast<uint32_t, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::cast<uint32_t, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::cast<uint32_t, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::cast<uint32_t, std::complex<double>>(in, out, size); break;

    case Bool: cpu::cast<uint32_t, bool>(in, out, size); break;
    }
    break;

  case Ulong:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::cast<uint64_t, float>(in, out, size); break;
    case Double: cpu::cast<uint64_t, double>(in, out, size); break;

    case Sbyte: cpu::cast<uint64_t, int8_t>(in, out, size);
    case Short: cpu::cast<uint64_t, int16_t>(in, out, size); break;
    case Int: cpu::cast<uint64_t, int32_t>(in, out, size); break;
    case Long: cpu::cast<uint64_t, int64_t>(in, out, size); break;

    case Byte: cpu::cast<uint64_t, uint8_t>(in, out, size);
    case Ushort: cpu::cast<uint64_t, uint16_t>(in, out, size);
    case Uint: cpu::cast<uint64_t, uint32_t>(in, out, size); break;

    case Cint: cpu::cast<uint64_t, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::cast<uint64_t, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::cast<uint64_t, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::cast<uint64_t, std::complex<double>>(in, out, size); break;

    case Bool: cpu::cast<uint64_t, bool>(in, out, size); break;
    }
    break;

  case Bool:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::cast<bool, float>(in, out, size); break;
    case Double: cpu::cast<bool, double>(in, out, size); break;

    case Sbyte: cpu::cast<bool, int8_t>(in, out, size);
    case Short: cpu::cast<bool, int16_t>(in, out, size); break;
    case Int: cpu::cast<bool, int32_t>(in, out, size); break;
    case Long: cpu::cast<bool, int64_t>(in, out, size); break;

    case Byte: cpu::cast<bool, uint8_t>(in, out, size);
    case Ushort: cpu::cast<bool, uint16_t>(in, out, size);
    case Uint: cpu::cast<bool, uint32_t>(in, out, size); break;
    case Ulong: cpu::cast<bool, uint64_t>(in, out, size); break;

    case Cint: cpu::cast<bool, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::cast<bool, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::cast<bool, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::cast<bool, std::complex<double>>(in, out, size); break;
    }
    break;

  case Cint:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::ccast<std::complex<int32_t>, float>(in, out, size); break;
    case Double: cpu::ccast<std::complex<int32_t>, double>(in, out, size); break;

    case Sbyte: cpu::ccast<std::complex<int32_t>, int8_t>(in, out, size);
    case Short: cpu::ccast<std::complex<int32_t>, int16_t>(in, out, size); break;
    case Int: cpu::ccast<std::complex<int32_t>, int32_t>(in, out, size); break;
    case Long: cpu::ccast<std::complex<int32_t>, int64_t>(in, out, size); break;

    case Byte: cpu::ccast<std::complex<int32_t>, uint8_t>(in, out, size);
    case Ushort: cpu::ccast<std::complex<int32_t>, uint16_t>(in, out, size);
    case Uint: cpu::ccast<std::complex<int32_t>, uint32_t>(in, out, size); break;
    case Ulong: cpu::ccast<std::complex<int32_t>, uint64_t>(in, out, size); break;

    case Cint: cpu::ccast<std::complex<int32_t>, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::ccast<std::complex<int32_t>, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::ccast<std::complex<int32_t>, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::ccast<std::complex<int32_t>, std::complex<double>>(in, out, size); break;

    case Bool: cpu::ccast<std::complex<int32_t>, bool>(in, out, size); break;
    }
    break;


  case Cuint:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::ccast<std::complex<uint32_t>, float>(in, out, size); break;
    case Double: cpu::ccast<std::complex<uint32_t>, double>(in, out, size); break;

    case Sbyte: cpu::ccast<std::complex<uint32_t>, int8_t>(in, out, size);
    case Short: cpu::ccast<std::complex<uint32_t>, int16_t>(in, out, size); break;
    case Int: cpu::ccast<std::complex<uint32_t>, int32_t>(in, out, size); break;
    case Long: cpu::ccast<std::complex<uint32_t>, int64_t>(in, out, size); break;

    case Byte: cpu::ccast<std::complex<uint32_t>, uint8_t>(in, out, size);
    case Ushort: cpu::ccast<std::complex<uint32_t>, uint16_t>(in, out, size);
    case Uint: cpu::ccast<std::complex<uint32_t>, uint32_t>(in, out, size); break;
    case Ulong: cpu::ccast<std::complex<uint32_t>, uint64_t>(in, out, size); break;

    case Cint: cpu::ccast<std::complex<uint32_t>, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::ccast<std::complex<uint32_t>, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::ccast<std::complex<uint32_t>, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::ccast<std::complex<uint32_t>, std::complex<double>>(in, out, size); break;

    case Bool: cpu::ccast<std::complex<uint32_t>, bool>(in, out, size); break;
    }
    break;

  case Cfloat:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::ccast<std::complex<float>, float>(in, out, size); break;
    case Double: cpu::ccast<std::complex<float>, double>(in, out, size); break;

    case Sbyte: cpu::ccast<std::complex<float>, int8_t>(in, out, size);
    case Short: cpu::ccast<std::complex<float>, int16_t>(in, out, size); break;
    case Int: cpu::ccast<std::complex<float>, int32_t>(in, out, size); break;
    case Long: cpu::ccast<std::complex<float>, int64_t>(in, out, size); break;

    case Byte: cpu::ccast<std::complex<float>, uint8_t>(in, out, size);
    case Ushort: cpu::ccast<std::complex<float>, uint16_t>(in, out, size);
    case Uint: cpu::ccast<std::complex<float>, uint32_t>(in, out, size); break;
    case Ulong: cpu::ccast<std::complex<float>, uint64_t>(in, out, size); break;

    case Cint: cpu::ccast<std::complex<float>, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::ccast<std::complex<float>, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::ccast<std::complex<float>, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::ccast<std::complex<float>, std::complex<double>>(in, out, size); break;

    case Bool: cpu::ccast<std::complex<float>, bool>(in, out, size); break;
    }
    break;

  case Cdouble:
    switch (to) {
    case Half: throw std::runtime_error("Half is not supported");
    case Float: cpu::ccast<std::complex<double>, float>(in, out, size); break;
    case Double: cpu::ccast<std::complex<double>, double>(in, out, size); break;

    case Sbyte: cpu::ccast<std::complex<double>, int8_t>(in, out, size);
    case Short: cpu::ccast<std::complex<double>, int16_t>(in, out, size); break;
    case Int: cpu::ccast<std::complex<double>, int32_t>(in, out, size); break;
    case Long: cpu::ccast<std::complex<double>, int64_t>(in, out, size); break;

    case Byte: cpu::ccast<std::complex<double>, uint8_t>(in, out, size);
    case Ushort: cpu::ccast<std::complex<double>, uint16_t>(in, out, size);
    case Uint: cpu::ccast<std::complex<double>, uint32_t>(in, out, size); break;
    case Ulong: cpu::ccast<std::complex<double>, uint64_t>(in, out, size); break;

    case Cint: cpu::ccast<std::complex<double>, std::complex<int32_t>>(in, out, size); break;
    case Cuint: cpu::ccast<std::complex<double>, std::complex<uint32_t>>(in, out, size); break;
    case Cfloat: cpu::ccast<std::complex<double>, std::complex<float>>(in, out, size); break;
    case Cdouble: cpu::ccast<std::complex<double>, std::complex<double>>(in, out, size); break;

    case Bool: cpu::ccast<std::complex<double>, bool>(in, out, size); break;
    }
    break;

  default:
    throw std::runtime_error("invalid Dtype");
  }
}


inline Buffer cast(Buffer& in, const Dtype& from, const Dtype& to, size_t size) {
  if (from == to) return in;
  Buffer out(size * to.size());
  cast(in, out, from, to, size);
  return out;
}

}