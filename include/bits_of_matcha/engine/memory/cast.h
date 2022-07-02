#pragma once

#include "bits_of_matcha/engine/memory/Buffer.h"
#include "bits_of_matcha/engine/cpu/kernels/cast.h"

#include <execution>
#include <numeric>
#include <algorithm>

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