#include "bits_of_matcha/engine/tensor/cast.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"


namespace matcha::engine {

inline void errHalf() {
  throw std::runtime_error("unsupported Dtype: Half");
}

Dtype promoteDtypes(Dtype a, Dtype b) {
  switch (a) {
  case Half: errHalf();

  case Float:
    switch (b) {
    case Half: errHalf();
    case Bool:
    case Sbyte:
    case Byte:
    case Short:
    case Ushort:
    case Float:
      return Float;
    case Int:
    case Uint:
    case Long:
    case Ulong:
    case Double:
      return Double;
    }
    break;

  case Double:
    return Double;

  case Sbyte:
    switch (b) {
    case Half: errHalf();
    case Bool:
    case Sbyte:
      return Sbyte;
    case Byte:
      return Short;
    case Short:
    case Int:
    case Long:
    case Float:
    case Double:
      return b;
    case Ushort:
      return Int;
    case Uint:
    case Ulong:
      return Long;
    }
    break;

  case Short:
    switch (b) {
    case Half: errHalf();
    case Bool:
    case Sbyte:
    case Short:
      return Short;
    case Int:
    case Long:
    case Float:
    case Double:
      return b;
    case Byte:
      return Short;
    case Ushort:
      return Int;
    case Uint:
    case Ulong:
      return Long;
    }
    break;

  case Int:
    switch (b) {
    case Half: errHalf();
    case Bool:
    case Sbyte:
    case Short:
    case Int:
      return Int;
    case Long:
      return Long;
    case Float:
    case Double:
      return Double;
    case Byte:
    case Ushort:
      return Int;
    case Uint:
    case Ulong:
      return Long;
    }
    break;

  case Long:
    switch (b) {
    case Half: errHalf();
    case Float:
    case Double:
      return Double;
    default:
      return Long;
    }
    break;

  case Byte:
    switch (b) {
    case Half: errHalf();
    case Bool:
      return Byte;
    case Sbyte:
      return Short;
    case Byte:
    case Short:
    case Int:
    case Long:
    case Float:
    case Double:
    case Ushort:
    case Uint:
    case Ulong:
      return Ulong;
    }
    break;

  case Ushort:
    switch (b) {
    case Half: errHalf();
    case Bool:
      return Ushort;
    case Sbyte:
    case Short:
      return Int;
    case Int:
    case Long:
    case Float:
    case Double:
      return b;
    case Byte:
    case Ushort:
      return Ushort;
    case Uint:
    case Ulong:
      return b;
    }
    break;

  case Uint:
    switch (b) {
    case Half: errHalf();
    case Bool:
      return Uint;
    case Sbyte:
    case Short:
      return Int;
    case Int:
    case Long:
      return Long;
    case Float:
    case Double:
      return Double;
    case Byte:
    case Ushort:
    case Uint:
      return Uint;
    case Ulong:
      return Ulong;
    }
    break;

  case Ulong:
    switch (b) {
    case Half: errHalf();
    case Float:
    case Double:
      return Double;
    case Sbyte:
    case Short:
    case Int:
    case Long:
      return Long;
    default:
      return Ulong;
    }
    break;

  case Bool:
    switch (b) {
    case Half: errHalf();
    default:
      return b;
    }
  }

  throw std::runtime_error("unknown Dtype");
}

Dtype promoteDtypes(Tensor* a, Tensor* b) {
  return promoteDtypes(a->dtype(), b->dtype());
}

}