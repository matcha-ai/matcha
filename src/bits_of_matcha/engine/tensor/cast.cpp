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

    case Cint:
    case Cuint:
    case Cfloat:
    case Cdouble:
      return Cdouble;
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

    case Cint:
      return Cint;
    case Cuint:
      return Cint;
    case Cfloat:
    case Cdouble:
      return Cdouble;
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
    case Cint:
    case Cuint:
      return Cint;
    case Cfloat:
    case Cdouble:
      return Cdouble;

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
    case Cint:
    case Cuint:
      return Cint;
    case Cfloat:
    case Cdouble:
      return Cdouble;
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
    case Cint:
    case Cuint:
      return Cint;
    case Cfloat:
    case Cdouble:
      return Cdouble;
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
    case Cint:
    case Cuint:
    case Cfloat:
    case Cdouble:
      return b;
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
    case Cint:
    case Cuint:
    case Cfloat:
    case Cdouble:
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
    case Cuint:
    case Cint:
      return b;
    case Cfloat:
    case Cdouble:
      return Cdouble;
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
    case Cint:
      return Cint;
    case Cuint:
      return Cuint;
    case Cfloat:
    case Cdouble:
      return Cdouble;
    default:
      return Ulong;
    }
    break;

  case Cint:
    switch (b) {
    case Float:
    case Double:
    case Cfloat:
    case Cdouble:
      return Cdouble;
    default:
      return Cint;
    }
    break;

  case Cuint:
    switch (b) {
    case Float:
    case Double:
    case Cfloat:
    case Cdouble:
      return Cdouble;
    case Cint:
    case Int:
    case Short:
    case Sbyte:
      return Cint;
    default:
      return Cuint;
    }
    break;

  case Cfloat:
    switch (b) {
    case Double:
    case Cdouble:
    case Cint:
    case Int:
    case Uint:
    case Cuint:
      return Cdouble;
    default:
      return Cfloat;
    }
    break;

  case Cdouble:
    return Cdouble;

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