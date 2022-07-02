#pragma once

#include <string>


namespace matcha {

class Dtype {
public:

  enum {
    Half,
    Float,
    Double,

    Sbyte,
    Short,
    Int,
    Long,

    Byte,
    Ushort,
    Uint,
    Ulong,

    Bool,
  };

public:
  Dtype(unsigned dtype);
  Dtype(const std::string& dtype);

  std::string string() const;
  size_t size() const;

  operator unsigned() const;

public:
  static constexpr unsigned getSystemFloat() {
    switch (sizeof(float)) {
    case 2:
      return Half;
    default:
    case 4:
      return Float;
    case 8:
      return Double;
    }
  }

  static constexpr unsigned getSystemInt() {
    switch (sizeof(int)) {
    case 1:
      return Sbyte;
    case 2:
      return Short;
    default:
    case 4:
      return Int;
    case 8:
      return Long;
    }
  }

  static constexpr unsigned getSystemUint() {
    switch (sizeof(int)) {
    case 1:
      return Byte;
    case 2:
      return Ushort;
    default:
    case 4:
      return Uint;
    case 8:
      return Ulong;
    }
  }

private:
  unsigned dtype_;

};

enum {
  Half = Dtype::Half,
  Float = Dtype::Float,
  Double = Dtype::Double,

  Sbyte = Dtype::Sbyte,
  Short = Dtype::Short,
  Int = Dtype::Int,
  Long = Dtype::Long,

  Byte = Dtype::Byte,
  Ushort = Dtype::Ushort,
  Uint = Dtype::Uint,
  Ulong = Dtype::Ulong,

  Bool = Dtype::Bool,
};

std::ostream& operator<<(std::ostream& os, const Dtype& dtype);

}
