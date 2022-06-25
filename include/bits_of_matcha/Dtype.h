#pragma once

#include <string>


namespace matcha {

class Dtype {
public:

  enum {
    Half,
    Float,
    Double,

    ShortShort,
    Short,
    Int,
    Long,

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
      return ShortShort;
    case 2:
      return Short;
    default:
    case 4:
      return Int;
    case 8:
      return Long;
    }
  }

private:
  unsigned dtype_;

};

enum {
  Half = Dtype::Half,
  Float = Dtype::Float,
  Double = Dtype::Double,

  ShortShort = Dtype::ShortShort,
  Short = Dtype::Short,
  Int = Dtype::Int,
  Long = Dtype::Long,

  Bool = Dtype::Bool,
};

std::ostream& operator<<(std::ostream& os, const Dtype& dtype);

}
