#include "bits_of_matcha/Dtype.h"

#include <stdexcept>
#include <map>


namespace matcha {


Dtype::Dtype(unsigned dtype)
  : dtype_{dtype}
{
  switch (dtype_) {
    case Half:
    case Float:
    case Double:
    case ShortShort:
    case Short:
    case Int:
    case Long:
    case Bool:
      break;
    default:
      throw std::invalid_argument("invalid dtype");
  }
}

Dtype::Dtype(const std::string& dtype) {
  std::string lower;
  std::transform(dtype.begin(), dtype.end(), std::back_inserter(lower), tolower);
  static std::map<std::string, unsigned> dtypeTable = {
    {"half", Half},
    {"float16", Half},
    {"float", Float},
    {"float32", Float},
    {"double", Double},
    {"float64", Double},
    {"shortshort", ShortShort},
    {"short short", ShortShort},
    {"int8", ShortShort},
    {"short", Short},
    {"int16", Short},
    {"int", Int},
    {"int32", Int},
    {"long", Long},
    {"int64", Long},
    {"bool", Bool},
  };

  try {
    dtype_ = dtypeTable.at(dtype);
  } catch (std::exception& e) {
    throw std::invalid_argument("invalid type");
  }
}

std::string Dtype::string() const {
  switch (dtype_) {
  case Half: return "Half";
  case Float: return "Float";
  case Double: return "Double";
  case ShortShort: return "ShortShort";
  case Short: return "Short";
  case Int: return "Int";
  case Long: return "Long";
  case Bool: return "Bool";
  default:  throw std::runtime_error("unknown dtype");
  }
}

size_t Dtype::size() const {
  switch (dtype_) {
  case Half: return 2;
  case Float: return 4;
  case Double: return 8;
  case ShortShort: return 1;
  case Short: return 2;
  case Int: return 4;
  case Long: return 8;
  case Bool: return 1;
  default:  throw std::runtime_error("invalid dtype");
  }
}

Dtype::operator unsigned() const {
  return dtype_;
}

std::ostream& operator<<(std::ostream& os, const Dtype& dtype) {
  os << dtype.string();
  return os;
}

}