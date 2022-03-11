#include "bits_of_matcha/dtype.h"

#include <stdexcept>
#include <map>


namespace matcha {


Dtype::Dtype(unsigned dtype)
  : dtype_{dtype}
{
  switch (dtype_) {
    case 0: // Float
      break;
    default: throw std::invalid_argument("unknown dtype");
  }
}

Dtype::Dtype(const std::string& dtype) {
  static std::map<std::string, unsigned> dtypeTable = {
    {"Float", Float}
  };

  try {
    dtype_ = dtypeTable.at(dtype);
  } catch (std::exception& e) {
    throw std::invalid_argument("unknown type");
  }
}

std::string Dtype::string() const {
  switch (dtype_) {
    case 0:   return "Float";
    default:  throw std::runtime_error("unknown dtype");
  }
}

size_t Dtype::size() const {
  switch (dtype_) {
    case 0:   return 4;
    default:  throw std::runtime_error("unknown type");
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