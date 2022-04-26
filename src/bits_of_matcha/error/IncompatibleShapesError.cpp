#include "bits_of_matcha/error/IncompatibleShapesError.h"

#include <sstream>


namespace matcha {

std::string incompatibleShapesMessage(const Shape& a, const Shape& b, const std::pair<int, int>& loci) {
  std::string buffer = "Shapes are incompatible: ";
  int offset1, offset2;
  int width1, width2;
  buffer += "[";
  for (int i = 0; i < a.rank(); i++) {
    if (i != 0) buffer += ", ";
    std::string dim = std::to_string(a[i]);
    if (i == loci.first) {
      offset1 = (int) buffer.size() - 1;
      width1 = (int) dim.size();
    }
    buffer += dim;
  }
  buffer += "]";
  buffer += " and ";
  buffer += "[";
  for (int i = 0; i < b.rank(); i++) {
    if (i != 0) buffer += ", ";
    std::string dim = std::to_string(b[i]);
    if (i == loci.second) {
      offset2 = (int) buffer.size() - 1;
      width2 = (int) dim.size();
    }
    buffer += dim;
  }
  buffer += "]";

  return buffer;
}

IncompatibleShapesError::IncompatibleShapesError(const Shape& a, const Shape& b, const std::pair<int, int>& loci)
  : Error(incompatibleShapesMessage(a, b, loci))
{}


}