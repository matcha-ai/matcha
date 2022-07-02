#pragma once

#include "bits_of_matcha/error/Error.h"
#include "bits_of_matcha/Frame.h"

#include <map>

namespace matcha {

struct IncompatibleShapesError : public Error {
  explicit IncompatibleShapesError(const Shape& a, const Shape& b, const std::pair<int, int>& loci = {});
  explicit IncompatibleShapesError() = default;
};

}