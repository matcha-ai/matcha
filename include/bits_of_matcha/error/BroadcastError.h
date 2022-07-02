#pragma once

#include "bits_of_matcha/error/IncompatibleShapesError.h"

namespace matcha {

struct BroadcastError : IncompatibleShapesError {
  explicit BroadcastError(const Shape& a, const Shape& b, int dim = 0);
  explicit BroadcastError() = default;
};

}