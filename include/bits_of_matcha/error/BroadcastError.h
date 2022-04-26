#pragma once

#include "bits_of_matcha/error/IncompatibleShapesError.h"

namespace matcha {

struct BroadcastError : IncompatibleShapesError {
  BroadcastError(const Shape& a, const Shape& b, int dim);
  BroadcastError() = default;
};

}