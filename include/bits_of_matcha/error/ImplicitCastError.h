#pragma once

#include "bits_of_matcha/error/Error.h"
#include "bits_of_matcha/Dtype.h"

namespace matcha {

struct ImplicitCastError : Error {
  explicit ImplicitCastError(const Dtype& from, const Dtype& to);
};

}