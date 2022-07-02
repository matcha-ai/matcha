#pragma once

#include "bits_of_matcha/error/Error.h"
#include "bits_of_matcha/Frame.h"

#include <map>

namespace matcha {

struct IncompatibleDtypesError : public Error {
  explicit IncompatibleDtypesError(const Dtype& a, const Dtype& b);
  explicit IncompatibleDtypesError() = default;
};

}
