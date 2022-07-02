#include "bits_of_matcha/error/ImplicitCastError.h"


namespace matcha {

ImplicitCastError::ImplicitCastError(const Dtype& from, const Dtype& to)
  : Error("Implicit cast from " + from.string() +
          " to " + to.string() + " is not allowed. Use explicit cast." )
{}

}